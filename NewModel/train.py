import torch
import sys
from transformers import MobileViTForImageClassification
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def choose_num_workers():
    # In notebook/IPython on Windows, multiprocessing workers often fail to spawn.
    in_interactive = hasattr(sys, "ps1") or "ipykernel" in sys.modules
    return 0 if in_interactive else 4

if __name__ == '__main__':

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Training on: {device}")

    # Augmentation applied only to training images.
    # Val/test use the same deterministic pipeline as before.
    #
    # NOTE: At inference time the upstream hand-cropper often produces looser
    # boxes that include extra wrist/forearm below the hand, so the hand is
    # smaller and shifted within the frame relative to our tight training
    # crops. We simulate this with:
    #   1) Pad + RandomCrop  → loose / off-center framing with border padding
    #   2) RandomResizedCrop → varying zoom levels (some tight, some loose)
    #   3) Stronger RandomAffine scale/translate ranges
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=3),
        # Geometry — hand signs are orientation-sensitive, so keep flips/rotations mild
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        # Simulate looser bounding boxes (hand smaller in frame + shifted).
        # Pad up to ~30% on each side with edge replication, then crop back to 256.
        # This reproduces the "extra forearm/wrist" framing seen at inference.
        transforms.RandomApply([
            transforms.RandomCrop(
                size=256,
                padding=(40, 40, 40, 80),  # left, top, right, bottom — extra at bottom for forearm
                padding_mode="edge",
            ),
        ], p=0.7),
        # Random zoom-out/zoom-in: scale<1 means hand occupies less of the frame.
        transforms.RandomResizedCrop(
            size=256,
            scale=(0.6, 1.0),
            ratio=(0.85, 1.15),
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.15),
            scale=(0.75, 1.1),
            fill=0,
        ),
        # Photometric — simulate varying lighting / skin tones
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),          # very occasionally fully gray
        # Mild blur to simulate motion / focus variation
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder("dataset_cropped/train", transform=train_transform)
    val_dataset   = datasets.ImageFolder("dataset_cropped/valid",  transform=val_transform)
    test_dataset  = datasets.ImageFolder("dataset_cropped/test",   transform=val_transform)

    num_workers = choose_num_workers()
    pin_memory = device.type == "cuda"
    print(f"DataLoader workers: {num_workers} | pin_memory: {pin_memory}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Sanity-check that all three splits agree on class order — mismatched class
    # indices would silently produce garbage test accuracy.
    assert train_dataset.classes == val_dataset.classes == test_dataset.classes, (
        "Class order mismatch between splits:\n"
        f"  train: {train_dataset.classes}\n"
        f"  valid: {val_dataset.classes}\n"
        f"  test:  {test_dataset.classes}"
    )

    print(f"Classes ({len(train_dataset.classes)}): {train_dataset.classes}")
    print(
        f"Train images: {len(train_dataset)} | "
        f"Val images: {len(val_dataset)} | "
        f"Test images: {len(test_dataset)}"
    )

    num_classes = len(train_dataset.classes)

    model = MobileViTForImageClassification.from_pretrained(
        "apple/mobilevit-xx-small",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    model = model.to(device)

    # Label smoothing discourages the model from making over-confident
    # predictions, which is the main symptom of memorizing a small dataset.
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Phase 1: freeze backbone, train classifier head only ──────────────────
    print("\nPhase 1 — Freezing backbone, training head only...")

    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-2,
        weight_decay=1e-2,
    )

    for epoch in range(10):
        model.train()
        correct, total, running_loss = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        print(f"  Epoch {epoch+1}/10 | Loss: {running_loss/len(train_loader):.3f} | Acc: {correct/total*100:.1f}%")

    # ── Phase 2: unfreeze all layers, full fine-tuning ────────────────────────
    print("\nPhase 2 — Unfreezing all layers, full fine-tuning...")

    for param in model.parameters():
        param.requires_grad = True

    # Stronger weight decay during full fine-tuning is the main lever against
    # memorization on a small dataset.
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=5e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)

    best_val_acc = 0

    for epoch in range(20):
        model.train()
        correct, total, running_loss = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            # Gradient clipping stabilizes training and acts as a mild regularizer.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total * 100
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_correct, val_total, val_running_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(pixel_values=images).logits
                val_running_loss += criterion(outputs, labels).item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total * 100
        val_loss = val_running_loss / len(val_loader)

        scheduler.step()

        # Logging both losses makes overfitting visible even when both accs hit 100%:
        # a healthy run has val_loss ≈ train_loss; overfit runs have val_loss ≫ train_loss.
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            marker = " ← best saved"
        print(
            f"  Epoch {epoch+1}/20 | "
            f"Train loss {train_loss:.3f} acc {train_acc:.1f}% | "
            f"Val loss {val_loss:.3f} acc {val_acc:.1f}%{marker}"
        )

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.1f}%")
    print(f"Class order: {train_dataset.classes}")

    # ── Held-out test evaluation ─────────────────────────────────────────
    # Re-load the best checkpoint (saved at the best val epoch) so the test
    # number reflects the model we'd actually deploy, not the last-epoch model.
    print("\nEvaluating best checkpoint on held-out test set...")
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    classes = train_dataset.classes
    num_classes = len(classes)
    per_class_correct = [0] * num_classes
    per_class_total = [0] * num_classes
    test_correct, test_total, test_running_loss = 0, 0, 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(pixel_values=images).logits
            test_running_loss += criterion(outputs, labels).item()
            preds = outputs.argmax(1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
            for label, pred in zip(labels.tolist(), preds.tolist()):
                per_class_total[label] += 1
                if label == pred:
                    per_class_correct[label] += 1

    test_acc = test_correct / test_total * 100
    test_loss = test_running_loss / len(test_loader)
    print(f"Test loss: {test_loss:.3f} | Test acc: {test_acc:.1f}% ({test_correct}/{test_total})")
    print("Per-class test accuracy:")
    for cls, c, t in zip(classes, per_class_correct, per_class_total):
        acc = (c / t * 100) if t > 0 else 0.0
        print(f"  {cls:<20s} {acc:5.1f}%  ({c}/{t})")
