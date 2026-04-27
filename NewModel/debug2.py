import torch
from transformers import MobileViTForImageClassification
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = datasets.ImageFolder("dataset_cropped/train",  transform=transform)
loader  = DataLoader(dataset, batch_size=4, num_workers=0, shuffle=True)

num_classes = len(dataset.classes)

model = MobileViTForImageClassification.from_pretrained(
    "apple/mobilevit-xx-small",
    num_labels=num_classes,
    ignore_mismatched_sizes=True
)

images, labels = next(iter(loader))
outputs = model(pixel_values=images).logits

print(f"Output shape: {outputs.shape}")
print(f"Output sample: {outputs[0].detach()}")
print(f"Predicted classes: {outputs.argmax(1)}")
print(f"Actual labels: {labels}")
print(f"Classes: {dataset.classes}")
