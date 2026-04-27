import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = datasets.ImageFolder("dataset_cropped/train",  transform=transform)
loader  = DataLoader(dataset, batch_size=4, num_workers=0)

images, labels = next(iter(loader))

print(f"Image shape: {images.shape}")
print(f"Image min: {images.min():.3f} | max: {images.max():.3f}")
print(f"Labels: {labels}")
print(f"Classes: {dataset.classes}")
print(f"Label distribution sample: {[dataset.classes[l] for l in labels]}")
