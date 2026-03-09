"""
Train EfficientNet-V2-S color classifier on torso crops.

Replaces SimpleColorCNN with pretrained EfficientNet-V2-S for better accuracy.
Strategy: freeze backbone 5 epochs, then unfreeze with differential LR.

Usage:
    python tools/train_color_classifier.py
    python tools/train_color_classifier.py --data data/torso_crops_v2 --epochs 30
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import numpy as np
from multiprocessing import freeze_support
from collections import Counter

# Defaults
DATA_PATH = "data/torso_crops_v2"
MODEL_SAVE_PATH = "models/color_classifier_v2.pt"
ONNX_SAVE_PATH = "models/color_classifier_v2.onnx"
BATCH_SIZE = 32
EPOCHS = 30
IMG_SIZE = 128
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
WEIGHT_DECAY = 1e-4
FREEZE_EPOCHS = 5


class TorsoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d))
                               and d != "unsorted"])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((
                        os.path.join(class_dir, img_name),
                        self.class_to_idx[class_name]
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def build_model(num_classes):
    """EfficientNet-V2-S with custom classifier head."""
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_sampler_weights(dataset):
    """Compute sample weights for balanced sampling."""
    labels = [label for _, label in dataset.samples]
    class_counts = Counter(labels)
    total = len(labels)
    class_weights = {cls: total / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels]
    return sample_weights


def train(args):
    print("=" * 60)
    print("COLOR CLASSIFIER TRAINING — EfficientNet-V2-S")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
        transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset
    full_dataset = TorsoDataset(args.data, transform=train_transform)
    classes = full_dataset.classes
    print(f"Classes: {classes}")
    print(f"Total samples: {len(full_dataset)}")

    # Print per-class counts
    label_counts = Counter([label for _, label in full_dataset.samples])
    for cls_name, cls_idx in full_dataset.class_to_idx.items():
        print(f"  {cls_name}: {label_counts[cls_idx]}")

    # Split train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Val uses non-augmented transform
    val_dataset_wrapper = []
    for idx in val_dataset.indices:
        img_path, label = full_dataset.samples[idx]
        val_dataset_wrapper.append((img_path, label))

    class ValDataset(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label

    val_ds = ValDataset(val_dataset_wrapper, val_transform)

    # Weighted sampler for training (balance classes)
    train_labels = [full_dataset.samples[idx][1] for idx in train_dataset.indices]
    train_class_counts = Counter(train_labels)
    train_weights = [1.0 / train_class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights))

    print(f"Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)

    # Model
    num_classes = len(classes)
    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    best_acc = 0

    print(f"\nPhase 1: Freeze backbone, train head ({FREEZE_EPOCHS} epochs)")

    # Phase 1: Freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.classifier.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

    for epoch in range(FREEZE_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_cm = validate(model, val_loader, device, num_classes)

        print(f"Epoch {epoch+1:2d}/{FREEZE_EPOCHS} | "
              f"Loss: {train_loss:.3f} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model, classes, MODEL_SAVE_PATH)

    # Phase 2: Unfreeze all, differential LR
    print(f"\nPhase 2: Full fine-tune ({EPOCHS - FREEZE_EPOCHS} epochs)")

    for param in model.features.parameters():
        param.requires_grad = True

    optimizer = optim.Adam([
        {'params': model.features.parameters(), 'lr': LR_BACKBONE},
        {'params': model.classifier.parameters(), 'lr': LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    for epoch in range(FREEZE_EPOCHS, args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_cm = validate(model, val_loader, device, num_classes)
        scheduler.step()

        lr_bb = optimizer.param_groups[0]['lr']
        lr_hd = optimizer.param_groups[1]['lr']
        print(f"Epoch {epoch+1:2d}/{args.epochs} | "
              f"Loss: {train_loss:.3f} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | "
              f"LR: bb={lr_bb:.6f} hd={lr_hd:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model, classes, MODEL_SAVE_PATH)

    print()
    print("=" * 60)
    print(f"TRAINING COMPLETE!")
    print(f"Best Val Accuracy: {best_acc:.1f}%")
    print(f"Model saved: {MODEL_SAVE_PATH}")
    print(f"Classes: {classes}")

    # Print confusion matrix for best model
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True)
    model_best = build_model(num_classes).to(device)
    model_best.load_state_dict(checkpoint['model_state_dict'])
    _, val_cm = validate(model_best, val_loader, device, num_classes)
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(f"{'':>10}", end="")
    for c in classes:
        print(f"{c:>10}", end="")
    print()
    for i, c in enumerate(classes):
        print(f"{c:>10}", end="")
        for j in range(num_classes):
            print(f"{val_cm[i][j]:>10}", end="")
        print()

    print(f"\nExport ONNX:")
    print(f"  python tools/train_color_classifier.py --export")
    print("=" * 60)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, device, num_classes):
    model.eval()
    correct = 0
    total = 0
    cm = [[0] * num_classes for _ in range(num_classes)]

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for t, p in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                cm[t][p] += 1

    acc = 100. * correct / total if total > 0 else 0
    return acc, cm


def save_model(model, classes, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'architecture': 'efficientnet_v2_s',
        'classes': classes,
        'img_size': IMG_SIZE,
    }, path)


def export_onnx(args):
    print("Exporting to ONNX...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True)
    classes = checkpoint['classes']
    num_classes = len(classes)

    model = build_model(num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    torch.onnx.export(
        model, dummy, ONNX_SAVE_PATH,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch'},
            'output': {0: 'batch'},
        },
        opset_version=16,
    )
    print(f"Exported: {ONNX_SAVE_PATH}")
    print(f"Input: [N, 3, {IMG_SIZE}, {IMG_SIZE}]")
    print(f"Output: [N, {num_classes}]")
    print(f"Classes: {classes}")


if __name__ == "__main__":
    freeze_support()

    parser = argparse.ArgumentParser(description="Train EfficientNet-V2-S color classifier")
    parser.add_argument("--data", default=DATA_PATH, help="Data directory with color folders")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--export", action="store_true", help="Export to ONNX")
    args = parser.parse_args()

    if args.export:
        export_onnx(args)
    else:
        train(args)
