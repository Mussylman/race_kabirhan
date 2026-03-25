#!/usr/bin/env python3
"""Train SimpleCNN color classifier v3 on cleaned dataset."""

import sys
import random
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

# ── Config ──
DATASET_DIR = Path("data/color_dataset")
COLORS = ["blue", "green", "purple", "red", "yellow"]
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001
VAL_SPLIT = 0.2
SAVE_PATH = "models/color_classifier_v3.pt"
ONNX_PATH = "models/color_classifier_v3.onnx"


# ── Model (same as v1) ──
class SimpleColorCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Dataset ──
class ColorDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Augmentation ──
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.3, hue=0.05),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.7, 1.3)),
    transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_samples():
    samples = []
    for idx, color in enumerate(COLORS):
        folder = DATASET_DIR / color
        if not folder.exists():
            continue
        files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        for f in files:
            samples.append((f, idx))
    return samples


def split_data(samples):
    random.seed(42)
    by_class = {i: [] for i in range(len(COLORS))}
    for path, label in samples:
        by_class[label].append((path, label))

    train, val = [], []
    for label, items in by_class.items():
        random.shuffle(items)
        n_val = max(1, int(len(items) * VAL_SPLIT))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


def train():
    samples = load_samples()
    print(f"Dataset: {DATASET_DIR}")
    print(f"Total: {len(samples)} samples")
    for i, c in enumerate(COLORS):
        n = sum(1 for _, l in samples if l == i)
        print(f"  {c:>8}: {n}")

    train_samples, val_samples = split_data(samples)
    print(f"\nTrain: {len(train_samples)}, Val: {len(val_samples)}")

    train_ds = ColorDataset(train_samples, train_transform)
    val_ds = ColorDataset(val_samples, val_transform)

    # Weighted sampler for balance
    train_labels = [s[1] for s in train_samples]
    counts = Counter(train_labels)
    weights = [1.0 / counts[l] for l in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SimpleColorCNN(num_classes=len(COLORS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            train_correct += (out.argmax(1) == labels).sum().item()
            train_total += len(labels)
        scheduler.step()

        # Val
        model.eval()
        val_correct, val_total = 0, 0
        per_class = {i: {"correct": 0, "total": 0} for i in range(len(COLORS))}
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                preds = out.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)
                for p, l in zip(preds, labels):
                    per_class[l.item()]["total"] += 1
                    if p == l:
                        per_class[l.item()]["correct"] += 1

        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        if (epoch + 1) % 5 == 0 or epoch == 0 or val_acc > best_acc:
            per_str = " | ".join(
                f"{COLORS[i][:3]}:{100*per_class[i]['correct']/max(1,per_class[i]['total']):.0f}%"
                for i in range(len(COLORS))
            )
            marker = " *" if val_acc > best_acc else ""
            print(f"  epoch {epoch+1:3d}/{EPOCHS} | train={train_acc:.1f}% | val={val_acc:.1f}%{marker} | {per_str}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": COLORS,
                "architecture": "simple_cnn",
                "img_size": IMG_SIZE,
                "best_val_acc": best_acc,
            }, SAVE_PATH)

    print(f"\nBest val accuracy: {best_acc:.1f}%")
    print(f"Saved: {SAVE_PATH}")

    # Export ONNX
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    torch.onnx.export(
        model, dummy, ONNX_PATH,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX: {ONNX_PATH}")

    # Build TRT engine
    import subprocess
    r = subprocess.run(
        ["/tmp/build_engine", ONNX_PATH,
         SAVE_PATH.replace(".pt", ".engine"), "--fp16"],
        capture_output=True, text=True, cwd="models"
    )
    if r.returncode == 0:
        print(f"Engine: {SAVE_PATH.replace('.pt', '.engine')}")
    else:
        print(f"Engine build failed: {r.stderr}")


if __name__ == "__main__":
    train()
