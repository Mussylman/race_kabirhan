"""
Fine-tune CLIP for jockey color classification.

This script fine-tunes only the image encoder's last layers + a linear probe
on top of CLIP features. This is much more data-efficient than training
EfficientNet from scratch.

Strategy:
  1. Extract CLIP features from all crops (frozen backbone)
  2. Train a linear classifier on top (fast, <1 min)
  3. Optionally fine-tune last 2 transformer blocks (if enough data)

Requires: pip install transformers torch torchvision

Usage:
    # Linear probe (recommended with <500 samples):
    python tools/train_clip_classifier.py --data data/torso_crops_v2

    # Fine-tune last layers (recommended with 500+ samples):
    python tools/train_clip_classifier.py --data data/torso_crops_v2 --finetune

    # Use SigLIP instead of CLIP:
    python tools/train_clip_classifier.py --data data/torso_crops_v2 --backend siglip
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from pathlib import Path
from collections import Counter

MODEL_SAVE_PATH = "models/clip_color_classifier.pt"


class TorsoDataset(Dataset):
    def __init__(self, root_dir, processor, backend="clip"):
        self.root_dir = root_dir
        self.processor = processor
        self.backend = backend
        self.samples = []
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and d != "unsorted"
        ])
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
        inputs = self.processor(images=image, return_tensors="pt")
        # Squeeze batch dim added by processor
        pixel_values = inputs["pixel_values"].squeeze(0)
        return pixel_values, label


class CLIPLinearProbe(nn.Module):
    """Linear classifier on frozen CLIP features."""
    def __init__(self, clip_model, num_classes, feature_dim, finetune_layers=0):
        super().__init__()
        self.clip_model = clip_model
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, num_classes),
        )
        self.finetune_layers = finetune_layers

        # Freeze all CLIP parameters by default
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Optionally unfreeze last N transformer blocks
        if finetune_layers > 0:
            self._unfreeze_last_layers(finetune_layers)

    def _unfreeze_last_layers(self, n):
        """Unfreeze last N encoder layers."""
        vision_model = None
        if hasattr(self.clip_model, 'vision_model'):
            vision_model = self.clip_model.vision_model
        elif hasattr(self.clip_model, 'visual'):
            vision_model = self.clip_model.visual

        if vision_model is None:
            return

        # Find encoder layers
        if hasattr(vision_model, 'encoder') and hasattr(vision_model.encoder, 'layers'):
            layers = vision_model.encoder.layers
        elif hasattr(vision_model, 'transformer') and hasattr(vision_model.transformer, 'resblocks'):
            layers = vision_model.transformer.resblocks
        else:
            return

        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, pixel_values):
        with torch.set_grad_enabled(self.finetune_layers > 0):
            features = self.clip_model.get_image_features(pixel_values=pixel_values)
        features = features / features.norm(dim=-1, keepdim=True)
        return self.classifier(features)


def train(args):
    print("=" * 60)
    print("CLIP COLOR CLASSIFIER — Fine-tuning")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Backend: {args.backend}")
    print(f"Mode: {'fine-tune' if args.finetune else 'linear probe'}")

    # Load CLIP/SigLIP
    if args.backend == "clip":
        from transformers import CLIPModel, CLIPProcessor
        model_id = "openai/clip-vit-base-patch32"
        clip_model = CLIPModel.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id)
        feature_dim = 512
    elif args.backend == "siglip":
        from transformers import AutoModel, AutoProcessor
        model_id = "google/siglip2-base-patch16-224"
        clip_model = AutoModel.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        feature_dim = 768
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    clip_model = clip_model.to(device)

    # Dataset
    dataset = TorsoDataset(args.data, processor, args.backend)
    classes = dataset.classes
    num_classes = len(classes)

    print(f"Classes: {classes}")
    print(f"Total samples: {len(dataset)}")
    label_counts = Counter([label for _, label in dataset.samples])
    for cls_name, cls_idx in dataset.class_to_idx.items():
        print(f"  {cls_name}: {label_counts[cls_idx]}")

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    # Model
    finetune_layers = 2 if args.finetune else 0
    model = CLIPLinearProbe(
        clip_model, num_classes, feature_dim, finetune_layers
    ).to(device)

    # Only train unfrozen params
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(args.epochs):
        # Train
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        train_acc = 100 * correct / total
        scheduler.step()

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0

        print(f"Epoch {epoch+1:2d}/{args.epochs} | "
              f"Loss: {total_loss/len(train_loader):.3f} | "
              f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'classifier_state_dict': model.classifier.state_dict(),
                'classes': classes,
                'backend': args.backend,
                'feature_dim': feature_dim,
                'finetune': args.finetune,
            }, MODEL_SAVE_PATH)

    print(f"\nBest Val Accuracy: {best_acc:.1f}%")
    print(f"Model saved: {MODEL_SAVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune CLIP for color classification")
    parser.add_argument("--data", default="data/torso_crops_v2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune", action="store_true", help="Fine-tune last 2 layers")
    parser.add_argument("--backend", default="clip", choices=["clip", "siglip"])
    args = parser.parse_args()
    train(args)
