#!/usr/bin/env python3
import os
import torch
import cv2
from PIL import Image
from torchvision import transforms
import torch.nn as nn

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

def main():
    print("Loading ColorCNN...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load("models/color_classifier.pt", map_location=device)
    model = SimpleColorCNN(num_classes=5)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    color_names = ckpt["classes"]
    print(f"Classes: {color_names}")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_images = [
        "crop_kamera_03_p0.jpg", "crop_kamera_03_p1.jpg", 
        "crop_kamera_05_p0.jpg", "crop_kamera_20_p0.jpg",
        "test_detection.jpg"
    ]
    
    print("\nEvaluating existing crop images with the model:")
    with torch.no_grad():
        for img_name in test_images:
            if not os.path.exists(img_name):
                continue
                
            image = Image.open(img_name).convert('RGB')
            tensor = transform(image).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            
            top_conf, top_idx = torch.max(probs, 0)
            color = color_names[top_idx.item()]
            
            print(f"File: {img_name:<25} -> Pred: {color:<10} (Conf: {top_conf:.2f})")
            for i, c in enumerate(color_names):
                print(f"  - {c}: {probs[i]:.2f}")

if __name__ == "__main__":
    main()
