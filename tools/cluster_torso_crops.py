"""
Cluster torso crops by dominant color using EfficientNet-V2-S features + KMeans.
Creates subfolders (cluster_00..cluster_N) for easy manual review.
"""
import os
import sys
import shutil
import numpy as np
from pathlib import Path
from PIL import Image

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/torso_crops_v2/unsorted")
    parser.add_argument("--output", default="data/torso_crops_v2/clustered")
    parser.add_argument("--n-clusters", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in ('.jpg', '.png')])
    print(f"Found {len(files)} images")

    if not files:
        return

    # Load EfficientNet-V2-S for feature extraction
    import torch
    from torchvision import models, transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    model.classifier = torch.nn.Identity()  # remove classification head
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Extract features
    print("Extracting features...")
    all_features = []
    for i in range(0, len(files), args.batch_size):
        batch_files = files[i:i + args.batch_size]
        batch_tensors = []
        for f in batch_files:
            try:
                img = Image.open(f).convert("RGB")
                batch_tensors.append(transform(img))
            except Exception:
                batch_tensors.append(torch.zeros(3, 128, 128))

        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            feats = model(batch).cpu().numpy()
        all_features.append(feats)
        print(f"  {min(i + args.batch_size, len(files))}/{len(files)}")

    features = np.concatenate(all_features, axis=0)
    print(f"Features shape: {features.shape}")

    # Also extract simple color histograms (HSV) for better color clustering
    print("Extracting color histograms...")
    import cv2
    color_features = []
    for f in files:
        try:
            img = cv2.imread(str(f))
            img = cv2.resize(img, (64, 64))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # H histogram (18 bins), S histogram (8 bins), V histogram (8 bins)
            h_hist = cv2.calcHist([hsv], [0], None, [18], [0, 180]).flatten()
            s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
            v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
            hist = np.concatenate([h_hist, s_hist, v_hist])
            hist = hist / (hist.sum() + 1e-8)
            color_features.append(hist)
        except Exception:
            color_features.append(np.zeros(34))

    color_features = np.array(color_features)

    # Combine: normalize both, weight color more
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    scaler_feat = StandardScaler()
    scaler_color = StandardScaler()

    feat_norm = scaler_feat.fit_transform(features)
    color_norm = scaler_color.fit_transform(color_features)

    # Weight color features 2x more (we care about color grouping)
    combined = np.concatenate([feat_norm * 0.5, color_norm * 2.0], axis=1)

    print(f"Clustering into {args.n_clusters} groups...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(combined)

    # Copy files to cluster folders
    for cluster_id in range(args.n_clusters):
        cluster_dir = output_dir / f"cluster_{cluster_id:02d}"
        cluster_dir.mkdir(exist_ok=True)

    for f, label in zip(files, labels):
        dst = output_dir / f"cluster_{label:02d}" / f.name
        shutil.copy2(f, dst)

    # Print stats
    from collections import Counter
    counts = Counter(labels)
    for cid in sorted(counts):
        print(f"  cluster_{cid:02d}: {counts[cid]} images")

    print(f"\nDone! Review clusters in {output_dir}/")
    print("Rename cluster folders to: blue, green, purple, red, yellow")
    print("Delete clusters that are garbage/mixed")

if __name__ == "__main__":
    main()
