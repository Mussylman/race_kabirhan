#!/usr/bin/env python3
"""Quick test: run color_classifier v1 and v2 on data/reid/ crops, show accuracy."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from collections import defaultdict

from pipeline.trt_inference import ColorClassifierInfer

REID_DIR = Path("data/reid")
COLORS = ["blue", "green", "purple", "red", "yellow"]

def load_crops():
    """Load all crops from data/reid/{color}/ with ground truth labels."""
    crops = []
    labels = []
    for color in COLORS:
        d = REID_DIR / color
        if not d.exists():
            continue
        files = sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))
        for f in files:
            img = cv2.imread(str(f))
            if img is not None:
                crops.append(img)
                labels.append(color)
    return crops, labels

def test_model(name, pt_path, crops, labels):
    print(f"\n{'='*60}")
    print(f"  Model: {name} ({pt_path})")
    print(f"{'='*60}")

    clf = ColorClassifierInfer(device="cuda:0")
    clf._load_pytorch(pt_path)
    print(f"  Classes: {clf.classes}, Input: {clf.INPUT_SIZE}x{clf.INPUT_SIZE}")

    results = clf.classify_batch(crops)

    # Per-class accuracy
    stats = defaultdict(lambda: {"correct": 0, "total": 0, "preds": defaultdict(int)})

    for label, (pred, conf, probs) in zip(labels, results):
        stats[label]["total"] += 1
        stats[label]["preds"][pred] += 1
        if pred == label:
            stats[label]["correct"] += 1

    total_correct = sum(s["correct"] for s in stats.values())
    total = len(labels)

    print(f"\n  Overall: {total_correct}/{total} = {100*total_correct/total:.1f}%\n")
    print(f"  {'GT':>8} | {'total':>5} | {'acc':>6} | predictions")
    print(f"  {'-'*60}")
    for color in COLORS:
        s = stats[color]
        if s["total"] == 0:
            continue
        acc = 100 * s["correct"] / s["total"]
        preds_str = ", ".join(f"{k}={v}" for k, v in sorted(s["preds"].items(), key=lambda x: -x[1]))
        print(f"  {color:>8} | {s['total']:5d} | {acc:5.1f}% | {preds_str}")

    # Show some confident wrong predictions
    print(f"\n  Worst mistakes (high conf, wrong pred):")
    mistakes = []
    for label, (pred, conf, probs) in zip(labels, results):
        if pred != label and conf > 0.5:
            mistakes.append((conf, label, pred, probs))
    mistakes.sort(reverse=True)
    for conf, label, pred, probs in mistakes[:5]:
        print(f"    GT={label:>8} → pred={pred:>8} conf={conf:.3f}  probs={probs}")


def main():
    crops, labels = load_crops()
    print(f"Loaded {len(crops)} crops: " + ", ".join(f"{c}={labels.count(c)}" for c in COLORS))

    # v1: SimpleColorCNN
    test_model("v1 (SimpleCNN 64x64)", "models/color_classifier.pt", crops, labels)

    # v2: EfficientNet-V2-S
    test_model("v2 (EfficientNet 128x128)", "models/color_classifier_v2.pt", crops, labels)


if __name__ == "__main__":
    main()
