#!/usr/bin/env python3
"""
Segmentation + HSV: use YOLOv8n-seg to mask jockey body,
then classify color only on person pixels.
"""
import cv2
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from ultralytics import YOLO

CROPS_DIR = Path("ds_results/exp2/crops")
SEG_MODEL = "models/yolov8n-seg.pt"

# HSV ranges (tuned for camera white balance)
COLOR_RANGES = {
    "red":    [(0, 12, 50, 40), (155, 179, 50, 40)],
    "yellow": [(13, 38, 40, 40)],
    "green":  [(39, 100, 20, 20)],
    "blue":   [(101, 135, 50, 40)],
    "purple": [(136, 154, 40, 40)],
}

GROUND_TRUTH = {0: "green", 1: "red", 2: "yellow"}


def classify_masked_hsv(hsv_pixels):
    """Classify color from HSV pixels (Nx3 array, only person pixels)."""
    if len(hsv_pixels) < 10:
        return "unknown", 0.0, {}

    h_ch = hsv_pixels[:, 0]
    s_ch = hsv_pixels[:, 1]
    v_ch = hsv_pixels[:, 2]

    chromatic = (s_ch > 20) & (v_ch > 20)
    if chromatic.sum() < 10:
        return "unknown", 0.0, {"reason": "no chromatic"}

    scores = {}
    for color, ranges in COLOR_RANGES.items():
        mask = np.zeros(len(h_ch), dtype=bool)
        for h_lo, h_hi, s_min, v_min in ranges:
            mask |= (h_ch >= h_lo) & (h_ch <= h_hi) & (s_ch >= s_min) & (v_ch >= v_min)
        scores[color] = mask.sum() / chromatic.sum()

    best = max(scores, key=scores.get)
    return best, scores[best], scores


def main():
    print("Loading YOLOv8n-seg...")
    seg = YOLO(SEG_MODEL)

    crop_files = sorted(CROPS_DIR.glob("*.jpg"))
    print(f"Crops: {len(crop_files)}")

    results = []  # (filename, track_id, color, conf, mask_pct)
    no_mask = 0

    BATCH = 32
    for i in range(0, len(crop_files), BATCH):
        batch_files = crop_files[i:i + BATCH]
        batch_imgs = [cv2.imread(str(f)) for f in batch_files]

        # Run segmentation
        seg_results = seg(batch_imgs, verbose=False, conf=0.25, classes=[0])  # class 0 = person

        for f, img, seg_r in zip(batch_files, batch_imgs, seg_results):
            parts = f.stem.split("_")
            track_id = None
            for p in parts:
                if p.startswith("t") and p[1:].isdigit():
                    track_id = int(p[1:])

            # Get person mask
            if seg_r.masks is not None and len(seg_r.masks) > 0:
                # Take largest person mask
                mask_data = seg_r.masks.data.cpu().numpy()
                areas = [m.sum() for m in mask_data]
                best_mask = mask_data[np.argmax(areas)]

                # Resize mask to image size
                h, w = img.shape[:2]
                mask_resized = cv2.resize(best_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_bool = mask_resized > 0.5

                # Extract only person pixels
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                person_pixels = hsv[mask_bool]
                mask_pct = mask_bool.sum() / (h * w) * 100

                # Optional: only upper portion of mask (torso)
                # Find mask bounding box and take upper 60%
                ys, xs = np.where(mask_bool)
                if len(ys) > 0:
                    my1, my2 = ys.min(), ys.max()
                    mask_h = my2 - my1
                    torso_mask = mask_bool.copy()
                    torso_mask[int(my1 + mask_h * 0.60):, :] = False  # cut lower 40%
                    person_pixels = hsv[torso_mask]

                color, conf, details = classify_masked_hsv(person_pixels)
                results.append((f.name, track_id, color, conf, mask_pct))
            else:
                no_mask += 1
                results.append((f.name, track_id, "unknown", 0.0, 0.0))

        if (i // BATCH) % 10 == 0:
            print(f"  processed {min(i + BATCH, len(crop_files))}/{len(crop_files)}")

    # === Statistics ===
    total = len(results)
    colors = Counter(c for _, _, c, _, _ in results)
    print(f"\n{'=' * 60}")
    print(f"RESULTS: Segmentation + HSV on {total} crops")
    print(f"{'=' * 60}")
    print(f"No person mask: {no_mask}/{total} ({100 * no_mask / total:.1f}%)")
    print(f"\nColor distribution:")
    for c, n in colors.most_common():
        print(f"  {c:10s}: {n:5d} ({100 * n / total:.1f}%)")

    # Per-track
    print(f"\nPer-track (ground truth: t0=green, t1=red, t2=yellow):")
    tracks = defaultdict(list)
    for _, tid, color, conf, _ in results:
        if tid is not None:
            tracks[tid].append((color, conf))

    correct_total = 0
    evaluated_total = 0
    for tid in sorted(tracks.keys()):
        entries = tracks[tid]
        dist = Counter(c for c, _ in entries)
        dominant = dist.most_common(1)[0]
        avg_conf = np.mean([c for _, c in entries])
        gt = GROUND_TRUTH.get(tid, "?")
        correct = sum(1 for c, _ in entries if c == gt)
        acc = 100 * correct / len(entries) if gt != "?" else 0
        if gt != "?":
            correct_total += correct
            evaluated_total += len(entries)
        marker = "✓" if dominant[0] == gt else "✗"
        print(f"  t{tid}: {len(entries):4d} dets, dominant={dominant[0]:8s}({dominant[1]:4d}/{len(entries):4d}), "
              f"gt={gt:8s}, acc={acc:5.1f}% {marker}, dist={dict(dist)}")

    if evaluated_total > 0:
        print(f"\n  OVERALL ACCURACY (t0+t1+t2): {correct_total}/{evaluated_total} "
              f"({100 * correct_total / evaluated_total:.1f}%)")

    # Avg mask coverage
    mask_pcts = [m for _, _, _, _, m in results if m > 0]
    if mask_pcts:
        print(f"\n  Avg person mask coverage: {np.mean(mask_pcts):.1f}% of bbox")


if __name__ == "__main__":
    main()
