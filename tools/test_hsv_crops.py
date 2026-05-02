#!/usr/bin/env python3
"""
HSV color classifier test on DeepStream crops.
No training needed — pure histogram analysis.

Ground truth for cam-05:
  t0 = green, t1 = red, t2 = yellow
"""
import cv2
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

CROPS_DIR = Path("ds_results/exp2/crops")

# HSV ranges for jockey colors (H is 0-179 in OpenCV)
# Tuned on real cam-05 crops:
#   green jacket → H~96 (camera shifts green toward teal/blue-green)
#   red jacket   → H~154-168 (deep red, closer to purple boundary)
#   yellow jacket → H~25 (classic yellow)
COLOR_RANGES = {
    "red":    [(0, 12, 50, 40), (155, 179, 50, 40)],
    "yellow": [(13, 38, 40, 40)],
    "green":  [(39, 100, 20, 20)],                      # cam shifts green → teal (H~81-100)
    "blue":   [(101, 135, 50, 40)],                     # H 101-135 = true blue
    "purple": [(136, 154, 40, 40)],
}

GROUND_TRUTH = {0: "green", 1: "red", 2: "yellow"}


def classify_hsv(img_bgr, method="histogram"):
    """Classify jockey color from BGR crop using HSV analysis."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]

    # Focus on upper chest area only (skip head cap, skip legs/horse)
    # Full bbox = jockey+horse. Upper 5-30% = head+shoulders+chest (jockey jacket)
    torso = hsv[int(h * 0.05):int(h * 0.30), int(w * 0.10):int(w * 0.90)]
    if torso.size == 0:
        torso = hsv

    if method == "histogram":
        return _classify_histogram(torso)
    elif method == "pixel_vote":
        return _classify_pixel_vote(torso)
    elif method == "dominant_hue":
        return _classify_dominant_hue(torso)
    return "unknown", 0.0, {}


def _classify_histogram(torso_hsv):
    """Count pixels matching each color range, pick dominant."""
    h_ch = torso_hsv[:, :, 0]
    s_ch = torso_hsv[:, :, 1]
    v_ch = torso_hsv[:, :, 2]
    total_px = h_ch.size

    # Filter out low-saturation (gray/black/white) pixels
    chromatic = (s_ch > 20) & (v_ch > 20)
    chromatic_count = chromatic.sum()

    if chromatic_count < total_px * 0.1:
        return "unknown", 0.0, {"reason": "too few chromatic pixels"}

    scores = {}
    for color, ranges in COLOR_RANGES.items():
        mask = np.zeros_like(h_ch, dtype=bool)
        for h_lo, h_hi, s_min, v_min in ranges:
            mask |= (h_ch >= h_lo) & (h_ch <= h_hi) & (s_ch >= s_min) & (v_ch >= v_min)
        scores[color] = mask.sum() / chromatic_count

    best = max(scores, key=scores.get)
    conf = scores[best]

    return best, conf, scores


def _classify_pixel_vote(torso_hsv):
    """Per-pixel classification, then majority vote."""
    h_ch = torso_hsv[:, :, 0].ravel()
    s_ch = torso_hsv[:, :, 1].ravel()
    v_ch = torso_hsv[:, :, 2].ravel()

    votes = Counter()
    for i in range(len(h_ch)):
        if s_ch[i] < 40 or v_ch[i] < 40:
            continue
        h = h_ch[i]
        for color, ranges in COLOR_RANGES.items():
            for h_lo, h_hi, s_min, v_min in ranges:
                if h_lo <= h <= h_hi and s_ch[i] >= s_min and v_ch[i] >= v_min:
                    votes[color] += 1
                    break

    total = sum(votes.values())
    if total == 0:
        return "unknown", 0.0, {}

    best = votes.most_common(1)[0]
    return best[0], best[1] / total, dict(votes)


def _classify_dominant_hue(torso_hsv):
    """Compute dominant hue via histogram peak, then map to color."""
    s_ch = torso_hsv[:, :, 1].ravel()
    v_ch = torso_hsv[:, :, 2].ravel()
    h_ch = torso_hsv[:, :, 0].ravel()

    # Only chromatic pixels
    mask = (s_ch > 40) & (v_ch > 40)
    if mask.sum() < 20:
        return "unknown", 0.0, {}

    h_filtered = h_ch[mask]

    # Histogram with 180 bins (1 per hue degree)
    hist, _ = np.histogram(h_filtered, bins=180, range=(0, 180))

    # Smooth
    kernel = np.ones(5) / 5
    hist_smooth = np.convolve(hist, kernel, mode='same')

    peak_hue = np.argmax(hist_smooth)
    peak_count = hist_smooth[peak_hue]
    total_chromatic = mask.sum()
    conf = peak_count / total_chromatic

    # Map hue to color (tuned for camera white balance)
    if peak_hue <= 12 or peak_hue >= 155:
        color = "red"
    elif 13 <= peak_hue <= 38:
        color = "yellow"
    elif 39 <= peak_hue <= 100:
        color = "green"
    elif 101 <= peak_hue <= 135:
        color = "blue"
    elif 136 <= peak_hue <= 154:
        color = "purple"
    else:
        color = "unknown"

    return color, conf, {"peak_hue": int(peak_hue), "chromatic_pct": mask.sum() / len(h_ch)}


def main():
    crop_files = sorted(CROPS_DIR.glob("*.jpg"))
    print(f"Crops: {len(crop_files)}")

    methods = ["histogram", "pixel_vote", "dominant_hue"]

    for method in methods:
        print(f"\n{'='*60}")
        print(f"METHOD: {method}")
        print(f"{'='*60}")

        results = []  # (filename, track_id, predicted, conf)
        for f in crop_files:
            img = cv2.imread(str(f))
            if img is None:
                continue

            parts = f.stem.split("_")
            track_id = None
            for p in parts:
                if p.startswith("t") and p[1:].isdigit():
                    track_id = int(p[1:])

            color, conf, details = classify_hsv(img, method=method)
            results.append((f.name, track_id, color, conf))

        # Overall distribution
        total = len(results)
        colors = Counter(c for _, _, c, _ in results)
        print(f"\nColor distribution:")
        for c, n in colors.most_common():
            print(f"  {c:10s}: {n:5d} ({100*n/total:.1f}%)")

        # Per-track accuracy
        print(f"\nPer-track (ground truth: t0=green, t1=red, t2=yellow):")
        tracks = defaultdict(list)
        for _, tid, color, conf in results:
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
                  f"gt={gt:8s}, acc={acc:5.1f}% {marker}, avg_conf={avg_conf:.2f}, dist={dict(dist)}")

        if evaluated_total > 0:
            print(f"\n  OVERALL ACCURACY (t0+t1+t2): {correct_total}/{evaluated_total} ({100*correct_total/evaluated_total:.1f}%)")


if __name__ == "__main__":
    main()
