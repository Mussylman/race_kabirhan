#!/usr/bin/env python3
"""
calibrate_hsv.py — derive HSV Hue centroids for each silk color from
ground-truth JSONL + original MP4 recordings.

Input:
  - Directory with per-camera JSONL files (one record per frame with
    detections[{color, conf, bbox, track_id}]).
  - Directory with original MP4 files (one per cam_id).

Output:
  - Prints per-color Hue distribution stats (median, IQR).
  - Writes a proposed `configs/hsv_centroids.calibrated.yml` with
    median Hues per color.
  - Optionally saves annotated crops for manual review
    (--save-crops DIR).

Usage:
  python tools/calibrate_hsv.py \\
      --jsonl-dir /tmp/logs2/frames \\
      --mp4-dir   /path/to/recording2 \\
      --min-conf  0.90 \\
      --save-crops /home/user/race_kabirhan_artifacts/hsv_calib
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

log = logging.getLogger("tools.calibrate_hsv")


def extract_frame(mp4: Path, frame_seq: int) -> np.ndarray | None:
    """Seek mp4 to given frame and return BGR frame as ndarray."""
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        log.warning("Cannot open %s", mp4)
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_seq)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def extract_silk_region(bgr: np.ndarray, bbox: tuple) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = bgr.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(x1 + 1, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(y1 + 1, min(h, y2))
    crop = bgr[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    # Upper central (silk torso area) — matches hsv_classifier.py defaults.
    y0 = int(ch * 0.10)
    y1r = int(ch * 0.55)
    x0 = int(cw * 0.25)
    x1r = int(cw * 0.75)
    if y1r <= y0 or x1r <= x0:
        return crop
    return crop[y0:y1r, x0:x1r]


def masked_hue_pixels(region: np.ndarray,
                      min_sat: int = 80,
                      min_val: int = 40,
                      max_val: int = 240) -> np.ndarray:
    if region.size == 0:
        return np.array([], dtype=np.int32)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (sat >= min_sat) & (val >= min_val) & (val <= max_val)
    return hsv[:, :, 0][mask]


def circular_median(hues: np.ndarray) -> float:
    """Median on circular [0,180) Hue space, handling wraparound near 0/180."""
    if len(hues) == 0:
        return float("nan")
    low = int((hues < 20).sum())
    high = int((hues > 160).sum())
    if low > 0.2 * len(hues) and high > 0.2 * len(hues):
        rotated = (hues.astype(np.int32) + 90) % 180
        med_rot = float(np.median(rotated))
        return float((med_rot - 90) % 180)
    return float(np.median(hues))


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--jsonl-dir", required=True, type=Path)
    p.add_argument("--mp4-dir", required=True, type=Path)
    p.add_argument("--min-conf", type=float, default=0.90)
    p.add_argument("--max-samples-per-color", type=int, default=200)
    p.add_argument("--save-crops", type=Path, default=None)
    p.add_argument("--output", type=Path,
                   default=Path("configs/hsv_centroids.calibrated.yml"))
    args = p.parse_args()

    # Gather (cam, frame_seq, bbox, color, conf) tuples from JSONL.
    color_samples: dict[str, list[tuple]] = defaultdict(list)
    for jsonl_file in sorted(args.jsonl_dir.glob("*.jsonl")):
        cam_id = jsonl_file.stem.replace("cam_", "").replace("cam-", "")
        if not cam_id.startswith("cam"):
            cam_id = f"cam-{cam_id}" if not cam_id.startswith("cam-") else cam_id
        with open(jsonl_file) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                frame_seq = rec.get("frame_seq", 0)
                for det in rec.get("detections", []):
                    color = det.get("color")
                    conf = float(det.get("conf", 0))
                    if color and conf >= args.min_conf * 100:
                        bbox = det.get("bbox")
                        if bbox and len(bbox) == 4:
                            color_samples[color].append(
                                (rec.get("cam_id", cam_id), frame_seq, tuple(bbox))
                            )

    log.info("Loaded samples: %s",
             {c: len(s) for c, s in color_samples.items()})

    if args.save_crops:
        args.save_crops.mkdir(parents=True, exist_ok=True)

    # For each color, sample up to N bboxes, extract crop, compute Hue median.
    per_color_hues: dict[str, list[float]] = defaultdict(list)
    for color, samples in color_samples.items():
        if len(samples) > args.max_samples_per_color:
            step = len(samples) // args.max_samples_per_color
            samples = samples[::step][: args.max_samples_per_color]
        log.info("Processing %s: %d samples", color, len(samples))
        for i, (cam_id, frame_seq, bbox) in enumerate(samples):
            mp4 = args.mp4_dir / f"{cam_id}.mp4"
            if not mp4.exists():
                mp4 = args.mp4_dir / f"{cam_id}.MP4"
            if not mp4.exists():
                continue
            frame = extract_frame(mp4, frame_seq)
            if frame is None:
                continue
            region = extract_silk_region(frame, bbox)
            hues = masked_hue_pixels(region)
            if len(hues) < 30:
                continue
            med = circular_median(hues)
            if not np.isnan(med):
                per_color_hues[color].append(med)
            if args.save_crops and i < 20:
                crop_path = args.save_crops / f"{color}_{cam_id}_{frame_seq}.jpg"
                cv2.imwrite(str(crop_path), region)

    # Summarise.
    centroids: dict[str, list[float]] = {}
    print("\n=== HSV Calibration Report ===")
    for color, hues in sorted(per_color_hues.items()):
        arr = np.array(hues, dtype=np.float32)
        if len(arr) == 0:
            continue
        med = float(np.median(arr))
        q25 = float(np.percentile(arr, 25))
        q75 = float(np.percentile(arr, 75))
        print(f"  {color:8s}  n={len(arr):4d}  median={med:6.1f}  IQR=[{q25:.1f}, {q75:.1f}]")
        # Special-case red: if bimodal around 0/180 → two centroids.
        if color == "red":
            low_mode = arr[arr < 30]
            high_mode = arr[arr > 150]
            cents: list[float] = []
            if len(low_mode) >= 5:
                cents.append(round(float(np.median(low_mode)), 1))
            if len(high_mode) >= 5:
                cents.append(round(float(np.median(high_mode)), 1))
            centroids[color] = cents or [round(med, 1)]
        else:
            centroids[color] = [round(med, 1)]

    # Write output YAML.
    output_cfg = {
        "version": 1,
        "mask": {
            "min_saturation": 80,
            "min_value": 40,
            "max_value": 240,
            "min_pixels": 30,
        },
        "region": {
            "y_start": 0.10,
            "y_end": 0.55,
            "x_start": 0.25,
            "x_end": 0.75,
        },
        "tolerance_h": 18,
        "max_distance_h": 30,
        "colors": {
            color: {
                "hue_centroids": centroids[color],
                "notes": f"Calibrated from {len(per_color_hues[color])} samples",
            }
            for color in centroids
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        yaml.safe_dump(output_cfg, f, sort_keys=False, allow_unicode=True)
    print(f"\nWrote calibrated config: {args.output}")
    print("Replace configs/hsv_centroids.yml with this file after review.")


if __name__ == "__main__":
    main()
