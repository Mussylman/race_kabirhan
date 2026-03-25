#!/usr/bin/env python3
"""
Build color classification dataset from all available videos.
Extracts jockey crops using YOLO, auto-labels with v1 model, saves to dataset/.
"""

import sys
import os
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from ultralytics import YOLO

OUT_DIR = Path("data/color_dataset")
COLORS = ["blue", "green", "purple", "red", "yellow", "unknown"]
MIN_CROP_PIXELS = 500  # skip tiny crops < ~22x22

# All video sources
VIDEOS = []

# 4K videos (best quality)
for f in sorted(Path("data/videos").glob("*.mp4")):
    VIDEOS.append(str(f))

# 720p recordings — sample from the longest ones
for session in sorted(Path("/home/user/recordings").iterdir()):
    if not session.is_dir():
        continue
    mp4s = sorted(session.glob("*.mp4"))
    for mp4 in mp4s:
        VIDEOS.append(str(mp4))


def extract_crops():
    print(f"Loading YOLO...")
    yolo = YOLO("models/jockey_yolov11s.pt")

    # Load v1 classifier for auto-labeling
    print(f"Loading color classifier v1...")
    from pipeline.trt_inference import ColorClassifierInfer
    clf = ColorClassifierInfer(device="cuda:0")
    clf._load_pytorch("models/color_classifier.pt")

    # Create output dirs
    for c in COLORS:
        (OUT_DIR / c).mkdir(parents=True, exist_ok=True)

    stats = defaultdict(int)
    total_crops = 0

    for vid_path in VIDEOS:
        vid_name = Path(vid_path).stem
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"  SKIP (can't open): {vid_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Sample rate: ~3 fps for all videos
        skip = max(1, int(fps / 3))

        vid_crops = 0
        fnum = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fnum += 1
            if fnum % skip != 0:
                continue

            # Detect
            results = yolo(frame, imgsz=800, conf=0.20, verbose=False)

            crops_batch = []
            crop_metas = []

            for r in results:
                for i, box in enumerate(r.boxes):
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    conf = box.conf.item()
                    crop = frame[y1:y2, x1:x2]
                    cw, ch = x2 - x1, y2 - y1

                    if crop.size < MIN_CROP_PIXELS:
                        continue

                    crops_batch.append(crop)
                    crop_metas.append((vid_name, fnum, i, cw, ch, conf))

            if not crops_batch:
                continue

            # Auto-label
            labels = clf.classify_batch(crops_batch)

            for crop, meta, (pred, pred_conf, probs) in zip(crops_batch, crop_metas, labels):
                vid_name_m, fn, di, cw, ch, yolo_conf = meta
                color = pred if pred_conf > 0.50 else "unknown"

                fname = f"{vid_name_m}_f{fn:05d}_d{di}_{cw}x{ch}.jpg"
                cv2.imwrite(str(OUT_DIR / color / fname), crop)
                stats[color] += 1
                vid_crops += 1
                total_crops += 1

        cap.release()
        if vid_crops > 0:
            print(f"  {vid_name}: {w}x{h}, {fnum} frames, {vid_crops} crops")

    print(f"\n{'='*50}")
    print(f"Total: {total_crops} crops")
    for c in sorted(stats.keys()):
        print(f"  {c:>10}: {stats[c]}")
    print(f"\nSaved to: {OUT_DIR}/")


if __name__ == "__main__":
    extract_crops()
