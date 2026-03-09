#!/usr/bin/env python3
"""
Visual test: YOLOv11s + EfficientNet-V2-S on a single camera video.
Shows OpenCV window with bounding boxes, torso crops, and color labels.

Usage:
    python tools/test_visual.py                          # cam-03 default
    python tools/test_visual.py /path/to/video.mp4       # specific video
    python tools/test_visual.py --cam 7                  # camera 7
"""

import sys
import os
import cv2
import time
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.trt_inference import ColorClassifierInfer, YOLODetector

# ── Config ─────────────────────────────────────────────────────────
TORSO_TOP = 0.10
TORSO_BOTTOM = 0.40
TORSO_LEFT = 0.20
TORSO_RIGHT = 0.20
MIN_BBOX_HEIGHT = 30
EDGE_MARGIN = 10
MIN_CROP_PIXELS = 200
MAX_CROP_PIXELS = 15000

COLORS_BGR = {
    "green":  (0, 200, 0),
    "red":    (0, 0, 255),
    "yellow": (0, 230, 230),
    "unknown": (128, 128, 128),
}

REC_DIR = "/home/user/recordings/yaris_20260303_162028"


def extract_torso(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = y2 - y1, x2 - x1
    ty1 = max(0, y1 + int(h * TORSO_TOP))
    ty2 = min(frame.shape[0], y1 + int(h * TORSO_BOTTOM))
    tx1 = max(0, x1 + int(w * TORSO_LEFT))
    tx2 = min(frame.shape[1], x2 - int(w * TORSO_RIGHT))
    if ty2 - ty1 < 10 or tx2 - tx1 < 10:
        return None
    return frame[ty1:ty2, tx1:tx2]


def main():
    # Parse args
    video_path = None
    cam_num = 3  # default

    for arg in sys.argv[1:]:
        if arg.startswith("--cam"):
            cam_num = int(sys.argv[sys.argv.index(arg) + 1])
        elif os.path.isfile(arg):
            video_path = arg

    if video_path is None:
        # Find camera file
        import glob
        pattern = os.path.join(REC_DIR, f"kamera_{cam_num:02d}_*.mp4")
        matches = glob.glob(pattern)
        if not matches:
            print(f"No video found for camera {cam_num} in {REC_DIR}")
            sys.exit(1)
        video_path = matches[0]

    print(f"\n{'='*60}")
    print(f"  VISUAL TEST: {os.path.basename(video_path)}")
    print(f"{'='*60}\n")

    # Load models
    print("Loading YOLOv11s jockey detector...")
    detector = YOLODetector(fallback_pt="models/jockey_yolov11s.pt", imgsz=800, conf=0.35)

    print("Loading EfficientNet-V2-S color classifier...")
    classifier = ColorClassifierInfer(fallback_pt="models/color_classifier_v2.pt")
    print(f"  Classes: {classifier.classes}")
    print(f"  Input size: {classifier.INPUT_SIZE}px")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video: {total_frames} frames @ {fps:.1f} fps")
    print(f"\n  Controls: SPACE=pause, Q=quit, N=next frame, ←→=seek ±30f\n")

    frame_idx = 0
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        fh, fw = frame.shape[:2]
        display = frame.copy()

        # Detect
        t0 = time.time()
        batch_dets = detector.detect_batch([frame])
        dets = batch_dets[0] if batch_dets else []
        det_ms = (time.time() - t0) * 1000

        # Filter and classify
        crops = []
        valid_dets = []
        for det in dets:
            x1, y1, x2, y2 = det['bbox']
            bh = y2 - y1
            bw = x2 - x1
            if x1 <= EDGE_MARGIN or x2 >= fw - EDGE_MARGIN:
                continue
            if bh < MIN_BBOX_HEIGHT:
                continue
            torso = extract_torso(frame, det['bbox'])
            if torso is None:
                continue
            pixels = torso.shape[0] * torso.shape[1]
            if pixels < MIN_CROP_PIXELS or pixels > MAX_CROP_PIXELS:
                continue
            crops.append(torso)
            valid_dets.append(det)

        # Classify colors
        t1 = time.time()
        if crops:
            classifications = classifier.classify_batch(crops)
        else:
            classifications = []
        cls_ms = (time.time() - t1) * 1000

        # Draw results
        for det, (color, conf, prob_dict) in zip(valid_dets, classifications):
            x1, y1, x2, y2 = det['bbox']
            bgr = COLORS_BGR.get(color, (128, 128, 128))

            # Bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), bgr, 3)

            # Torso region (yellow dashed)
            h, w = y2 - y1, x2 - x1
            ty1 = y1 + int(h * TORSO_TOP)
            ty2 = y1 + int(h * TORSO_BOTTOM)
            tx1 = x1 + int(w * TORSO_LEFT)
            tx2 = x2 - int(w * TORSO_RIGHT)
            cv2.rectangle(display, (tx1, ty1), (tx2, ty2), (0, 255, 255), 1)

            # Label with probabilities
            label = f"{color} {conf:.0%}"
            cv2.putText(display, label, (x1, y1 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2)

            # Probability bar under bbox
            bar_y = y2 + 5
            for i, (c, p) in enumerate(sorted(prob_dict.items())):
                bar_w = int(p * 100)
                bar_bgr = COLORS_BGR.get(c, (128, 128, 128))
                cv2.rectangle(display, (x1, bar_y), (x1 + bar_w, bar_y + 12), bar_bgr, -1)
                cv2.putText(display, f"{c[:1]}:{p:.0%}", (x1 + bar_w + 3, bar_y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                bar_y += 15

        # Info panel
        cv2.rectangle(display, (0, 0), (400, 80), (0, 0, 0), -1)
        cv2.putText(display, f"Frame {frame_idx}/{total_frames}  |  {len(valid_dets)} jockeys",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display, f"YOLO: {det_ms:.0f}ms  Color: {cls_ms:.0f}ms",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        colors_found = [c for c, _, _ in classifications if c != "unknown"]
        cv2.putText(display, f"Colors: {', '.join(colors_found) or 'none'}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # Show
        cv2.imshow("Race Vision - Visual Test", display)

        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('n'):
            paused = True
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        elif key == 83:  # right arrow
            new_pos = min(frame_idx + 30, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            ret, frame = cap.read()
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        elif key == 81:  # left arrow
            new_pos = max(frame_idx - 30, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            ret, frame = cap.read()
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()
