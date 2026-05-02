#!/usr/bin/env python3
"""
test_reid_video.py — Run YOLO + ReID on a video, draw results.

Usage:
    python tools/test_reid_video.py --video data/videos/exp10_cam1.mp4
    python tools/test_reid_video.py --video data/videos/exp10_cam1.mp4 --save output_reid.mp4
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

COLORS_MAP = {
    "blue":   (255, 100, 0),
    "green":  (0, 200, 0),
    "purple": (200, 0, 200),
    "red":    (0, 0, 255),
    "yellow": (0, 230, 230),
    "unknown":(128, 128, 128),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--gallery", default="data/reid", help="Gallery directory")
    parser.add_argument("--backend", default="dinov2", choices=["clip", "dinov2"])
    parser.add_argument("--yolo-model", default="yolov8s.pt", help="YOLO model")
    parser.add_argument("--save", default=None, help="Save output video to file")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0=all)")
    parser.add_argument("--skip", type=int, default=2, help="Process every N-th frame")
    parser.add_argument("--conf", type=float, default=0.3, help="YOLO confidence threshold")
    args = parser.parse_args()

    # Load YOLO
    from ultralytics import YOLO
    yolo = YOLO(args.yolo_model)
    print(f"YOLO loaded: {args.yolo_model}")

    # Load ReID
    from pipeline.jockey_reid import JockeyReID
    reid = JockeyReID(gallery_dir=args.gallery, backend=args.backend, device="cuda:0")
    print(f"ReID loaded: {args.backend}, gallery={args.gallery}")
    stats = reid.get_gallery_stats()
    print(f"  Jockeys: {stats.get('n_jockeys', 0)}, embeddings: {stats.get('n_embeddings', 0)}")

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} @ {fps:.1f}fps, {total} frames")

    # Output writer
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = fps / args.skip
        writer = cv2.VideoWriter(args.save, fourcc, out_fps, (w, h))
        print(f"Saving to: {args.save} @ {out_fps:.1f}fps")

    frame_idx = 0
    processed = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % args.skip != 0:
            continue

        # YOLO detect (class 0 = person)
        results = yolo.predict(frame, conf=args.conf, classes=[0], verbose=False)
        boxes = results[0].boxes

        if len(boxes) > 0:
            crops = []
            bboxes = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                # Crop upper body (torso) for better ReID
                bh = y2 - y1
                torso_y2 = y1 + int(bh * 0.55)
                crop = frame[y1:torso_y2, x1:x2]
                if crop.size > 0:
                    crops.append(crop)
                    bboxes.append((x1, y1, x2, y2))

            if crops:
                identities = reid.identify_batch(crops)

                for (x1, y1, x2, y2), (name, score, details) in zip(bboxes, identities):
                    color = COLORS_MAP.get(name, (128, 128, 128))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{name} {score:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Frame counter
        cv2.putText(frame, f"Frame {frame_idx}/{total}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if writer:
            writer.write(frame)

        processed += 1
        if processed % 20 == 0:
            elapsed = time.time() - t_start
            fps_actual = processed / elapsed
            print(f"  Frame {frame_idx}/{total} | {processed} processed | {fps_actual:.1f} fps | "
                  f"{len(boxes)} detections")

        if args.max_frames and processed >= args.max_frames:
            break

    cap.release()
    if writer:
        writer.release()

    elapsed = time.time() - t_start
    print(f"\nDone: {processed} frames in {elapsed:.1f}s ({processed/elapsed:.1f} fps)")
    if args.save:
        print(f"Output saved: {args.save}")


if __name__ == "__main__":
    main()
