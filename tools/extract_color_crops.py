#!/usr/bin/env python3
"""
Extract torso crops from race recordings for color classifier training.

Uses the trained jockey YOLO model to detect jockeys, crops the torso
region (same parameters as analyzer.py), and saves crops for manual sorting.

Step 1: Extract crops
    python tools/extract_color_crops.py \
        --videos /home/user/recordings/yaris_20260303_162028 \
        --videos /home/user/recordings/yaris_20260303_163257 \
        --videos /home/user/recordings/yaris_20260303_164835 \
        --output data/torso_crops_v2

Step 2: Sort crops into color folders (manually)
    data/torso_crops_v2/unsorted/  ->  blue/ green/ purple/ red/ yellow/

Step 3: Train classifier
    python tools/train_color_classifier.py
"""

import argparse
import cv2
import glob
import sys
from pathlib import Path

# Torso extraction params (same as analyzer.py)
TORSO_TOP = 0.10
TORSO_BOTTOM = 0.40
TORSO_LEFT = 0.20
TORSO_RIGHT = 0.20

MIN_BBOX_HEIGHT = 50
MIN_ASPECT_RATIO = 1.0
EDGE_MARGIN = 5


def extract_torso(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = y2 - y1, x2 - x1
    fh, fw = frame.shape[:2]
    ty1 = max(0, y1 + int(h * TORSO_TOP))
    ty2 = min(fh, y1 + int(h * TORSO_BOTTOM))
    tx1 = max(0, x1 + int(w * TORSO_LEFT))
    tx2 = min(fw, x2 - int(w * TORSO_RIGHT))
    if ty2 - ty1 < 10 or tx2 - tx1 < 10:
        return None
    return frame[ty1:ty2, tx1:tx2]


def main():
    parser = argparse.ArgumentParser(description="Extract torso crops for color classifier")
    parser.add_argument("--videos", required=True, action="append",
                        help="Video directory (can specify multiple times)")
    parser.add_argument("--output", default="data/torso_crops_v2",
                        help="Output directory")
    parser.add_argument("--model", default="models/jockey_yolov8s.onnx",
                        help="YOLO model for jockey detection")
    parser.add_argument("--interval", type=float, default=0.2,
                        help="Seconds between frames to process")
    parser.add_argument("--conf", type=float, default=0.30,
                        help="Detection confidence threshold")
    args = parser.parse_args()

    from ultralytics import YOLO

    out_dir = Path(args.output) / "unsorted"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect videos from all directories
    video_files = []
    for videos_dir in args.videos:
        vdir = Path(videos_dir)
        found = sorted(glob.glob(str(vdir / "*.mp4")))
        print(f"{vdir.name}: {len(found)} videos")
        video_files.extend(found)

    if not video_files:
        print("No .mp4 files found")
        sys.exit(1)

    print(f"\nTotal: {len(video_files)} videos")
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    total_crops = 0

    for vid_path in video_files:
        vid_name = Path(vid_path).stem
        parts = vid_name.split("_")
        cam_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else vid_name
        race_id = Path(vid_path).parent.name

        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_interval = max(1, int(fps * args.interval))

        frame_idx = 0
        cam_crops = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            fh, fw = frame.shape[:2]
            results = model(frame, verbose=False, conf=args.conf, imgsz=800)

            for r in results:
                if r.boxes is None:
                    continue
                for i, box in enumerate(r.boxes):
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                    bh = y2 - y1
                    bw = x2 - x1

                    if x1 <= EDGE_MARGIN or x2 >= fw - EDGE_MARGIN:
                        continue
                    if bh < MIN_BBOX_HEIGHT:
                        continue
                    if bh / max(bw, 1) < MIN_ASPECT_RATIO:
                        continue

                    torso = extract_torso(frame, (x1, y1, x2, y2))
                    if torso is None:
                        continue

                    pixels = torso.shape[0] * torso.shape[1]
                    if pixels < 400:
                        continue

                    crop_name = f"{race_id}__{cam_id}__f{frame_idx:06d}__d{i}.jpg"
                    cv2.imwrite(str(out_dir / crop_name), torso)
                    cam_crops += 1
                    total_crops += 1

            frame_idx += 1

        cap.release()
        if cam_crops > 0:
            print(f"  {race_id}/{cam_id}: {cam_crops} crops")

    print(f"\n{'='*60}")
    print(f"Total: {total_crops} torso crops in {out_dir}")
    print(f"\nСледующий шаг:")
    print(f"  1. Открой {out_dir}")
    colors = "blue green purple red yellow"
    print(f"  2. Создай папки: mkdir -p {out_dir.parent}/{{{colors}}}")
    print(f"  3. Разложи кропы по цветам жокеев")
    print(f"  4. Обучи: python tools/train_color_classifier.py")


if __name__ == "__main__":
    main()
