#!/usr/bin/env python3
"""
Two-step dataset preparation for jockey detector training (v2).

Step 1: Extract crops from multiple recordings (auto-detect with YOLO person)
    python tools/prepare_training_data.py extract \
        --videos /home/user/recordings/yaris_20260303_162028 \
        --videos /home/user/recordings/yaris_20260303_163257 \
        --videos /home/user/recordings/yaris_20260303_164835 \
        --output dataset/jockey_v2

    → Opens dataset/jockey_v2/crops/ with cropped detections
    → USER: delete wrong crops (spectators, staff, etc.), keep only jockeys

Step 2: Build YOLO labels from remaining crops
    python tools/prepare_training_data.py build \
        --output dataset/jockey_v2 \
        --val-split 0.2

    → Creates images/{train,val} + labels/{train,val} + data.yaml
"""

import argparse
import glob
import os
import random
import sys
import cv2
from pathlib import Path
from collections import defaultdict


def compute_iou(box1, box2):
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def extract(args):
    from ultralytics import YOLO

    output_dir = Path(args.output)
    interval = args.interval

    # Collect videos from all --videos directories
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

    model = YOLO(args.model)
    print(f"Loaded {args.model}")

    crops_dir = output_dir / "crops"
    frames_dir = output_dir / "frames"
    crops_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    MIN_BBOX_HEIGHT = 50
    MIN_ASPECT_RATIO = 1.0
    EDGE_MARGIN = 5

    total_crops = 0
    total_frames = 0
    crowded_frames = 0

    for vid_path in video_files:
        vid_name = Path(vid_path).stem
        # Camera id from filename: kamera_01_164835_END165759 -> kamera_01
        parts = vid_name.split("_")
        cam_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else vid_name
        # Race id from parent dir
        race_id = Path(vid_path).parent.name

        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_interval = max(1, int(fps * interval))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n{race_id}/{cam_id}: {frame_count} frames, every {interval}s")

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
            results = model(frame, verbose=False, conf=args.conf, classes=[0])

            det_idx = 0
            frame_saved = False
            frame_bboxes = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                    bw = x2 - x1
                    bh = y2 - y1

                    if x1 <= EDGE_MARGIN or x2 >= fw - EDGE_MARGIN:
                        continue
                    if bh < MIN_BBOX_HEIGHT:
                        continue
                    if bh / max(bw, 1) < MIN_ASPECT_RATIO:
                        continue

                    frame_bboxes.append((x1, y1, x2, y2))

                    # Crop with small padding
                    pad = 10
                    cx1 = max(0, x1 - pad)
                    cy1 = max(0, y1 - pad)
                    cx2 = min(fw, x2 + pad)
                    cy2 = min(fh, y2 + pad)
                    crop = frame[cy1:cy2, cx1:cx2]

                    # Filename: {race}_{cam}__f{frame}__d{det}__{x1}_{y1}_{x2}_{y2}.jpg
                    crop_name = f"{race_id}_{cam_id}__f{frame_idx:06d}__d{det_idx}__{x1}_{y1}_{x2}_{y2}.jpg"
                    cv2.imwrite(str(crops_dir / crop_name), crop)

                    if not frame_saved:
                        frame_name = f"{race_id}_{cam_id}__f{frame_idx:06d}.jpg"
                        cv2.imwrite(str(frames_dir / frame_name), frame)
                        frame_saved = True
                        total_frames += 1

                    det_idx += 1
                    cam_crops += 1
                    total_crops += 1

            # Check if this is a crowded frame (>=3 detections with high IoU)
            if len(frame_bboxes) >= 3:
                has_overlap = False
                for i in range(len(frame_bboxes)):
                    for j in range(i + 1, len(frame_bboxes)):
                        if compute_iou(frame_bboxes[i], frame_bboxes[j]) > 0.2:
                            has_overlap = True
                            break
                    if has_overlap:
                        break
                if has_overlap:
                    crowded_frames += 1

            frame_idx += 1

        cap.release()
        print(f"  -> {cam_crops} crops")

    print(f"\n{'='*60}")
    print(f"Crops: {total_crops} in {crops_dir}")
    print(f"Frames: {total_frames} in {frames_dir}")
    print(f"Crowded frames (>=3 det with overlap): {crowded_frames}")
    print(f"\nСейчас:")
    print(f"  1. Открой {crops_dir}")
    print(f"  2. Удали НЕ жокеев (зрители, персонал, мусор)")
    print(f"  3. Запусти: python tools/prepare_training_data.py build --output {output_dir}")


def build(args):
    """Build YOLO dataset from remaining crops with train/val split."""
    import shutil

    output_dir = Path(args.output)
    crops_dir = output_dir / "crops"
    frames_dir = output_dir / "frames"
    val_split = args.val_split

    if not crops_dir.exists():
        print(f"No crops dir: {crops_dir}")
        sys.exit(1)

    # Create train/val dirs
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Parse crop filenames -> group by frame
    # Format: {race}_{cam}__f{frame}__d{det}__{x1}_{y1}_{x2}_{y2}.jpg
    frame_bboxes = defaultdict(list)

    crop_files = sorted(crops_dir.glob("*.jpg"))
    print(f"Remaining crops: {len(crop_files)}")

    for crop_path in crop_files:
        name = crop_path.stem
        parts = name.split("__")
        if len(parts) != 4:
            print(f"  Skipping {name} (bad format)")
            continue

        prefix = parts[0]       # race_cam
        frame_part = parts[1]   # f000123
        bbox_part = parts[3]    # x1_y1_x2_y2

        frame_key = f"{prefix}__{frame_part}"
        coords = bbox_part.split("_")
        if len(coords) != 4:
            continue
        x1, y1, x2, y2 = [int(c) for c in coords]
        frame_bboxes[frame_key].append((x1, y1, x2, y2))

    print(f"Frames with annotations: {len(frame_bboxes)}")

    # Split by camera: extract cam number from frame_key
    # frame_key = "yaris_..._kamera_05__f000123"
    def get_cam_num(frame_key):
        try:
            parts = frame_key.split("_kamera_")
            if len(parts) >= 2:
                num_part = parts[1].split("__")[0]
                return int(num_part)
        except (ValueError, IndexError):
            pass
        return 0

    # Cameras 21-25 go to val, rest to train
    val_cams = set(range(21, 26))

    train_count = 0
    val_count = 0
    total_labels = 0

    for frame_key, bboxes in sorted(frame_bboxes.items()):
        frame_path = frames_dir / f"{frame_key}.jpg"
        if not frame_path.exists():
            print(f"  Missing frame: {frame_path}")
            continue

        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        fh, fw = img.shape[:2]

        cam_num = get_cam_num(frame_key)
        split = "val" if cam_num in val_cams else "train"

        # If not enough val cameras, use random split as fallback
        if val_split > 0 and not val_cams:
            split = "val" if random.random() < val_split else "train"

        # Copy frame
        shutil.copy2(frame_path, output_dir / "images" / split / f"{frame_key}.jpg")

        # Write YOLO labels
        labels = []
        for x1, y1, x2, y2 in bboxes:
            x_center = ((x1 + x2) / 2) / fw
            y_center = ((y1 + y2) / 2) / fh
            w_norm = (x2 - x1) / fw
            h_norm = (y2 - y1) / fh
            labels.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            total_labels += 1

        lbl_path = output_dir / "labels" / split / f"{frame_key}.txt"
        with open(lbl_path, "w") as f:
            f.write("\n".join(labels))

        if split == "val":
            val_count += 1
        else:
            train_count += 1

    # Add negative samples (frames without remaining crops)
    all_frames = set(f.stem for f in frames_dir.glob("*.jpg"))
    annotated_frames = set(frame_bboxes.keys())
    negative_frames = all_frames - annotated_frames

    neg_count = min(len(negative_frames), max(50, int(len(annotated_frames) * 0.3)))
    neg_sample = random.sample(sorted(negative_frames), neg_count) if negative_frames else []

    neg_train = 0
    neg_val = 0
    for frame_key in neg_sample:
        cam_num = get_cam_num(frame_key)
        split = "val" if cam_num in val_cams else "train"

        shutil.copy2(frames_dir / f"{frame_key}.jpg",
                      output_dir / "images" / split / f"{frame_key}.jpg")
        with open(output_dir / "labels" / split / f"{frame_key}.txt", "w") as f:
            f.write("")

        if split == "val":
            neg_val += 1
        else:
            neg_train += 1

    # data.yaml
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(f"""path: {output_dir.resolve()}
train: images/train
val: images/val

names:
  0: jockey

# Train: {train_count} positive + {neg_train} negative = {train_count + neg_train}
# Val: {val_count} positive + {neg_val} negative = {val_count + neg_val}
# Total bounding boxes: {total_labels}
""")

    print(f"\n{'='*60}")
    print(f"Dataset built!")
    print(f"  Train: {train_count} positive + {neg_train} negative = {train_count + neg_train}")
    print(f"  Val:   {val_count} positive + {neg_val} negative = {val_count + neg_val}")
    print(f"  Labels: {total_labels} bboxes")
    print(f"  Config: {data_yaml}")
    print(f"\nТренировка:")
    print(f"  python tools/train_jockey.py --data {data_yaml} --model yolo11s.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_ext = sub.add_parser("extract", help="Extract crops from videos")
    p_ext.add_argument("--videos", required=True, action="append",
                        help="Video directory (can specify multiple times)")
    p_ext.add_argument("--output", required=True)
    p_ext.add_argument("--interval", type=float, default=0.15,
                        help="Seconds between frames (default: 0.15)")
    p_ext.add_argument("--model", default="yolov8s.pt")
    p_ext.add_argument("--conf", type=float, default=0.20,
                        help="Detection confidence (default: 0.20, lower = more candidates)")

    p_build = sub.add_parser("build", help="Build YOLO dataset from reviewed crops")
    p_build.add_argument("--output", required=True)
    p_build.add_argument("--val-split", type=float, default=0.2,
                          help="Validation split ratio (default: 0.2, cameras 21-25)")

    args = parser.parse_args()
    if args.cmd == "extract":
        extract(args)
    elif args.cmd == "build":
        build(args)
    else:
        parser.print_help()
