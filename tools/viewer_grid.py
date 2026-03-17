"""
viewer_grid.py — All-in-one: 10 cameras grid + YOLO + color classifier + display.

No Docker, no SHM — just Python + YOLO + OpenCV.

Usage:
    source venv/bin/activate
    python tools/viewer_grid.py --config cameras_test_10.json
"""

import sys
import json
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ── Color classifier ──
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.trt_inference import SimpleColorCNN

COLOR_NAMES = ["blue", "green", "purple", "red", "yellow"]
COLOR_BGR = {
    "blue":   (255, 100, 0),
    "green":  (0, 220, 0),
    "purple": (200, 0, 200),
    "red":    (0, 0, 255),
    "yellow": (0, 230, 255),
}

# ImageNet normalization
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

# Detection filters (same as C++ pipeline)
MIN_BBOX_HEIGHT = 50
MIN_ASPECT_RATIO = 1.0
EDGE_MARGIN = 10

# Torso crop params
TORSO_TOP = 0.10
TORSO_BOTTOM = 0.40
TORSO_LEFT = 0.20
TORSO_RIGHT = 0.20


def load_color_model(path="models/color_classifier.pt"):
    ckpt = torch.load(path, map_location="cuda:0", weights_only=False)
    model = SimpleColorCNN(num_classes=len(ckpt["classes"])).cuda()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def classify_colors(color_model, frame_bgr, boxes):
    """Crop torso regions, classify colors."""
    if len(boxes) == 0:
        return []

    crops = []
    h, w = frame_bgr.shape[:2]

    for (x1, y1, x2, y2) in boxes:
        bw, bh = x2 - x1, y2 - y1
        # Torso crop
        tx1 = max(0, int(x1 + bw * TORSO_LEFT))
        ty1 = max(0, int(y1 + bh * TORSO_TOP))
        tx2 = min(w, int(x2 - bw * TORSO_RIGHT))
        ty2 = min(h, int(y1 + bh * TORSO_BOTTOM))

        if tx2 <= tx1 or ty2 <= ty1:
            crops.append(np.zeros((64, 64, 3), dtype=np.uint8))
            continue

        crop = frame_bgr[ty1:ty2, tx1:tx2]
        crop = cv2.resize(crop, (64, 64))
        crops.append(crop)

    # Batch to tensor (BGR→RGB, normalize)
    batch = np.stack(crops)  # (N, 64, 64, 3)
    batch = batch[:, :, :, ::-1].copy()  # BGR→RGB
    batch = torch.from_numpy(batch).permute(0, 3, 1, 2).float().cuda() / 255.0
    batch = (batch - MEAN) / STD

    with torch.no_grad():
        logits = color_model(batch)
        probs = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)

    results = []
    for i in range(len(boxes)):
        color_id = preds[i].item()
        conf = confs[i].item()
        name = COLOR_NAMES[color_id] if color_id < len(COLOR_NAMES) else "?"
        results.append((name, conf))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cameras_test_10.json")
    parser.add_argument("--data", default="/home/user/recordings/yaris_20260303_162028")
    parser.add_argument("--yolo", default="yolov8s.pt")
    parser.add_argument("--colors", default="models/color_classifier.pt")
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--cell-w", type=int, default=384)
    parser.add_argument("--cell-h", type=int, default=216)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=800)
    args = parser.parse_args()

    # Load models
    print("Loading YOLO...")
    yolo = YOLO(args.yolo)
    print("Loading color classifier...")
    color_model = load_color_model(args.colors)
    print("Models loaded!")

    # Load camera config
    with open(args.config) as f:
        cfg = json.load(f)
    cameras = cfg.get("analytics", [])
    n_cams = len(cameras)
    cols = args.cols
    rows = (n_cams + cols - 1) // cols

    # Open video files
    caps = []
    cam_ids = []
    for cam in cameras:
        cam_id = cam["id"]
        cam_ids.append(cam_id)
        url = cam["url"]
        if url.startswith("file:///data/"):
            filename = url.replace("file:///data/", "")
            filepath = str(Path(args.data) / filename)
        else:
            filepath = url
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            print(f"WARNING: Cannot open {filepath}")
        caps.append(cap)

    cell_w, cell_h = args.cell_w, args.cell_h
    grid_w = cols * cell_w
    grid_h = rows * cell_h

    print(f"Grid: {cols}x{rows} = {n_cams} cameras, {cell_w}x{cell_h}")
    print("Press 'q' to quit")

    frame_count = 0
    fps_start = time.time()
    fps_display = 0.0

    while True:
        frames = []
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            frames.append(frame if ret else np.zeros((720, 1280, 3), dtype=np.uint8))

        # Batch YOLO inference on all cameras
        results = yolo.predict(frames, imgsz=args.imgsz, conf=args.conf,
                               classes=[0], verbose=False, device=0)

        # Build grid
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for idx in range(n_cams):
            frame = frames[idx]
            r = results[idx]
            row = idx // cols
            col = idx % cols

            # Get person boxes
            boxes = []
            confs_det = []
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].item()
                    bh = y2 - y1
                    bw = x2 - x1
                    # Filter: height, aspect ratio, edges
                    if bh < MIN_BBOX_HEIGHT:
                        continue
                    if bh / max(bw, 1) < MIN_ASPECT_RATIO:
                        continue
                    h_frame, w_frame = frame.shape[:2]
                    if x1 < EDGE_MARGIN or x2 > w_frame - EDGE_MARGIN:
                        continue
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))
                    confs_det.append(conf)

            # Classify colors
            if boxes:
                color_results = classify_colors(color_model, frame, boxes)
            else:
                color_results = []

            # Draw detections
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                if i < len(color_results):
                    color_name, color_conf = color_results[i]
                else:
                    color_name, color_conf = "?", 0.0

                bgr = COLOR_BGR.get(color_name, (200, 200, 200))
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                label = f"{color_name} {int(color_conf * 100)}%"
                # Background for text
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), bgr, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Resize to cell
            cell = cv2.resize(frame, (cell_w, cell_h))
            y_off = row * cell_h
            x_off = col * cell_w
            grid[y_off:y_off + cell_h, x_off:x_off + cell_w] = cell

            # Camera label (top-left)
            cv2.putText(grid, cam_ids[idx], (x_off + 5, y_off + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Detection count (bottom-left)
            if boxes:
                cv2.putText(grid, f"{len(boxes)} det", (x_off + 5, y_off + cell_h - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # FPS counter
        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()

        cv2.putText(grid, f"FPS: {fps_display:.1f}", (grid_w - 150, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Race Vision — YOLO + Color", grid)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        if cap and cap.isOpened():
            cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
