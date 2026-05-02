"""
test_detection_video.py — Person detection + color classification on a video file.

Usage:
    python3 tools/test_detection_video.py <video_path> [--out output.mp4] [--conf 0.35] [--every N]

Models:
    - YOLO:  models/jockey_yolov11s.pt   (person detector)
    - Color: models/color_classifier_v3.pt (5 classes: blue/green/purple/red/yellow)

Torso crop (matches C++ color_infer.h):
    top=10%, bottom=40%, left/right margin=20%
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
# cuDNN 9.21.0 on system vs torch built with 9.2.0 — disable to avoid SUBLIBRARY_VERSION_MISMATCH
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torchvision.transforms as T
from ultralytics import YOLO


class SimpleColorCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

RACE_VISION_ROOT = Path(__file__).resolve().parent.parent

YOLO_MODEL  = RACE_VISION_ROOT / "yolo11s.pt"
COLOR_MODEL = RACE_VISION_ROOT / "models" / "color_classifier_v4.pt"

COLOR_NAMES = ["blue", "green", "purple", "red", "yellow"]
COLOR_BGR   = {
    "blue":   (200,  80,  10),
    "green":  ( 30, 180,  30),
    "purple": (180,  30, 180),
    "red":    ( 20,  20, 200),
    "yellow": ( 20, 200, 220),
}

TORSO_TOP    = 0.10
TORSO_BOTTOM = 0.40
TORSO_LEFT   = 0.20
TORSO_RIGHT  = 0.20
CROP_SIZE    = 128

# Detection filters (match pipeline.h)
MIN_BBOX_HEIGHT  = 40
MIN_ASPECT_RATIO = 0.2
EDGE_MARGIN      = 5
MIN_CROP_PIXELS  = 200
MAX_CROP_PIXELS  = 30000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

color_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((CROP_SIZE, CROP_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_color_model(path: Path):
    # Instantiate fresh architecture, load only state_dict — avoids cuDNN serialization issues
    saved = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(saved, dict) and "model_state_dict" in saved:
        state = saved["model_state_dict"]
    elif hasattr(saved, "state_dict"):
        state = saved.state_dict()
    else:
        state = saved
    model = SimpleColorCNN(num_classes=5)
    model.load_state_dict(state)
    model.eval()
    return model.to(DEVICE)


def torso_roi(x1, y1, x2, y2, fw, fh):
    bw, bh = x2 - x1, y2 - y1
    tx1 = max(0, min(fw - 1, int(x1 + bw * TORSO_LEFT)))
    ty1 = max(0, min(fh - 1, int(y1 + bh * TORSO_TOP)))
    tx2 = max(0, min(fw - 1, int(x2 - bw * TORSO_RIGHT)))
    ty2 = max(0, min(fh - 1, int(y1 + bh * TORSO_BOTTOM)))
    return tx1, ty1, tx2, ty2


def classify_colors(model, crops_bgr: list) -> list:
    if not crops_bgr:
        return []
    tensors = []
    for crop in crops_bgr:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensors.append(color_transform(rgb))
    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        logits = model(batch)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
    results = []
    for p in probs:
        idx = int(np.argmax(p))
        results.append((COLOR_NAMES[idx], float(p[idx]), p))
    return results


def filter_det(x1, y1, x2, y2, fw, fh) -> bool:
    w, h = x2 - x1, y2 - y1
    if h < MIN_BBOX_HEIGHT:
        return False
    if h > 0 and w / h < MIN_ASPECT_RATIO:
        return False
    if x1 < EDGE_MARGIN or x2 > fw - EDGE_MARGIN:
        return False
    area = w * h
    if area < MIN_CROP_PIXELS or area > MAX_CROP_PIXELS:
        return False
    return True


def draw_results(frame, dets, color_results, frame_idx, fps_est):
    fh, fw = frame.shape[:2]
    for i, (x1, y1, x2, y2, conf) in enumerate(dets):
        cn, cc, _ = color_results[i]
        bgr = COLOR_BGR.get(cn, (128, 128, 128))
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
        tx1, ty1, tx2, ty2 = torso_roi(x1, y1, x2, y2, fw, fh)
        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (255, 255, 0), 1)
        label = f"{cn} {cc:.2f} | {conf:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1, cv2.LINE_AA)
    info = f"frame={frame_idx}  dets={len(dets)}  fps~{fps_est:.1f}"
    cv2.putText(frame, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out",        default="",   help="output video path")
    ap.add_argument("--conf",       type=float, default=0.35)
    ap.add_argument("--every",      type=int,   default=1)
    ap.add_argument("--max-frames",  type=int,   default=0)
    ap.add_argument("--display-fps", type=float, default=30.0, help="playback fps in window")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        sys.exit(f"ERROR: {video_path}")

    print(f"Device: {DEVICE}")
    print(f"[YOLO]  {YOLO_MODEL}")
    yolo = YOLO(str(YOLO_MODEL))

    print(f"[Color] {COLOR_MODEL}")
    color_model = load_color_model(COLOR_MODEL)

    cap     = cv2.VideoCapture(str(video_path))
    fw      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Video] {fw}x{fh}  {src_fps:.1f}fps  {total} frames  ({total/src_fps:.1f}s)")

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, src_fps / max(args.every, 1), (fw, fh))

    total_dets   = 0
    color_counts = {c: 0 for c in COLOR_NAMES}
    frame_idx    = 0
    proc_count   = 0
    t0           = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % args.every != 0:
            continue
        if args.max_frames and proc_count >= args.max_frames:
            break
        proc_count += 1
        frame_t = time.time()

        results = yolo.predict(frame, conf=args.conf, classes=[0],
                               imgsz=960, verbose=False)
        boxes = results[0].boxes

        dets = []
        if boxes is not None and len(boxes):
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                if filter_det(x1, y1, x2, y2, fw, fh):
                    dets.append((x1, y1, x2, y2, conf))

        crops = []
        for x1, y1, x2, y2, _ in dets:
            tx1, ty1, tx2, ty2 = torso_roi(x1, y1, x2, y2, fw, fh)
            crop = frame[ty1:ty2, tx1:tx2]
            crops.append(crop if crop.size > 0 else np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8))

        color_results = classify_colors(color_model, crops)

        total_dets += len(dets)
        for cn, _, _ in color_results:
            color_counts[cn] += 1

        fps_est = proc_count / max(time.time() - t0, 0.001)
        draw_results(frame, dets, color_results, frame_idx, fps_est)

        if writer:
            writer.write(frame)

        cv2.imshow("Detection", frame)
        # throttle to 30 fps display
        elapsed_frame = time.time() - frame_t
        wait_ms = max(1, int((1000 / args.display_fps) - elapsed_frame * 1000))
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC to quit
            break

        if proc_count % 50 == 0 or proc_count == 1:
            print(f"  frame {frame_idx:5d}/{total}  dets={len(dets):2d}  "
                  f"total={total_dets}  fps~{fps_est:.1f}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"Processed {proc_count} frames in {elapsed:.1f}s  ({proc_count/elapsed:.1f} fps)")
    print(f"Total detections: {total_dets}  ({total_dets/max(proc_count,1):.2f}/frame)")
    print(f"\nColor distribution:")
    for c, n in sorted(color_counts.items(), key=lambda x: -x[1]):
        pct = n / max(total_dets, 1) * 100
        bar = "█" * int(pct / 2.5)
        print(f"  {c:8s} {n:5d}  {pct:5.1f}%  {bar}")
    if args.out:
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
