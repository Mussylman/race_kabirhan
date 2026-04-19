"""
Extract N frames from a video, run YOLO + color classifier on each,
and save two images per frame:

  frameNN_det.jpg   — full frame with all YOLO bboxes
  frameNN_cropNN.jpg — per-detection crop, with color probs overlaid

Bypasses DeepStream entirely — pure PyTorch/ultralytics + torch. Run this
to diagnose color classifier behaviour without the full pipeline.

Usage:
    python tools/debug_detect_classify.py <video.mp4> [--frames 5] [--out /tmp/dbg]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from ultralytics import YOLO

# ------------------------------------------------------------------
# Force CPU for color classifier — avoids cuDNN version mismatch on
# this box. YOLO still runs on GPU via ultralytics.
torch.backends.cudnn.enabled = False
DEVICE_YOLO  = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_COLOR = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parent.parent
YOLO_PT    = ROOT / "yolo11s.pt"       # COCO person detector
COLOR_PT   = ROOT / "models" / "color_classifier_v4.pt"

COLOR_NAMES = ["blue", "green", "purple", "red", "yellow"]
COLOR_BGR   = {
    "blue":   (200,  80,  10),
    "green":  ( 30, 180,  30),
    "purple": (180,  30, 180),
    "red":    ( 20,  20, 200),
    "yellow": ( 20, 200, 220),
}

CROP_SIZE     = 128
TORSO_TOP     = 0.10
TORSO_BOTTOM  = 0.40
TORSO_LEFT    = 0.20
TORSO_RIGHT   = 0.20

color_tf = T.Compose([
    T.ToPILImage(),
    T.Resize((CROP_SIZE, CROP_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


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


def load_color_model() -> nn.Module:
    saved = torch.load(str(COLOR_PT), map_location="cpu", weights_only=False)
    state = saved["model_state_dict"] if isinstance(saved, dict) and "model_state_dict" in saved else saved
    m = SimpleColorCNN(num_classes=5)
    m.load_state_dict(state)
    m.eval()
    return m.to(DEVICE_COLOR)


def torso_roi(x1, y1, x2, y2, fw, fh):
    bw, bh = x2 - x1, y2 - y1
    tx1 = max(0, min(fw - 1, int(x1 + bw * TORSO_LEFT)))
    ty1 = max(0, min(fh - 1, int(y1 + bh * TORSO_TOP)))
    tx2 = max(0, min(fw - 1, int(x2 - bw * TORSO_RIGHT)))
    ty2 = max(0, min(fh - 1, int(y1 + bh * TORSO_BOTTOM)))
    return tx1, ty1, tx2, ty2


def classify(model: nn.Module, crop_bgr: np.ndarray) -> tuple[str, float, np.ndarray]:
    if crop_bgr.size == 0:
        return "?", 0.0, np.zeros(5, dtype=np.float32)
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    t   = color_tf(rgb).unsqueeze(0).to(DEVICE_COLOR)
    with torch.no_grad():
        logits = model(t)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return COLOR_NAMES[idx], float(probs[idx]), probs


def draw_probs_panel(crop_full: np.ndarray, crop_torso: np.ndarray,
                     full_name: str, full_conf: float, full_probs: np.ndarray,
                     torso_name: str, torso_conf: float, torso_probs: np.ndarray) -> np.ndarray:
    # Panel: left=full crop, mid=torso crop, right=probs bar chart
    H = 256
    def scale(img):
        h, w = img.shape[:2]
        s = H / max(h, 1)
        return cv2.resize(img, (int(w * s), H)) if img.size else np.zeros((H, H, 3), dtype=np.uint8)

    fh_px, fw_px = crop_full.shape[:2] if crop_full.size else (0, 0)
    th_px, tw_px = crop_torso.shape[:2] if crop_torso.size else (0, 0)

    cf = scale(crop_full)
    ct = scale(crop_torso)
    # Label each crop with its raw pixel size
    cv2.rectangle(cf, (0, 0), (cf.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(cf, f"{fw_px}x{fh_px}", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(ct, (0, 0), (ct.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(ct, f"{tw_px}x{th_px}", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    bar = np.full((H, 300, 3), 30, dtype=np.uint8)
    cv2.putText(bar, f"FULL {fw_px}x{fh_px}: {full_name} {full_conf:.2f}", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    for i, (name, p) in enumerate(zip(COLOR_NAMES, full_probs)):
        y = 30 + i * 18
        w = int(p * 200)
        cv2.rectangle(bar, (80, y - 10), (80 + w, y + 2), COLOR_BGR[name], -1)
        cv2.putText(bar, f"{name}: {p:.2f}", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(bar, f"TORSO {tw_px}x{th_px}: {torso_name} {torso_conf:.2f}", (8, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    for i, (name, p) in enumerate(zip(COLOR_NAMES, torso_probs)):
        y = 152 + i * 18
        w = int(p * 200)
        cv2.rectangle(bar, (80, y - 10), (80 + w, y + 2), COLOR_BGR[name], -1)
        cv2.putText(bar, f"{name}: {p:.2f}", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return np.hstack([cf, ct, bar])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frames",     type=int, default=5,
                    help="how many evenly-spaced frames to pick (ignored if --every-sec set)")
    ap.add_argument("--every-sec",  type=float, default=None,
                    help="pick one frame every N seconds (covers the whole range)")
    ap.add_argument("--start-sec",  type=float, default=10.0)
    ap.add_argument("--end-sec",    type=float, default=None)
    ap.add_argument("--conf",       type=float, default=0.30)
    ap.add_argument("--resize",     default=None,
                    help="resize frame to WxH before detection (mimics mux, e.g. 800x800)")
    ap.add_argument("--label",      default=None,
                    help="label suffix for runs/NN_<label> folder; default derived from video name")
    ap.add_argument("--out-dir",    default=None,
                    help="explicit output folder (bypasses runs/ auto-creation + tee)")
    args = ap.parse_args()

    resize_wh = None
    if args.resize:
        rw, rh = args.resize.lower().split("x")
        resize_wh = (int(rw), int(rh))

    sys.path.insert(0, str(ROOT))
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        from tools.rv_run import new_run_dir, tee_stdout, write_meta
        vid_name = Path(args.video).stem
        label = f"debug_{args.label}" if args.label else f"debug_{vid_name}"
        out_dir = new_run_dir(label)
        tee_stdout(out_dir / "log.txt")
        write_meta(out_dir, {
            "kind":      "debug_detect_classify",
            "video":     args.video,
            "frames":    args.frames,
            "every_sec": args.every_sec,
            "start_sec": args.start_sec,
            "end_sec":   args.end_sec,
            "conf":      args.conf,
            "resize":    args.resize or "native",
        })

    cap = cv2.VideoCapture(args.video)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    end_sec = args.end_sec if args.end_sec else total / fps
    print(f"video: {fw}x{fh} @ {fps:.1f}fps, {total} frames, {total/fps:.1f}s")
    print(f"saving to: {out_dir}")

    print(f"loading YOLO from {YOLO_PT} ...")
    yolo = YOLO(str(YOLO_PT))

    print(f"loading color model from {COLOR_PT} ...")
    color_model = load_color_model()

    if args.every_sec:
        picks = np.arange(args.start_sec * fps, end_sec * fps, args.every_sec * fps).astype(int)
    else:
        picks = np.linspace(args.start_sec * fps, end_sec * fps, args.frames).astype(int)
    print(f"picking {len(picks)} frames: {list(picks[:10])}{'...' if len(picks) > 10 else ''}")

    for i, fidx in enumerate(picks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ok, frame = cap.read()
        if not ok:
            print(f"  frame {fidx}: read failed")
            continue

        if resize_wh is not None:
            frame = cv2.resize(frame, resize_wh, interpolation=cv2.INTER_LINEAR)
            fh, fw = frame.shape[:2]

        # YOLO
        res = yolo.predict(frame, conf=args.conf, classes=[0], imgsz=960,
                           device=DEVICE_YOLO, verbose=False)
        boxes = res[0].boxes
        dets = []
        if boxes is not None and len(boxes):
            for b in boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                c = float(b.conf[0])
                dets.append((x1, y1, x2, y2, c))

        # First: classify all dets so we can colour bboxes
        classifications = []
        for j, (x1, y1, x2, y2, c) in enumerate(dets):
            tx1, ty1, tx2, ty2 = torso_roi(x1, y1, x2, y2, fw, fh)
            full_crop  = frame[y1:y2, x1:x2]
            torso_crop = frame[ty1:ty2, tx1:tx2]
            f_name, f_conf, f_probs = classify(color_model, full_crop)
            t_name, t_conf, t_probs = classify(color_model, torso_crop)
            classifications.append((f_name, f_conf, f_probs, t_name, t_conf, t_probs,
                                    full_crop, torso_crop))

        # Draw bboxes using full-bbox classification colour
        ann = frame.copy()
        for j, ((x1, y1, x2, y2, c), (f_name, f_conf, *_)) in enumerate(zip(dets, classifications)):
            color = COLOR_BGR.get(f_name, (128, 128, 128))
            cv2.rectangle(ann, (x1, y1), (x2, y2), color, 3)
            tx1, ty1, tx2, ty2 = torso_roi(x1, y1, x2, y2, fw, fh)
            cv2.rectangle(ann, (tx1, ty1), (tx2, ty2), (0, 255, 255), 1)
            bw, bh = x2 - x1, y2 - y1
            label = f"#{j} {f_name} {f_conf:.2f} {bw}x{bh}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(ann, (x1, max(y1 - th - 6, 0)),
                          (x1 + tw + 6, max(y1, th + 4)), color, -1)
            cv2.putText(ann, label, (x1 + 3, max(y1 - 4, th + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        det_path = out_dir / f"frame{i:02d}_t{fidx/fps:.1f}s_det.jpg"
        cv2.imwrite(str(det_path), ann)
        print(f"  frame {i} t={fidx/fps:.1f}s: {len(dets)} detections -> {det_path.name}")

        for j, (f_name, f_conf, f_probs, t_name, t_conf, t_probs, full_crop, torso_crop) in enumerate(classifications):
            panel = draw_probs_panel(full_crop, torso_crop,
                                     f_name, f_conf, f_probs,
                                     t_name, t_conf, t_probs)
            panel_path = out_dir / f"frame{i:02d}_t{fidx/fps:.1f}s_crop{j:02d}.jpg"
            cv2.imwrite(str(panel_path), panel)
            fh_px, fw_px = full_crop.shape[:2] if full_crop.size else (0, 0)
            th_px, tw_px = torso_crop.shape[:2] if torso_crop.size else (0, 0)
            print(f"    crop{j}: full {fw_px}x{fh_px} → {f_name} {f_conf:.2f}  "
                  f"torso {tw_px}x{th_px} → {t_name} {t_conf:.2f}")

    cap.release()
    print(f"\ndone. open {out_dir} to see images")


if __name__ == "__main__":
    main()
