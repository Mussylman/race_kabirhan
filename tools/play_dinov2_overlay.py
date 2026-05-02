#!/usr/bin/env python3
"""play_dinov2_overlay.py — proigrat' video s overlay'em DINOv2 prod
predictions (bbox + color label + sim) cherez cv2.imshow (X DISPLAY).

Vizualnaya verifikatsiya kachestva production SGIE bez gallery.

Klyuchi:
  SPACE  - pauza/play
  q/ESC  - vyhod
  s      - sohranit' tekushchiy frame v output/...
  +/-    - speed up/down (skip frames)
  [/]    - prev/next single frame (when paused)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = REPO / "data/videos/test_full_loop/yaris_20260421_174240/kamera_24_174252_END174647.mp4"
YOLO_ONNX = REPO / "models" / "yolo11s_person_960.onnx"
PROD_BENCH = Path("/home/ipodrom/race_vision_bench/dinov2_prod")
ENGINE = PROD_BENCH / "dinov2_base_b1_gpu0_fp16.engine"
PROTOS_NPZ = PROD_BENCH / "prototypes_dinov2_v1.npz"

DET_CONF = 0.25
DET_IOU  = 0.5
DET_IMGSZ = 960
MIN_BBOX_H = 25
MIN_SIM = 0.55

LABELS_COLOR_TXT = {"blue": 0, "green": 1, "purple": 2, "red": 3, "yellow": 4}
NVINFER_OFFSETS = np.array([123.675, 116.28, 103.53], dtype=np.float32)
NVINFER_SCALE   = np.float32(0.01735)

# BGR colors for overlay
COLOR_BGR = {
    "blue":    (255,  80,  80),
    "green":   ( 60, 200,  60),
    "red":     ( 40,  40, 220),
    "yellow":  ( 40, 220, 220),
    "unknown": (128, 128, 128),
}


def preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32)
    x = (x - NVINFER_OFFSETS) * NVINFER_SCALE
    return np.transpose(x, (2, 0, 1))[None, :].copy()


def cpp_parser(emb, protos, classes, thr):
    norm = float(np.linalg.norm(emb)) + 1e-12
    sims = (protos @ emb) / norm
    best = int(np.argmax(sims)); max_sim = float(sims[best])
    if max_sim < thr:
        return "unknown", max_sim
    return classes[best], max_sim


class TrtRunner:
    def __init__(self, p):
        import tensorrt as trt
        from cuda.bindings import runtime as cudart
        self.c = cudart
        rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.eng = rt.deserialize_cuda_engine(open(p, "rb").read())
        self.ctx = self.eng.create_execution_context()
        i = self.eng.get_tensor_name(0); o = self.eng.get_tensor_name(1)
        ish = tuple(self.eng.get_tensor_shape(i))
        osh = tuple(self.eng.get_tensor_shape(o))
        self.in_size = int(np.prod(ish)) * 4
        self.out_size = int(np.prod(osh)) * 4
        self.osh = osh
        _, self.d_in = cudart.cudaMalloc(self.in_size)
        _, self.d_out = cudart.cudaMalloc(self.out_size)
        self.ctx.set_tensor_address(i, int(self.d_in))
        self.ctx.set_tensor_address(o, int(self.d_out))
        _, self.stream = cudart.cudaStreamCreate()

    def infer(self, x):
        c = self.c
        c.cudaMemcpyAsync(self.d_in, x.ctypes.data, self.in_size,
                          c.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
        self.ctx.execute_async_v3(self.stream)
        out = np.empty(self.osh, dtype=np.float32)
        c.cudaMemcpyAsync(out.ctypes.data, self.d_out, self.out_size,
                          c.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)
        c.cudaStreamSynchronize(self.stream)
        return out[0]


def draw_overlay(frame, dets, frame_idx, fps_actual, paused, stride):
    out = frame.copy()
    h, w = out.shape[:2]
    counts = {}
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        color_name = d["color"]
        sim = d["sim"]
        det_conf = d["det_conf"]
        bgr = COLOR_BGR.get(color_name, (200, 200, 200))
        thick = 2 if color_name != "unknown" else 1
        cv2.rectangle(out, (x1, y1), (x2, y2), bgr, thick)
        label = f"{color_name} {sim:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        ly = max(th + 2, y1 - 2)
        cv2.rectangle(out, (x1, ly - th - 2), (x1 + tw + 2, ly + 2), bgr, -1)
        text_col = (0, 0, 0) if color_name in ("yellow", "unknown") else (255, 255, 255)
        cv2.putText(out, label, (x1 + 1, ly - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_col, 1, cv2.LINE_AA)
        counts[color_name] = counts.get(color_name, 0) + 1

    # HUD bar
    hud_h = 28
    cv2.rectangle(out, (0, 0), (w, hud_h), (0, 0, 0), -1)
    status = "PAUSED" if paused else f"{fps_actual:5.1f} fps"
    summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    text = f"frame {frame_idx:6d}  | {status} | stride={stride} | {summary or 'no det'}"
    cv2.putText(out, text, (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(DEFAULT_VIDEO))
    ap.add_argument("--start", type=int, default=0, help="start frame")
    ap.add_argument("--stride", type=int, default=1,
                    help="process every N-th frame")
    ap.add_argument("--scale", type=float, default=0.7,
                    help="display scale (0<s<=1)")
    args = ap.parse_args()

    print(f"Video:  {args.video}")
    print(f"Engine: {ENGINE}")
    print()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"FATAL: cannot open {args.video}", file=sys.stderr)
        sys.exit(1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_src = cap.get(cv2.CAP_PROP_FPS)
    print(f"  total: {total} frames @ {fps_src:.1f} fps")
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    from ultralytics import YOLO
    yolo = YOLO(str(YOLO_ONNX), task="detect")
    runner = TrtRunner(ENGINE)
    data = np.load(PROTOS_NPZ, allow_pickle=False)
    protos = data["prototypes"].astype(np.float32)
    classes = [str(c) for c in data["classes"]]
    print(f"  protos: {classes}")
    print()
    print("KEYS: SPACE=pause/play  q/ESC=quit  s=save  +/-=stride  [/]=step")
    print()

    win = "DINOv2 prod overlay (cam-24 test_full_loop)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    save_dir = REPO / "output" / "play_overlay_saves"
    save_dir.mkdir(parents=True, exist_ok=True)

    paused = False
    stride = max(1, args.stride)
    frame_idx = args.start
    last_t = time.time()
    fps_smooth = 0.0
    last_render = None
    last_dets = []

    def grab_and_process(idx):
        # Skip stride-1 frames cheaply, then read+process current
        nonlocal frame_idx
        for _ in range(stride - 1):
            if not cap.grab(): return None, None
            frame_idx += 1
        ok, frame = cap.read()
        if not ok or frame is None: return None, None
        frame_idx += 1

        h_img, w_img = frame.shape[:2]
        results = yolo.predict(source=frame, imgsz=DET_IMGSZ,
                               conf=DET_CONF, iou=DET_IOU,
                               classes=[0], device=0, half=False, verbose=False)
        dets = []
        if results and results[0].boxes is not None:
            r = results[0]
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for box, det_conf in zip(boxes, confs):
                x1, y1, x2, y2 = box.tolist()
                if (y2 - y1) < MIN_BBOX_H: continue
                x1 = max(0, int(x1)); y1 = max(0, int(y1))
                x2 = min(w_img, int(x2)); y2 = min(h_img, int(y2))
                if x2 <= x1 or y2 <= y1: continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue
                emb = runner.infer(preprocess(crop))
                color, sim = cpp_parser(emb, protos, classes, MIN_SIM)
                dets.append({
                    "bbox": (x1, y1, x2, y2),
                    "color": color,
                    "sim": sim,
                    "det_conf": float(det_conf),
                })
        return frame, dets

    while True:
        if not paused:
            frame, dets = grab_and_process(frame_idx)
            if frame is None:
                print("\nEnd of video.")
                break
            now = time.time()
            dt = now - last_t
            if dt > 0:
                fps_smooth = 0.85 * fps_smooth + 0.15 * (1.0 / dt)
            last_t = now
            last_render = draw_overlay(frame, dets, frame_idx, fps_smooth,
                                       paused, stride)
            last_dets = dets
        if last_render is not None:
            disp = last_render
            if args.scale != 1.0:
                disp = cv2.resize(disp, None, fx=args.scale, fy=args.scale,
                                  interpolation=cv2.INTER_AREA)
            cv2.imshow(win, disp)

        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key in (ord('q'), 27):  # q or ESC
            break
        elif key == ord(' '):
            paused = not paused
            print(f"  {'PAUSED' if paused else 'PLAY'} @ frame {frame_idx}")
        elif key == ord('s'):
            sp = save_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(sp), last_render)
            print(f"  saved: {sp}")
        elif key in (ord('+'), ord('=')):
            stride = min(stride + 1, 30)
            print(f"  stride={stride}")
        elif key in (ord('-'), ord('_')):
            stride = max(1, stride - 1)
            print(f"  stride={stride}")
        elif key == ord(']') and paused:
            frame, dets = grab_and_process(frame_idx)
            if frame is not None:
                last_render = draw_overlay(frame, dets, frame_idx,
                                           fps_smooth, paused, stride)
                last_dets = dets
        elif key == ord('[') and paused:
            target = max(0, frame_idx - 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            frame_idx = target
            frame, dets = grab_and_process(frame_idx)
            if frame is not None:
                last_render = draw_overlay(frame, dets, frame_idx,
                                           fps_smooth, paused, stride)
                last_dets = dets

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nLast frame_idx: {frame_idx}")


if __name__ == "__main__":
    main()
