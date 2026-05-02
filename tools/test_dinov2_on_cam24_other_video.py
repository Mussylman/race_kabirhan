#!/usr/bin/env python3
"""test_dinov2_on_cam24_other_video.py — in-domain test для production
DINOv2 SGIE на ДРУГОМ cam-24 видео (не том, что использовалось для
сбора prototypes).

Цель — отделить эффект distributional shift между камерами от собственно
качества модели. Если на cam-24 (in-domain) accuracy ≥ 78%, значит
production setup OK, нужны cross-camera prototypes для cam-13.
Если и тут < 78% — модель в принципе плоха.

Шаги:
  1. Sample frames (stride=30) из kamera_24_174252_END174647.mp4
  2. YOLO11s person detect (same params как в diagnose_color_classifier.py)
  3. На каждый detection: re-crop, prep for nvinfer, infer TRT, parser
  4. Report: distribution (color + unknown), confidence histogram, gallery
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
VIDEO = REPO / "data/videos/test_full_loop/yaris_20260421_174240/kamera_24_174252_END174647.mp4"
YOLO_ONNX = REPO / "models" / "yolo11s_person_960.onnx"
PROD_BENCH = Path("/home/ipodrom/race_vision_bench/dinov2_prod")
ENGINE = PROD_BENCH / "dinov2_base_b1_gpu0_fp16.engine"
PROTOS_NPZ = PROD_BENCH / "prototypes_dinov2_v1.npz"
OUT_DIR = REPO / "output" / "phase4_cam24_indomain_2026-04-27"

SAMPLE_STRIDE = 30
DET_CONF = 0.25
DET_IOU  = 0.5
DET_IMGSZ = 960
MIN_BBOX_H = 25
MIN_SIM = 0.55

LABELS_COLOR_TXT = {"blue": 0, "green": 1, "purple": 2, "red": 3, "yellow": 4}
COLOR_UNKNOWN_ID = 255

NVINFER_OFFSETS = np.array([123.675, 116.28, 103.53], dtype=np.float32)
NVINFER_SCALE   = np.float32(0.01735)


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
        return COLOR_UNKNOWN_ID, max_sim, "unknown"
    return LABELS_COLOR_TXT[classes[best]], max_sim, classes[best]


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
        self.in_size = int(np.prod(ish)) * 4; self.out_size = int(np.prod(osh)) * 4
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


def write_gallery(out_path: Path, items_by_color: dict, crops_subdir: str):
    """Группировка по pred class — чтобы быстро глазами оценить
    что попало в каждый bucket."""
    parts = [
        '<!doctype html><html><head><meta charset="utf-8">',
        '<title>cam-24 in-domain — DINOv2 prod predictions</title>',
        '<style>',
        '  body { font-family: system-ui, monospace; background:#1a1a1a; color:#ddd; margin:0; padding:12px; }',
        '  h2 { color:#fff; font-size:14px; margin:14px 0 6px 0; padding:4px 8px; background:#333; }',
        '  .grid { display:grid; grid-template-columns: repeat(8, 1fr); gap:4px; margin-bottom:14px; }',
        '  .card { background:#2a2a2a; border-radius:3px; padding:3px; }',
        '  .card img { max-width:100%; max-height:200px; display:block; margin:0 auto; }',
        '  .meta { font-size:10px; color:#aaa; padding:2px; word-break:break-all; }',
        '</style></head><body>',
        '<h1 style="color:#fff">cam-24 in-domain test (DINOv2 prod)</h1>',
    ]
    for color in ["green", "yellow", "red", "blue", "unknown"]:
        items = items_by_color.get(color, [])
        if not items: continue
        parts.append(f'<h2>{color} — {len(items)} crops</h2>')
        parts.append('<div class="grid">')
        for it in items[:160]:
            sim = it["max_sim"]
            parts.append(
                f'<div class="card">'
                f'<img src="{escape(crops_subdir)}/{escape(it["crop"])}" loading="lazy">'
                f'<div class="meta">{escape(it["crop"])}<br>sim={sim:.3f}</div>'
                f'</div>'
            )
        parts.append('</div>')
    parts.append('</body></html>')
    out_path.write_text("\n".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-crops", action="store_true",
                    help="save per-detection crops + gallery")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    crops_dir = OUT_DIR / "crops"
    if args.save_crops:
        crops_dir.mkdir(exist_ok=True)

    print("=" * 76)
    print("CAM-24 in-domain test (DINOv2 prod TRT)")
    print("=" * 76)
    print(f"  video:        {VIDEO.name}")
    print(f"  yolo:         {YOLO_ONNX.name}")
    print(f"  engine:       {ENGINE.name}")
    print(f"  prototypes:   {PROTOS_NPZ.name} (built from cam-24 OTHER video)")
    print(f"  threshold:    {MIN_SIM}")
    print(f"  save crops:   {args.save_crops}")
    print()

    # Load video
    cap = cv2.VideoCapture(str(VIDEO))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  total frames: {total}, fps={fps:.2f}, "
          f"duration={total/fps:.1f}s")

    # Sample frames
    print("=== sampling frames (stride=30) ===")
    frames: list[tuple[int, np.ndarray]] = []
    idx = 0
    while True:
        ok = cap.grab()
        if not ok: break
        if idx % SAMPLE_STRIDE == 0:
            ok2, f = cap.retrieve()
            if ok2 and f is not None:
                frames.append((idx, f))
        idx += 1
    cap.release()
    print(f"  sampled {len(frames)} frames")
    print()

    # Load YOLO + TRT + protos
    print("=== loading YOLO + TRT + prototypes ===")
    from ultralytics import YOLO
    yolo = YOLO(str(YOLO_ONNX), task="detect")
    runner = TrtRunner(ENGINE)
    data = np.load(PROTOS_NPZ, allow_pickle=False)
    protos = data["prototypes"].astype(np.float32)
    classes = [str(c) for c in data["classes"]]
    print()

    # Process
    print("=== running YOLO + TRT + parser ===")
    t0 = time.time()
    items_by_color = defaultdict(list)
    counts = Counter()
    sims_by_color = defaultdict(list)
    confs_by_color = defaultdict(list)
    n_dets_total = 0

    for frame_idx, frame in frames:
        h_img, w_img = frame.shape[:2]
        results = yolo.predict(source=frame, imgsz=DET_IMGSZ,
                               conf=DET_CONF, iou=DET_IOU,
                               classes=[0], device=0, half=False, verbose=False)
        if not results: continue
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0: continue
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for det_i, (box, det_conf) in enumerate(zip(boxes, confs)):
            x1, y1, x2, y2 = box.tolist()
            bbox_h = y2 - y1
            if bbox_h < MIN_BBOX_H: continue
            x1 = max(0, int(x1)); y1 = max(0, int(y1))
            x2 = min(w_img, int(x2)); y2 = min(h_img, int(y2))
            if x2 <= x1 or y2 <= y1: continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue

            x = preprocess(crop)
            emb = runner.infer(x)
            pred_id, max_sim, pred_name = cpp_parser(emb, protos, classes, MIN_SIM)

            n_dets_total += 1
            counts[pred_name] += 1
            sims_by_color[pred_name].append(max_sim)
            confs_by_color[pred_name].append(float(det_conf))

            if args.save_crops:
                crop_name = f"frame_{frame_idx:06d}_det_{det_i:02d}.jpg"
                cv2.imwrite(str(crops_dir / crop_name), crop,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                items_by_color[pred_name].append({
                    "crop": crop_name,
                    "max_sim": max_sim,
                    "det_conf": float(det_conf),
                    "h": int(bbox_h),
                    "w": int(x2 - x1),
                })

    elapsed = time.time() - t0
    print(f"  done: {n_dets_total} detections in {elapsed:.1f}s")
    print()

    # Report
    print("=== distribution ===")
    print(f"  total detections: {n_dets_total}")
    for c in ["green", "yellow", "red", "blue", "unknown"]:
        n = counts[c]
        pct = n / max(n_dets_total, 1) * 100
        sims = sims_by_color[c]
        if sims:
            print(f"  {c:8s}: {n:5d} ({pct:5.1f}%) "
                  f"sim mean={np.mean(sims):.3f} median={np.median(sims):.3f} "
                  f"min={np.min(sims):.3f} max={np.max(sims):.3f}")
        else:
            print(f"  {c:8s}: {n:5d} ({pct:5.1f}%)")
    print()

    # Confidence buckets
    print("=== max_sim distribution buckets (all detections) ===")
    all_sims = []
    for sims in sims_by_color.values(): all_sims.extend(sims)
    all_sims = np.array(all_sims)
    if len(all_sims) > 0:
        bins = [0.0, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.01]
        labels = ["<0.4", "0.4-0.5", "0.5-0.55", "0.55-0.6", "0.6-0.7",
                  "0.7-0.8", "0.8-0.9", "≥0.9"]
        h, _ = np.histogram(all_sims, bins=bins)
        for label, n in zip(labels, h):
            pct = n / len(all_sims) * 100
            bar = "█" * int(pct / 2)
            print(f"  {label:>10s}: {n:5d} ({pct:5.1f}%) {bar}")
    print()

    # Save artifacts
    summary = {
        "video": str(VIDEO),
        "n_frames_sampled": len(frames),
        "n_detections":     n_dets_total,
        "counts":           dict(counts),
        "threshold":        MIN_SIM,
        "engine":           str(ENGINE),
        "prototypes":       str(PROTOS_NPZ),
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  summary: {summary_path}")

    if args.save_crops:
        gallery_path = OUT_DIR / "gallery.html"
        write_gallery(gallery_path, items_by_color, crops_dir.name)
        print(f"  gallery: {gallery_path}")

    print()
    print("=" * 76)
    reject_pct = counts["unknown"] / max(n_dets_total, 1) * 100
    accept_pct = 100 - reject_pct
    print(f"=== HEADLINE: accepted as colour: {accept_pct:.1f}% / rejected: {reject_pct:.1f}% ===")
    print("=" * 76)


if __name__ == "__main__":
    main()
