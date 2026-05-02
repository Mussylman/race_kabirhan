#!/usr/bin/env python3
"""analyze_dinov2_preprocess_variants.py — диагностика Phase 4 fail.

Сравниваем 3 preprocessing варианта на тех же 420 crops:
  V1 (PROD):     stretch resize 224×224 + nvinfer offsets/scale  ← текущий sgie_color.txt
  V2 (DS-AR1):   maintain-aspect-ratio=1 emulation: letterbox to 224 + symmetric pad mean
  V3 (HF-LIKE):  resize shortest_edge=256 + center_crop 224×224 + ImageNet normalize

Цель — понять, какой preprocessing fix вернёт jockey accuracy к 78%+.
Если V2 ≈ V1 → DS option недостаточен, нужен HF-path
Если V3 >> V1 → preprocessing root cause, надо менять sgie_color.txt
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DIAGNOSE = REPO / "output" / "diagnose_v4_2026-04-25"
RAW_FRAMES = DIAGNOSE / "raw_frames"
LABELS_JSON = DIAGNOSE / "labels.json"
CROPS_INDEX = DIAGNOSE / "crops_index.json"

PROD_BENCH = Path("/home/ipodrom/race_vision_bench/dinov2_prod")
ENGINE = PROD_BENCH / "dinov2_base_b1_gpu0_fp16.engine"
PROTOS_NPZ = PROD_BENCH / "prototypes_dinov2_v1.npz"

OUT_DIR = REPO / "output" / "phase4_regression_2026-04-27"

LABELS_COLOR_TXT = {"blue": 0, "green": 1, "purple": 2, "red": 3, "yellow": 4}
COLOR_UNKNOWN_ID = 255
MIN_SIM = 0.55

NVINFER_OFFSETS = np.array([123.675, 116.28, 103.53], dtype=np.float32)
NVINFER_SCALE   = np.float32(0.01735)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# nvinfer offsets/scale converted to per-channel mean/std equivalents
# (для V2 используем ту же ImageNet single-scale approximation)


def to_chw_batch(rgb_norm: np.ndarray) -> np.ndarray:
    return np.transpose(rgb_norm, (2, 0, 1))[None, :].copy()


def preprocess_v1_prod(crop_bgr: np.ndarray) -> np.ndarray:
    """V1 = текущий sgie_color.txt (stretch + nvinfer single-scale)."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32)
    x = (x - NVINFER_OFFSETS) * NVINFER_SCALE
    return to_chw_batch(x)


def preprocess_v2_ds_ar1(crop_bgr: np.ndarray) -> np.ndarray:
    """V2 = maintain-aspect-ratio=1 + symmetric-padding=1 (как DS).
    Resize так чтобы long_dim=224, short_dim padded mean colour."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if h >= w:
        new_h = 224
        new_w = max(1, int(round(w * 224 / h)))
    else:
        new_w = 224
        new_h = max(1, int(round(h * 224 / w)))
    rgb_resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # symmetric pad with offsets value (≈ mean image colour)
    pad_h = 224 - new_h
    pad_w = 224 - new_w
    top = pad_h // 2; bottom = pad_h - top
    left = pad_w // 2; right = pad_w - left
    pad_value = (
        int(round(NVINFER_OFFSETS[0])),
        int(round(NVINFER_OFFSETS[1])),
        int(round(NVINFER_OFFSETS[2])),
    )
    rgb_padded = cv2.copyMakeBorder(
        rgb_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=pad_value
    )
    x = rgb_padded.astype(np.float32)
    x = (x - NVINFER_OFFSETS) * NVINFER_SCALE
    return to_chw_batch(x)


def preprocess_v3_hf_like(crop_bgr: np.ndarray) -> np.ndarray:
    """V3 = HF AutoImageProcessor стиль:
    resize shortest_edge=256 (зум) + center_crop 224×224 + exact ImageNet."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    # zoom so shortest_edge=256
    if h <= w:
        scale = 256.0 / h
        new_h, new_w = 256, max(1, int(round(w * scale)))
    else:
        scale = 256.0 / w
        new_w, new_h = 256, max(1, int(round(h * scale)))
    rgb_resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # center crop 224×224
    cy = (new_h - 224) // 2
    cx = (new_w - 224) // 2
    crop = rgb_resized[cy:cy+224, cx:cx+224]
    x = crop.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return to_chw_batch(x)


def cpp_parser(emb, protos, classes, thr):
    norm = float(np.linalg.norm(emb)) + 1e-12
    sims = (protos @ emb) / norm
    best = int(np.argmax(sims)); max_sim = float(sims[best])
    if max_sim < thr:
        return COLOR_UNKNOWN_ID, max_sim, "unknown"
    return LABELS_COLOR_TXT[classes[best]], max_sim, classes[best]


class TrtRunner:
    def __init__(self, engine_path):
        import tensorrt as trt
        from cuda.bindings import runtime as cudart
        self._cudart = cudart
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(open(engine_path, "rb").read())
        self.ctx = self.engine.create_execution_context()
        self.in_name = self.engine.get_tensor_name(0)
        self.out_name = self.engine.get_tensor_name(1)
        self.in_shape = tuple(self.engine.get_tensor_shape(self.in_name))
        self.out_shape = tuple(self.engine.get_tensor_shape(self.out_name))
        self.in_size = int(np.prod(self.in_shape)) * 4
        self.out_size = int(np.prod(self.out_shape)) * 4
        _, self.d_in = cudart.cudaMalloc(self.in_size)
        _, self.d_out = cudart.cudaMalloc(self.out_size)
        self.ctx.set_tensor_address(self.in_name, int(self.d_in))
        self.ctx.set_tensor_address(self.out_name, int(self.d_out))
        _, self.stream = cudart.cudaStreamCreate()

    def infer(self, x):
        c = self._cudart
        c.cudaMemcpyAsync(self.d_in, x.ctypes.data, self.in_size,
                          c.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
        self.ctx.execute_async_v3(self.stream)
        out = np.empty(self.out_shape, dtype=np.float32)
        c.cudaMemcpyAsync(out.ctypes.data, self.d_out, self.out_size,
                          c.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)
        c.cudaStreamSynchronize(self.stream)
        return out[0]


def evaluate(name, prep_fn, runner, protos, classes, items, label_map, frame_cache):
    n_jockey = sum(1 for cid, _ in items if label_map[cid] != "not_jockey")
    n_nj     = sum(1 for cid, _ in items if label_map[cid] == "not_jockey")
    n_jc_correct = 0; n_jc_wrong = 0; n_jc_reject = 0
    n_nj_reject = 0; n_nj_passed = 0; n_nj_conf_wrong = 0
    cm_rows = ["blue","green","red","yellow","not_jockey"]
    cm_cols = ["blue","green","red","yellow","unknown"]
    cm = {r:{c:0 for c in cm_cols} for r in cm_rows}

    for cid, entry in items:
        true_lbl = label_map[cid]
        crop = entry["__crop__"]
        x = prep_fn(crop)
        emb = runner.infer(x)
        pred_id, max_sim, pred_name = cpp_parser(emb, protos, classes, MIN_SIM)
        if true_lbl in cm and pred_name in cm[true_lbl]:
            cm[true_lbl][pred_name] += 1
        if true_lbl != "not_jockey":
            if pred_name == true_lbl:   n_jc_correct += 1
            elif pred_name == "unknown":n_jc_reject  += 1
            else:                       n_jc_wrong   += 1
        else:
            if pred_name == "unknown":  n_nj_reject  += 1
            else:
                n_nj_passed += 1
                if max_sim > 0.9: n_nj_conf_wrong += 1

    print(f"\n=== {name} ===")
    print(f"  jockey accuracy: {n_jc_correct}/{n_jockey} = {n_jc_correct/max(n_jockey,1):.1%} "
          f"(wrong={n_jc_wrong}, rejected={n_jc_reject})")
    print(f"  NJ reject rate:  {n_nj_reject}/{n_nj} = {n_nj_reject/max(n_nj,1):.1%} "
          f"(passed_color={n_nj_passed}, conf_wrong={n_nj_conf_wrong})")
    print(f"  confusion matrix:")
    print("       " + "".join(f"{c:>10s}" for c in cm_cols))
    for r in cm_rows:
        print(f"  {r:6s}" + "".join(f"{cm[r][c]:>10d}" for c in cm_cols))
    return {
        "name": name,
        "jockey_acc": n_jc_correct/max(n_jockey,1),
        "nj_reject":  n_nj_reject/max(n_nj,1),
        "nj_conf_wrong": n_nj_conf_wrong/max(n_nj,1),
        "cm": cm,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 76)
    print("PHASE 4 — preprocessing variant comparison")
    print("=" * 76)

    crops_index = json.loads(CROPS_INDEX.read_text())
    labels_raw = json.loads(LABELS_JSON.read_text())
    label_map = {}
    for color, ids in labels_raw.items():
        for cid in ids: label_map[cid] = color

    runner = TrtRunner(ENGINE)
    data = np.load(PROTOS_NPZ, allow_pickle=False)
    protos = data["prototypes"].astype(np.float32)
    classes = [str(c) for c in data["classes"]]

    # Pre-load crops once (faster than re-reading 3x)
    print("=== loading crops from raw_frames ===")
    items = []
    last_path = None; cached = None
    for entry in crops_index:
        cid = entry["crop"].replace(".jpg","")
        if cid not in label_map: continue
        fp = str(RAW_FRAMES / entry["frame"])
        if fp != last_path:
            cached = cv2.imread(fp); last_path = fp
        if cached is None: continue
        x1,y1,x2,y2 = entry["bbox"]
        h_img,w_img = cached.shape[:2]
        x1=max(0,int(x1)); y1=max(0,int(y1))
        x2=min(w_img,int(x2)); y2=min(h_img,int(y2))
        if x2<=x1 or y2<=y1: continue
        crop = cached[y1:y2, x1:x2]
        if crop.size == 0: continue
        e2 = dict(entry); e2["__crop__"] = crop
        items.append((cid, e2))
    print(f"  loaded {len(items)} labeled crops with valid bbox+frame")

    results = []
    for name, fn in [
        ("V1 (PROD: stretch + nvinfer offsets)", preprocess_v1_prod),
        ("V2 (DS-AR1: maintain-aspect-ratio=1 + symmetric pad)", preprocess_v2_ds_ar1),
        ("V3 (HF-LIKE: resize-256 + center-crop-224 + ImageNet)", preprocess_v3_hf_like),
    ]:
        results.append(evaluate(name, fn, runner, protos, classes, items, label_map, None))

    print()
    print("=" * 76)
    print("=== SUMMARY ===")
    print(f"  {'variant':70s} {'jockey':>10s} {'NJ rej':>10s} {'conf_wr':>10s}")
    for r in results:
        print(f"  {r['name']:70s} {r['jockey_acc']:>10.1%} "
              f"{r['nj_reject']:>10.1%} {r['nj_conf_wrong']:>10.1%}")
    print()
    print("  Acceptance gates: jockey ≥ 78%, NJ_reject ≥ 96%, NJ_conf_wrong ≤ 2%")

    (OUT_DIR / "preprocess_variants.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    print(f"  saved: {OUT_DIR / 'preprocess_variants.json'}")


if __name__ == "__main__":
    main()
