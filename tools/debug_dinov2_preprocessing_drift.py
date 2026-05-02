#!/usr/bin/env python3
"""debug_dinov2_preprocessing_drift.py — pinpoint preprocessing bug.

Берём 5 known crops каждого класса из blue_collection (тех же, что
строили prototypes), прогоняем через 4 разных пути, смотрим:
  - cosine similarity к каждому из 4 prototypes
  - winner class
  - совпадает ли с истинной меткой

Пути:
  P1: HF-preproc + HF-model      (REF — built prototypes, должно быть 100%)
  P2: HF-preproc + TRT-engine    (parity path — было 100/100 на parity test)
  P3: cv2-HF-emulation + TRT     (мой V3 attempt)
  P4: cv2-stretch + TRT          (PROD — текущий sgie_color.txt)

Если P1 = P2 = 100% и P3, P4 < 100% — preprocessing root cause confirmed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

WORK = Path("/home/ipodrom/race_vision_bench/dinov2_prod")
ENGINE = WORK / "dinov2_base_b1_gpu0_fp16.engine"
PROTOS_NPZ = WORK / "prototypes_dinov2_v1.npz"
LABELS_JSON = WORK / "blue_collection" / "labels.json"
CROPS_DIR = WORK / "blue_collection" / "crops_raw"

NVINFER_OFFSETS = np.array([123.675, 116.28, 103.53], dtype=np.float32)
NVINFER_SCALE   = np.float32(0.01735)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_batch_chw(rgb_norm: np.ndarray) -> np.ndarray:
    return np.transpose(rgb_norm, (2, 0, 1))[None, :].copy()


def cv2_stretch_v1(crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32)
    x = (x - NVINFER_OFFSETS) * NVINFER_SCALE
    return to_batch_chw(x)


def cv2_hf_emulation_v3(crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if h <= w:
        scale = 256.0 / h; new_h, new_w = 256, max(1, int(round(w * scale)))
    else:
        scale = 256.0 / w; new_w, new_h = 256, max(1, int(round(h * scale)))
    rgb_resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    cy = (new_h - 224) // 2; cx = (new_w - 224) // 2
    crop = rgb_resized[cy:cy+224, cx:cx+224]
    x = crop.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return to_batch_chw(x)


class Trt:
    def __init__(self, p):
        import tensorrt as trt
        from cuda.bindings import runtime as cudart
        self.c = cudart
        rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.eng = rt.deserialize_cuda_engine(open(p, "rb").read())
        self.ctx = self.eng.create_execution_context()
        i = self.eng.get_tensor_name(0); o = self.eng.get_tensor_name(1)
        self.ish = tuple(self.eng.get_tensor_shape(i))
        self.osh = tuple(self.eng.get_tensor_shape(o))
        self.in_size = int(np.prod(self.ish)) * 4
        self.out_size = int(np.prod(self.osh)) * 4
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


def parser(emb, protos, classes, thr=0.55):
    norm = float(np.linalg.norm(emb)) + 1e-12
    sims = (protos @ emb) / norm
    best = int(np.argmax(sims)); max_sim = float(sims[best])
    if max_sim < thr: return "unknown", max_sim, sims
    return classes[best], max_sim, sims


def main():
    print("=" * 80)
    print("DINOv2 preprocessing drift diagnostic")
    print("=" * 80)

    # Sample 5 known crops per class from blue_collection (the ones that BUILT
    # the prototypes — should give near-100% accuracy on REF path)
    labels = json.loads(LABELS_JSON.read_text())
    rng = np.random.default_rng(42)
    sample = []
    for cls, ids in labels.items():
        chosen = rng.choice(ids, size=min(5, len(ids)), replace=False)
        for cid in chosen: sample.append((cid, cls))
    print(f"  sampled {len(sample)} crops "
          f"({len([s for s in sample if s[1]=='blue'])} blue, "
          f"{len([s for s in sample if s[1]=='green'])} green, "
          f"{len([s for s in sample if s[1]=='yellow'])} yellow, "
          f"{len([s for s in sample if s[1]=='red'])} red)")

    # Load prototypes + TRT
    data = np.load(PROTOS_NPZ, allow_pickle=False)
    protos = data["prototypes"].astype(np.float32)
    classes = [str(c) for c in data["classes"]]
    print(f"  protos: {protos.shape}, order={classes}")

    trt = Trt(ENGINE)

    # Load HF model + processor for REF + parity paths
    print("  loading HF AutoImageProcessor + AutoModel (FP16)...")
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    hf_model = AutoModel.from_pretrained("facebook/dinov2-base",
                                         dtype=torch.float16).to("cuda").eval()

    # Pre-load PIL images for HF + cv2 mats for cv2 paths
    pil_imgs = []
    cv2_mats = []
    for cid, cls in sample:
        fp = CROPS_DIR / f"{cid}.jpg"
        pil_imgs.append(Image.open(fp).convert("RGB"))
        cv2_mats.append(cv2.imread(str(fp)))

    # === P1: HF preproc + HF model
    print("\n=== P1: HF preproc + HF model (REF — what built prototypes) ===")
    inputs = processor(images=pil_imgs, return_tensors="pt").to("cuda", torch.float16)
    with torch.inference_mode():
        out = hf_model(**inputs)
    p1_emb = out.pooler_output.float().cpu().numpy()  # [N, 768]
    p1_results = [parser(p1_emb[i], protos, classes) for i in range(len(sample))]

    # === P2: HF preproc + TRT engine
    print("=== P2: HF preproc + TRT engine (parity test path) ===")
    px = inputs["pixel_values"].float().cpu().numpy()  # [N, 3, 224, 224] FP32
    p2_emb = np.stack([trt.infer(px[i:i+1].copy()) for i in range(len(sample))])
    p2_results = [parser(p2_emb[i], protos, classes) for i in range(len(sample))]

    # === P3: cv2 HF-emulation + TRT
    print("=== P3: cv2 HF-emulation + TRT engine ===")
    p3_emb = []
    for mat in cv2_mats:
        p3_emb.append(trt.infer(cv2_hf_emulation_v3(mat)))
    p3_emb = np.stack(p3_emb)
    p3_results = [parser(p3_emb[i], protos, classes) for i in range(len(sample))]

    # === P4: cv2 stretch (PROD) + TRT
    print("=== P4: cv2 stretch + nvinfer offsets + TRT (PROD CURRENT) ===")
    p4_emb = []
    for mat in cv2_mats:
        p4_emb.append(trt.infer(cv2_stretch_v1(mat)))
    p4_emb = np.stack(p4_emb)
    p4_results = [parser(p4_emb[i], protos, classes) for i in range(len(sample))]

    # Compare
    print()
    print("=" * 80)
    print("=== per-crop verdict ===")
    print(f"  {'crop_id':<32s} {'true':<7s} {'P1':<10s} {'P2':<10s} {'P3':<10s} {'P4':<10s}")
    print("  " + "-" * 78)
    correct = {1:0, 2:0, 3:0, 4:0}
    for i, (cid, true_cls) in enumerate(sample):
        p1 = p1_results[i][0]; p2 = p2_results[i][0]
        p3 = p3_results[i][0]; p4 = p4_results[i][0]
        if p1 == true_cls: correct[1] += 1
        if p2 == true_cls: correct[2] += 1
        if p3 == true_cls: correct[3] += 1
        if p4 == true_cls: correct[4] += 1
        mark = lambda p: "✓" if p == true_cls else "✗"
        print(f"  {cid:<32s} {true_cls:<7s} "
              f"{p1:<7s}{mark(p1):<3s} {p2:<7s}{mark(p2):<3s} "
              f"{p3:<7s}{mark(p3):<3s} {p4:<7s}{mark(p4):<3s}")
    print()
    n = len(sample)
    print(f"=== accuracy ===")
    print(f"  P1 (HF preproc + HF model)    : {correct[1]:2d}/{n}  ({correct[1]/n:.0%})")
    print(f"  P2 (HF preproc + TRT engine)  : {correct[2]:2d}/{n}  ({correct[2]/n:.0%})")
    print(f"  P3 (cv2 HF-emulation + TRT)   : {correct[3]:2d}/{n}  ({correct[3]/n:.0%})")
    print(f"  P4 (cv2 stretch + TRT — PROD) : {correct[4]:2d}/{n}  ({correct[4]/n:.0%})")
    print()

    # Embedding cosine drift between paths (P1 = reference)
    print("=== embedding cosine similarity to P1 (REF) ===")
    def cossim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    for label, embs in [("P2", p2_emb), ("P3", p3_emb), ("P4", p4_emb)]:
        sims = [cossim(p1_emb[i], embs[i]) for i in range(n)]
        print(f"  {label} vs P1:  mean={np.mean(sims):.4f}  "
              f"median={np.median(sims):.4f}  min={np.min(sims):.4f}")
    print()

    print("=" * 80)
    if correct[3] >= correct[4] + 5:
        print("VERDICT: P3 (cv2 HF emulation) significantly better than P4 (PROD).")
        print("→ Production preprocessing is the bug. Need: maintain-aspect-ratio=1")
        print("  + per-channel-mean (а не single offsets), либо custom preproc layer.")
    elif correct[4] < n // 2:
        print("VERDICT: All cv2 paths fail. Maybe cv2 INTER_LINEAR ≠ PIL.Image bilinear,")
        print("или DINOv2 чувствителен к JPEG re-encode artifacts. Нужен HF processor.")
    else:
        print("VERDICT: see numbers above.")


if __name__ == "__main__":
    main()
