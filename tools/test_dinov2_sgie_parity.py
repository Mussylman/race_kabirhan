#!/usr/bin/env python3
"""test_dinov2_sgie_parity.py — parity между TRT engine + Python emulation
C++ parser logic, vs HF transformers + same Python emulation.

Шкала проверки:
  TRT path (production)   = embedding(TRT, FP16) → L2 + cosine to 4 protos
                            + argmax + threshold → class_id из labels_color.txt
  HF reference path       = embedding(HF, FP32)  → same downstream logic

Acceptance: agreement ≥ 99/100 на 100 random crops.

Notes:
- Не вызываем libnvdsinfer_racevision.so напрямую — эмулируем
  параллельную математику в Python (она 1:1 с C++ кодом).
- Этот тест проверяет что FP16 TRT embedding не меняет class_id
  decision на нашей 4-prototype схеме.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
WORK = Path("/home/ipodrom/race_vision_bench/dinov2_prod")
ENGINE = WORK / "dinov2_base_b1_gpu0_fp16.engine"
PROTOS_NPZ = WORK / "prototypes_dinov2_v1.npz"
LABELS_JSON = WORK / "blue_collection" / "labels.json"
CROPS_DIR = WORK / "blue_collection" / "crops_raw"

# labels_color.txt: blue=0, green=1, purple=2, red=3, yellow=4
LABELS_COLOR_TXT = {"blue": 0, "green": 1, "purple": 2, "red": 3, "yellow": 4}
# Same as RV_DINOV2_MIN_SIM default in C++ parser
MIN_SIM = 0.55
COLOR_UNKNOWN_ID = 255


def load_protos():
    data = np.load(PROTOS_NPZ, allow_pickle=False)
    protos = data["prototypes"].astype(np.float32)
    classes = [str(c) for c in data["classes"]]
    return protos, classes


def cpp_logic(emb: np.ndarray, protos: np.ndarray, classes: list[str], thr: float):
    """1:1 эмуляция C++ parser logic: L2 norm + cosine sim + argmax + threshold."""
    norm = float(np.linalg.norm(emb)) + 1e-12
    sims = (protos @ emb) / norm
    best = int(np.argmax(sims))
    max_sim = float(sims[best])
    if max_sim < thr:
        return COLOR_UNKNOWN_ID, max_sim, "unknown"
    return LABELS_COLOR_TXT[classes[best]], max_sim, classes[best]


def trt_infer_batch(engine_path: Path, pixel_values: np.ndarray) -> np.ndarray:
    """Run TRT engine batch=1, return [N, 768] embeddings."""
    import tensorrt as trt
    from cuda.bindings import runtime as cudart
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    in_name = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)
    in_shape = tuple(engine.get_tensor_shape(in_name))
    out_shape = tuple(engine.get_tensor_shape(out_name))
    in_size = int(np.prod(in_shape)) * 4
    out_size = int(np.prod(out_shape)) * 4
    err1, d_in = cudart.cudaMalloc(in_size)
    err2, d_out = cudart.cudaMalloc(out_size)
    context.set_tensor_address(in_name, int(d_in))
    context.set_tensor_address(out_name, int(d_out))
    err3, stream = cudart.cudaStreamCreate()
    embs = []
    for i in range(pixel_values.shape[0]):
        single = pixel_values[i:i+1].astype(np.float32)
        cudart.cudaMemcpyAsync(d_in, single.ctypes.data, in_size,
                               cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        context.execute_async_v3(stream)
        out_host = np.empty(out_shape, dtype=np.float32)
        cudart.cudaMemcpyAsync(out_host.ctypes.data, d_out, out_size,
                               cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cudart.cudaStreamSynchronize(stream)
        embs.append(out_host[0].copy())
    cudart.cudaStreamDestroy(stream)
    cudart.cudaFree(d_in)
    cudart.cudaFree(d_out)
    return np.stack(embs)


def main():
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    print("=" * 76)
    print("DINOv2 SGIE parity test: TRT vs HF на 100 random crops + 4 prototypes")
    print("=" * 76)
    print(f"  threshold (RV_DINOV2_MIN_SIM): {MIN_SIM}")
    print(f"  prototypes: {PROTOS_NPZ}")
    print(f"  engine:     {ENGINE}")
    print()

    # Load 100 random labeled crops
    labels = json.loads(LABELS_JSON.read_text())
    rng = np.random.default_rng(42)
    all_ids = []
    for cls, ids in labels.items():
        for cid in ids:
            all_ids.append((cid, cls))
    sample = rng.choice(len(all_ids), 100, replace=False)
    sample_items = [all_ids[i] for i in sample]
    print(f"  loaded {len(sample_items)} crops "
          f"({sum(1 for _, c in sample_items if c == 'blue')} blue, "
          f"{sum(1 for _, c in sample_items if c == 'green')} green, "
          f"{sum(1 for _, c in sample_items if c == 'yellow')} yellow, "
          f"{sum(1 for _, c in sample_items if c == 'red')} red)")

    protos, classes = load_protos()
    print(f"  prototypes: {protos.shape}, classes: {classes}")

    # Preprocess all
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    imgs = [Image.open(CROPS_DIR / f"{cid}.jpg").convert("RGB") for cid, _ in sample_items]
    inputs = processor(images=imgs, return_tensors="pt")
    px = inputs["pixel_values"].cpu().numpy()
    print(f"  preprocess: pixel_values shape {px.shape}")

    # HF baseline
    print()
    print("=== running HF (FP32) baseline ===")
    t0 = time.time()
    model = AutoModel.from_pretrained("facebook/dinov2-base", dtype=torch.float32).to("cuda").eval()
    with torch.inference_mode():
        hf_emb = model(pixel_values=torch.from_numpy(px).cuda()).pooler_output.cpu().numpy()
    print(f"  HF embeddings: {hf_emb.shape}, {time.time()-t0:.1f}s")

    # TRT production path
    print()
    print("=== running TRT (FP16) production path ===")
    t0 = time.time()
    trt_emb = trt_infer_batch(ENGINE, px)
    print(f"  TRT embeddings: {trt_emb.shape}, {time.time()-t0:.1f}s")

    # Apply our C++ parser logic to both
    print()
    print("=== applying C++ parser logic to both ===")
    hf_decisions  = [cpp_logic(hf_emb[i],  protos, classes, MIN_SIM) for i in range(100)]
    trt_decisions = [cpp_logic(trt_emb[i], protos, classes, MIN_SIM) for i in range(100)]

    # Compare class_id (production-relevant decision)
    hf_ids  = [d[0] for d in hf_decisions]
    trt_ids = [d[0] for d in trt_decisions]
    agree = sum(1 for h, t in zip(hf_ids, trt_ids) if h == t)
    print(f"  class_id agreement: {agree}/100 ({agree}%)")

    # Show disagreements
    disagrees = [i for i in range(100) if hf_ids[i] != trt_ids[i]]
    if disagrees:
        print()
        print("=== Disagreements ===")
        for i in disagrees:
            cid, true_lbl = sample_items[i]
            h_id, h_sim, h_name = hf_decisions[i]
            t_id, t_sim, t_name = trt_decisions[i]
            print(f"  {cid:35s} true={true_lbl:7s} | "
                  f"HF: {h_name:8s}(sim={h_sim:.4f}, id={h_id})  "
                  f"TRT: {t_name:8s}(sim={t_sim:.4f}, id={t_id})")
            # margin analysis
            diff = abs(h_sim - t_sim)
            print(f"    sim diff: {diff:.4f} (boundary case если diff < 0.01)")
    else:
        print(f"  ✅ NO disagreements")

    # Also report agreement on "rejected_or_class" sets
    hf_reject = sum(1 for x in hf_ids if x == COLOR_UNKNOWN_ID)
    trt_reject = sum(1 for x in trt_ids if x == COLOR_UNKNOWN_ID)
    print()
    print(f"  HF rejected (sim<{MIN_SIM}):  {hf_reject}/100")
    print(f"  TRT rejected (sim<{MIN_SIM}): {trt_reject}/100")

    print()
    if agree >= 99:
        print(f"✅ PARITY OK ({agree}/100 ≥ 99/100)")
        sys.exit(0)
    else:
        print(f"❌ PARITY FAIL ({agree}/100 < 99/100)")
        sys.exit(1)


if __name__ == "__main__":
    main()
