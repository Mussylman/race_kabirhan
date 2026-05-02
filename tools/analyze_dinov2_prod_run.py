#!/usr/bin/env python3
"""analyze_dinov2_prod_run.py — Phase 4 regression test для DINOv2-base
prototype-based color SGIE на golden cam-13 video.

Подход:
  1. Берём существующие crops_index.json + labels.json из
     output/diagnose_v4_2026-04-25/ (420+ размеченных crops с
     frame_NNNNNN_det_NN ID-системой).
  2. Re-crop из raw_frames по точным YOLO bbox'ам (NO margin) →
     resize 224×224 (стрейч, как nvinfer maintain-aspect-ratio=0).
  3. Прогоняем через прод-TRT engine (dinov2_base_b1_gpu0_fp16.engine).
  4. Применяем C++ parser logic в Python (1:1 эмуляция, как в
     test_dinov2_sgie_parity.py): L2 norm + cosine sim к 4 prototypes
     + argmax + threshold.
  5. Mатчим pred class_id → labels.json по crop ID (frame+det), без IoU.
  6. Считаем acceptance criteria + confusion matrix + 23 HARD case audit.

Acceptance (Phase 4 verdict):
  - jockey color accuracy ≥ 78%   (Phase 3 HF был 80.8%)
  - not_jockey reject rate ≥ 96%  (Phase 3 HF был 98.8%)
  - confident wrong на NJ (>0.9) ≤ 2%
  - 23 HARD test cases rejected ≥ 18/23 (тот чел + КАМАЗ +
    frame_002310-370)

ESC: Если accuracy < 78% — НЕ STOP, доложить + предложить fix через
     maintain-aspect-ratio=1 в sgie_color.txt. Ждать решения.
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

# labels_color.txt: blue=0, green=1, purple=2, red=3, yellow=4
LABELS_COLOR_TXT = {"blue": 0, "green": 1, "purple": 2, "red": 3, "yellow": 4}
COLOR_UNKNOWN_ID = 255
MIN_SIM = 0.55  # = RV_DINOV2_MIN_SIM default = classifier-threshold в sgie_color.txt

# Production nvinfer preprocessing (как в sgie_color.txt — нам важна
# 1:1 эмуляция, а не exact ImageNet, чтобы Phase 4 матчил production).
#   net-scale-factor = 0.01735     (≈ 1/(255*0.226), single scale)
#   offsets          = 123.675; 116.28; 103.53  (RGB)
#   formula nvinfer: y = (px - offset) * scale
NVINFER_OFFSETS = np.array([123.675, 116.28, 103.53], dtype=np.float32)
NVINFER_SCALE   = np.float32(0.01735)

# 23 HARD NJ cases — frame ranges с твердым false positive у v4 CNN:
#   frame_002310-002370 = "тот чел" + KAMAZ
HARD_NJ_FRAMES = set()
for f in range(2310, 2371, 30):
    HARD_NJ_FRAMES.add(f"frame_{f:06d}")


def preprocess_for_nvinfer(crop_bgr: np.ndarray) -> np.ndarray:
    """1:1 эмуляция nvinfer preprocessing (sgie_color.txt):
       - maintain-aspect-ratio=0 → стрейч до 224×224
       - model-color-format=0 → RGB
       - net-scale-factor + offsets → y = (px - offset) * scale
       - CHW + batch → [1, 3, 224, 224] FP32
    """
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32)
    x = (x - NVINFER_OFFSETS) * NVINFER_SCALE
    x = np.transpose(x, (2, 0, 1))[None, :]  # [1, 3, 224, 224]
    return x.copy()


def cpp_parser_logic(emb: np.ndarray, protos: np.ndarray,
                     classes: list[str], thr: float):
    """1:1 эмуляция C++ NvDsInferClassifierParseCustomDinov2:
       L2 norm + cosine sim + argmax + threshold reject."""
    norm = float(np.linalg.norm(emb)) + 1e-12
    sims = (protos @ emb) / norm
    best = int(np.argmax(sims))
    max_sim = float(sims[best])
    if max_sim < thr:
        return COLOR_UNKNOWN_ID, max_sim, "unknown"
    return LABELS_COLOR_TXT[classes[best]], max_sim, classes[best]


class TrtRunner:
    def __init__(self, engine_path: Path):
        import tensorrt as trt
        from cuda.bindings import runtime as cudart
        self._cudart = cudart
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.in_name = self.engine.get_tensor_name(0)
        self.out_name = self.engine.get_tensor_name(1)
        self.in_shape = tuple(self.engine.get_tensor_shape(self.in_name))
        self.out_shape = tuple(self.engine.get_tensor_shape(self.out_name))
        self.in_size = int(np.prod(self.in_shape)) * 4
        self.out_size = int(np.prod(self.out_shape)) * 4
        _, self.d_in = cudart.cudaMalloc(self.in_size)
        _, self.d_out = cudart.cudaMalloc(self.out_size)
        self.context.set_tensor_address(self.in_name, int(self.d_in))
        self.context.set_tensor_address(self.out_name, int(self.d_out))
        _, self.stream = cudart.cudaStreamCreate()

    def infer(self, x: np.ndarray) -> np.ndarray:
        """x: [1, 3, 224, 224] FP32 → emb [768] FP32"""
        cudart = self._cudart
        cudart.cudaMemcpyAsync(self.d_in, x.ctypes.data, self.in_size,
                               cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                               self.stream)
        self.context.execute_async_v3(self.stream)
        out = np.empty(self.out_shape, dtype=np.float32)
        cudart.cudaMemcpyAsync(out.ctypes.data, self.d_out, self.out_size,
                               cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                               self.stream)
        cudart.cudaStreamSynchronize(self.stream)
        return out[0]


def load_protos():
    data = np.load(PROTOS_NPZ, allow_pickle=False)
    protos = data["prototypes"].astype(np.float32)
    classes = [str(c) for c in data["classes"]]
    return protos, classes


def build_label_map(labels: dict) -> dict[str, str]:
    """labels.json {color: [crop_id, ...]} → {crop_id: color}"""
    out = {}
    for color, ids in labels.items():
        for cid in ids:
            out[cid] = color
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("PHASE 4 — DINOv2 prod regression на golden cam-13 video")
    print("=" * 76)
    print(f"  engine:        {ENGINE}")
    print(f"  prototypes:    {PROTOS_NPZ}")
    print(f"  labels:        {LABELS_JSON}")
    print(f"  crops_index:   {CROPS_INDEX}")
    print(f"  raw_frames:    {RAW_FRAMES}")
    print(f"  threshold:     {MIN_SIM}")
    print()

    # 1. Load index + labels
    crops_index = json.loads(CROPS_INDEX.read_text())
    labels_raw = json.loads(LABELS_JSON.read_text())
    label_map = build_label_map(labels_raw)
    print(f"  crops in index:   {len(crops_index)}")
    print(f"  labeled crops:    {len(label_map)}")
    print(f"  unlabeled (skip): {len(crops_index) - len(label_map)}")
    print()

    # 2. Load TRT engine + prototypes
    print("=== loading TRT engine + prototypes ===")
    runner = TrtRunner(ENGINE)
    protos, classes = load_protos()
    print(f"  engine: in={runner.in_shape} out={runner.out_shape}")
    print(f"  protos: {protos.shape}, order={classes}")
    print()

    # 3. Per-crop inference: read raw frame, crop at exact bbox, preprocess,
    #    infer, parser logic.
    print("=== running prod TRT + parser on crops ===")
    t0 = time.time()
    predictions: list[dict] = []
    n_processed = 0
    n_missing_frame = 0
    n_unlabeled = 0
    last_frame_path: str | None = None
    cached_frame: np.ndarray | None = None

    for entry in crops_index:
        crop_id = entry["crop"].replace(".jpg", "")
        true_label = label_map.get(crop_id)
        if true_label is None:
            n_unlabeled += 1
            continue

        frame_name = entry["frame"]
        frame_path = str(RAW_FRAMES / frame_name)
        if frame_path != last_frame_path:
            cached_frame = cv2.imread(frame_path)
            last_frame_path = frame_path
            if cached_frame is None:
                n_missing_frame += 1
                continue

        x1, y1, x2, y2 = entry["bbox"]
        h_img, w_img = cached_frame.shape[:2]
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(w_img, int(x2)); y2 = min(h_img, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = cached_frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        x = preprocess_for_nvinfer(crop)
        emb = runner.infer(x)
        pred_id, max_sim, pred_name = cpp_parser_logic(
            emb, protos, classes, MIN_SIM
        )

        predictions.append({
            "id": crop_id,
            "true": true_label,
            "pred": pred_name,            # "blue/green/red/yellow/unknown"
            "pred_id": pred_id,           # labels_color.txt id или 255
            "max_sim": max_sim,
            "frame": frame_name,
            "bbox": [x1, y1, x2, y2],
            "h": entry.get("h", 0),
            "w": entry.get("w", 0),
            "det_conf": entry.get("det_conf", 0.0),
        })
        n_processed += 1
        if n_processed % 50 == 0:
            print(f"  {n_processed}/{len(label_map)} processed "
                  f"({time.time()-t0:.1f}s)")

    elapsed = time.time() - t0
    print(f"  done: {n_processed} predictions in {elapsed:.1f}s "
          f"({n_processed/max(elapsed,1e-3):.1f} crops/s)")
    if n_missing_frame:
        print(f"  WARN: {n_missing_frame} missing raw frames")
    print()

    # 4. Save raw predictions
    pred_path = OUT_DIR / "step4_predict_dinov2_prod.json"
    pred_path.write_text(json.dumps(predictions, indent=2))
    print(f"  saved: {pred_path}")
    print()

    # 5. Headline metrics
    print("=== headline metrics ===")
    n_total = len(predictions)
    n_jockey_true = sum(1 for p in predictions if p["true"] != "not_jockey")
    n_nj_true = sum(1 for p in predictions if p["true"] == "not_jockey")

    # Jockey accuracy: для true=jockey-color, pred должен быть тот же color
    n_jockey_correct = sum(
        1 for p in predictions
        if p["true"] != "not_jockey" and p["pred"] == p["true"]
    )
    n_jockey_wrong_color = sum(
        1 for p in predictions
        if p["true"] != "not_jockey"
        and p["pred"] not in ("unknown", p["true"])
    )
    n_jockey_rejected = sum(
        1 for p in predictions
        if p["true"] != "not_jockey" and p["pred"] == "unknown"
    )

    # NJ reject rate
    n_nj_rejected = sum(
        1 for p in predictions
        if p["true"] == "not_jockey" and p["pred"] == "unknown"
    )
    n_nj_passed_as_color = n_nj_true - n_nj_rejected

    # Confident wrong на NJ: NJ accepted as color с max_sim > 0.9
    n_nj_confident_wrong = sum(
        1 for p in predictions
        if p["true"] == "not_jockey" and p["pred"] != "unknown"
        and p["max_sim"] > 0.9
    )

    jockey_acc   = n_jockey_correct / max(n_jockey_true, 1)
    nj_reject    = n_nj_rejected     / max(n_nj_true,    1)
    nj_conf_wrong= n_nj_confident_wrong / max(n_nj_true, 1)

    print(f"  total predictions:     {n_total}")
    print(f"  true jockey (color):   {n_jockey_true}")
    print(f"  true not_jockey:       {n_nj_true}")
    print()
    print(f"  jockey color accuracy: {n_jockey_correct}/{n_jockey_true} "
          f"= {jockey_acc:.1%}")
    print(f"    wrong color:         {n_jockey_wrong_color}")
    print(f"    rejected (unknown):  {n_jockey_rejected}")
    print()
    print(f"  not_jockey reject rate:{n_nj_rejected}/{n_nj_true} "
          f"= {nj_reject:.1%}")
    print(f"    NJ passed as color:  {n_nj_passed_as_color}")
    print(f"    NJ confident wrong   "
          f"(>0.9): {n_nj_confident_wrong}/{n_nj_true} = {nj_conf_wrong:.1%}")
    print()

    # 6. Confusion matrix
    print("=== confusion matrix (true → pred counts) ===")
    rows_order = ["blue", "green", "red", "yellow", "not_jockey"]
    cols_order = ["blue", "green", "red", "yellow", "unknown"]
    cm = {r: {c: 0 for c in cols_order} for r in rows_order}
    for p in predictions:
        t = p["true"]
        pr = p["pred"]
        if t in cm and pr in cm[t]:
            cm[t][pr] += 1
    header = "  " + " " * 12 + "".join(f"{c:>10s}" for c in cols_order)
    print(header)
    for r in rows_order:
        row = f"  {r:12s}" + "".join(f"{cm[r][c]:>10d}" for c in cols_order)
        print(row)
    print()

    # 7. 23 HARD NJ cases
    hard_cases = [
        p for p in predictions
        if p["true"] == "not_jockey"
        and any(p["frame"].startswith(f) for f in HARD_NJ_FRAMES)
    ]
    hard_rejected = [p for p in hard_cases if p["pred"] == "unknown"]
    hard_passed   = [p for p in hard_cases if p["pred"] != "unknown"]
    print(f"=== 23 HARD NJ cases (frame_002310..002370) ===")
    print(f"  total HARD NJ found:   {len(hard_cases)}")
    print(f"  rejected:              {len(hard_rejected)}/{len(hard_cases)}")
    print(f"  passed as color:       {len(hard_passed)}")
    if hard_passed:
        print(f"  PASSED (false positives) — заслуживают reject:")
        for p in hard_passed[:10]:
            print(f"    {p['id']:35s} pred={p['pred']:7s} "
                  f"sim={p['max_sim']:.3f}")
    print()

    # 8. Acceptance verdict
    print("=" * 76)
    print("=== ACCEPTANCE CHECKPOINTS ===")
    chk1 = jockey_acc >= 0.78
    chk2 = nj_reject  >= 0.96
    chk3 = nj_conf_wrong <= 0.02
    chk4 = (len(hard_cases) == 0) or (len(hard_rejected) >= 18 if len(hard_cases) >= 23 else len(hard_rejected) / len(hard_cases) >= 18/23)
    mark = lambda b: "✅ PASS" if b else "❌ FAIL"
    print(f"  [1] jockey color accuracy ≥ 78%:    "
          f"{jockey_acc:.1%}  {mark(chk1)}")
    print(f"  [2] not_jockey reject ≥ 96%:        "
          f"{nj_reject:.1%}  {mark(chk2)}")
    print(f"  [3] NJ confident wrong (>0.9) ≤ 2%: "
          f"{nj_conf_wrong:.1%}  {mark(chk3)}")
    print(f"  [4] 23 HARD NJ rejected ≥ 18/23:    "
          f"{len(hard_rejected)}/{len(hard_cases)}  {mark(chk4)}")
    all_pass = chk1 and chk2 and chk3 and chk4
    print()
    print(f"  OVERALL: {'✅ PASS — ready for Phase 5' if all_pass else '❌ FAIL — see details'}")
    print("=" * 76)

    # 9. Comparison table v4 vs Phase 3 HF vs prod TRT
    print()
    print("=== comparison vs prior baselines ===")
    print(
        "  | metric                  | v4 CNN | DINOv2 P3 (HF) | DINOv2 prod (TRT) |"
    )
    print(
        "  |-------------------------|--------|----------------|-------------------|"
    )
    print(
        f"  | jockey color accuracy   | 100%   | 80.8%          | {jockey_acc:.1%}             |"
    )
    print(
        f"  | not_jockey reject rate  | 0%     | 98.8%          | {nj_reject:.1%}             |"
    )
    print(
        f"  | confident wrong on NJ   | 21.6%  | 0%             | {nj_conf_wrong:.1%}             |"
    )
    print(
        f"  | 23 HARD cases rejected  | 0/23   | (no measure)   | {len(hard_rejected)}/{len(hard_cases)}             |"
    )
    print()

    # 10. Save summary report
    summary = {
        "n_total": n_total,
        "n_jockey_true": n_jockey_true,
        "n_nj_true": n_nj_true,
        "jockey_correct": n_jockey_correct,
        "jockey_wrong_color": n_jockey_wrong_color,
        "jockey_rejected": n_jockey_rejected,
        "nj_rejected": n_nj_rejected,
        "nj_passed_as_color": n_nj_passed_as_color,
        "nj_confident_wrong": n_nj_confident_wrong,
        "jockey_accuracy": jockey_acc,
        "nj_reject_rate": nj_reject,
        "nj_confident_wrong_rate": nj_conf_wrong,
        "hard_cases_total": len(hard_cases),
        "hard_cases_rejected": len(hard_rejected),
        "confusion_matrix": cm,
        "checkpoints": {
            "jockey_acc_78":   chk1,
            "nj_reject_96":    chk2,
            "nj_conf_wrong_2": chk3,
            "hard_18_of_23":   chk4,
            "overall_pass":    all_pass,
        },
        "threshold": MIN_SIM,
        "engine":     str(ENGINE),
        "prototypes": str(PROTOS_NPZ),
    }
    summary_path = OUT_DIR / "phase4_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"  summary saved: {summary_path}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
