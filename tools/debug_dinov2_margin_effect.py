#!/usr/bin/env python3
"""debug_dinov2_margin_effect.py — testing margin hypothesis.

In-domain blue_collection crops have 5% margin → look balanced after stretch.
Real live YOLO bboxes have NO margin → tall-narrow, stretch destroys them.

Re-run debug using bboxes from crops_index.json (NO margin) on raw_frames.
Compare: cropped-with-margin (P4_marg) vs cropped-no-margin (P4_raw)
through PROD preprocessing.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import cv2
import numpy as np

WORK = Path("/home/ipodrom/race_vision_bench/dinov2_prod")
ENGINE = WORK / "dinov2_base_b1_gpu0_fp16.engine"
PROTOS_NPZ = WORK / "prototypes_dinov2_v1.npz"
LABELS_JSON = WORK / "blue_collection" / "labels.json"
CROPS_INDEX = WORK / "blue_collection" / "crops_index.json"
RAW_FRAMES  = WORK / "blue_collection" / "raw_frames"
CROPS_DIR_MARGIN = WORK / "blue_collection" / "crops_raw"  # с 5% margin

NVINFER_OFFSETS = np.array([123.675, 116.28, 103.53], dtype=np.float32)
NVINFER_SCALE   = np.float32(0.01735)
IMAGENET_MEAN   = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD    = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_batch(rgb): return np.transpose(rgb, (2,0,1))[None,:].copy()


def prep_stretch(crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32)
    return to_batch((x - NVINFER_OFFSETS) * NVINFER_SCALE)


def prep_hf_emulation(crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if h <= w: scale=256/h; new_h,new_w=256,max(1,int(round(w*scale)))
    else:      scale=256/w; new_w,new_h=256,max(1,int(round(h*scale)))
    rgb_r = cv2.resize(rgb, (new_w,new_h), interpolation=cv2.INTER_LINEAR)
    cy=(new_h-224)//2; cx=(new_w-224)//2
    crop = rgb_r[cy:cy+224, cx:cx+224]
    x = crop.astype(np.float32)/255.0
    return to_batch((x - IMAGENET_MEAN) / IMAGENET_STD)


class Trt:
    def __init__(self, p):
        import tensorrt as trt
        from cuda.bindings import runtime as cudart
        self.c = cudart
        rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.eng = rt.deserialize_cuda_engine(open(p, "rb").read())
        self.ctx = self.eng.create_execution_context()
        i = self.eng.get_tensor_name(0); o = self.eng.get_tensor_name(1)
        ish = tuple(self.eng.get_tensor_shape(i))
        self.osh = tuple(self.eng.get_tensor_shape(o))
        self.in_size = int(np.prod(ish))*4; self.out_size = int(np.prod(self.osh))*4
        _, self.d_in = cudart.cudaMalloc(self.in_size)
        _, self.d_out = cudart.cudaMalloc(self.out_size)
        self.ctx.set_tensor_address(i, int(self.d_in))
        self.ctx.set_tensor_address(o, int(self.d_out))
        _, self.s = cudart.cudaStreamCreate()

    def infer(self, x):
        c = self.c
        c.cudaMemcpyAsync(self.d_in, x.ctypes.data, self.in_size,
                          c.cudaMemcpyKind.cudaMemcpyHostToDevice, self.s)
        self.ctx.execute_async_v3(self.s)
        out = np.empty(self.osh, dtype=np.float32)
        c.cudaMemcpyAsync(out.ctypes.data, self.d_out, self.out_size,
                          c.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.s)
        c.cudaStreamSynchronize(self.s)
        return out[0]


def parser(emb, protos, classes, thr=0.55):
    n = float(np.linalg.norm(emb))+1e-12
    sims = (protos @ emb)/n
    b = int(np.argmax(sims)); ms = float(sims[b])
    if ms < thr: return "unknown", ms
    return classes[b], ms


def main():
    print("=" * 80)
    print("DINOv2 margin effect diagnostic (in-domain blue_collection)")
    print("=" * 80)

    if not CROPS_INDEX.is_file():
        print(f"FATAL: {CROPS_INDEX} missing", file=sys.stderr); sys.exit(1)
    if not RAW_FRAMES.is_dir():
        print(f"FATAL: {RAW_FRAMES} missing", file=sys.stderr); sys.exit(1)

    labels = json.loads(LABELS_JSON.read_text())
    label_map = {cid: cls for cls, ids in labels.items() for cid in ids}
    crops_index = json.loads(CROPS_INDEX.read_text())
    print(f"  labels: {len(label_map)}")
    print(f"  crops_index: {len(crops_index)}")

    # Random sample 20 (5 per class), compute aspect ratio (raw bbox)
    rng = np.random.default_rng(42)
    sample = []
    for cls, ids in labels.items():
        chosen = rng.choice(ids, size=min(8, len(ids)), replace=False)
        for cid in chosen: sample.append((cid, cls))
    print(f"  sample: {len(sample)} crops "
          f"(5+ per class)")

    # Build entry lookup
    by_id = {e["crop"].replace(".jpg",""): e for e in crops_index}

    data = np.load(PROTOS_NPZ, allow_pickle=False)
    protos = data["prototypes"].astype(np.float32)
    classes = [str(c) for c in data["classes"]]

    trt = Trt(ENGINE)

    print()
    print(f"  {'crop_id':<32s} {'true':<7s} {'h×w_raw':<10s} {'AR':<5s} "
          f"{'P_marg_str':<12s} {'P_raw_str':<12s} {'P_marg_HF':<12s} {'P_raw_HF':<12s}")
    print("  " + "-" * 110)

    n = 0
    correct = {"marg_str":0, "raw_str":0, "marg_HF":0, "raw_HF":0}
    aspect_buckets = {"square (<1.5)":[], "tall (1.5-2.5)":[], "very_tall (>2.5)":[]}

    for cid, true_cls in sample:
        entry = by_id.get(cid)
        if not entry: continue
        crop_marg = cv2.imread(str(CROPS_DIR_MARGIN / entry["crop"]))
        if crop_marg is None: continue

        # Re-crop without margin from raw_frame using bbox
        raw_path = RAW_FRAMES / entry["frame"]
        raw = cv2.imread(str(raw_path))
        if raw is None: continue
        x1,y1,x2,y2 = entry["bbox"]
        h_img,w_img = raw.shape[:2]
        x1=max(0,int(x1)); y1=max(0,int(y1))
        x2=min(w_img,int(x2)); y2=min(h_img,int(y2))
        crop_raw = raw[y1:y2, x1:x2]
        if crop_raw.size == 0: continue

        h_raw, w_raw = crop_raw.shape[:2]
        ar = h_raw / max(w_raw, 1)

        e_ms = trt.infer(prep_stretch(crop_marg))
        e_rs = trt.infer(prep_stretch(crop_raw))
        e_mh = trt.infer(prep_hf_emulation(crop_marg))
        e_rh = trt.infer(prep_hf_emulation(crop_raw))

        p_ms = parser(e_ms, protos, classes)
        p_rs = parser(e_rs, protos, classes)
        p_mh = parser(e_mh, protos, classes)
        p_rh = parser(e_rh, protos, classes)

        if p_ms[0] == true_cls: correct["marg_str"] += 1
        if p_rs[0] == true_cls: correct["raw_str"] += 1
        if p_mh[0] == true_cls: correct["marg_HF"] += 1
        if p_rh[0] == true_cls: correct["raw_HF"] += 1
        n += 1

        bucket = ("square (<1.5)" if ar < 1.5
                  else "tall (1.5-2.5)" if ar < 2.5
                  else "very_tall (>2.5)")
        aspect_buckets[bucket].append((true_cls, p_ms[0], p_rs[0], p_mh[0], p_rh[0]))

        m = lambda p: "✓" if p[0]==true_cls else "✗"
        hw = f"{h_raw}x{w_raw}"
        print(f"  {cid:<32s} {true_cls:<7s} {hw:<10s} {ar:<5.2f} "
              f"{p_ms[0]:<7s}{p_ms[1]:.2f}{m(p_ms):<2s} "
              f"{p_rs[0]:<7s}{p_rs[1]:.2f}{m(p_rs):<2s} "
              f"{p_mh[0]:<7s}{p_mh[1]:.2f}{m(p_mh):<2s} "
              f"{p_rh[0]:<7s}{p_rh[1]:.2f}{m(p_rh):<2s}")

    print()
    print(f"=== accuracy on {n} crops ===")
    for k, c in correct.items():
        print(f"  {k:<12s}: {c:2d}/{n}  ({c/max(n,1):.0%})")
    print()
    print(f"=== by aspect ratio bucket ===")
    for bucket, items in aspect_buckets.items():
        if not items: continue
        c_ms = sum(1 for x in items if x[1]==x[0])
        c_rs = sum(1 for x in items if x[2]==x[0])
        c_mh = sum(1 for x in items if x[3]==x[0])
        c_rh = sum(1 for x in items if x[4]==x[0])
        print(f"  {bucket:<20s} (n={len(items):2d}): "
              f"marg_str={c_ms}/{len(items)}  raw_str={c_rs}/{len(items)}  "
              f"marg_HF={c_mh}/{len(items)}  raw_HF={c_rh}/{len(items)}")
    print()
    print("Read columns: P_marg_str = with-margin + stretch (PROD path)")
    print("              P_raw_str  = no-margin + stretch (PROD on live YOLO bbox)")
    print("              P_marg_HF  = with-margin + HF resize-256 + center-crop")
    print("              P_raw_HF   = no-margin + HF resize-256 + center-crop")


if __name__ == "__main__":
    main()
