#!/usr/bin/env python3
"""smoke_compare_mux_resolutions.py — мини-эксперимент: проверить
гипотезу что mux 720p downsample ухудшает SGIE качество.

Симулирует production pipeline двумя путями:
  Conf A: pre-resize frame to mux dims (--mux-w 1280 --mux-h 720)
          → YOLO детектит на downsampled frame → bbox на mux res
          → crop по bbox на downsampled frame → stretch 224×224 → TRT/HF
  Conf B: native source resolution (no pre-resize)
          → YOLO + crop + stretch на native frame → TRT/HF

Используется v2_multicam prototypes (БЕЗ tight). Tight НЕ применяется.

REF path = HF AutoImageProcessor + DINOv2 FP16 (gold standard)
PROD path = cv2 stretch + nvinfer offsets/scale + same TRT engine

Output: stats JSON only (для скорости — без overlay video).
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = REPO / "data/videos/test_full_loop/yaris_20260421_174240/kamera_24_174252_END174647.mp4"
YOLO_ONNX = REPO / "models" / "yolo11s_person_960.onnx"
WORK = Path("/home/ipodrom/race_vision_bench/dinov2_prod")
ENGINE = WORK / "dinov2_base_b1_gpu0_fp16.engine"
PROTOS_NPZ = WORK / "v2_build" / "prototypes_v2_multicam.npz"   # explicit no-tight

DET_CONF=0.25; DET_IOU=0.5; DET_IMGSZ=960; MIN_BBOX_H=25; MIN_SIM=0.55
NVINFER_OFFSETS = np.array([123.675,116.28,103.53], dtype=np.float32)
NVINFER_SCALE   = np.float32(0.01735)


def to_batch(rgb): return np.transpose(rgb,(2,0,1))[None,:].copy()


def prep_stretch(crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb,(224,224), interpolation=cv2.INTER_LINEAR)
    return to_batch(((rgb.astype(np.float32) - NVINFER_OFFSETS) * NVINFER_SCALE))


class Trt:
    def __init__(self, p):
        import tensorrt as trt
        from cuda.bindings import runtime as cudart
        self.c = cudart
        rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.eng = rt.deserialize_cuda_engine(open(p,"rb").read())
        self.ctx = self.eng.create_execution_context()
        i = self.eng.get_tensor_name(0); o = self.eng.get_tensor_name(1)
        self.osh = tuple(self.eng.get_tensor_shape(o))
        self.in_size = int(np.prod(tuple(self.eng.get_tensor_shape(i))))*4
        self.out_size = int(np.prod(self.osh))*4
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


def parser(emb, protos, classes, thr=MIN_SIM):
    n = float(np.linalg.norm(emb))+1e-12
    sims = (protos @ emb)/n
    b = int(np.argmax(sims)); ms = float(sims[b])
    if ms < thr: return "unknown", ms
    return classes[b], ms


def run(video_path, mux_w, mux_h, label, max_frames=0):
    print()
    print("=" * 78)
    print(f"  CONFIG '{label}': mux {mux_w}×{mux_h if mux_h else 'native'}")
    print("=" * 78)

    from ultralytics import YOLO
    yolo = YOLO(str(YOLO_ONNX), task="detect")
    trt  = Trt(ENGINE)
    data = np.load(PROTOS_NPZ, allow_pickle=False)
    protos = data["prototypes"].astype(np.float32)
    classes = [str(c) for c in data["classes"]]

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    hf_model = AutoModel.from_pretrained("facebook/dinov2-base",
                                         dtype=torch.float16).to("cuda").eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"FATAL: cannot open {video_path}", file=sys.stderr); sys.exit(1)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0: src_n = min(src_n, max_frames)
    do_resize = (mux_w is not None) and (mux_w, mux_h) != (src_w, src_h)
    print(f"  source video: {src_w}×{src_h}, {src_n} frames")
    print(f"  pre-resize:   {do_resize} ({src_w}×{src_h} → {mux_w}×{mux_h})" if do_resize
          else f"  pre-resize:   no (native {src_w}×{src_h})")

    counts_prod = Counter(); counts_ref = Counter()
    sims_prod = defaultdict(list); sims_ref = defaultdict(list)
    n_dets = 0; n_disagree = 0
    crop_pixel_areas = []

    t0 = time.time()
    last_print = time.time()
    for fidx in range(src_n):
        ok, frame = cap.read()
        if not ok or frame is None: break

        # Simulate mux: downsample to mux dims if specified
        if do_resize:
            frame = cv2.resize(frame, (mux_w, mux_h), interpolation=cv2.INTER_LINEAR)

        h_img, w_img = frame.shape[:2]
        results = yolo.predict(source=frame, imgsz=DET_IMGSZ,
                               conf=DET_CONF, iou=DET_IOU,
                               classes=[0], device=0, half=False, verbose=False)
        if not results or results[0].boxes is None: continue
        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        crops_bgr = []
        for box in boxes:
            x1,y1,x2,y2 = box.tolist()
            if (y2-y1) < MIN_BBOX_H: continue
            x1=max(0,int(x1)); y1=max(0,int(y1))
            x2=min(w_img,int(x2)); y2=min(h_img,int(y2))
            if x2<=x1 or y2<=y1: continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue
            crops_bgr.append(crop)
            crop_pixel_areas.append((x2-x1) * (y2-y1))
        if not crops_bgr: continue
        pils = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
        inputs = processor(images=pils, return_tensors="pt").to("cuda", torch.float16)
        with torch.inference_mode():
            ref_emb = hf_model(**inputs).pooler_output.float().cpu().numpy()
        for i, crop in enumerate(crops_bgr):
            ref = parser(ref_emb[i], protos, classes)
            prod = parser(trt.infer(prep_stretch(crop)), protos, classes)
            counts_prod[prod[0]] += 1; counts_ref[ref[0]] += 1
            sims_prod[prod[0]].append(prod[1])
            sims_ref[ref[0]].append(ref[1])
            if ref[0] != prod[0]: n_disagree += 1
            n_dets += 1
        if time.time() - last_print > 8:
            cur_fps = (fidx+1) / max(time.time()-t0, 1e-3)
            eta = (time.time()-t0)/max(fidx+1,1) * (src_n-fidx-1)
            print(f"  {fidx+1}/{src_n}  ({cur_fps:.1f} fps proc, ETA {eta:.0f}s)")
            last_print = time.time()
    cap.release()
    elapsed = time.time() - t0
    print(f"  done: {n_dets} dets in {elapsed:.0f}s")

    print()
    print(f"  REF/PROD disagreements: {n_disagree}/{n_dets} ({100*n_disagree/max(n_dets,1):.1f}%)")
    print(f"  median crop pixel area: {int(np.median(crop_pixel_areas))} px²")
    print(f"  PROD distribution:")
    for c in ["green","yellow","red","blue","unknown"]:
        n = counts_prod[c]; pct = 100*n/max(n_dets,1)
        s = sims_prod[c]
        print(f"    {c:<8s} {n:>6d} ({pct:>5.1f}%) "
              f"{('sim_med=' + format(np.median(s),'.3f')) if s else ''}")
    print(f"  REF distribution:")
    for c in ["green","yellow","red","blue","unknown"]:
        n = counts_ref[c]; pct = 100*n/max(n_dets,1)
        s = sims_ref[c]
        print(f"    {c:<8s} {n:>6d} ({pct:>5.1f}%) "
              f"{('sim_med=' + format(np.median(s),'.3f')) if s else ''}")

    return {
        "label": label,
        "mux_w": mux_w, "mux_h": mux_h,
        "src_w": src_w, "src_h": src_h,
        "do_resize": do_resize,
        "n_dets": n_dets,
        "n_disagree": n_disagree,
        "disagree_pct": 100*n_disagree/max(n_dets,1),
        "median_crop_area_px2": int(np.median(crop_pixel_areas)),
        "counts_prod": dict(counts_prod),
        "counts_ref": dict(counts_ref),
        "sim_med_prod": {c: float(np.median(s)) for c,s in sims_prod.items() if s},
        "sim_med_ref":  {c: float(np.median(s)) for c,s in sims_ref.items() if s},
        "elapsed_sec": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(DEFAULT_VIDEO))
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    print(f"  PROTOTYPES: {PROTOS_NPZ}")
    print(f"  ENGINE:     {ENGINE}")
    print(f"  VIDEO:      {args.video}")

    # Conf A: mux 1280×720 (production default)
    a = run(args.video, 1280, 720, "A_mux720p", args.max_frames)
    # Conf B: native (no resize)
    b = run(args.video, None, None, "B_native_1080p", args.max_frames)

    out = {"A_mux720p": a, "B_native_1080p": b}
    out_path = WORK / "v2_build" / "smoke_mux_compare.json"
    out_path.write_text(json.dumps(out, indent=2))

    print()
    print("=" * 78)
    print("=== COMPARISON ===")
    print("=" * 78)
    rows = [
        ("REF/PROD disagreement %", a['disagree_pct'], b['disagree_pct']),
        ("median crop pixel area",  a['median_crop_area_px2'], b['median_crop_area_px2']),
        ("green % PROD",  100*a['counts_prod'].get('green',0)/max(a['n_dets'],1),
                           100*b['counts_prod'].get('green',0)/max(b['n_dets'],1)),
        ("yellow % PROD", 100*a['counts_prod'].get('yellow',0)/max(a['n_dets'],1),
                           100*b['counts_prod'].get('yellow',0)/max(b['n_dets'],1)),
        ("red % PROD",    100*a['counts_prod'].get('red',0)/max(a['n_dets'],1),
                           100*b['counts_prod'].get('red',0)/max(b['n_dets'],1)),
        ("blue % PROD",   100*a['counts_prod'].get('blue',0)/max(a['n_dets'],1),
                           100*b['counts_prod'].get('blue',0)/max(b['n_dets'],1)),
        ("unknown % PROD",100*a['counts_prod'].get('unknown',0)/max(a['n_dets'],1),
                           100*b['counts_prod'].get('unknown',0)/max(b['n_dets'],1)),
    ]
    print(f"  {'metric':<28s} {'A 720p':>10s} {'B 1080p':>10s} {'Δ':>9s}")
    for name, A, B in rows:
        delta = B - A
        if isinstance(A, float):
            print(f"  {name:<28s} {A:>10.1f} {B:>10.1f} {delta:>+9.1f}")
        else:
            print(f"  {name:<28s} {A:>10d} {B:>10d} {delta:>+9d}")
    print()
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
