#!/usr/bin/env python3
"""record_dual_overlay_mp4.py — visual production smoke test.

Запись mp4 с dual-path overlay (REF=HF reference, PROD=cv2 stretch+TRT)
на тестовом видео + статистика по detection distribution.

Использует prototypes из /home/ipodrom/race_vision_bench/dinov2_prod/
prototypes_dinov2_v1.npz (текущий "production" .npz — может быть
v1 или v2 в зависимости от того что закопировано).
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
PROTOS_NPZ = WORK / "prototypes_dinov2_v1.npz"   # production location
DEFAULT_OUT = WORK / "visual_smoke_v2_multicam.mp4"

DET_CONF=0.25; DET_IOU=0.5; DET_IMGSZ=960; MIN_BBOX_H=25; MIN_SIM=0.55
NVINFER_OFFSETS = np.array([123.675,116.28,103.53], dtype=np.float32)
NVINFER_SCALE   = np.float32(0.01735)

COLOR_BGR = {"blue":(255,80,80),"green":(60,200,60),"red":(40,40,220),
             "yellow":(40,220,220),"unknown":(128,128,128)}


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


def draw_overlay(frame, dets, frame_idx, fps_actual):
    out = frame.copy()
    h, w = out.shape[:2]
    counts_ref, counts_prod = {}, {}
    n_disagree = 0
    for d in dets:
        x1,y1,x2,y2 = d["bbox"]
        ref = d["ref"]; prod = d["prod"]
        bgr = COLOR_BGR.get(prod[0], (200,200,200))
        thick = 3 if ref[0] != prod[0] else 2
        cv2.rectangle(out,(x1,y1),(x2,y2), bgr, thick)
        if ref[0] != prod[0]:
            n_disagree += 1
            cv2.line(out,(x2-12,y1+2),(x2-2,y1+12),(0,0,255),2)
            cv2.line(out,(x2-2,y1+2),(x2-12,y1+12),(0,0,255),2)
        lbl_ref  = f"REF:  {ref[0]} {ref[1]:.2f}"
        lbl_prod = f"PROD: {prod[0]} {prod[1]:.2f}"
        (tw1,th1),_ = cv2.getTextSize(lbl_ref,  cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        (tw2,th2),_ = cv2.getTextSize(lbl_prod, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        max_tw = max(tw1, tw2)
        ly = max(th1+th2+8, y1-2)
        bg_y1 = ly - th1 - th2 - 8
        cv2.rectangle(out,(x1,bg_y1),(x1+max_tw+4, ly+2), (0,0,0), -1)
        cv2.putText(out, lbl_ref,  (x1+2, ly-th2-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60,255,60),  1, cv2.LINE_AA)
        cv2.putText(out, lbl_prod, (x1+2, ly-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60,180,255), 1, cv2.LINE_AA)
        counts_ref[ref[0]]   = counts_ref.get(ref[0],0)+1
        counts_prod[prod[0]] = counts_prod.get(prod[0],0)+1
    hud_h = 44
    cv2.rectangle(out,(0,0),(w,hud_h), (0,0,0), -1)
    sum_ref  = "  ".join(f"{k}={v}" for k,v in sorted(counts_ref.items()))
    sum_prod = "  ".join(f"{k}={v}" for k,v in sorted(counts_prod.items()))
    line1 = f"frame {frame_idx:6d}  | {fps_actual:5.1f} fps proc | dets={len(dets)} | DISAGREE={n_disagree}"
    line2 = f"REF:  {sum_ref or 'no det'}"
    line3 = f"PROD: {sum_prod or 'no det'}"
    cv2.putText(out, line1, (8,14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(out, line2, (8,28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60,255,60), 1, cv2.LINE_AA)
    cv2.putText(out, line3, (8,40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60,180,255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(DEFAULT_VIDEO))
    ap.add_argument("--out",   default=str(DEFAULT_OUT))
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0 = entire video")
    args = ap.parse_args()

    print(f"  video:  {args.video}")
    print(f"  out:    {args.out}")
    print(f"  protos: {PROTOS_NPZ}")
    print()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"FATAL: cannot open {args.video}", file=sys.stderr); sys.exit(1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_src = cap.get(cv2.CAP_PROP_FPS)
    w_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.max_frames > 0: total = min(total, args.max_frames)
    print(f"  total: {total} frames @ {fps_src:.1f} fps, {w_src}×{h_src}")

    print("=== loading YOLO + TRT + HF + protos ===")
    from ultralytics import YOLO
    yolo = YOLO(str(YOLO_ONNX), task="detect")
    trt = Trt(ENGINE)
    data = np.load(PROTOS_NPZ, allow_pickle=False)
    protos = data["prototypes"].astype(np.float32)
    classes = [str(c) for c in data["classes"]]
    n_per_class = data.get("n_per_class")
    print(f"  protos:    {protos.shape}, classes={classes}")
    if n_per_class is not None:
        print(f"  n per class: {dict(zip(classes, n_per_class.tolist()))}")
    if "source" in data: print(f"  source:    {data['source']}")
    print()

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    hf_model = AutoModel.from_pretrained("facebook/dinov2-base",
                                         dtype=torch.float16).to("cuda").eval()
    print("  HF model loaded")
    print()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps_src, (w_src, h_src))
    if not writer.isOpened():
        print(f"FATAL: VideoWriter cannot open {args.out}", file=sys.stderr); sys.exit(1)

    # Stats
    counts_prod = Counter(); counts_ref = Counter()
    sims_prod = defaultdict(list); sims_ref = defaultdict(list)
    n_dets_total = 0
    frames_with_ge4 = 0
    n_disagree_total = 0

    t0 = time.time()
    frame_idx = 0
    last_print = time.time()
    while frame_idx < total:
        ok, frame = cap.read()
        if not ok or frame is None: break

        h_img, w_img = frame.shape[:2]
        results = yolo.predict(source=frame, imgsz=DET_IMGSZ,
                               conf=DET_CONF, iou=DET_IOU,
                               classes=[0], device=0, half=False, verbose=False)
        dets = []
        if results and results[0].boxes is not None:
            r = results[0]
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            crops_bgr = []; valid = []
            for box, dc in zip(boxes, confs):
                x1,y1,x2,y2 = box.tolist()
                if (y2-y1) < MIN_BBOX_H: continue
                x1=max(0,int(x1)); y1=max(0,int(y1))
                x2=min(w_img,int(x2)); y2=min(h_img,int(y2))
                if x2<=x1 or y2<=y1: continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue
                crops_bgr.append(crop); valid.append((x1,y1,x2,y2,float(dc)))
            if crops_bgr:
                pils = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
                inputs = processor(images=pils, return_tensors="pt").to("cuda", torch.float16)
                with torch.inference_mode():
                    ref_emb = hf_model(**inputs).pooler_output.float().cpu().numpy()
                for i,(x1,y1,x2,y2,dc) in enumerate(valid):
                    ref = parser(ref_emb[i], protos, classes)
                    prod = parser(trt.infer(prep_stretch(crops_bgr[i])), protos, classes)
                    dets.append({"bbox":(x1,y1,x2,y2),"ref":ref,"prod":prod,"det_conf":dc})
                    counts_prod[prod[0]] += 1; counts_ref[ref[0]] += 1
                    sims_prod[prod[0]].append(prod[1])
                    sims_ref[ref[0]].append(ref[1])
                    if ref[0] != prod[0]: n_disagree_total += 1
        n_dets_total += len(dets)
        if len(dets) >= 4: frames_with_ge4 += 1

        cur_fps = frame_idx / max(time.time()-t0, 1e-3)
        out_frame = draw_overlay(frame, dets, frame_idx, cur_fps)
        writer.write(out_frame)
        frame_idx += 1
        if time.time() - last_print > 5:
            eta = (time.time()-t0)/max(frame_idx,1) * (total-frame_idx)
            print(f"  {frame_idx}/{total}  ({cur_fps:.1f} fps proc, ETA {eta:.0f}s)")
            last_print = time.time()

    writer.release(); cap.release()
    elapsed = time.time() - t0
    print(f"  done: {frame_idx} frames in {elapsed:.0f}s ({frame_idx/elapsed:.1f} fps proc)")
    print()

    # Stats
    print("=" * 78)
    print("=== DETECTION STATS ===")
    print("=" * 78)
    print(f"  total detections:    {n_dets_total}")
    print(f"  frames with ≥4 dets: {frames_with_ge4}/{frame_idx} "
          f"({100*frames_with_ge4/max(frame_idx,1):.1f}%)")
    print(f"  REF/PROD disagreements: {n_disagree_total} "
          f"({100*n_disagree_total/max(n_dets_total,1):.1f}%)")
    print()
    print(f"  Distribution by color (PROD path = production):")
    print(f"    {'color':<8s} {'n':>6s} {'%':>6s} {'sim_med':>8s} {'sim_min':>8s} {'sim_max':>8s}")
    for c in ["green","yellow","red","blue","unknown"]:
        n = counts_prod[c]
        pct = 100*n/max(n_dets_total,1)
        sims = sims_prod[c]
        if sims:
            print(f"    {c:<8s} {n:>6d} {pct:>5.1f}% {np.median(sims):>8.3f} "
                  f"{np.min(sims):>8.3f} {np.max(sims):>8.3f}")
        else:
            print(f"    {c:<8s} {n:>6d} {pct:>5.1f}%")
    print()
    print(f"  Distribution by color (REF path = HF reference):")
    print(f"    {'color':<8s} {'n':>6s} {'%':>6s} {'sim_med':>8s}")
    for c in ["green","yellow","red","blue","unknown"]:
        n = counts_ref[c]; sims = sims_ref[c]
        pct = 100*n/max(n_dets_total,1)
        if sims:
            print(f"    {c:<8s} {n:>6d} {pct:>5.1f}% {np.median(sims):>8.3f}")
        else:
            print(f"    {c:<8s} {n:>6d} {pct:>5.1f}%")
    print()

    # Save stats JSON
    stats = {
        "video": args.video,
        "out_mp4": args.out,
        "n_frames": frame_idx,
        "n_detections_total": n_dets_total,
        "frames_with_ge4_dets": frames_with_ge4,
        "n_disagreements_ref_vs_prod": n_disagree_total,
        "counts_prod": dict(counts_prod),
        "counts_ref": dict(counts_ref),
        "sim_stats_prod": {c: {"median": float(np.median(s)), "min": float(np.min(s)),
                                "max": float(np.max(s)), "mean": float(np.mean(s))}
                            for c, s in sims_prod.items() if s},
        "elapsed_sec": elapsed,
        "prototypes_source": str(PROTOS_NPZ),
    }
    stats_path = Path(args.out).with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2))
    out_sz_mb = Path(args.out).stat().st_size / 1024 / 1024
    print(f"  mp4: {args.out}  ({out_sz_mb:.1f} MB)")
    print(f"  stats: {stats_path}")


if __name__ == "__main__":
    main()
