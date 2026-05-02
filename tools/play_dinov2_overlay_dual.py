#!/usr/bin/env python3
"""play_dinov2_overlay_dual.py — visualization s DVUMYA paralelynymi
classifier paths:
  REF  (zelenyi tekst): HF AutoImageProcessor + HF DINOv2 model (FP16)
                        ETO TO chto STROILO prototypes — gold standard.
  PROD (krasnyi tekst): cv2 stretch + nvinfer offsets + TRT engine
                        ETO chto sgie_color.txt zapuskaet v production.

Esli REF == PROD vsegda → kod OK, problema v modeli/domenne.
Esli REF != PROD na konkretnom frame → bug v prod-path/preprocess.

Klavishi:
  SPACE  - pauza/play
  q/ESC  - exit
  s      - sohranit' overlay PNG + raw frame JPG + predictions JSON
  +/-    - speed up/down
  [/]    - prev/next single frame (when paused)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = REPO / "data/videos/test_full_loop/yaris_20260421_174240/kamera_24_174252_END174647.mp4"
YOLO_ONNX = REPO / "models" / "yolo11s_person_960.onnx"
WORK = Path("/home/ipodrom/race_vision_bench/dinov2_prod")
ENGINE = WORK / "dinov2_base_b1_gpu0_fp16.engine"
PROTOS_NPZ = WORK / "prototypes_dinov2_v1.npz"

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


def draw_overlay(frame, dets, frame_idx, fps_actual, paused, stride):
    out = frame.copy()
    h, w = out.shape[:2]
    counts_ref, counts_prod = {}, {}
    n_disagree = 0
    for d in dets:
        x1,y1,x2,y2 = d["bbox"]
        ref = d["ref"]; prod = d["prod"]
        # bbox color = PROD color (production behaviour)
        bgr = COLOR_BGR.get(prod[0], (200,200,200))
        # diagonal lines if disagree
        thick = 3 if ref[0] != prod[0] else 2
        cv2.rectangle(out,(x1,y1),(x2,y2), bgr, thick)
        if ref[0] != prod[0]:
            n_disagree += 1
            # red 'X' marker on top-right corner
            cv2.line(out,(x2-12,y1+2),(x2-2,y1+12),(0,0,255),2)
            cv2.line(out,(x2-2,y1+2),(x2-12,y1+12),(0,0,255),2)
        # two-line label
        lbl_ref  = f"REF:  {ref[0]} {ref[1]:.2f}"
        lbl_prod = f"PROD: {prod[0]} {prod[1]:.2f}"
        (tw1,th1),_ = cv2.getTextSize(lbl_ref,  cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        (tw2,th2),_ = cv2.getTextSize(lbl_prod, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        max_tw = max(tw1, tw2)
        ly = max(th1+th2+8, y1-2)
        bg_y1 = ly - th1 - th2 - 8
        cv2.rectangle(out,(x1,bg_y1),(x1+max_tw+4, ly+2), (0,0,0), -1)
        col_ref  = (60,255,60)   # bright green
        col_prod = (60,180,255)  # orange
        cv2.putText(out, lbl_ref,  (x1+2, ly-th2-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col_ref,  1, cv2.LINE_AA)
        cv2.putText(out, lbl_prod, (x1+2, ly-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col_prod, 1, cv2.LINE_AA)
        counts_ref[ref[0]]   = counts_ref.get(ref[0],0)+1
        counts_prod[prod[0]] = counts_prod.get(prod[0],0)+1
    # HUD
    hud_h = 44
    cv2.rectangle(out,(0,0),(w,hud_h), (0,0,0), -1)
    status = "PAUSED" if paused else f"{fps_actual:5.1f} fps"
    sum_ref  = "  ".join(f"{k}={v}" for k,v in sorted(counts_ref.items()))
    sum_prod = "  ".join(f"{k}={v}" for k,v in sorted(counts_prod.items()))
    line1 = f"frame {frame_idx:6d}  | {status} | stride={stride} | dets={len(dets)} | DISAGREE={n_disagree}"
    line2 = f"REF:  {sum_ref or 'no det'}"
    line3 = f"PROD: {sum_prod or 'no det'}"
    cv2.putText(out, line1, (8,14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(out, line2, (8,28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60,255,60), 1, cv2.LINE_AA)
    cv2.putText(out, line3, (8,40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60,180,255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(DEFAULT_VIDEO))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--scale", type=float, default=0.7)
    args = ap.parse_args()

    save_dir = REPO / "output" / "play_overlay_dual_saves"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Video: {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"FATAL: cannot open {args.video}", file=sys.stderr); sys.exit(1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_src = cap.get(cv2.CAP_PROP_FPS)
    print(f"  total: {total} frames @ {fps_src:.1f} fps")
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    print("=== loading YOLO + TRT + HF ===")
    from ultralytics import YOLO
    yolo = YOLO(str(YOLO_ONNX), task="detect")
    trt = Trt(ENGINE)
    data = np.load(PROTOS_NPZ, allow_pickle=False)
    protos = data["prototypes"].astype(np.float32)
    classes = [str(c) for c in data["classes"]]
    print(f"  protos classes: {classes}")

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    print("  loading HF DINOv2-base FP16 (REF path)...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    hf_model = AutoModel.from_pretrained("facebook/dinov2-base",
                                         dtype=torch.float16).to("cuda").eval()
    print()
    print("KEYS: SPACE=pause/play  q/ESC=quit  s=save (overlay+raw+json)  +/-=stride  [/]=step")
    print("HUD legend: green=REF (HF reference)  orange=PROD (cv2 stretch + TRT)")
    print()

    win = "DUAL: REF (HF) vs PROD (cv2-stretch+TRT) — DINOv2 prod overlay"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    paused = False
    stride = max(1, args.stride)
    frame_idx = args.start
    last_t = time.time()
    fps_smooth = 0.0
    last_render = None
    last_dets = []
    last_frame_raw = None

    def grab_and_process(idx):
        nonlocal frame_idx
        for _ in range(stride - 1):
            if not cap.grab(): return None, None
            frame_idx += 1
        ok, frame = cap.read()
        if not ok or frame is None: return None, None
        frame_idx += 1

        h_img, w_img = frame.shape[:2]
        results = yolo.predict(source=frame, imgsz=DET_IMGSZ, conf=DET_CONF,
                               iou=DET_IOU, classes=[0], device=0,
                               half=False, verbose=False)
        dets = []
        if results and results[0].boxes is not None:
            r = results[0]
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            # batch HF inference for all detections in this frame
            crops_bgr = []
            valid_boxes = []
            for box, det_conf in zip(boxes, confs):
                x1,y1,x2,y2 = box.tolist()
                if (y2-y1) < MIN_BBOX_H: continue
                x1=max(0,int(x1)); y1=max(0,int(y1))
                x2=min(w_img,int(x2)); y2=min(h_img,int(y2))
                if x2<=x1 or y2<=y1: continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue
                crops_bgr.append(crop)
                valid_boxes.append((x1,y1,x2,y2,float(det_conf)))

            if crops_bgr:
                # REF: batched HF
                pils = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
                inputs = processor(images=pils, return_tensors="pt").to("cuda", torch.float16)
                with torch.inference_mode():
                    ref_emb = hf_model(**inputs).pooler_output.float().cpu().numpy()  # [N,768]
                for i, (x1,y1,x2,y2,dc) in enumerate(valid_boxes):
                    ref = parser(ref_emb[i], protos, classes)
                    prod = parser(trt.infer(prep_stretch(crops_bgr[i])), protos, classes)
                    dets.append({
                        "bbox":(x1,y1,x2,y2),
                        "ref":  ref,
                        "prod": prod,
                        "det_conf": dc,
                    })
        return frame, dets

    while True:
        if not paused:
            frame, dets = grab_and_process(frame_idx)
            if frame is None:
                print("\nEnd of video."); break
            now = time.time(); dt = now - last_t
            if dt > 0:
                fps_smooth = 0.85 * fps_smooth + 0.15 * (1.0/dt)
            last_t = now
            last_render = draw_overlay(frame, dets, frame_idx, fps_smooth, paused, stride)
            last_dets = dets
            last_frame_raw = frame
        if last_render is not None:
            disp = last_render
            if args.scale != 1.0:
                disp = cv2.resize(disp, None, fx=args.scale, fy=args.scale,
                                  interpolation=cv2.INTER_AREA)
            cv2.imshow(win, disp)

        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key in (ord('q'), 27): break
        elif key == ord(' '):
            paused = not paused
            print(f"  {'PAUSED' if paused else 'PLAY'} @ frame {frame_idx}")
        elif key == ord('s'):
            stem = f"frame_{frame_idx:06d}"
            if last_render is not None:
                cv2.imwrite(str(save_dir / f"{stem}_overlay.png"), last_render)
            if last_frame_raw is not None:
                cv2.imwrite(str(save_dir / f"{stem}_raw.jpg"), last_frame_raw,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            preds = [{
                "bbox": d["bbox"],
                "det_conf": d["det_conf"],
                "ref":  {"name": d["ref"][0],  "sim": d["ref"][1]},
                "prod": {"name": d["prod"][0], "sim": d["prod"][1]},
                "agree": d["ref"][0] == d["prod"][0],
            } for d in last_dets]
            (save_dir / f"{stem}_predictions.json").write_text(
                json.dumps({"frame_idx": frame_idx, "video": args.video,
                            "predictions": preds}, indent=2))
            print(f"  SAVED: {stem}_overlay.png + raw.jpg + predictions.json"
                  f"   ({len(preds)} dets, "
                  f"{sum(1 for p in preds if not p['agree'])} disagree)")
        elif key in (ord('+'), ord('=')):
            stride = min(stride+1, 30); print(f"  stride={stride}")
        elif key in (ord('-'), ord('_')):
            stride = max(1, stride-1); print(f"  stride={stride}")
        elif key == ord(']') and paused:
            frame, dets = grab_and_process(frame_idx)
            if frame is not None:
                last_render = draw_overlay(frame, dets, frame_idx, fps_smooth,
                                           paused, stride)
                last_dets = dets; last_frame_raw = frame
        elif key == ord('[') and paused:
            target = max(0, frame_idx - 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            frame_idx = target
            frame, dets = grab_and_process(frame_idx)
            if frame is not None:
                last_render = draw_overlay(frame, dets, frame_idx, fps_smooth,
                                           paused, stride)
                last_dets = dets; last_frame_raw = frame

    cap.release(); cv2.destroyAllWindows()
    print(f"\nLast frame_idx: {frame_idx}")
    print(f"Saved files in: {save_dir}")


if __name__ == "__main__":
    main()
