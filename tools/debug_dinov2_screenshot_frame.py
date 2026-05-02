#!/usr/bin/env python3
"""debug_dinov2_screenshot_frame.py — взять конкретный кадр из
cam-24 second session video (где user видел wrong predictions),
прогнать ВСЕ детекции через 4 пути:
  P1: HF processor + HF model       (REF: что строило prototypes)
  P2: HF processor + TRT engine     (parity path)
  P3: cv2 HF emulation + TRT        (matching prototype build preproc)
  P4: cv2 stretch + nvinfer + TRT   (PROD, текущий sgie_color.txt)

Сохраняем gallery с per-detection 4-path predictions для visual verify.

ЕСЛИ P1 даёт правильные ответы, а P4 нет → preprocessing/prod-path bug.
ЕСЛИ P1 тоже неправильно → cross-domain failure (модель в принципе
не работает на этом домене), не bug в коде.
"""
from __future__ import annotations
import argparse, json, sys
from html import escape
from pathlib import Path
import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
VIDEO = REPO / "data/videos/test_full_loop/yaris_20260421_174240/kamera_24_174252_END174647.mp4"
YOLO_ONNX = REPO / "models" / "yolo11s_person_960.onnx"
WORK = Path("/home/ipodrom/race_vision_bench/dinov2_prod")
ENGINE = WORK / "dinov2_base_b1_gpu0_fp16.engine"
PROTOS_NPZ = WORK / "prototypes_dinov2_v1.npz"

OUT_DIR = REPO / "output" / "phase4_screenshot_diag_2026-04-27"

DET_CONF = 0.25; DET_IOU = 0.5; DET_IMGSZ = 960; MIN_BBOX_H = 25
MIN_SIM = 0.55

NVINFER_OFFSETS = np.array([123.675, 116.28, 103.53], dtype=np.float32)
NVINFER_SCALE   = np.float32(0.01735)
IMAGENET_MEAN   = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD    = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_batch(rgb): return np.transpose(rgb, (2,0,1))[None,:].copy()


def prep_stretch(crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    return to_batch(((rgb.astype(np.float32) - NVINFER_OFFSETS) * NVINFER_SCALE))


def prep_hf_emul(crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if h <= w: scale=256/h; new_h,new_w=256,max(1,int(round(w*scale)))
    else:      scale=256/w; new_w,new_h=256,max(1,int(round(h*scale)))
    rgb_r = cv2.resize(rgb, (new_w,new_h), interpolation=cv2.INTER_LINEAR)
    cy=(new_h-224)//2; cx=(new_w-224)//2
    crop = rgb_r[cy:cy+224, cx:cx+224]
    return to_batch(((crop.astype(np.float32)/255.0) - IMAGENET_MEAN) / IMAGENET_STD)


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


def parser(emb, protos, classes, thr=0.55):
    n = float(np.linalg.norm(emb))+1e-12
    sims = (protos @ emb)/n
    b = int(np.argmax(sims)); ms = float(sims[b])
    if ms < thr: return "unknown", ms, sims
    return classes[b], ms, sims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=1775,
                    help="frame index to extract (default 1775 — last frame_idx of overlay)")
    ap.add_argument("--range", type=int, default=0,
                    help="±N frames around target (default 0 = single frame)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    crops_dir = OUT_DIR / "crops"; crops_dir.mkdir(exist_ok=True)

    print(f"Video: {VIDEO.name}")
    print(f"Frame target: {args.frame}  ±{args.range}")

    cap = cv2.VideoCapture(str(VIDEO))
    target_frames = list(range(max(0, args.frame - args.range),
                               args.frame + args.range + 1))
    frames = {}
    for tgt in target_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, tgt)
        ok, f = cap.read()
        if ok and f is not None:
            frames[tgt] = f
    cap.release()
    if not frames:
        print("FATAL: could not read any target frame", file=sys.stderr); sys.exit(1)
    print(f"  read {len(frames)} frames")

    # Setup
    print("=== loading YOLO + TRT + HF + protos ===")
    from ultralytics import YOLO
    yolo = YOLO(str(YOLO_ONNX), task="detect")
    trt = Trt(ENGINE)
    data = np.load(PROTOS_NPZ, allow_pickle=False)
    protos = data["prototypes"].astype(np.float32)
    classes = [str(c) for c in data["classes"]]
    print(f"  protos: {protos.shape}  classes={classes}")

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    hf_model = AutoModel.from_pretrained("facebook/dinov2-base",
                                         dtype=torch.float16).to("cuda").eval()

    # For each frame: detect + predict 4 ways
    rows = []
    for fidx, frame in frames.items():
        h_img, w_img = frame.shape[:2]
        results = yolo.predict(source=frame, imgsz=DET_IMGSZ, conf=DET_CONF,
                               iou=DET_IOU, classes=[0], device=0, half=False, verbose=False)
        if not results or results[0].boxes is None: continue
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for det_i, (box, det_conf) in enumerate(zip(boxes, confs)):
            x1,y1,x2,y2 = box.tolist()
            if (y2-y1) < MIN_BBOX_H: continue
            x1=max(0,int(x1)); y1=max(0,int(y1))
            x2=min(w_img,int(x2)); y2=min(h_img,int(y2))
            if x2<=x1 or y2<=y1: continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue

            # Save crop
            crop_name = f"f{fidx:06d}_d{det_i:02d}.jpg"
            cv2.imwrite(str(crops_dir / crop_name), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

            # P1: HF + HF
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            inputs = processor(images=[pil], return_tensors="pt").to("cuda", torch.float16)
            with torch.inference_mode():
                e1 = hf_model(**inputs).pooler_output.float().cpu().numpy()[0]
            p1 = parser(e1, protos, classes)

            # P2: HF preproc + TRT
            px = inputs["pixel_values"].float().cpu().numpy()
            e2 = trt.infer(px.copy())
            p2 = parser(e2, protos, classes)

            # P3: cv2 HF emul + TRT
            e3 = trt.infer(prep_hf_emul(crop))
            p3 = parser(e3, protos, classes)

            # P4: PROD
            e4 = trt.infer(prep_stretch(crop))
            p4 = parser(e4, protos, classes)

            rows.append({
                "frame": fidx, "det": det_i, "crop": crop_name,
                "bbox": [x1,y1,x2,y2], "h": y2-y1, "w": x2-x1,
                "det_conf": float(det_conf),
                "P1": {"name": p1[0], "sim": p1[1], "all_sims": [float(x) for x in p1[2]]},
                "P2": {"name": p2[0], "sim": p2[1], "all_sims": [float(x) for x in p2[2]]},
                "P3": {"name": p3[0], "sim": p3[1], "all_sims": [float(x) for x in p3[2]]},
                "P4": {"name": p4[0], "sim": p4[1], "all_sims": [float(x) for x in p4[2]]},
            })

    print()
    print(f"  total detections: {len(rows)}")
    print()
    print(f"  {'crop':<22s} {'h×w':<10s} {'AR':<5s} {'P1 (REF)':<14s} {'P2':<14s} {'P3':<14s} {'P4 (PROD)':<14s}")
    print("  " + "-"*100)
    for r in rows:
        ar = r["h"] / max(r["w"], 1)
        hw = f"{r['h']}x{r['w']}"
        print(f"  {r['crop']:<22s} {hw:<10s} {ar:<5.2f} "
              f"{r['P1']['name']:<7s}{r['P1']['sim']:.2f}    "
              f"{r['P2']['name']:<7s}{r['P2']['sim']:.2f}    "
              f"{r['P3']['name']:<7s}{r['P3']['sim']:.2f}    "
              f"{r['P4']['name']:<7s}{r['P4']['sim']:.2f}    ")

    # Save JSON + gallery
    (OUT_DIR / "predictions.json").write_text(json.dumps(rows, indent=2))

    # Gallery
    parts = [
        '<!doctype html><html><head><meta charset="utf-8">',
        '<title>cam-24 second session — 4-path predictions</title>',
        '<style>',
        'body{font-family:system-ui;background:#1a1a1a;color:#ddd;margin:0;padding:12px;}',
        '.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;}',
        '.card{background:#2a2a2a;border-radius:4px;padding:6px;}',
        '.card img{max-width:100%;max-height:300px;display:block;margin:0 auto;}',
        '.meta{font-size:11px;color:#aaa;padding:4px;line-height:1.5;}',
        '.row{display:flex;justify-content:space-between;font-family:monospace;}',
        '.label{color:#88ddff;}.green{color:#3a3;}.yellow{color:#aa3;}.red{color:#a33;}.blue{color:#4af;}.unknown{color:#888;}',
        '</style></head><body>',
        f'<h1 style="color:#fff">cam-24 second session — frame {args.frame}±{args.range}</h1>',
        '<p style="color:#aaa">P1=HF+HF (REF) · P2=HF+TRT · P3=cv2-HF+TRT · P4=PROD (cv2-stretch+TRT)</p>',
        '<div class="grid">'
    ]
    for r in rows:
        ar = r["h"] / max(r["w"], 1)
        rows_html = ""
        for tag in ["P1","P2","P3","P4"]:
            p = r[tag]
            cls = p["name"]
            rows_html += (f'<div class="row"><span class="label">{tag}</span>'
                          f'<span class="{cls}">{cls} {p["sim"]:.3f}</span></div>')
        parts.append(
            f'<div class="card">'
            f'<img src="crops/{escape(r["crop"])}" loading="lazy">'
            f'<div class="meta">{escape(r["crop"])}<br>'
            f'h×w={r["h"]}×{r["w"]} AR={ar:.2f} det_conf={r["det_conf"]:.2f}<br>'
            f'{rows_html}</div></div>'
        )
    parts.append('</div></body></html>')
    (OUT_DIR / "gallery.html").write_text("\n".join(parts))
    print()
    print(f"  saved: {OUT_DIR / 'predictions.json'}")
    print(f"         {OUT_DIR / 'gallery.html'}")
    print(f"         {crops_dir} ({len(rows)} crops)")


if __name__ == "__main__":
    main()
