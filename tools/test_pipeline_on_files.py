#!/usr/bin/env python3
"""
Test full YOLO + Color pipeline on recorded video files.
Uses YOLOv11s (jockey detector) + EfficientNet-V2-S (color classifier).
Scans all 25 cameras, finds jockeys, classifies colors.
"""

import os
import sys
import glob
import time
import cv2
import torch
import numpy as np
import torch.nn as nn

# ── Detection filters ─────────────────────────────────────────────

MIN_BBOX_HEIGHT = 30
EDGE_MARGIN = 10
MIN_CROP_PIXELS = 200
MAX_CROP_PIXELS = 15000
CONF_THRESHOLD = 0.30

# Torso extraction
TORSO_TOP = 0.10
TORSO_BOTTOM = 0.40
TORSO_LEFT = 0.20
TORSO_RIGHT = 0.20


def extract_torso(frame, bbox):
    """Extract torso region from jockey bounding box."""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = y2 - y1, x2 - x1
    ty1 = max(0, y1 + int(h * TORSO_TOP))
    ty2 = min(frame.shape[0], y1 + int(h * TORSO_BOTTOM))
    tx1 = max(0, x1 + int(w * TORSO_LEFT))
    tx2 = min(frame.shape[1], x2 - int(w * TORSO_RIGHT))
    if ty2 - ty1 < 10 or tx2 - tx1 < 10:
        return None
    return frame[ty1:ty2, tx1:tx2]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rec_dir = args[0] if args else "/home/user/recordings/yaris_20260303_162028/"

    print(f"\n{'='*70}")
    print(f"  PIPELINE TEST: {os.path.basename(rec_dir)}")
    print(f"  Models: YOLOv11s (jockey) + EfficientNet-V2-S (color)")
    print(f"{'='*70}\n")

    # Load YOLO jockey detector
    from ultralytics import YOLO
    yolo = YOLO("models/jockey_yolov11s.pt")
    print(f"[OK] YOLOv11s loaded (classes: {yolo.names})")

    # Load EfficientNet-V2-S color classifier
    from torchvision import models
    ckpt = torch.load("models/color_classifier_v2.pt", map_location="cpu", weights_only=False)
    COLOR_NAMES = ckpt["classes"]
    img_size = ckpt.get("img_size", 128)

    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, len(COLOR_NAMES)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().cuda()
    print(f"[OK] EfficientNet-V2-S loaded, classes: {COLOR_NAMES}, input: {img_size}x{img_size}")

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def preprocess_crop(crop_bgr):
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (img_size, img_size))
        t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        for c in range(3):
            t[c] = (t[c] - MEAN[c]) / STD[c]
        return t

    # Find all camera files
    all_files = sorted(glob.glob(os.path.join(rec_dir, "kamera_*.mp4")))
    print(f"[OK] Found {len(all_files)} camera recordings\n")

    all_results = {}
    scan_step = 50     # Every 50th frame (~2 sec at 25fps)
    show_display = "--show" in sys.argv

    for fpath in all_files:
        fname = os.path.basename(fpath)
        cam_num = int(fname.split("_")[1])
        cam_id = f"cam-{cam_num:02d}"

        cap = cv2.VideoCapture(fpath)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        cam_results = []
        t0 = time.time()

        for pos in range(0, total_frames, scan_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO jockey detection
            results = yolo(frame, imgsz=800, conf=CONF_THRESHOLD, verbose=False)

            jockeys = results[0].boxes
            if jockeys is None or len(jockeys) == 0:
                continue

            frame_detections = []
            fh, fw = frame.shape[:2]

            # Collect valid torso crops for batch classification
            crops = []
            crop_meta = []  # (bbox, det_conf)

            for b in jockeys:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                bh = y2 - y1
                bw = x2 - x1
                det_conf = float(b.conf)

                if x1 <= EDGE_MARGIN or x2 >= fw - EDGE_MARGIN:
                    continue
                if bh < MIN_BBOX_HEIGHT:
                    continue

                torso = extract_torso(frame, (x1, y1, x2, y2))
                if torso is None:
                    continue

                crop_pixels = torso.shape[0] * torso.shape[1]
                if crop_pixels < MIN_CROP_PIXELS or crop_pixels > MAX_CROP_PIXELS:
                    continue

                crops.append(torso)
                crop_meta.append(((x1, y1, x2, y2), det_conf, f"{bw}x{bh}"))

            # Batch classify
            if crops:
                tensors = torch.stack([preprocess_crop(c) for c in crops]).cuda()
                with torch.no_grad():
                    logits = model(tensors)
                    probs = torch.softmax(logits, dim=1)

                for i, ((bbox, det_conf, size), prob) in enumerate(zip(crop_meta, probs)):
                    color_id = prob.argmax().item()
                    color_conf = prob[color_id].item()
                    color_name = COLOR_NAMES[color_id]
                    x1, y1, x2, y2 = bbox

                    frame_detections.append({
                        "bbox": bbox,
                        "det_conf": det_conf,
                        "color": color_name,
                        "color_conf": color_conf,
                        "size": size,
                        "center_x": (x1 + x2) / 2 / fw,
                        "probs": {COLOR_NAMES[j]: f"{prob[j].item():.2%}" for j in range(len(COLOR_NAMES))},
                    })

            if frame_detections:
                cam_results.append({
                    "frame": pos,
                    "time_sec": pos / fps,
                    "detections": frame_detections,
                })

                if show_display:
                    COLOR_BGR = {
                        "green": (0, 255, 0),
                        "red": (0, 0, 255),
                        "yellow": (0, 255, 255),
                    }
                    for d in frame_detections:
                        bx1, by1, bx2, by2 = d["bbox"]
                        bgr = COLOR_BGR.get(d["color"], (255, 255, 255))
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), bgr, 2)
                        label = f"{d['color']} {d['color_conf']:.0%} det={d['det_conf']:.0%}"
                        cv2.putText(frame, label, (bx1, by1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
                        # Draw torso region
                        h, w = by2 - by1, bx2 - bx1
                        ty1 = by1 + int(h * TORSO_TOP)
                        ty2 = by1 + int(h * TORSO_BOTTOM)
                        tx1 = bx1 + int(w * TORSO_LEFT)
                        tx2 = bx2 - int(w * TORSO_RIGHT)
                        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 255, 255), 1)
                    title = f"{cam_id} frame={pos} t={pos/fps:.1f}s"
                    cv2.putText(frame, title, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.imshow("Race Vision Test", frame)
                    key = cv2.waitKey(0)
                    if key == ord('q'):
                        show_display = False

        elapsed = time.time() - t0
        cap.release()
        all_results[cam_num] = cam_results

        n_frames = len(cam_results)
        total_jockeys = sum(len(fr["detections"]) for fr in cam_results)

        if n_frames > 0:
            first_t = cam_results[0]["time_sec"]
            last_t = cam_results[-1]["time_sec"]

            color_counts = {}
            for fr in cam_results:
                for d in fr["detections"]:
                    if d["color_conf"] > 0.5:
                        color_counts[d["color"]] = color_counts.get(d["color"], 0) + 1

            colors_str = ", ".join(f"{c}:{n}" for c, n in sorted(color_counts.items()))
            print(f"  {cam_id}: {n_frames} frames, jockeys={total_jockeys}, "
                  f"window={first_t:.1f}-{last_t:.1f}s, "
                  f"colors=[{colors_str}]  ({elapsed:.1f}s)")
        else:
            print(f"  {cam_id}: no detections  ({elapsed:.1f}s)")

    # ── Global timeline ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  RACE TIMELINE (jockey passage through cameras)")
    print(f"{'='*70}\n")

    for cam_num in sorted(all_results.keys()):
        results = all_results[cam_num]
        if not results:
            continue

        cam_id = f"cam-{cam_num:02d}"
        first_t = results[0]["time_sec"]
        last_t = results[-1]["time_sec"]

        best = max(results, key=lambda r: len(r["detections"]))
        best_dets = best["detections"]
        jockeys_str = " | ".join(
            f"{d['color']}({d['color_conf']:.0%})"
            for d in best_dets
        )

        bar_start = int(first_t / 2)
        bar_end = int(last_t / 2)
        bar = " " * bar_start + "\u2588" * max(1, bar_end - bar_start)

        print(f"  {cam_id} [{first_t:5.1f}s-{last_t:5.1f}s] {bar}")
        if jockeys_str:
            print(f"           best: {jockeys_str}")

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
