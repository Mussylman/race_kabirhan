#!/usr/bin/env python3
"""
Test full YOLO + ColorCNN pipeline on recorded video files.
Scans all 25 cameras, finds frames with horses, classifies jockey colors.
"""

import os
import sys
import glob
import time
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn as nn

# ── Color classifier model ──────────────────────────────────────────

class SimpleColorCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


# ── Detection filters (same as C++ pipeline) ────────────────────────

MIN_BBOX_HEIGHT = 40       # Lowered from 65 for testing
MIN_ASPECT_RATIO = 1.0     # Lowered from 1.2 — riders may be wider
EDGE_MARGIN = 10
MIN_CROP_PIXELS = 200
MAX_CROP_PIXELS = 15000
PERSON_CLASS = 0
HORSE_CLASS = 17
CONF_THRESHOLD = 0.30

# Only these 3 jockey colors are in the race
ACTIVE_COLORS = {"green", "red", "yellow"}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rec_dir = args[0] if args else "/home/user/recordings/yaris_20260303_162028/"

    print(f"\n{'='*70}")
    print(f"  PIPELINE TEST ON FILES: {os.path.basename(rec_dir)}")
    print(f"{'='*70}\n")

    # Load YOLO
    from ultralytics import YOLO
    yolo = YOLO("models/yolov8s.pt")
    print(f"[OK] YOLOv8s loaded")

    # Load color classifier
    ckpt = torch.load("models/color_classifier.pt", map_location="cpu")
    color_model = SimpleColorCNN(num_classes=5)
    color_model.load_state_dict(ckpt["model_state_dict"])
    color_model.eval().cuda()
    COLOR_NAMES = ckpt["classes"]
    print(f"[OK] ColorCNN loaded, classes: {COLOR_NAMES}")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Find all camera files
    all_files = sorted(glob.glob(os.path.join(rec_dir, "kamera_*.mp4")))
    print(f"[OK] Found {len(all_files)} camera recordings\n")

    # Results storage
    all_results = {}   # cam_num -> list of (frame_pos, time_sec, detections)

    scan_step = 50     # Check every 50th frame (~2 sec at 25fps)
    show_display = "--show" in sys.argv  # --show to display frames with detections

    for fpath in all_files:
        fname = os.path.basename(fpath)
        cam_num = int(fname.split("_")[1])
        cam_id = f"cam-{cam_num:02d}"

        cap = cv2.VideoCapture(fpath)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        duration = total_frames / fps

        cam_results = []
        t0 = time.time()

        for pos in range(0, total_frames, scan_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO inference
            results = yolo(frame, imgsz=800, conf=CONF_THRESHOLD, verbose=False)

            persons = []
            horses = []
            for b in results[0].boxes:
                cls = int(b.cls)
                if cls == PERSON_CLASS:
                    persons.append(b)
                elif cls == HORSE_CLASS:
                    horses.append(b)

            if not persons and not horses:
                continue

            # Get horse bboxes for proximity filter
            horse_bboxes = []
            for b in horses:
                hx1, hy1, hx2, hy2 = map(int, b.xyxy[0].tolist())
                horse_bboxes.append((hx1, hy1, hx2, hy2))

            # Filter and classify persons
            frame_detections = []
            fh, fw = frame.shape[:2]

            for b in persons:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                bh = y2 - y1
                bw = x2 - x1
                det_conf = float(b.conf)

                # Filters
                if x1 <= EDGE_MARGIN or x2 >= fw - EDGE_MARGIN:
                    continue
                if bh < MIN_BBOX_HEIGHT:
                    continue

                # Must be near a horse — filters out spectators/staff
                if horse_bboxes:
                    px = (x1 + x2) / 2
                    py = (y1 + y2) / 2
                    margin = max(bh, 80)  # margin scales with person size
                    near_horse = False
                    for hx1, hy1, hx2, hy2 in horse_bboxes:
                        if hx1 - margin < px < hx2 + margin and hy1 - margin < py < hy2 + margin:
                            near_horse = True
                            break
                    if not near_horse:
                        continue

                # Torso crop (upper half)
                torso_y2 = y1 + bh // 2
                crop = frame[y1:torso_y2, x1:x2]
                if crop.shape[0] < 5 or crop.shape[1] < 5:
                    continue

                crop_pixels = crop.shape[0] * crop.shape[1]
                if crop_pixels < MIN_CROP_PIXELS or crop_pixels > MAX_CROP_PIXELS:
                    continue

                # Color classify
                pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                tensor = transform(pil).unsqueeze(0).cuda()
                with torch.no_grad():
                    logits = color_model(tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    color_id = probs.argmax().item()
                    color_conf = probs[color_id].item()

                color_name = COLOR_NAMES[color_id]
                # Skip colors not in this race
                if color_name not in ACTIVE_COLORS:
                    continue

                frame_detections.append({
                    "type": "person",
                    "bbox": (x1, y1, x2, y2),
                    "det_conf": det_conf,
                    "color": color_name,
                    "color_conf": color_conf,
                    "size": f"{bw}x{bh}",
                    "center_x": (x1 + x2) / 2 / fw,  # normalized
                })

            # Also track horse detections (for awareness)
            for b in horses:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                bh = y2 - y1
                bw = x2 - x1
                frame_detections.append({
                    "type": "horse",
                    "bbox": (x1, y1, x2, y2),
                    "det_conf": float(b.conf),
                    "color": "-",
                    "color_conf": 0,
                    "size": f"{bw}x{bh}",
                    "center_x": (x1 + x2) / 2 / fw,
                })

            if frame_detections:
                cam_results.append({
                    "frame": pos,
                    "time_sec": pos / fps,
                    "detections": frame_detections,
                })

                # Draw detections on frame and show
                if show_display:
                    COLOR_BGR = {
                        "green": (0, 255, 0),
                        "red": (0, 0, 255),
                        "yellow": (0, 255, 255),
                    }
                    for d in frame_detections:
                        bx1, by1, bx2, by2 = d["bbox"]
                        if d["type"] == "horse":
                            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (180, 130, 70), 1)
                            cv2.putText(frame, "horse", (bx1, by2 + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 130, 70), 1)
                        else:
                            bgr = COLOR_BGR.get(d["color"], (255, 255, 255))
                            cv2.rectangle(frame, (bx1, by1), (bx2, by2), bgr, 2)
                            label = f"{d['color']} {d['color_conf']:.0%}"
                            cv2.putText(frame, label, (bx1, by1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)
                    title = f"{cam_id} frame={pos} t={pos/fps:.1f}s"
                    cv2.putText(frame, title, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.imshow("Race Vision Test", frame)
                    key = cv2.waitKey(0)  # Press any key for next, 'q' to quit
                    if key == ord('q'):
                        show_display = False

        elapsed = time.time() - t0
        cap.release()

        all_results[cam_num] = cam_results

        # Print summary for this camera
        n_frames_with_dets = len(cam_results)
        total_persons = sum(
            len([d for d in fr["detections"] if d["type"] == "person"])
            for fr in cam_results
        )
        total_horses = sum(
            len([d for d in fr["detections"] if d["type"] == "horse"])
            for fr in cam_results
        )

        if n_frames_with_dets > 0:
            first_t = cam_results[0]["time_sec"]
            last_t = cam_results[-1]["time_sec"]

            # Color summary
            color_counts = {}
            for fr in cam_results:
                for d in fr["detections"]:
                    if d["type"] == "person" and d["color_conf"] > 0.5:
                        color_counts[d["color"]] = color_counts.get(d["color"], 0) + 1

            colors_str = ", ".join(f"{c}:{n}" for c, n in sorted(color_counts.items()))
            print(f"  {cam_id}: {n_frames_with_dets} frames, "
                  f"persons={total_persons} horses={total_horses}, "
                  f"window={first_t:.1f}-{last_t:.1f}s, "
                  f"colors=[{colors_str}]  ({elapsed:.1f}s)")
        else:
            print(f"  {cam_id}: no detections  ({elapsed:.1f}s)")

    # ── Global timeline ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  RACE TIMELINE (horse passage through cameras)")
    print(f"{'='*70}\n")

    for cam_num in sorted(all_results.keys()):
        results = all_results[cam_num]
        if not results:
            continue

        cam_id = f"cam-{cam_num:02d}"
        first_t = results[0]["time_sec"]
        last_t = results[-1]["time_sec"]

        # Best frame (most detections)
        best = max(results, key=lambda r: len(r["detections"]))
        best_dets = best["detections"]
        persons_str = " | ".join(
            f"{d['color']}({d['color_conf']:.0%})"
            for d in best_dets if d["type"] == "person"
        )

        bar_start = int(first_t / 2)
        bar_end = int(last_t / 2)
        bar = " " * bar_start + "█" * max(1, bar_end - bar_start)

        print(f"  {cam_id} [{first_t:5.1f}s-{last_t:5.1f}s] {bar}")
        if persons_str:
            print(f"           best: {persons_str}")

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
