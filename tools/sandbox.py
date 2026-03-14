#!/usr/bin/env python3
"""
Sandbox — jockey detection & color on ORIGINAL resolution with ROI crop.

Key differences from test_visual.py:
  - Works on ORIGINAL frame resolution (no 800px downscale for color)
  - ROI crop: select region of interest with mouse (skip empty areas)
  - Strict filtering: min 100px bbox height, aspect ratio > 1.0
  - Large torso crop panel: see exactly what the color model sees
  - Auto-save torso crops for training dataset
  - YOLO runs on full frame, color classifier gets high-res crops

Usage:
    python tools/sandbox.py video.mp4
    python tools/sandbox.py video.mp4 --roi                    # select ROI with mouse
    python tools/sandbox.py video.mp4 --min-height 120         # stricter filter
    python tools/sandbox.py video.mp4 --save-crops crops_out/  # save torso crops
    python tools/sandbox.py video.mp4 --model models/race_classifier.pt
    python tools/sandbox.py video.mp4 --imgsz 1280             # YOLO input size

Controls:
    SPACE   = pause/resume
    N       = next frame (while paused)
    S       = save current crops to disk
    R       = select ROI with mouse
    C       = clear ROI (use full frame)
    +/-     = increase/decrease min bbox height by 10
    Q/ESC   = quit
    RIGHT   = seek +30 frames
    LEFT    = seek -30 frames
"""

import sys
import os
import cv2
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Torso extraction (same params as C++ pipeline) ──────────────────

TORSO_TOP = 0.10
TORSO_BOTTOM = 0.40
TORSO_LEFT = 0.20
TORSO_RIGHT = 0.20


def extract_torso(frame, x1, y1, x2, y2):
    """Extract torso crop from original resolution frame."""
    h, w = y2 - y1, x2 - x1
    ty1 = max(0, y1 + int(h * TORSO_TOP))
    ty2 = min(frame.shape[0], y1 + int(h * TORSO_BOTTOM))
    tx1 = max(0, x1 + int(w * TORSO_LEFT))
    tx2 = min(frame.shape[1], x2 - int(w * TORSO_RIGHT))
    if ty2 - ty1 < 5 or tx2 - tx1 < 5:
        return None, (tx1, ty1, tx2, ty2)
    return frame[ty1:ty2, tx1:tx2].copy(), (tx1, ty1, tx2, ty2)


def make_crop_panel(crops, labels, panel_h=200, max_crops=8):
    """Build a horizontal panel of torso crops with labels for side-by-side view."""
    if not crops:
        panel = np.zeros((panel_h, 400, 3), dtype=np.uint8)
        cv2.putText(panel, "No detections", (10, panel_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        return panel

    cell_w = 150
    n = min(len(crops), max_crops)
    panel = np.zeros((panel_h, cell_w * n, 3), dtype=np.uint8)

    for i in range(n):
        crop = crops[i]
        label = labels[i]
        if crop is None:
            continue

        # Resize crop to fill cell (maintain aspect ratio)
        ch, cw = crop.shape[:2]
        target_h = panel_h - 40  # leave space for label
        scale = target_h / ch
        new_w = min(int(cw * scale), cell_w - 4)
        new_h = min(int(ch * scale), target_h)
        resized = cv2.resize(crop, (new_w, new_h))

        # Center in cell
        x_off = (cell_w * i) + (cell_w - new_w) // 2
        y_off = 2
        panel[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        # Label below crop
        text_x = cell_w * i + 4
        text_y = panel_h - 8
        cv2.putText(panel, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return panel


def save_crops_to_disk(crops, labels, save_dir, frame_idx):
    """Save torso crops with labels as filenames."""
    os.makedirs(save_dir, exist_ok=True)
    saved = 0
    for i, (crop, label) in enumerate(zip(crops, labels)):
        if crop is None:
            continue
        # Save as: unsorted/frame_00150_det_0_green_87.jpg
        fname = f"frame_{frame_idx:06d}_det_{i}_{label.replace(' ', '_')}.jpg"
        path = os.path.join(save_dir, fname)
        cv2.imwrite(path, crop)
        saved += 1
    return saved


def select_roi(frame, window_name="Select ROI"):
    """Let user draw a rectangle on the frame to select ROI."""
    print("\n  Draw a rectangle on the frame. Press ENTER to confirm, C to cancel.")

    # Scale down for display if frame is huge
    fh, fw = frame.shape[:2]
    scale = min(1.0, 1600 / fw, 900 / fh)
    if scale < 1.0:
        display = cv2.resize(frame, (int(fw * scale), int(fh * scale)))
    else:
        display = frame.copy()

    roi = cv2.selectROI(window_name, display, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)

    if roi[2] == 0 or roi[3] == 0:
        return None

    # Scale back to original coords
    x, y, w, h = roi
    x1 = int(x / scale)
    y1 = int(y / scale)
    x2 = int((x + w) / scale)
    y2 = int((y + h) / scale)

    print(f"  ROI selected: ({x1}, {y1}) → ({x2}, {y2})  [{x2-x1}x{y2-y1}]")
    return (x1, y1, x2, y2)


COLORS_BGR = {
    "green":  (0, 200, 0),
    "red":    (0, 0, 255),
    "yellow": (0, 230, 230),
    "blue":   (255, 150, 0),
    "purple": (200, 0, 200),
    "unknown": (128, 128, 128),
}


def main():
    parser = argparse.ArgumentParser(description="Sandbox: jockey detection on original quality")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--model", default="models/color_classifier_v2.pt",
                        help="Color classifier model (.pt)")
    parser.add_argument("--yolo", default="models/jockey_yolov11s.pt",
                        help="YOLO detector model (.pt)")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="YOLO input size (default: 1280 for better quality)")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="YOLO confidence threshold")
    parser.add_argument("--min-height", type=int, default=100,
                        help="Min bbox height in pixels (original resolution)")
    parser.add_argument("--min-aspect", type=float, default=1.0,
                        help="Min aspect ratio h/w (person = taller than wide)")
    parser.add_argument("--edge-margin", type=int, default=15,
                        help="Ignore detections near frame edges (pixels)")
    parser.add_argument("--roi", action="store_true",
                        help="Select ROI with mouse before starting")
    parser.add_argument("--save-crops", default=None,
                        help="Directory to auto-save torso crops")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"Video not found: {args.video}")
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  SANDBOX: {os.path.basename(args.video)}")
    print(f"  Original resolution, strict filtering, crop panel")
    print(f"{'='*65}\n")

    # ── Load models ──────────────────────────────────────────────────

    from pipeline.trt_inference import ColorClassifierInfer, YOLODetector

    print(f"Loading YOLO: {args.yolo}  (imgsz={args.imgsz})")
    detector = YOLODetector(fallback_pt=args.yolo, imgsz=args.imgsz, conf=args.conf)

    print(f"Loading color classifier: {args.model}")
    classifier = ColorClassifierInfer(fallback_pt=args.model)
    print(f"  Classes: {classifier.classes}")

    # ── Open video ───────────────────────────────────────────────────

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Cannot open {args.video}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Video: {orig_w}x{orig_h}, {total_frames} frames @ {fps:.1f} fps")

    # ── ROI selection ────────────────────────────────────────────────

    roi = None
    if args.roi:
        ret, first_frame = cap.read()
        if ret:
            roi = select_roi(first_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── Filtering params (mutable at runtime) ────────────────────────

    min_height = args.min_height
    min_aspect = args.min_aspect
    edge_margin = args.edge_margin

    print(f"\n  Filters: min_height={min_height}px, min_aspect={min_aspect:.1f}, edge={edge_margin}px")
    if roi:
        print(f"  ROI: ({roi[0]}, {roi[1]}) → ({roi[2]}, {roi[3]})")
    print(f"\n  Controls: SPACE=pause  N=next  R=ROI  C=clearROI  S=save  +/-=height  Q=quit\n")

    # ── Main loop ────────────────────────────────────────────────────

    frame_idx = 0
    paused = False
    frame = None
    total_saved = 0

    while True:
        if not paused or frame is None:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        fh, fw = frame.shape[:2]

        # Apply ROI: crop frame for detection
        if roi:
            rx1, ry1, rx2, ry2 = roi
            detect_frame = frame[ry1:ry2, rx1:rx2].copy()
            offset_x, offset_y = rx1, ry1
        else:
            detect_frame = frame
            offset_x, offset_y = 0, 0

        dfh, dfw = detect_frame.shape[:2]

        # ── Detect ───────────────────────────────────────────────────

        t0 = time.time()
        batch_dets = detector.detect_batch([detect_frame])
        dets = batch_dets[0] if batch_dets else []
        det_ms = (time.time() - t0) * 1000

        # ── Filter + extract torso crops ─────────────────────────────

        valid_dets = []
        torso_crops = []
        torso_rects = []

        for det in dets:
            x1, y1, x2, y2 = det['bbox']
            bh = y2 - y1
            bw = x2 - x1

            # Filter: edges (relative to detect_frame)
            if x1 <= edge_margin or x2 >= dfw - edge_margin:
                continue

            # Filter: min height (on ORIGINAL resolution)
            if bh < min_height:
                continue

            # Filter: aspect ratio (person should be taller than wide)
            if bw > 0 and bh / bw < min_aspect:
                continue

            # Extract torso from ORIGINAL frame (not resized)
            abs_x1 = x1 + offset_x
            abs_y1 = y1 + offset_y
            abs_x2 = x2 + offset_x
            abs_y2 = y2 + offset_y

            torso, torso_rect = extract_torso(frame, abs_x1, abs_y1, abs_x2, abs_y2)
            if torso is None:
                continue

            pixels = torso.shape[0] * torso.shape[1]
            if pixels < 300:
                continue

            valid_dets.append({
                'bbox': (abs_x1, abs_y1, abs_x2, abs_y2),
                'conf': det['conf'],
                'bbox_roi': (x1, y1, x2, y2),  # relative to ROI
            })
            torso_crops.append(torso)
            torso_rects.append(torso_rect)

        # ── Classify colors on ORIGINAL resolution crops ─────────────

        t1 = time.time()
        if torso_crops:
            classifications = classifier.classify_batch(torso_crops)
        else:
            classifications = []
        cls_ms = (time.time() - t1) * 1000

        # ── Build display ────────────────────────────────────────────

        # Scale frame for display (fit in ~1600x900)
        disp_scale = min(1.0, 1600 / fw, 800 / fh)
        display = cv2.resize(frame, (int(fw * disp_scale), int(fh * disp_scale)))

        # Draw ROI rectangle
        if roi:
            rx1, ry1, rx2, ry2 = roi
            cv2.rectangle(display,
                          (int(rx1 * disp_scale), int(ry1 * disp_scale)),
                          (int(rx2 * disp_scale), int(ry2 * disp_scale)),
                          (0, 255, 0), 2)

        # Draw detections
        crop_labels = []
        for i, (det, (color, conf, prob_dict)) in enumerate(zip(valid_dets, classifications)):
            x1, y1, x2, y2 = det['bbox']
            bgr = COLORS_BGR.get(color, (128, 128, 128))

            # Bbox on display
            dx1 = int(x1 * disp_scale)
            dy1 = int(y1 * disp_scale)
            dx2 = int(x2 * disp_scale)
            dy2 = int(y2 * disp_scale)
            cv2.rectangle(display, (dx1, dy1), (dx2, dy2), bgr, 2)

            # Torso rect
            tx1, ty1t, tx2, ty2t = torso_rects[i]
            cv2.rectangle(display,
                          (int(tx1 * disp_scale), int(ty1t * disp_scale)),
                          (int(tx2 * disp_scale), int(ty2t * disp_scale)),
                          (0, 255, 255), 1)

            # Label
            bh = y2 - y1
            label = f"{color} {conf:.0%} [{bh}px]"
            cv2.putText(display, label, (dx1, dy1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr, 2)

            # Detection confidence
            det_label = f"det:{det['conf']:.0%}"
            cv2.putText(display, det_label, (dx1, dy2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            crop_labels.append(f"{color} {conf:.0%}")

        # ── Info panel (top) ─────────────────────────────────────────

        panel_h = 55
        cv2.rectangle(display, (0, 0), (display.shape[1], panel_h), (0, 0, 0), -1)
        cv2.putText(display,
                    f"Frame {frame_idx}/{total_frames}  |  {len(valid_dets)} jockeys  |"
                    f"  YOLO: {det_ms:.0f}ms  Color: {cls_ms:.0f}ms",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(display,
                    f"min_h={min_height}px  aspect>{min_aspect:.1f}  "
                    f"res={orig_w}x{orig_h}  imgsz={args.imgsz}"
                    f"{'  ROI' if roi else ''}",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        # ── Crop panel (bottom) ──────────────────────────────────────

        crop_panel = make_crop_panel(torso_crops, crop_labels, panel_h=180)

        # Combine: display + crop panel
        # Match widths
        dw = display.shape[1]
        pw = crop_panel.shape[1]
        if pw < dw:
            pad = np.zeros((crop_panel.shape[0], dw - pw, 3), dtype=np.uint8)
            crop_panel = np.hstack([crop_panel, pad])
        elif pw > dw:
            crop_panel = crop_panel[:, :dw]

        combined = np.vstack([display, crop_panel])

        cv2.imshow("Sandbox", combined)

        # ── Auto-save crops ──────────────────────────────────────────

        if args.save_crops and torso_crops and not paused:
            n = save_crops_to_disk(torso_crops, crop_labels, args.save_crops, frame_idx)
            total_saved += n

        # ── Keyboard ─────────────────────────────────────────────────

        wait_ms = 1 if not paused else 0
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord('q') or key == 27:
            break

        elif key == ord(' '):
            paused = not paused

        elif key == ord('n'):
            paused = True
            ret, frame = cap.read()
            if ret:
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        elif key == ord('s') and torso_crops:
            save_dir = args.save_crops or "sandbox_crops"
            n = save_crops_to_disk(torso_crops, crop_labels, save_dir, frame_idx)
            total_saved += n
            print(f"  Saved {n} crops to {save_dir}/ (total: {total_saved})")

        elif key == ord('r'):
            # Select new ROI
            roi = select_roi(frame)

        elif key == ord('c'):
            roi = None
            print("  ROI cleared — using full frame")

        elif key == ord('+') or key == ord('='):
            min_height += 10
            print(f"  min_height = {min_height}px")

        elif key == ord('-') and min_height > 20:
            min_height -= 10
            print(f"  min_height = {min_height}px")

        elif key == 83 or key == ord('d'):  # right arrow
            new_pos = min(frame_idx + 30, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            ret, frame = cap.read()
            if ret:
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        elif key == 81 or key == ord('a'):  # left arrow
            new_pos = max(frame_idx - 30, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            ret, frame = cap.read()
            if ret:
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Total crops saved: {total_saved}")


if __name__ == "__main__":
    main()
