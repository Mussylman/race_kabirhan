#!/usr/bin/env python3
"""
Test BoT-SORT-ReID tracker on cam-05 video.
Uses our YOLO jockey detector + BoT-SORT with fast-reid appearance features.
"""
import sys
import os
import cv2
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

# Add BoT-SORT to path
BOTSORT_DIR = "/tmp/YOLOv12-BoT-SORT-ReID/BoT-SORT"
sys.path.insert(0, BOTSORT_DIR)
os.chdir(BOTSORT_DIR)

VIDEO = "/home/user/recordings/yaris_20260303_162028/kamera_05_162030_END162507.mp4"
YOLO_MODEL = "/home/user/race_vision/models/jockey_yolov11s.pt"
REID_CONFIG = "logs/sbs_S50/config.yaml"
REID_WEIGHTS = "logs/sbs_S50/model_0016.pth"
OUTPUT_DIR = "/home/user/race_vision/ds_results/botsort_reid"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    from ultralytics import YOLO
    from tracker.mc_bot_sort import BoTSORT
    from types import SimpleNamespace

    print("Loading YOLO detector...")
    yolo = YOLO(YOLO_MODEL)

    # BoT-SORT config — use same params as their working pipeline
    tracker_opts = SimpleNamespace(
        # ReID
        with_reid=True,
        fast_reid_config=REID_CONFIG,
        fast_reid_weights=REID_WEIGHTS,
        # Tracker params (from predict_track3.py defaults)
        track_high_thresh=0.3,
        track_low_thresh=0.05,
        new_track_thresh=0.4,
        track_buffer=60,       # frames to keep lost tracks (same as their script)
        match_thresh=0.7,
        proximity_thresh=0.5,
        appearance_thresh=0.25,
        # Camera motion compensation
        cmc_method="none",
        # Other
        mot20=False,
        ablation=False,
        name="botsort_reid",
        device="cuda:0",
    )

    print("Creating BoT-SORT tracker with ReID...")
    tracker = BoTSORT(tracker_opts, frame_rate=15)

    # Open video
    cap = cv2.VideoCapture(VIDEO)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {w}x{h}, {fps}fps, {total_frames} frames")

    # Output video
    out_path = os.path.join(OUTPUT_DIR, "botsort_reid_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # Track color by ID
    track_colors = {}
    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
              (0, 255, 255), (128, 0, 255), (255, 128, 0), (0, 128, 255), (128, 255, 0)]

    # Collect results
    all_results = []  # (frame_id, track_id, x1, y1, x2, y2, score)
    max_frames = min(total_frames, 1200)  # first 1200 frames (~48s, while jockeys visible)

    frame_id = 0
    while cap.isOpened() and frame_id < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection — low conf like their pipeline (0.09), tracker filters later
        results = yolo(frame, verbose=False, conf=0.1, imgsz=1280)
        dets = results[0].boxes

        if len(dets) > 0:
            # Format: [x1, y1, x2, y2, score, class]
            xyxy = dets.xyxy.cpu().numpy()
            conf = dets.conf.cpu().numpy()
            cls = dets.cls.cpu().numpy()

            det_array = np.column_stack([xyxy, conf, cls])

            # BoT-SORT update (returns tracked, lost)
            online_targets, _ = tracker.update(det_array, frame)
        else:
            online_targets, _ = tracker.update(np.empty((0, 6)), frame)

        # Draw results
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            x1, y1, tw, th = tlwh
            x2, y2 = x1 + tw, y1 + th

            all_results.append((frame_id, tid, x1, y1, x2, y2, t.score))

            # Color per track
            if tid not in track_colors:
                track_colors[tid] = COLORS[len(track_colors) % len(COLORS)]
            color = track_colors[tid]

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"ID:{tid} {t.score:.2f}",
                        (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out_writer.write(frame)

        frame_id += 1
        if frame_id % 100 == 0:
            n_active = len(online_targets)
            print(f"  frame {frame_id}/{max_frames}, active tracks: {n_active}, total IDs: {len(track_colors)}")

    cap.release()
    out_writer.release()

    # === Statistics ===
    print(f"\n{'='*60}")
    print(f"BoT-SORT-ReID RESULTS ({frame_id} frames)")
    print(f"{'='*60}")
    print(f"Total unique track IDs: {len(track_colors)}")
    print(f"Output video: {out_path}")

    # Per-track stats
    tracks = defaultdict(list)
    for fid, tid, x1, y1, x2, y2, score in all_results:
        tracks[tid].append((fid, x1, y1, x2, y2, score))

    print(f"\nPer-track summary:")
    for tid in sorted(tracks.keys()):
        entries = tracks[tid]
        frames = [e[0] for e in entries]
        scores = [e[5] for e in entries]
        print(f"  ID {tid:3d}: {len(entries):5d} dets, frames {min(frames):4d}-{max(frames):4d}, "
              f"avg_score={np.mean(scores):.2f}")

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "tracks.csv")
    with open(csv_path, "w") as f:
        f.write("frame,track_id,x1,y1,x2,y2,score\n")
        for fid, tid, x1, y1, x2, y2, score in all_results:
            f.write(f"{fid},{tid},{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f},{score:.4f}\n")
    print(f"CSV: {csv_path}")

    # Save crops of each track (first 5 frames)
    crops_dir = os.path.join(OUTPUT_DIR, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    # Re-read video for crop saving
    cap2 = cv2.VideoCapture(VIDEO)
    saved_per_track = defaultdict(int)
    result_idx = 0

    for fid in range(min(frame_id, 200)):  # first 200 frames
        ret, frame = cap2.read()
        if not ret:
            break
        for _, tid, x1, y1, x2, y2, score in [r for r in all_results if r[0] == fid]:
            if saved_per_track[tid] < 5:
                crop = frame[max(0,int(y1)):int(y2), max(0,int(x1)):int(x2)]
                if crop.size > 0:
                    cv2.imwrite(f"{crops_dir}/id{tid:03d}_f{fid:04d}.jpg", crop)
                    saved_per_track[tid] += 1
    cap2.release()
    print(f"Crops saved: {crops_dir}/")


if __name__ == "__main__":
    main()
