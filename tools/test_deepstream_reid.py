#!/usr/bin/env python3
"""
test_deepstream_reid.py — Run DeepStream C++ (YOLO detection) + Python ReID.

Two-phase approach:
  Phase 1: DeepStream processes video → collects all detections from SHM
  Phase 2: Re-read video, crop detections, run ReID, save annotated video

Usage:
    python tools/test_deepstream_reid.py
    python tools/test_deepstream_reid.py --save /tmp/reid_output.mp4
"""

import argparse
import sys
import time
import signal
import subprocess
import threading
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VIDEO_PATH = "/home/user/recordings/yaris_20260303_162028/kamera_05_162030_END162507.mp4"
GALLERY_DIR = "data/reid"
REID_BACKEND = "dinov2"

DS_BINARY = "deepstream/build/race_vision_deepstream"
DS_CONFIG = "cameras_1cam.json"
DS_YOLO_ENGINE = "deepstream/configs/nvinfer_jockey_1cam.txt"
DS_COLOR_ENGINE = "models/color_classifier_v2.engine"

COLORS_BGR = {
    "blue":    (255, 100, 0),
    "green":   (0, 200, 0),
    "purple":  (200, 0, 200),
    "red":     (0, 0, 255),
    "yellow":  (0, 230, 230),
    "unknown": (128, 128, 128),
}


def _log(msg, log_file=None):
    """Print and optionally write to log file."""
    print(msg)
    if log_file:
        log_file.write(msg + '\n')
        log_file.flush()


def phase1_collect_detections(log_file=None):
    """Run DeepStream and collect all detections from SHM."""
    ds_cmd = [
        DS_BINARY,
        "--config", DS_CONFIG,
        "--yolo-engine", DS_YOLO_ENGINE,
        "--color-engine", DS_COLOR_ENGINE,
        "--mux-width", "1520",
        "--mux-height", "1520",
        "--file-mode",
    ]
    _log(f"[Phase 1] Starting DeepStream: {' '.join(ds_cmd)}", log_file)
    ds_proc = subprocess.Popen(ds_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    ds_lines = []
    def _forward():
        for line in iter(ds_proc.stdout.readline, b''):
            text = line.decode('utf-8', errors='replace').rstrip()
            ds_lines.append(text)
            if '[Pipeline]' in text or '[Main]' in text or 'ERROR' in text:
                _log(f"  [DS] {text}", log_file)
    ds_thread = threading.Thread(target=_forward, daemon=True)
    ds_thread.start()

    # Wait for SHM
    _log("[Phase 1] Waiting for SHM...", log_file)
    from pipeline.shm_reader import SharedMemoryReader
    reader = SharedMemoryReader(timeout_ms=500)
    for _ in range(30):
        if reader.attach():
            break
        time.sleep(0.5)
    else:
        _log("ERROR: SHM not available", log_file)
        ds_proc.terminate()
        sys.exit(1)
    _log("[Phase 1] SHM attached, collecting detections...", log_file)

    # Collect all detections keyed by SHM write_seq
    all_detections = []  # list of (seq, cam_results)
    seq_count = 0

    while True:
        results = reader.read()
        if results:
            seq_count += 1
            for cam_det in results:
                if cam_det.n_detections > 0:
                    all_detections.append({
                        'seq': seq_count,
                        'cam_id': cam_det.cam_id,
                        'frame_w': cam_det.frame_width,
                        'frame_h': cam_det.frame_height,
                        'detections': list(cam_det.detections),
                    })
        elif ds_proc.poll() is not None:
            # DS exited, drain remaining
            for _ in range(10):
                results = reader.read()
                if results:
                    seq_count += 1
                    for cam_det in results:
                        if cam_det.n_detections > 0:
                            all_detections.append({
                                'seq': seq_count,
                                'cam_id': cam_det.cam_id,
                                'frame_w': cam_det.frame_width,
                                'frame_h': cam_det.frame_height,
                                'detections': list(cam_det.detections),
                            })
            break

    reader.detach()
    ds_proc.wait(timeout=5)

    # Parse frame numbers from DS log lines
    # Format: [DET] frm=7     cam=cam-05 trk=0     yolo=0.73  bbox=  37x77   @( 608, 618)  class=UNKNWN ...
    import re
    det_pattern = re.compile(
        r'frm=(\d+)\s+cam=(\S+)\s+trk=(\d+)\s+yolo=([\d.]+)\s+bbox=\s*(\d+)x(\d+)\s+@\(\s*(\d+),\s*(\d+)\)\s+class=(\S+)'
    )
    frame_detections = defaultdict(list)
    for line in ds_lines:
        if '[DET]' not in line:
            continue
        m = det_pattern.search(line)
        if m:
            frm = int(m.group(1))
            det_info = {
                'track_id': int(m.group(3)),
                'yolo_conf': float(m.group(4)),
                'bbox_w': int(m.group(5)),
                'bbox_h': int(m.group(6)),
                'center_x': int(m.group(7)),
                'center_y': int(m.group(8)),
                'ds_color': m.group(9),
            }
            frame_detections[frm].append(det_info)

    _log(f"[Phase 1] Done: {len(frame_detections)} frames with detections, "
         f"{sum(len(v) for v in frame_detections.values())} total detections", log_file)
    return frame_detections


def phase2_reid(frame_detections, save_path=None, display=False, log_file=None, crops_dir=None):
    """Re-read video, crop detections, run ReID with detailed diagnostics."""
    _log(f"\n[Phase 2] Loading ReID ({REID_BACKEND})...", log_file)
    from pipeline.jockey_reid import JockeyReID
    reid = JockeyReID(gallery_dir=GALLERY_DIR, backend=REID_BACKEND, device="cuda:0")
    stats = reid.get_gallery_stats()
    _log(f"  {stats['n_jockeys']} jockeys, {stats['n_embeddings']} embeddings", log_file)
    _log(f"  Gallery: {stats.get('jockeys', {})}", log_file)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _log(f"  Video: {vid_w}x{vid_h} @ {fps:.1f}fps, {total} frames", log_file)

    # Mux resolution from DS
    mux_w, mux_h = 1520, 1520
    scale_x = vid_w / mux_w
    scale_y = vid_h / mux_h

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (vid_w, vid_h))
        _log(f"  Saving to: {save_path}", log_file)

    # CSV for detailed analysis
    csv_path = None
    csv_file = None
    if log_file:
        csv_path = log_file.name.replace('.log', '.csv') if hasattr(log_file, 'name') else None
    if csv_path:
        csv_file = open(csv_path, 'w')
        csv_file.write("frame,track_id,ds_color,reid_name,reid_score,"
                       "crop_w,crop_h,crop_pixels,yolo_conf,"
                       "top1_name,top1_score,top2_name,top2_score,top3_name,top3_score\n")

    header = (f"\n{'frame':>6} | {'trk':>3} | {'DS':>7} | {'ReID':>8} | {'score':>5} | "
              f"{'crop':>7} | {'top1':>14} | {'top2':>14} | {'top3':>14}")
    _log(header, log_file)
    _log("-" * 110, log_file)

    frame_idx = 0
    reid_total = 0
    crop_count = 0
    jockey_stats = defaultdict(lambda: {'count': 0, 'scores': [], 'tracks': set(), 'crop_sizes': []})
    track_timeline = defaultdict(list)  # track_id -> [(frame, name, score)]
    crop_size_buckets = {'tiny': 0, 'small': 0, 'medium': 0, 'large': 0}  # crop quality

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        dets = frame_detections.get(frame_idx)
        if not dets:
            if writer:
                writer.write(frame)
            if display:
                cv2.imshow("ReID", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        crops = []
        valid_dets = []

        for det in dets:
            bw = det.get('bbox_w', 0)
            bh = det.get('bbox_h', 0)
            cx = det.get('center_x', 0)
            cy = det.get('center_y', 0)

            x1_mux = cx - bw // 2
            y1_mux = cy - bh // 2
            x2_mux = cx + bw // 2
            y2_mux = cy + bh // 2

            x1 = max(0, int(x1_mux * scale_x))
            y1 = max(0, int(y1_mux * scale_y))
            x2 = min(vid_w, int(x2_mux * scale_x))
            y2 = min(vid_h, int(y2_mux * scale_y))

            body_h = y2 - y1
            torso_y2 = y1 + int(body_h * 0.55)
            crop = frame[y1:torso_y2, x1:x2]

            if crop.size > 100:
                crops.append(crop)
                valid_dets.append((det, x1, y1, x2, y2))

        if not crops:
            if writer:
                writer.write(frame)
            continue

        identities = reid.identify_batch(crops)
        reid_total += len(identities)

        for i, ((det, x1, y1, x2, y2), (name, score, details)) in enumerate(zip(valid_dets, identities)):
            ds_color = det.get('ds_color', '?').lower()
            track_id = det.get('track_id', 0)
            yolo_conf = det.get('yolo_conf', 0)
            cw, ch = x2 - x1, y2 - y1
            torso_h = int(ch * 0.55)
            crop_px = cw * torso_h

            # Crop size bucket
            if crop_px < 500:
                crop_size_buckets['tiny'] += 1
            elif crop_px < 2000:
                crop_size_buckets['small'] += 1
            elif crop_px < 5000:
                crop_size_buckets['medium'] += 1
            else:
                crop_size_buckets['large'] += 1

            # Top-3 from details
            sorted_details = sorted(details.items(), key=lambda x: -x[1]) if details else []
            top3 = [(n, s) for n, s in sorted_details[:3]]
            while len(top3) < 3:
                top3.append(('-', 0.0))

            top3_str = ' | '.join(f"{n:>6} {s:.3f}" for n, s in top3)
            _log(f"{frame_idx:6d} | {track_id:3d} | {ds_color:>7} | {name:>8} | {score:.3f} | "
                 f"{cw:3d}x{torso_h:<3d} | {top3_str}", log_file)

            # CSV
            if csv_file:
                csv_file.write(f"{frame_idx},{track_id},{ds_color},{name},{score:.4f},"
                               f"{cw},{torso_h},{crop_px},{yolo_conf:.3f},"
                               f"{top3[0][0]},{top3[0][1]:.4f},"
                               f"{top3[1][0]},{top3[1][1]:.4f},"
                               f"{top3[2][0]},{top3[2][1]:.4f}\n")

            # Stats
            jockey_stats[name]['count'] += 1
            jockey_stats[name]['scores'].append(score)
            jockey_stats[name]['tracks'].add(track_id)
            jockey_stats[name]['crop_sizes'].append(crop_px)
            track_timeline[track_id].append((frame_idx, name, score))

            # Save crops
            if crops_dir and crop_count < 500:
                crop_img = crops[i]
                fname = f"f{frame_idx:05d}_t{track_id}_{name}_{score:.3f}_{cw}x{torso_h}.jpg"
                cv2.imwrite(str(crops_dir / fname), crop_img)
                crop_count += 1

            # Draw on frame
            color = COLORS_BGR.get(name, (128, 128, 128))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if writer:
            writer.write(frame)
        if display:
            cv2.imshow("ReID", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if writer:
        writer.release()
        _log(f"\nSaved: {save_path}", log_file)
    if csv_file:
        csv_file.close()
        _log(f"CSV saved: {csv_path}", log_file)
    if display:
        cv2.destroyAllWindows()

    # ── Summary ──────────────────────────────────────────────────────
    _log(f"\n{'='*70}", log_file)
    _log(f"=== SUMMARY ===", log_file)
    _log(f"{'='*70}", log_file)
    _log(f"Frames total: {frame_idx}, with detections: {len(frame_detections)}", log_file)
    _log(f"DS detections: {sum(len(v) for v in frame_detections.values())}", log_file)
    _log(f"ReID identifications: {reid_total}", log_file)

    # Crop size distribution
    _log(f"\n--- Crop size distribution ---", log_file)
    _log(f"  tiny  (<500px):  {crop_size_buckets['tiny']:5d}", log_file)
    _log(f"  small (<2000px): {crop_size_buckets['small']:5d}", log_file)
    _log(f"  medium(<5000px): {crop_size_buckets['medium']:5d}", log_file)
    _log(f"  large (5000px+): {crop_size_buckets['large']:5d}", log_file)

    # Per-jockey breakdown
    _log(f"\n--- Per-jockey breakdown ---", log_file)
    _log(f"  {'jockey':>10} | {'count':>5} | {'avg':>5} | {'min':>5} | {'max':>5} | {'avg_crop':>8} | tracks", log_file)
    _log(f"  {'-'*75}", log_file)
    for name in sorted(jockey_stats.keys()):
        s = jockey_stats[name]
        avg = np.mean(s['scores'])
        mn = np.min(s['scores'])
        mx = np.max(s['scores'])
        avg_crop = np.mean(s['crop_sizes'])
        trks = ','.join(str(t) for t in sorted(s['tracks']))
        _log(f"  {name:>10} | {s['count']:5d} | {avg:.3f} | {mn:.3f} | {mx:.3f} | {avg_crop:7.0f}px | {trks}", log_file)

    # Track stability
    _log(f"\n--- Track stability (per track_id) ---", log_file)
    _log(f"  {'trk':>3} | {'frames':>6} | {'dominant':>10} | {'dom%':>5} | identities seen", log_file)
    _log(f"  {'-'*70}", log_file)
    for tid in sorted(track_timeline.keys()):
        entries = track_timeline[tid]
        id_counts = defaultdict(int)
        for _, name, _ in entries:
            id_counts[name] += 1
        dominant = max(id_counts, key=id_counts.get)
        dom_pct = id_counts[dominant] / len(entries) * 100
        id_list = ', '.join(f"{n}({c})" for n, c in sorted(id_counts.items(), key=lambda x: -x[1]))
        _log(f"  {tid:3d} | {len(entries):6d} | {dominant:>10} | {dom_pct:4.0f}% | {id_list}", log_file)

    if crops_dir:
        _log(f"\nCrops saved: {crop_count} images in {crops_dir}/", log_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default=None, help="Save annotated video")
    parser.add_argument("--display", action="store_true", help="Show output in cv2 window")
    parser.add_argument("--log", default=None, help="Save detection log to file")
    parser.add_argument("--save-crops", default=None, help="Save crop images to directory")
    args = parser.parse_args()

    # Set up logging to file
    log_file = None
    if args.log:
        log_file = open(args.log, 'w')

    # Create crops dir
    crops_dir = None
    if args.save_crops:
        crops_dir = Path(args.save_crops)
        crops_dir.mkdir(parents=True, exist_ok=True)

    frame_dets = phase1_collect_detections(log_file=log_file)
    phase2_reid(frame_dets, save_path=args.save, display=args.display,
                log_file=log_file, crops_dir=crops_dir)

    if log_file:
        log_file.close()
        print(f"Log saved: {args.log}")


if __name__ == "__main__":
    main()
