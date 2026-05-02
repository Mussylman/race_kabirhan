"""
extract_analysis_frames.py — Post-run frame extractor for Race Vision analysis.

Reads /tmp/race_analysis/detections.jsonl (written during the run),
extracts the actual video frame at that moment, draws colored bboxes,
saves annotated JPEG to /tmp/race_analysis/frames/.

Usage:
    python tools/extract_analysis_frames.py [--jsonl /tmp/race_analysis/detections.jsonl]
                                             [--config configs/cameras_all_25.json]
                                             [--out /tmp/race_analysis/frames]
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Color map: jockey color name → BGR
COLOR_BGR = {
    "red":    (0,   30,  220),
    "green":  (30,  180, 30),
    "yellow": (0,   210, 230),
    "blue":   (220, 80,  0),
    "purple": (200, 0,   180),
    "unknown":(128, 128, 128),
}

FPS = 25.0  # DeepStream pipeline fps — used to convert frame_seq → video time


def load_cam_files(config_path: str) -> dict[str, str]:
    """Returns {cam_id: /path/to/video.mp4}."""
    with open(config_path) as f:
        cfg = json.load(f)
    result = {}
    for cam in cfg.get("analytics", []):
        url = cam.get("url", "")
        # file:///path → /path
        if url.startswith("file://"):
            path = url[7:]
        elif url.startswith("/"):
            path = url
        else:
            continue
        if os.path.exists(path):
            result[cam["id"]] = path
    return result


def extract_frame_ffmpeg(video_path: str, seek_secs: float) -> np.ndarray | None:
    """Extract a single frame at seek_secs using ffmpeg. Returns BGR numpy array."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{seek_secs:.3f}",
        "-i", video_path,
        "-frames:v", "1",
        "-f", "image2pipe",
        "-vcodec", "bmp",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=10)
    if result.returncode != 0 or len(result.stdout) < 100:
        return None
    buf = np.frombuffer(result.stdout, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return frame


def draw_bboxes(frame: np.ndarray, detections: list, frame_w: int, frame_h: int) -> np.ndarray:
    """Draw detection bboxes on frame. Handles scale difference."""
    h, w = frame.shape[:2]
    scale_x = w / frame_w if frame_w else 1.0
    scale_y = h / frame_h if frame_h else 1.0

    for det in detections:
        bbox = det.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = int(x1 * scale_x); y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)

        color_name = det.get("color", "unknown")
        conf = det.get("conf", 0)
        bgr = COLOR_BGR.get(color_name, COLOR_BGR["unknown"])
        track_id = det.get("track_id", 0)

        # bbox rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)

        # label
        label = f"{color_name.upper()} {conf}% t{track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ly = max(y1 - 4, th + 4)
        cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 4, ly), bgr, -1)
        cv2.putText(frame, label, (x1 + 2, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def add_info_bar(frame: np.ndarray, event: dict) -> np.ndarray:
    """Add top info bar with cam_id, frame_seq, ts."""
    cam_id = event["cam_id"]
    frame_seq = event["frame_seq"]
    ts = event["ts_capture"]
    n_dets = len(event.get("detections", []))
    text = f"{cam_id}  frame={frame_seq}  ts={ts:.3f}  dets={n_dets}"

    bar = np.zeros((30, frame.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([bar, frame])


def process(jsonl_path: str, config_path: str, out_dir: str, max_frames_per_cam: int = 20):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cam_files = load_cam_files(config_path)
    print(f"Camera files found: {len(cam_files)}")
    print(f"Output dir: {out_dir}")

    # Group events by cam_id — take max_frames_per_cam evenly spaced
    events_by_cam: dict[str, list] = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            cam_id = ev.get("cam_id")
            if cam_id:
                events_by_cam.setdefault(cam_id, []).append(ev)

    print(f"Cameras with detections: {sorted(events_by_cam.keys())}")
    total_saved = 0

    for cam_id, events in sorted(events_by_cam.items()):
        video_path = cam_files.get(cam_id)
        if not video_path:
            print(f"  [{cam_id}] no video file — skip")
            continue

        # Pick evenly spaced events
        step = max(1, len(events) // max_frames_per_cam)
        selected = events[::step][:max_frames_per_cam]
        print(f"  [{cam_id}] {len(events)} events → extracting {len(selected)} frames")

        for ev in selected:
            frame_seq = ev["frame_seq"]
            seek_secs = frame_seq / FPS

            frame = extract_frame_ffmpeg(video_path, seek_secs)
            if frame is None:
                print(f"    frame_seq={frame_seq} → ffmpeg failed")
                continue

            frame = draw_bboxes(frame, ev.get("detections", []),
                                ev.get("frame_w", frame.shape[1]),
                                ev.get("frame_h", frame.shape[0]))
            frame = add_info_bar(frame, ev)

            fname = f"{cam_id}_f{frame_seq:05d}_t{int(ev['ts_capture'])}.jpg"
            out_path = os.path.join(out_dir, fname)
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            total_saved += 1

    print(f"\nDone. Saved {total_saved} annotated frames → {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl",   default="/tmp/race_analysis/detections.jsonl")
    ap.add_argument("--config",  default="configs/cameras_all_25.json")
    ap.add_argument("--out",     default="/tmp/race_analysis/frames")
    ap.add_argument("--max-per-cam", type=int, default=20)
    args = ap.parse_args()

    if not os.path.exists(args.jsonl):
        print(f"ERROR: {args.jsonl} not found — run the pipeline first")
        return

    process(args.jsonl, args.config, args.out, args.max_per_cam)


if __name__ == "__main__":
    main()
