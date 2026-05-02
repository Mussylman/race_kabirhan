"""
live_frame_saver.py — читает SHM в реальном времени, сохраняет кадры с bbox.

Запускать параллельно с DeepStream.
Кадры берёт из видеофайла по frame_seq, рисует bbox, сохраняет JPEG.

Usage:
    # Single camera:
    python tools/live_frame_saver.py --cam cam-01 --config cameras.json --out /tmp/logs/frames
    # All cameras:
    python tools/live_frame_saver.py --config cameras_all.json --out /tmp/logs/frames
"""

import argparse
import json
import os
import sys
import time
import subprocess
import struct
import ctypes
import ctypes.util
import mmap
import numpy as np
import cv2
from pathlib import Path

# ── SHM constants (must match config.h) ──────────────────────────────
SHM_NAME      = "/rv_detections"
SEM_NAME      = "/rv_detections_sem"
MAX_CAMERAS   = 25
MAX_DETS      = 20
NUM_COLORS    = 5
CAM_ID_LEN    = 16
DETECTION_SIZE     = 56
CAMERA_SLOT_SIZE   = CAM_ID_LEN + 8 + 4 + 4 + 4 + 4 + DETECTION_SIZE * MAX_DETS
SHM_HEADER_SIZE    = 16
SHM_TOTAL_SIZE     = SHM_HEADER_SIZE + CAMERA_SLOT_SIZE * MAX_CAMERAS
DETECTION_FMT      = "<6fIf5fI"
SLOT_HEADER_FMT    = "<16sQIIII"
SLOT_HEADER_SIZE   = struct.calcsize(SLOT_HEADER_FMT)
SHM_HEADER_FMT     = "<QII"
COLOR_NAMES        = ["blue", "green", "purple", "red", "yellow"]

COLOR_BGR = {
    "red":    (0,   30,  220),
    "green":  (30,  180,  30),
    "yellow": (0,   210, 230),
    "blue":   (220,  80,   0),
    "purple": (200,   0, 180),
    "unknown":(128, 128, 128),
}
FPS = 25.0

class _Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


def load_video_paths(config_path: str, cam_filter: str = None) -> dict:
    """Returns {cam_id: video_path} for all or filtered cameras."""
    cfg = json.load(open(config_path))
    paths = {}
    for cam in cfg.get("analytics", []):
        cid = cam["id"]
        if cam_filter and cid != cam_filter:
            continue
        url = cam.get("url", "")
        paths[cid] = url[7:] if url.startswith("file://") else url
    return paths


def extract_frame(video_path: str, seek_secs: float) -> np.ndarray | None:
    cmd = ["ffmpeg", "-y", "-ss", f"{seek_secs:.3f}", "-i", video_path,
           "-frames:v", "1", "-f", "image2pipe", "-vcodec", "bmp", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True, timeout=8)
    if r.returncode != 0 or len(r.stdout) < 100:
        return None
    buf = np.frombuffer(r.stdout, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def draw(frame, detections, frame_w, frame_h, cam_id, frame_seq, ts_capture):
    h, w = frame.shape[:2]
    sx = w / (frame_w or w)
    sy = h / (frame_h or h)

    for det in detections:
        bbox = det.get("bbox")
        if not bbox: continue
        x1, y1, x2, y2 = int(bbox[0]*sx), int(bbox[1]*sy), int(bbox[2]*sx), int(bbox[3]*sy)
        color = det.get("color", "unknown")
        conf  = det.get("conf", 0)
        tid   = det.get("track_id", 0)
        bgr   = COLOR_BGR.get(color, COLOR_BGR["unknown"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
        label = f"{color.upper()} {conf}% t{tid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ly = max(y1 - 4, th + 4)
        cv2.rectangle(frame, (x1, ly-th-4), (x1+tw+4, ly), bgr, -1)
        cv2.putText(frame, label, (x1+2, ly-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

    # info bar
    bar = np.zeros((28, w, 3), dtype=np.uint8)
    info = f"{cam_id}  frame={frame_seq}  ts={ts_capture:.3f}  dets={len(detections)}"
    cv2.putText(bar, info, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    return np.vstack([bar, frame])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam",    default=None, help="Single camera filter (omit for all)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out",    default="/tmp/logs/frames")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    video_paths = load_video_paths(args.config, args.cam)
    if not video_paths:
        print(f"ERROR: no cameras found in {args.config}"); sys.exit(1)

    cam_ids = sorted(video_paths.keys())
    print(f"[frame_saver] Tracking {len(cam_ids)} cameras: {', '.join(cam_ids)}")
    print(f"[frame_saver] Output: {args.out}")

    # Open per-camera JSONL files
    jsonl_files = {}
    for cid in cam_ids:
        path = os.path.join(args.out, f"{cid}.jsonl")
        jsonl_files[cid] = open(path, "w")
        print(f"  {cid} → {video_paths[cid]}")

    # Attach SHM
    librt_path = ctypes.util.find_library("rt")
    librt = ctypes.CDLL(librt_path or "librt.so.1", use_errno=True)

    for _ in range(30):
        fd = librt.shm_open(SHM_NAME.encode(), 0, 0o666)
        if fd >= 0: break
        print("Waiting for DeepStream SHM..."); time.sleep(1)
    else:
        print("ERROR: SHM not found"); sys.exit(1)

    shm_buf = mmap.mmap(fd, SHM_TOTAL_SIZE, mmap.MAP_SHARED, mmap.PROT_READ)

    librt.sem_open.restype = ctypes.c_void_p
    sem = librt.sem_open(SEM_NAME.encode(), 0)

    last_seq = 0
    saved = {cid: 0 for cid in cam_ids}
    total_saved = 0

    print(f"[frame_saver] Reading SHM — Ctrl+C to stop\n")

    try:
        while True:
            now = time.time() + 0.2
            ts = _Timespec(int(now), int((now % 1) * 1e9))
            librt.sem_timedwait.restype = ctypes.c_int
            librt.sem_timedwait.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Timespec)]
            librt.sem_timedwait(ctypes.c_void_p(sem), ctypes.byref(ts))

            shm_buf.seek(0)
            write_seq, num_cams, _ = struct.unpack(SHM_HEADER_FMT, shm_buf.read(16))
            if write_seq == last_seq:
                continue
            last_seq = write_seq

            for i in range(min(num_cams, MAX_CAMERAS)):
                offset = SHM_HEADER_SIZE + i * CAMERA_SLOT_SIZE
                shm_buf.seek(offset)
                slot = shm_buf.read(SLOT_HEADER_SIZE)
                cam_raw, ts_us, fw, fh, nd, _ = struct.unpack(SLOT_HEADER_FMT, slot)
                cam_id = cam_raw.rstrip(b'\x00').decode('ascii', errors='replace')

                if cam_id not in video_paths or nd == 0:
                    for _ in range(nd): shm_buf.read(DETECTION_SIZE)
                    continue

                dets = []
                for _ in range(min(nd, MAX_DETS)):
                    d = struct.unpack(DETECTION_FMT, shm_buf.read(DETECTION_SIZE))
                    x1,y1,x2,y2,cx,dconf,cid_color,cconf,p0,p1,p2,p3,p4,tid = d
                    color = COLOR_NAMES[cid_color] if cid_color < NUM_COLORS else "unknown"
                    dets.append({
                        "color": color, "conf": round(cconf*100),
                        "track_id": int(tid),
                        "bbox": (int(x1),int(y1),int(x2),int(y2))
                    })

                ts_cap = ts_us / 1e6
                print(f"  [{cam_id}] frame={write_seq} ts={ts_cap:.3f} dets={len(dets)} "
                      f"colors={[d['color'] for d in dets]}")

                # Save to JSONL
                video_sec = round(write_seq / FPS, 2)
                rec = {"cam_id": cam_id, "frame_seq": write_seq, "video_sec": video_sec,
                       "ts_capture": ts_cap, "frame_w": fw, "frame_h": fh, "detections": dets}
                jsonl_files[cam_id].write(json.dumps(rec) + "\n")
                jsonl_files[cam_id].flush()

                # Extract and save frame
                seek = write_seq / FPS
                frame = extract_frame(video_paths[cam_id], seek)
                if frame is not None:
                    frame = draw(frame, dets, fw, fh, cam_id, write_seq, ts_cap)
                    fname = f"{cam_id}_f{write_seq:05d}.jpg"
                    cv2.imwrite(os.path.join(args.out, fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    saved[cam_id] += 1
                    total_saved += 1
                    print(f"    → saved {fname}  (total={total_saved})")

    except KeyboardInterrupt:
        pass
    finally:
        for f in jsonl_files.values():
            f.close()
        shm_buf.close()
        print(f"\n[frame_saver] Done. Total saved: {total_saved}")
        for cid in cam_ids:
            if saved[cid] > 0:
                print(f"  {cid}: {saved[cid]} frames")


if __name__ == "__main__":
    main()
