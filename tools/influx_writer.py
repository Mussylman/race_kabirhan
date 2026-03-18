"""
influx_writer.py — Reads DeepStream SHM detections and writes to InfluxDB v2.

Usage:
    python3 tools/influx_writer.py [--url http://localhost:8086] [--token <token>]
                                   [--org race_vision] [--bucket race_vision]

Metrics written:
    detections       — per-camera detection count + color distribution
    track_events     — per track_id: color, confidence, position
    pipeline_health  — FPS, total detections
"""

import time
import struct
import ctypes
import ctypes.util
import mmap
import argparse
import sys
import os
import logging
from collections import defaultdict
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("influx_writer")

# ── SHM constants (must match config.h) ─────────────────────────────

SHM_NAME       = "/rv_detections"
SEM_NAME       = "/rv_detections_sem"
MAX_CAMERAS    = 25
MAX_DETECTIONS = 20
NUM_COLORS     = 5
CAM_ID_LEN     = 16
COLOR_NAMES    = ["blue", "green", "purple", "red", "yellow"]

DETECTION_SIZE         = 56
CAMERA_SLOT_HEADER_FMT = "<16sQIIII"
CAMERA_SLOT_HEADER_SIZE = struct.calcsize(CAMERA_SLOT_HEADER_FMT)
DETECTION_FMT          = "<6fIf5fI"
SHM_HEADER_FMT         = "<QII"
SHM_HEADER_SIZE        = struct.calcsize(SHM_HEADER_FMT)
CAMERA_SLOT_SIZE       = CAMERA_SLOT_HEADER_SIZE + DETECTION_SIZE * MAX_DETECTIONS

# ── POSIX SHM / SEM ──────────────────────────────────────────────────

def _load_libs():
    librt_name = ctypes.util.find_library("rt") or "librt.so.1"
    librt = ctypes.CDLL(librt_name, use_errno=True)
    libc  = ctypes.CDLL(None, use_errno=True)
    return librt, libc

def attach_shm(librt):
    O_RDONLY = 0
    PROT_READ = 1
    MAP_SHARED = 1
    shm_size = SHM_HEADER_SIZE + CAMERA_SLOT_SIZE * MAX_CAMERAS

    librt.shm_open.restype = ctypes.c_int
    fd = librt.shm_open(SHM_NAME.encode(), O_RDONLY, 0)
    if fd < 0:
        raise OSError(f"shm_open failed: {ctypes.get_errno()}")

    libc = ctypes.CDLL(None)
    libc.mmap.restype = ctypes.c_void_p
    ptr = libc.mmap(None, shm_size, PROT_READ, MAP_SHARED, fd, 0)
    if ptr == ctypes.c_void_p(-1).value:
        raise OSError("mmap failed")

    import os
    os.close(fd)
    return ptr, shm_size

def open_sem(librt):
    librt.sem_open.restype = ctypes.c_void_p
    sem = librt.sem_open(SEM_NAME.encode(), 0)
    if sem == ctypes.c_void_p(-1).value:
        raise OSError("sem_open failed")
    return sem

def sem_timedwait(librt, sem, timeout_ms=200):
    import ctypes
    class Timespec(ctypes.Structure):
        _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]
    ts = Timespec()
    librt.clock_gettime(1, ctypes.byref(ts))  # CLOCK_REALTIME=1
    ns = ts.tv_nsec + timeout_ms * 1_000_000
    ts.tv_sec  += ns // 1_000_000_000
    ts.tv_nsec  = ns %  1_000_000_000
    return librt.sem_timedwait(ctypes.c_void_p(sem), ctypes.byref(ts))

def read_shm(ptr, shm_size):
    raw = (ctypes.c_char * shm_size).from_address(ptr)
    buf = bytes(raw)

    write_seq, num_cameras, _ = struct.unpack_from(SHM_HEADER_FMT, buf, 0)

    results = []
    offset = SHM_HEADER_SIZE
    for _ in range(num_cameras):
        hdr = struct.unpack_from(CAMERA_SLOT_HEADER_FMT, buf, offset)
        cam_id_raw, ts_us, fw, fh, num_dets, _pad = hdr
        cam_id = cam_id_raw.rstrip(b"\x00").decode(errors="replace")

        dets = []
        det_offset = offset + CAMERA_SLOT_HEADER_SIZE
        for di in range(num_dets):
            d = struct.unpack_from(DETECTION_FMT, buf, det_offset)
            x1, y1, x2, y2, cx, det_conf, color_id, color_conf = d[:8]
            color_probs = d[8:13]
            track_id = d[13]
            dets.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": cx, "det_conf": det_conf,
                "color_id": color_id, "color_conf": color_conf,
                "color_probs": color_probs, "track_id": track_id,
            })
            det_offset += DETECTION_SIZE

        if cam_id:
            results.append({"cam_id": cam_id, "ts_us": ts_us, "dets": dets})

        offset += CAMERA_SLOT_SIZE

    return write_seq, results


# ── InfluxDB writer (line protocol over HTTP) ────────────────────────

def write_to_influx(url: str, token: str, org: str, bucket: str, lines: list[str]):
    import urllib.request
    import urllib.error
    body = "\n".join(lines).encode()
    req = urllib.request.Request(
        f"{url}/api/v2/write?org={org}&bucket={bucket}&precision=ms",
        data=body,
        headers={
            "Authorization": f"Token {token}",
            "Content-Type": "text/plain; charset=utf-8",
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 204
    except urllib.error.HTTPError as e:
        log.warning(f"InfluxDB write error: {e.code} {e.read()[:200]}")
        return False
    except Exception as e:
        log.warning(f"InfluxDB write failed: {e}")
        return False


def build_line_protocol(cam_results: list, ts_ms: int) -> list[str]:
    """Build InfluxDB line protocol strings from detection results."""
    lines = []

    total_dets = 0
    color_counts = defaultdict(int)

    for cam in cam_results:
        cam_id = cam["cam_id"]
        dets   = cam["dets"]
        n      = len(dets)
        total_dets += n

        # Per-camera detection count
        lines.append(
            f'detections,camera={cam_id} count={n}i {ts_ms}'
        )

        for d in dets:
            color_id   = d["color_id"]
            color_conf = d["color_conf"]
            det_conf   = d["det_conf"]
            track_id   = d["track_id"]
            cx         = d["cx"]
            bbox_h     = d["y2"] - d["y1"]

            color_name = COLOR_NAMES[color_id] if 0 <= color_id < NUM_COLORS else "unknown"
            color_counts[color_name] += 1

            # Per-track event
            lines.append(
                f'track_events,camera={cam_id},color={color_name},track_id={track_id} '
                f'det_conf={det_conf:.3f},color_conf={color_conf:.3f},'
                f'center_x={cx:.1f},bbox_height={bbox_h:.1f} {ts_ms}'
            )

            # Per-color confidence breakdown (all 5 probs)
            probs = d["color_probs"]
            prob_fields = ",".join(
                f"prob_{COLOR_NAMES[i]}={probs[i]:.4f}" for i in range(NUM_COLORS)
            )
            lines.append(
                f'color_probs,camera={cam_id},track_id={track_id} {prob_fields} {ts_ms}'
            )

    # Pipeline health
    color_fields = ",".join(f"{k}={v}i" for k, v in color_counts.items()) or "none=0i"
    lines.append(
        f'pipeline_health total_detections={total_dets}i,{color_fields} {ts_ms}'
    )

    return lines


# ── Main loop ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DeepStream SHM → InfluxDB writer")
    parser.add_argument("--url",    default="http://localhost:8086")
    parser.add_argument("--token",  default=os.environ.get("INFLUX_TOKEN", ""))
    parser.add_argument("--org",    default="race_vision")
    parser.add_argument("--bucket", default="race_vision")
    parser.add_argument("--interval-ms", type=int, default=200,
                        help="SHM poll interval in ms (default: 200)")
    args = parser.parse_args()

    if not args.token:
        log.error("No InfluxDB token. Set --token or INFLUX_TOKEN env var")
        sys.exit(1)

    log.info(f"Connecting to InfluxDB at {args.url}, org={args.org}, bucket={args.bucket}")

    librt, libc = _load_libs()

    # Wait for SHM to appear (DeepStream may start later)
    log.info(f"Waiting for SHM {SHM_NAME}...")
    while True:
        try:
            ptr, shm_size = attach_shm(librt)
            break
        except OSError:
            time.sleep(1.0)

    sem = open_sem(librt)
    log.info("Attached to SHM and semaphore. Starting write loop...")

    last_seq = None
    write_count = 0
    err_count = 0

    while True:
        ret = sem_timedwait(librt, sem, args.interval_ms)
        if ret != 0:
            continue  # timeout, no new data

        write_seq, cam_results = read_shm(ptr, shm_size)

        if write_seq == last_seq:
            continue
        last_seq = write_seq

        # Only write if there are detections
        ts_ms = int(time.time() * 1000)
        lines = build_line_protocol(cam_results, ts_ms)

        ok = write_to_influx(args.url, args.token, args.org, args.bucket, lines)
        if ok:
            write_count += 1
            if write_count % 100 == 0:
                total = sum(len(c["dets"]) for c in cam_results)
                log.info(f"Written {write_count} batches to InfluxDB | current dets: {total}")
        else:
            err_count += 1
            if err_count % 10 == 0:
                log.warning(f"InfluxDB write errors: {err_count}")


if __name__ == "__main__":
    main()
