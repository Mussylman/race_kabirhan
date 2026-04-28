"""Snapshot saver — debug JPEG dumper triggered by classified-jockey events.

FrameSaver runs a background thread; the probe just calls maybe_trigger()
on each frame and the worker pulls a frame from go2rtc and writes it to
the experiment directory. _make_exp_dir() creates a fresh exp_NNN dir
under runs/.

Extracted from deepstream/pipeline.py during the refactor.
"""
from __future__ import annotations

import os
import queue
import threading
import time
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ── FrameSaver: snapshot когда на камере ≥N классифицированных жокеев ──
_SNAP_MIN_COUNT  = int(os.environ.get("RV_SNAP_MIN", "3"))
_SNAP_RATE_SEC   = float(os.environ.get("RV_SNAP_RATE_SEC", "2.0"))
_SNAP_GO2RTC_API = os.environ.get("RV_GO2RTC_API", "http://localhost:1984")
_SNAP_INVERT     = os.environ.get("RV_INVERT_TRACK", "1") == "1"


def _make_exp_dir() -> Path:
    """Create runs/exp_NNN_YYYYMMDD_HHMMSS/ with next sequential N."""
    base = REPO_ROOT / "runs"
    base.mkdir(exist_ok=True)
    nums = []
    for p in base.iterdir():
        if p.is_dir() and p.name.startswith("exp_"):
            parts = p.name.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                nums.append(int(parts[1]))
    next_num = (max(nums) + 1) if nums else 1
    ts = time.strftime("%Y%m%d_%H%M%S")
    d = base / f"exp_{next_num:03d}_{ts}"
    d.mkdir()
    return d


class FrameSaver:
    """Async snapshot saver. Non-blocking for the probe thread: puts jobs
    on a bounded queue; a worker thread fetches JPEG from go2rtc and writes
    to disk. Rate-limited per camera."""

    def __init__(self, exp_dir: Path, min_count: int, rate_sec: float):
        self.exp_dir   = exp_dir
        self.min_count = min_count
        self.rate_sec  = rate_sec
        self.last_save = {}   # cam_id -> monotonic ts
        self.q         = queue.Queue(maxsize=50)
        self.stop      = threading.Event()
        self.thread    = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        print(f"[saver] experiment dir: {exp_dir}  trigger ≥{min_count} jockeys, rate {rate_sec}s/cam",
              flush=True)

    def maybe_trigger(self, cam_id: str, classified: list[tuple[float, str]]):
        """classified = [(x1, color_name), ...] — only active-color dets."""
        if len(classified) < self.min_count:
            return
        now = time.monotonic()
        if now - self.last_save.get(cam_id, 0) < self.rate_sec:
            return
        self.last_save[cam_id] = now
        try:
            self.q.put_nowait((cam_id, list(classified), time.time()))
        except queue.Full:
            pass  # drop: under overload we'd rather skip than block

    def _worker(self):
        while not self.stop.is_set():
            try:
                cam_id, dets, ts = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                url = f"{_SNAP_GO2RTC_API}/api/frame.jpeg?src={cam_id}"
                with urllib.request.urlopen(url, timeout=3) as r:
                    jpg = r.read()
                # ranking within this frame: leader = leftmost if inverted
                ordered = sorted(dets, key=lambda d: d[0], reverse=not _SNAP_INVERT)
                seq = "-".join(f"{i+1}{c[:3]}" for i, (_, c) in enumerate(ordered))
                tstr = time.strftime("%H%M%S", time.localtime(ts))
                fname = f"{cam_id}_{tstr}_{len(ordered)}j_{seq}.jpg"
                (self.exp_dir / fname).write_bytes(jpg)
            except Exception as e:
                # silent-ish: one line per failure, no stack spam
                print(f"[saver] {cam_id}: {e}", flush=True)

    def close(self):
        self.stop.set()
