"""Audit logger — thread-safe append-only JSONL sinks + async annotated
snapshot saver. Activated via RV_AUDIT=1.

Files under ${RV_AUDIT_DIR}/ (default output/audit/):
  detections.jsonl    — every person detection from DetectionProbe
  tracker_events.jsonl — every TimeTracker.ingest decision
  cam-XX/frame_*.jpg   — annotated snapshots (rate-limited)
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


class AuditLogger:
    """Singleton-ish sink. Safe to construct with RV_AUDIT off (becomes no-op)."""

    _instance: "AuditLogger | None" = None

    def __init__(self, enabled: bool, audit_dir: Path | None = None,
                 snap_rate_sec: float = 1.0,
                 go2rtc_api: str = "http://localhost:1984",
                 roi_polygons: dict | None = None):
        self.enabled = enabled
        if not enabled:
            return

        self.audit_dir = audit_dir or Path("output/audit")
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.snap_rate_sec = snap_rate_sec
        self.go2rtc_api = go2rtc_api
        self.roi_polygons = roi_polygons or {}

        self._det_fp = open(self.audit_dir / "detections.jsonl", "a",
                            buffering=1)
        self._trk_fp = open(self.audit_dir / "tracker_events.jsonl", "a",
                            buffering=1)
        self._log_lock = threading.Lock()

        # Snapshot worker
        self._snap_q: queue.Queue = queue.Queue(maxsize=200)
        self._snap_last_ts: dict[str, float] = {}
        self._stop = threading.Event()
        self._snap_thread = threading.Thread(
            target=self._snap_worker, daemon=True, name="AuditSnapWorker"
        )
        self._snap_thread.start()

        # Touch run marker
        (self.audit_dir / "RUN_START.txt").write_text(
            f"started={time.time()}\n"
            f"snap_rate={snap_rate_sec}\n"
            f"pid={os.getpid()}\n"
        )

    @classmethod
    def get(cls) -> "AuditLogger":
        if cls._instance is None:
            enabled = os.environ.get("RV_AUDIT", "0") == "1"
            audit_dir = Path(os.environ.get("RV_AUDIT_DIR", "output/audit"))
            snap_rate = float(os.environ.get("RV_AUDIT_SNAP_RATE_SEC", "1.0"))
            go2rtc = os.environ.get("RV_GO2RTC_API", "http://localhost:1984")
            cls._instance = cls(enabled, audit_dir, snap_rate, go2rtc)
        return cls._instance

    def set_roi_polygons(self, polys: dict):
        if self.enabled:
            self.roi_polygons = polys

    # ------------------------------------------------------------------
    # DETECTION LOG — one line per person detection from DetectionProbe
    # ------------------------------------------------------------------
    def log_detection(self, *, cam_id: str, ts: float,
                      frame_idx: int,
                      bbox: tuple[float, float, float, float],
                      frame_w: int, frame_h: int,
                      color: str, color_conf: float,
                      det_conf: float,
                      passed_filters: bool,
                      inside_roi: bool,
                      written_to_shm: bool,
                      track_id: int = 0,
                      reject_reason: str = ""):
        if not self.enabled:
            return
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        rec = {
            "ts": ts,
            "cam": cam_id,
            "frame_idx": frame_idx,
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "center_px": [round(cx, 1), round(cy, 1)],
            "center_norm": [round(cx / max(1, frame_w), 3),
                            round(cy / max(1, frame_h), 3)],
            "frame_wh": [frame_w, frame_h],
            "color": color,
            "color_conf": round(color_conf, 3),
            "det_conf": round(det_conf, 3),
            "passed_filters": passed_filters,
            "inside_roi": inside_roi,
            "written_to_shm": written_to_shm,
            "track_id": int(track_id),
            "reject_reason": reject_reason,
        }
        with self._log_lock:
            self._det_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Maybe queue a snapshot (rate-limited per camera).
        if color and color != "?" and inside_roi and passed_filters:
            self._maybe_enqueue_snap(cam_id, ts, bbox, frame_w, frame_h, color,
                                    color_conf, track_id)

    # ------------------------------------------------------------------
    # TRACKER LOG — one line per TimeTracker.ingest decision
    # ------------------------------------------------------------------
    def log_tracker_event(self, *, ts: float, cam_id: str, cam_idx: int | None,
                          color: str, accepted: bool, reason: str,
                          bbox_x: float | None = None,
                          state_before_cam_idx: int | None = None):
        if not self.enabled:
            return
        rec = {
            "ts": ts,
            "cam": cam_id,
            "cam_idx": cam_idx,
            "color": color,
            "accepted": accepted,
            "reason": reason,
            "bbox_x": bbox_x,
            "state_before_cam_idx": state_before_cam_idx,
        }
        with self._log_lock:
            self._trk_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # SNAPSHOT QUEUE
    # ------------------------------------------------------------------
    def _maybe_enqueue_snap(self, cam_id: str, ts: float, bbox, frame_w,
                            frame_h, color, color_conf, track_id):
        now = time.monotonic()
        last = self._snap_last_ts.get(cam_id, 0.0)
        if now - last < self.snap_rate_sec:
            return
        self._snap_last_ts[cam_id] = now
        job = {
            "cam_id": cam_id, "ts": ts, "bbox": bbox,
            "frame_w": frame_w, "frame_h": frame_h,
            "color": color, "color_conf": color_conf,
            "track_id": track_id,
        }
        try:
            self._snap_q.put_nowait(job)
        except queue.Full:
            pass  # drop under overload

    def _snap_worker(self):
        while not self._stop.is_set():
            try:
                job = self._snap_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process_snap(job)
            except Exception as e:
                # non-fatal — audit must not crash pipeline
                with self._log_lock:
                    self._trk_fp.write(json.dumps({
                        "ts": time.time(), "snap_err": str(e)[:200],
                        "cam": job.get("cam_id"),
                    }) + "\n")

    def _process_snap(self, job):
        cam_id = job["cam_id"]
        url = f"{self.go2rtc_api}/api/frame.jpeg?src={cam_id}"
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                jpg = r.read()
            if len(jpg) < 200:
                return  # empty/error response
        except Exception:
            return

        cam_dir = self.audit_dir / cam_id
        cam_dir.mkdir(parents=True, exist_ok=True)
        fname_raw = cam_dir / f"frame_{job['ts']:.3f}_{job['color']}_raw.jpg"

        if not _HAS_PIL:
            fname_raw.write_bytes(jpg)
            return

        try:
            from io import BytesIO
            img = Image.open(BytesIO(jpg)).convert("RGB")
            draw = ImageDraw.Draw(img)
            W, H = img.size

            # scale bbox to snapshot size (detections were in mux coords)
            sx = W / max(1, job["frame_w"])
            sy = H / max(1, job["frame_h"])
            x1, y1, x2, y2 = [c * (sx if i % 2 == 0 else sy)
                              for i, c in enumerate(job["bbox"])]

            # bbox
            col_rgb = _color_rgb(job["color"])
            draw.rectangle([x1, y1, x2, y2], outline=col_rgb, width=3)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=col_rgb)

            # ROI polygon (in normalized coords — rescale to snapshot)
            polys = self.roi_polygons.get(cam_id, [])
            for poly in polys:
                pts = [(p[0] * W, p[1] * H) for p in poly]
                if len(pts) >= 3:
                    draw.line(pts + [pts[0]], fill=(0, 255, 0), width=2)

            # Label text
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            lines = [
                f"cam: {cam_id}",
                f"t: {job['ts']:.2f}",
                f"color: {job['color']} ({job['color_conf']:.2f})",
                f"bbox: {int(x1)},{int(y1)},{int(x2)},{int(y2)}",
                f"track_id: {job['track_id']}",
            ]
            y = 6
            for line in lines:
                draw.rectangle([6, y, 260, y + 14], fill=(0, 0, 0, 180))
                draw.text((10, y), line, fill=(255, 255, 255), font=font)
                y += 16

            out = cam_dir / f"frame_{job['ts']:.3f}_{job['color']}.jpg"
            img.save(out, "JPEG", quality=80)
        except Exception:
            fname_raw.write_bytes(jpg)  # fallback: raw

    def close(self):
        if not self.enabled:
            return
        self._stop.set()
        try:
            self._det_fp.close()
            self._trk_fp.close()
        except Exception:
            pass


def _color_rgb(name: str) -> tuple[int, int, int]:
    return {
        "blue":   (30, 100, 255),
        "green":  (50, 220, 50),
        "red":    (255, 40, 40),
        "yellow": (255, 220, 30),
        "purple": (180, 50, 200),
    }.get(name, (200, 200, 200))
