#!/usr/bin/env python3
"""run_record.py — replay a recorded multi-camera session through the
DeepStream pipeline, dump every detection to CSV, and (optionally) save
annotated snapshots for manual inspection.

Architecture (mode (a), no api/server.py):
  - subprocess: python -m deepstream.main --cameras <config>
                writes detections to /rv_detections SHM
  - this script: SharedMemoryReader (with seqlock retry) → harvest detections
                + cv2.VideoCapture per camera → annotated JPEG snapshots
                + TimeTracker → pos_along_track_m

Usage:
    python tools/run_record.py \\
        --record records-test/yaris_20260421_141439/ \\
        --config configs/cameras_yaris4.json \\
        --run-id test_run_1 \\
        --output /tmp/yaris_analysis/ \\
        --save-snapshots

Identity assumption: Race Vision runs without an nvtracker by default
(deepstream/main.py: --tracker=""), so SHM track_id is not stable across
frames. We treat (cam_id, color) as the per-camera identity for snapshot
grouping; transitions use color as the cross-camera identity.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.shm_reader import SharedMemoryReader  # noqa: E402
from pipeline.time_tracker import TimeTracker  # noqa: E402

# Color → BGR for bbox drawing
_COLOR_BGR = {
    "blue":   (255, 80, 80),
    "green":  (60, 200, 60),
    "purple": (200, 60, 200),
    "red":    (40, 40, 220),
    "yellow": (40, 220, 220),
    "white":  (240, 240, 240),
    "pink":   (180, 105, 255),
    "black":  (40, 40, 40),
    "unknown": (128, 128, 128),
}
JPEG_QUALITY = 85
MAX_SNAPSHOTS = 2000
LOW_CONF_THRESHOLD = 0.7
LOW_CONF_SAMPLE_RATE = 10  # 1 of N
SHM_WAIT_TIMEOUT_SEC = 30.0
DRAIN_AFTER_EXIT_SEC = 2.0

log = logging.getLogger("run_record")


# ─── Frame store: cv2 captures per camera, lockstep frame advance ──────

class FrameStore:
    """Opens cv2.VideoCapture per file:// camera and seeks to the frame that
    corresponds to a given SHM write_seq. Uses cv2.grab() to skip frames fast
    (no decode) and cv2.retrieve() to decode only when a snapshot is needed.

    Per-camera frame index = cam_det.frame_seq - first_seq_observed, since
    DeepStream commits one batch per N source frames (one frame from each
    camera per batch). If the reader falls behind, we grab() forward to catch
    up before retrieving."""

    def __init__(self, cameras: list[dict]):
        self._caps: dict[str, cv2.VideoCapture] = {}
        self._idx: dict[str, int] = {}  # next frame to be grabbed (0-indexed)
        self._latest: dict[str, "cv2.Mat"] = {}
        self._latest_idx: dict[str, int] = {}  # frame idx the cached frame is at
        self._eof: dict[str, bool] = {}
        for c in cameras:
            cam_id = c["id"]
            url = c["url"]
            parsed = urlparse(url)
            if parsed.scheme != "file":
                log.warning("camera %s: non-file URL (%s) — snapshots disabled", cam_id, url)
                continue
            path = parsed.path
            if not Path(path).is_file():
                log.warning("camera %s: video file missing (%s)", cam_id, path)
                continue
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                log.warning("camera %s: VideoCapture failed to open %s", cam_id, path)
                continue
            self._caps[cam_id] = cap
            self._idx[cam_id] = 0
            self._latest[cam_id] = None
            self._latest_idx[cam_id] = -1
            self._eof[cam_id] = False
        log.info("FrameStore: opened %d / %d captures", len(self._caps), len(cameras))

    def grab_to(self, cam_id: str, target_idx: int) -> bool:
        """Fast-forward (grab() without decode) until next-to-grab is target_idx."""
        cap = self._caps.get(cam_id)
        if cap is None or self._eof.get(cam_id, True):
            return False
        while self._idx[cam_id] < target_idx:
            if not cap.grab():
                self._eof[cam_id] = True
                return False
            self._idx[cam_id] += 1
        return True

    def retrieve(self, cam_id: str) -> Optional["cv2.Mat"]:
        """Decode the frame at the current position (last one grabbed)."""
        cap = self._caps.get(cam_id)
        if cap is None or self._eof.get(cam_id, True):
            return self._latest.get(cam_id)
        target_idx = self._idx[cam_id] - 1
        if self._latest_idx.get(cam_id, -1) == target_idx:
            return self._latest[cam_id]
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            return self._latest.get(cam_id)
        self._latest[cam_id] = frame
        self._latest_idx[cam_id] = target_idx
        return frame

    def current_idx(self, cam_id: str) -> int:
        """Last-grabbed frame index (what retrieve() would return)."""
        return self._idx.get(cam_id, 0) - 1

    def seek_retrieve(self, cam_id: str, frame_id: int) -> Optional["cv2.Mat"]:
        """Random-access decode of frame_id. Slow (seeks to nearest keyframe
        then decodes forward). Use only for finalize/last_seen since the main
        loop uses fast sequential grab()/retrieve()."""
        cap = self._caps.get(cam_id)
        if cap is None or frame_id < 0:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_id))
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        # Invalidate lockstep state — subsequent grab_to would be off.
        # Safe because seek_retrieve is only called after the main loop ends.
        self._idx[cam_id] = frame_id + 1
        self._latest[cam_id] = frame
        self._latest_idx[cam_id] = frame_id
        return frame

    def close(self) -> None:
        for cap in self._caps.values():
            cap.release()


# ─── Snapshot policy ──────────────────────────────────────────────────

@dataclass
class TrackState:
    first_seen_done: bool = False
    last_seen_frame: int = -1
    last_seen_cam: str = ""
    last_seen_det: dict = field(default_factory=dict)
    last_seen_src_w: int = 0
    last_seen_src_h: int = 0
    last_cam_for_color: str = ""  # for transition detection (per-color)
    low_conf_streak: int = 0


class SnapshotPolicy:
    """Decides which (camera, color, frame) tuples deserve a snapshot.

    Identity is keyed by (cam_id, color) since the pipeline runs without an
    nvtracker (track_id is the uint32 sentinel 4294967295 for every detection).
    With color as identity, each unique silk color gets its own first_seen +
    last_seen pair per camera, which matches the per-color racing semantics."""

    def __init__(self):
        self._track_state: dict[tuple[str, str], TrackState] = defaultdict(TrackState)
        self._color_last_cam: dict[str, str] = {}  # color → last camera_id
        self._low_conf_count: dict[tuple[str, str], int] = defaultdict(int)
        self.counts = defaultdict(int)  # event_type → count
        self.total_saved = 0

    def evaluate(self, cam_id: str, det: dict, frame_id: int,
                 src_w: int = 0, src_h: int = 0) -> list[str]:
        """Return list of event types that fire for this detection."""
        events: list[str] = []
        color = det.get("color", "unknown")
        conf = float(det.get("conf", 0.0))

        key = (cam_id, color)
        st = self._track_state[key]

        if not st.first_seen_done:
            events.append("first_seen")
            st.first_seen_done = True

        st.last_seen_frame = frame_id
        st.last_seen_cam = cam_id
        st.last_seen_det = det
        st.last_seen_src_w = src_w
        st.last_seen_src_h = src_h

        # camera transition: color appeared on a different cam than last time
        prev_cam = self._color_last_cam.get(color)
        if prev_cam and prev_cam != cam_id:
            events.append("camera_transition")
        self._color_last_cam[color] = cam_id

        if conf < LOW_CONF_THRESHOLD:
            n = self._low_conf_count[key]
            if n % LOW_CONF_SAMPLE_RATE == 0:
                events.append("low_confidence")
            self._low_conf_count[key] = n + 1
        else:
            self._low_conf_count[key] = 0

        return events

    def finalize_last_seen(self) -> list[tuple[str, str, dict, int, int, int]]:
        """At end of run: yield (cam_id, color, det, frame_id, src_w, src_h)."""
        out = []
        for (cam_id, color), st in self._track_state.items():
            if st.last_seen_det:
                out.append((cam_id, color, st.last_seen_det,
                            st.last_seen_frame, st.last_seen_src_w, st.last_seen_src_h))
        return out


# ─── Annotation ───────────────────────────────────────────────────────

def annotate_frame(frame, dets: list[dict], cam_id: str, frame_id: int,
                   ts: float, torn_retries: Optional[int] = None,
                   pos_lookup: Optional[dict] = None,
                   src_w: int = 0, src_h: int = 0) -> "cv2.Mat":
    """Draw bboxes + metadata. Returns a copy with overlays.

    src_w/src_h are the frame dims the bboxes were computed against
    (CameraDetections.frame_width/height). If they differ from the cv2
    frame size, bboxes are scaled. 0 disables scaling."""
    img = frame.copy()
    h, w = img.shape[:2]
    sx = (w / src_w) if src_w > 0 else 1.0
    sy = (h / src_h) if src_h > 0 else 1.0

    for det in dets:
        x1, y1, x2, y2 = det["bbox"]
        x1 = max(0, min(w - 1, int(x1 * sx)))
        y1 = max(0, min(h - 1, int(y1 * sy)))
        x2 = max(0, min(w - 1, int(x2 * sx)))
        y2 = max(0, min(h - 1, int(y2 * sy)))
        if x2 <= x1 or y2 <= y1:
            continue
        color = det.get("color", "unknown")
        bgr = _COLOR_BGR.get(color, _COLOR_BGR["unknown"])
        cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)

        track_id = int(det.get("track_id", 0))
        conf = float(det.get("conf", 0.0))
        label = f"ID:{track_id} | {color} ({conf:.2f})"
        if pos_lookup and color in pos_lookup:
            label += f" | pos:{pos_lookup[color]:.0f}m"
        cv2.putText(img, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)

    header = f"{cam_id} | frame:{frame_id} | t:{ts:.3f}"
    cv2.putText(img, header, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, header, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 1, cv2.LINE_AA)

    if torn_retries is not None:
        warn = f"TORN_READ retry={torn_retries}"
        cv2.putText(img, warn, (w - 280, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return img


# ─── Output management ────────────────────────────────────────────────

class OutputManager:
    EVENT_DIRS = ["track_events", "camera_transitions",
                  "low_confidence", "torn_read_events"]

    def __init__(self, run_dir: Path, save_snapshots: bool):
        self.run_dir = run_dir
        self.save_snapshots = save_snapshots
        run_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = run_dir / "detections.csv"
        self._csv_fh = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._csv_fh)
        self._csv.writerow([
            "timestamp", "camera_id", "frame_id", "track_id",
            "bbox_x", "bbox_y", "bbox_w", "bbox_h",
            "color_class", "color_confidence", "pos_along_track_m",
        ])

        self.snap_root = run_dir / "snapshots"
        self.snap_index_path: Optional[Path] = None
        self._snap_idx_fh = None
        self._snap_idx = None
        if save_snapshots:
            for d in self.EVENT_DIRS:
                (self.snap_root / d).mkdir(parents=True, exist_ok=True)
            self.snap_index_path = self.snap_root / "index.csv"
            self._snap_idx_fh = open(self.snap_index_path, "w", newline="", encoding="utf-8")
            self._snap_idx = csv.writer(self._snap_idx_fh)
            self._snap_idx.writerow([
                "snapshot_path", "event_type", "camera_id", "frame_id",
                "track_id", "color_class", "color_confidence",
                "pos_along_track_m", "timestamp", "notes",
            ])

    def write_detection(self, ts: float, cam_id: str, frame_id: int, det: dict,
                        pos_m: Optional[float]) -> None:
        x1, y1, x2, y2 = det["bbox"]
        self._csv.writerow([
            f"{ts:.6f}", cam_id, frame_id, int(det.get("track_id", 0)),
            int(x1), int(y1), int(x2 - x1), int(y2 - y1),
            det.get("color", "unknown"), f"{float(det.get('conf', 0)):.4f}",
            f"{pos_m:.2f}" if pos_m is not None else "",
        ])

    def save_snapshot(self, frame, event_type: str, cam_id: str, frame_id: int,
                      track_id: int, det: dict, pos_m: Optional[float],
                      ts: float, notes: str = "") -> Optional[str]:
        if not self.save_snapshots or self._snap_idx is None:
            return None
        color = det.get("color", "unknown")
        suffix = (notes or event_type).replace(" ", "_").replace("/", "_")
        slug = f"{cam_id}_{frame_id:06d}_{color}_{suffix}.jpg"
        out_path = self.snap_root / event_type / slug
        ok = cv2.imwrite(str(out_path), frame,
                         [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            log.warning("imwrite failed for %s", out_path)
            return None
        rel = out_path.relative_to(self.run_dir).as_posix()
        self._snap_idx.writerow([
            rel, event_type, cam_id, frame_id, track_id,
            det.get("color", "unknown"), f"{float(det.get('conf', 0)):.4f}",
            f"{pos_m:.2f}" if pos_m is not None else "",
            f"{ts:.6f}", notes,
        ])
        return rel

    def close(self) -> None:
        self._csv_fh.close()
        if self._snap_idx_fh:
            self._snap_idx_fh.close()


# ─── Pipeline subprocess ──────────────────────────────────────────────

def spawn_pipeline(camera_config: Path, run_dir: Path) -> subprocess.Popen:
    ds_log_dir = run_dir / "ds_log"
    ds_log_dir.mkdir(exist_ok=True)
    cmd = [
        sys.executable, "-m", "deepstream.main",
        "--cameras", str(camera_config),
        "--log-dir", str(ds_log_dir),
        "--log-level", "INFO",
    ]
    log.info("spawning pipeline: %s", " ".join(cmd))
    return subprocess.Popen(cmd, cwd=str(ROOT),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)


def wait_for_shm(reader: SharedMemoryReader, timeout_sec: float) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if reader.attach():
            return True
        time.sleep(0.5)
    return False


# ─── Main loop ────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> int:
    record_path = Path(args.record)
    config_path = Path(args.config)
    if not config_path.is_file():
        log.error("config not found: %s", config_path)
        return 2
    if not record_path.is_dir():
        log.error("record dir not found: %s", record_path)
        return 2

    cfg = json.loads(config_path.read_text())
    cameras = cfg.get("analytics") or cfg.get("cameras") or []
    if not cameras:
        log.error("config %s has no analytics/cameras list", config_path)
        return 2

    run_dir = Path(args.output) / args.run_id
    if run_dir.exists():
        log.warning("output dir %s exists — removing", run_dir)
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    log.info("run output dir: %s", run_dir)

    cam_order = [c["id"] for c in cameras]
    cam_pos_m: dict[str, float] = {}
    for c in cameras:
        if "track_start" in c and "track_end" in c:
            cam_pos_m[c["id"]] = 0.5 * (float(c["track_start"]) + float(c["track_end"]))
    if cam_pos_m:
        log.info("camera positions (m): %s",
                 ", ".join(f"{cid}={p:.0f}" for cid, p in cam_pos_m.items()))
    else:
        log.warning("no track_start/track_end in config — pos_along_track_m will be empty")

    out = OutputManager(run_dir, args.save_snapshots)
    policy = SnapshotPolicy()
    fstore = FrameStore(cameras) if args.save_snapshots else None
    tracker = TimeTracker(cam_order=cam_order)

    proc = spawn_pipeline(config_path, run_dir)
    pipe_log_path = run_dir / "ds_log" / "subprocess.log"
    pipe_log_fh = open(pipe_log_path, "w", encoding="utf-8")

    def _drain_stdout():
        try:
            for line in proc.stdout:
                pipe_log_fh.write(line)
                pipe_log_fh.flush()
        except Exception:
            pass

    drain_thread = threading.Thread(target=_drain_stdout, daemon=True)
    drain_thread.start()

    reader = SharedMemoryReader(timeout_ms=20)
    log.info("waiting for SHM to appear (up to %.0fs)...", SHM_WAIT_TIMEOUT_SEC)
    if not wait_for_shm(reader, SHM_WAIT_TIMEOUT_SEC):
        log.error("SHM did not appear; subprocess may have crashed")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 3
    log.info("attached to SHM, harvesting...")

    detections_total = 0
    snapshots_saved = 0
    first_seen_pending: list[tuple[str, dict, int, float]] = []
    snapshot_capped = False

    last_snapshot_warn_ts = 0.0
    drain_started = None
    started_at = time.time()

    try:
        while True:
            batch = reader.read()
            if batch is None:
                if proc.poll() is not None:
                    if drain_started is None:
                        log.info("subprocess exited (rc=%s); draining SHM for %.1fs",
                                 proc.returncode, DRAIN_AFTER_EXIT_SEC)
                        drain_started = time.time()
                    elif time.time() - drain_started > DRAIN_AFTER_EXIT_SEC:
                        break
                continue

            stats = reader.get_stats()
            exhausted_now = stats["retry_histogram"].get("exhausted", 0)

            for cam_det in batch:
                cam_id = cam_det.cam_id
                # source_frame_num is the per-source decoder frame index written
                # by the pipeline probe from frame_meta.frame_number — exact
                # alignment with the MP4 frames cv2 reads.
                target_idx = int(getattr(cam_det, "source_frame_num", 0))
                if fstore is not None:
                    fstore.grab_to(cam_id, target_idx + 1)
                frame_id = target_idx
                ts = float(cam_det.timestamp)
                # pos_along_track for this batch = midpoint of the camera the
                # detection is observed on (track_start..track_end from config)
                pos_m = cam_pos_m.get(cam_id) if cam_pos_m else None
                # pos_lookup keyed by color, used by annotate_frame label
                pos_lookup = {d.get("color", "unknown"): pos_m
                              for d in cam_det.detections} if pos_m is not None else {}

                for det in cam_det.detections:
                    detections_total += 1
                    color = det.get("color", "unknown")
                    tracker.ingest(ts, cam_id, color)
                    out.write_detection(ts, cam_id, frame_id, det, pos_m)

                    if not args.save_snapshots:
                        continue

                    events = policy.evaluate(cam_id, det, frame_id,
                                             src_w=cam_det.frame_width,
                                             src_h=cam_det.frame_height)

                    if not events:
                        continue

                    frame = fstore.retrieve(cam_id)
                    if frame is None:
                        continue

                    track_id = int(det.get("track_id", 0))
                    for ev in events:
                        if snapshots_saved >= MAX_SNAPSHOTS:
                            if not snapshot_capped:
                                log.warning("snapshot cap %d reached — stopping snapshots",
                                            MAX_SNAPSHOTS)
                                snapshot_capped = True
                            break
                        target_ev = ev
                        if ev == "first_seen":
                            target_ev = "track_events"
                        elif ev == "camera_transition":
                            target_ev = "camera_transitions"
                        elif ev == "low_confidence":
                            target_ev = "low_confidence"

                        ann = annotate_frame(frame, [det], cam_id, frame_id, ts,
                                             pos_lookup=pos_lookup,
                                             src_w=cam_det.frame_width,
                                             src_h=cam_det.frame_height)
                        notes = ev
                        out.save_snapshot(ann, target_ev, cam_id, frame_id,
                                          track_id, det, pos_m, ts, notes=notes)
                        snapshots_saved += 1
                        policy.counts[target_ev] += 1

            if args.save_snapshots and not snapshot_capped \
                    and exhausted_now > policy.counts.get("torn_read_events", 0):
                for cam_det in batch:
                    if not cam_det.detections or fstore is None:
                        continue
                    frame = fstore.retrieve(cam_det.cam_id)
                    frame_id = fstore.current_idx(cam_det.cam_id)
                    if frame is None:
                        continue
                    ann = annotate_frame(frame, cam_det.detections, cam_det.cam_id,
                                         frame_id, cam_det.timestamp,
                                         torn_retries=exhausted_now,
                                         pos_lookup=pos_lookup,
                                         src_w=cam_det.frame_width,
                                         src_h=cam_det.frame_height)
                    out.save_snapshot(ann, "torn_read_events", cam_det.cam_id,
                                      frame_id, 0, cam_det.detections[0],
                                      None, cam_det.timestamp,
                                      notes=f"exhausted_total={exhausted_now}")
                    snapshots_saved += 1
                    policy.counts["torn_read_events"] += 1
                    if snapshots_saved >= MAX_SNAPSHOTS:
                        break

            if time.time() - last_snapshot_warn_ts > 10.0:
                last_snapshot_warn_ts = time.time()
                log.info("running… dets=%d snaps=%d reads=%d torn=%s",
                         detections_total, snapshots_saved, stats["reads_total"],
                         stats["retry_histogram"].get("exhausted", 0))

    except KeyboardInterrupt:
        log.warning("interrupted; terminating subprocess")
        proc.terminate()
    finally:
        # finalize last_seen snapshots
        if args.save_snapshots and not snapshot_capped and fstore is not None:
            for cam_id, color, det, frame_id, src_w, src_h in policy.finalize_last_seen():
                if snapshots_saved >= MAX_SNAPSHOTS:
                    break
                # Seek back to the exact frame the detection was computed on —
                # the sequential grab/retrieve loop has moved past it by now.
                frame = fstore.seek_retrieve(cam_id, frame_id)
                if frame is None:
                    continue
                ann = annotate_frame(frame, [det], cam_id, frame_id,
                                     time.time(), src_w=src_w, src_h=src_h)
                out.save_snapshot(ann, "track_events", cam_id, frame_id,
                                  int(det.get("track_id", 0)), det, None,
                                  time.time(), notes="last_seen")
                snapshots_saved += 1
                policy.counts["track_events"] += 1

        if proc.poll() is None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=5)
        pipe_log_fh.close()
        out.close()
        if fstore is not None:
            fstore.close()
        reader.detach()

    elapsed = time.time() - started_at
    stats = reader.get_stats()
    print()
    print("=" * 60)
    print(f"Run finished in {elapsed:.1f}s")
    print(f"Output dir:    {run_dir}")
    print(f"CSV:           {out.csv_path}")
    if args.save_snapshots:
        print(f"Snapshots:     {out.snap_root}")
        print(f"Index CSV:     {out.snap_index_path}")
    print()
    print(f"Subprocess rc: {proc.returncode}")
    print(f"Detections:    {detections_total}")
    print(f"SHM reads:     {stats['reads_total']}")
    print(f"SHM last_seq:  {stats['last_seq']} (commits writer made)")
    hist = stats["retry_histogram"]
    print(f"Torn-read retry histogram: {dict(sorted(hist.items()))}")
    if args.save_snapshots:
        print(f"Snapshots saved by category:")
        for k in OutputManager.EVENT_DIRS:
            print(f"  {k}: {policy.counts.get(k, 0)}")
        print(f"  TOTAL: {snapshots_saved}{' (CAPPED)' if snapshot_capped else ''}")
    print("=" * 60)
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", required=True, help="path to record dir (only used for sanity check)")
    ap.add_argument("--config", required=True, help="camera config JSON (with file:// URLs)")
    ap.add_argument("--run-id", required=True, help="run identifier (subdir name)")
    ap.add_argument("--output", required=True, help="parent output directory")
    ap.add_argument("--save-snapshots", action="store_true",
                    help="annotate and save JPEG snapshots for events")
    return ap.parse_args()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
