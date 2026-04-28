"""
deepstream/pipeline.py — Phase 5: multi-camera Race Vision pipeline.

Replaces ~900 lines of C++ (pipeline.cpp + analysis_pipeline.cpp +
trigger_pipeline.cpp + dual_pipeline.cpp + main.cpp) with one Python module.

Topology per pipeline:
    [src_i = uridecodebin]_{i=0..N-1}
            \\
             nvstreammux (batch=N, 800x800)
                \\
                 nvinfer (libnvdsinfer_racevision.so, YOLO person)
                    \\
                     probe → filter + (later) color + SHM
                        \\
                         fakesink

Step 5.1 deliverable: pipeline runs N cameras end-to-end and the probe
counts detections per camera. Color classification + SHM writing land in
later steps (5.2 = SHM, 5.3 = color).
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pyservicemaker import BatchMetadataOperator, Pipeline, Probe, osd

# Local plugin binding
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepstream.rv_plugin import (
    RVPlugin,
    COLOR_UNKNOWN,
    MAX_DETECTIONS,
    make_camera_slot,
    make_detection,
)

REPO_ROOT        = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG   = REPO_ROOT / "deepstream" / "configs" / "nvinfer_racevision.txt"
DEFAULT_SGIE     = REPO_ROOT / "deepstream" / "configs" / "sgie_color.txt"
DEFAULT_CAMERAS  = REPO_ROOT / "configs"   / "cameras_test_files.json"

PERSON_CLASS_ID = 0

# Tight crop preprocessing for DINOv2 SGIE — отрезаем голову/ноги/лошадь/
# соседей, оставляем только торс с силком. Применяется через bbox-rewrite
# в pre-SGIE probe; post-SGIE probe восстанавливает original bbox через
# обратную математику (no shared state, no race conditions).
#  X: 10-70% (центр сдвинут влево, отрезает соседей)
#  Y: 0-55% (от верха до середины, отрезает ноги/лошадь)
TIGHT_X_LO, TIGHT_X_HI = 0.10, 0.70   # → ширина = 0.60 от original
TIGHT_Y_LO, TIGHT_Y_HI = 0.00, 0.55   # → высота = 0.55 от original
_TIGHT_W_FRAC = TIGHT_X_HI - TIGHT_X_LO   # 0.60
_TIGHT_H_FRAC = TIGHT_Y_HI - TIGHT_Y_LO   # 0.55

# Tight bbox preprocessing was added for DINOv2 (which needed torso-only crops
# to avoid head/legs/horse contamination of the embedding). OSNet was trained
# on full-body person crops at 256x128 stretch and works on full bbox directly.
# Disable via RV_TIGHT_SGIE=0 when running OSNet SGIE; default 1 keeps the
# behaviour for DINOv2 fallback. Gates BOTH the pre-SGIE shrink probe AND the
# post-SGIE reverse-math in DetectionProbe — they must move together.
TIGHT_SGIE_ENABLED = os.environ.get("RV_TIGHT_SGIE", "1") == "1"


class TightBboxShrinkProbe(BatchMetadataOperator):
    """Pre-SGIE probe: shrink rect_params per-obj to torso-only zone.
    SGIE (process-mode=2) crop'ит NvBufSurface по obj.rect_params, поэтому
    модификация bbox здесь = SGIE подаст в TRT engine только torso region.
    DetectionProbe потом восстанавливает original bbox через reverse math."""

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            for obj in frame_meta.object_items:
                if obj.class_id != PERSON_CLASS_ID:
                    continue
                rp = obj.rect_params
                orig_w = rp.width
                orig_h = rp.height
                rp.left   = rp.left + TIGHT_X_LO * orig_w
                rp.top    = rp.top  + TIGHT_Y_LO * orig_h
                rp.width  = _TIGHT_W_FRAC * orig_w
                rp.height = _TIGHT_H_FRAC * orig_h

# Color label → SHM color id (matches deepstream/src/config.h ColorId enum
# and deepstream/configs/labels_color.txt order).
_COLOR_NAME_TO_ID = {
    "blue":   0,
    "green":  1,
    "purple": 2,
    "red":    3,
    "yellow": 4,
}
_COLOR_ID_TO_NAME = {v: k for k, v in _COLOR_NAME_TO_ID.items()}

# Race-specific: today only these colors exist on the track; everything else
# is a spectator / false positive. Override with env RV_ACTIVE_COLORS.
_ACTIVE_COLORS = set(
    os.environ.get("RV_ACTIVE_COLORS", "blue,green,red,yellow").split(",")
)
_ACTIVE_IDS = {cid for name, cid in _COLOR_NAME_TO_ID.items() if name in _ACTIVE_COLORS}


def _load_roi_polygons(path: Path) -> dict[str, list[list[tuple[float, float]]]]:
    """Load normalized ROI polygons from JSON. Returns
    {cam_id: [polygon1, polygon2, ...]} where each polygon is a list of
    (x_norm, y_norm) pairs in [0, 1]."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    out: dict[str, list[list[tuple[float, float]]]] = {}
    for cam_id, polys in data.items():
        cam_polys = []
        for p in polys:
            pts = [(float(pt["x"]), float(pt["y"])) for pt in p]
            if len(pts) >= 3:
                cam_polys.append(pts)
        if cam_polys:
            out[cam_id] = cam_polys
    return out


def _point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside

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


# RGB (0-1) for OSD border_color — bright so it shows well on video
_COLOR_RGB_NORMALISED = {
    "blue":   (0.1, 0.4, 1.0),
    "green":  (0.2, 0.9, 0.2),
    "purple": (0.8, 0.2, 0.8),
    "red":    (1.0, 0.1, 0.1),
    "yellow": (1.0, 0.9, 0.1),
    "?":      (0.6, 0.6, 0.6),
}


@dataclass
class CameraEntry:
    cam_id: str
    uri: str
    track_start: float = 0.0
    track_end: float   = 100.0


def load_cameras(config_path: Path, limit: Optional[int] = None) -> list[CameraEntry]:
    with open(config_path) as fp:
        cfg = json.load(fp)
    raw = cfg.get("analytics", cfg if isinstance(cfg, list) else [])
    cams: list[CameraEntry] = []
    for entry in raw:
        url = entry["url"]
        # Accept both file:// URIs and bare paths
        if not (url.startswith("rtsp://") or url.startswith("file://") or url.startswith("http://")):
            url = f"file://{os.path.abspath(url)}"
        cams.append(CameraEntry(
            cam_id=entry["id"],
            uri=url,
            track_start=float(entry.get("track_start", 0)),
            track_end=float(entry.get("track_end", 100)),
        ))
    if limit is not None:
        cams = cams[:limit]
    return cams


class DetectionProbe(BatchMetadataOperator):
    """Per-frame probe: filter detections, build CameraSlots, write to SHM.

    Color classification is not yet wired in (Step 5.3) — color_id is set
    to UNKNOWN. Once we extract crops from the GPU buffer, this is the
    place to call rv_color_classify.
    """

    # Detection filters (env-tunable for distant jockeys on substream).
    MIN_BBOX_HEIGHT  = int(os.environ.get("RV_MIN_BBOX_H", "20"))
    MIN_ASPECT_RATIO = float(os.environ.get("RV_MIN_AR", "0.2"))
    EDGE_MARGIN      = int(os.environ.get("RV_EDGE_MARGIN", "5"))

    # Temporal stability: color must be seen N consecutive times on same
    # track_id before ingesting to TimeTracker. Filters 1-off false classifications.
    MIN_CONSEC = int(os.environ.get("RV_MIN_CONSEC", "3"))

    def __init__(self, plugin: RVPlugin, shm_handle, cam_ids: list[str],
                 mux_width: int, mux_height: int, log_every: int = 50,
                 cam_uris: list[str] | None = None,
                 sgie_active: bool = False):
        super().__init__()
        self.plugin       = plugin
        self.shm_handle   = shm_handle
        self.cam_ids      = cam_ids
        self.cam_uris: list[str] = list(cam_uris or [""] * len(cam_ids))
        # When True: pre-SGIE probe shrinks bbox per-obj, this probe restores
        # it via reverse math at start of _handle_metadata_impl. When False
        # (no SGIE), bbox is untouched both ways.
        self.sgie_active  = sgie_active
        # Stability state: (cam_id, track_id) -> {color, count}
        self._stability: dict[tuple[str, int], dict] = {}
        # Extract last IP octet per cam for OSD display
        import re as _re
        self.cam_ips: list[str] = []
        for u in self.cam_uris:
            m = _re.search(r'@10\.223\.70\.(\d+)', u or "")
            self.cam_ips.append(m.group(1) if m else "")
        self.mux_width    = mux_width
        self.mux_height   = mux_height
        self.log_every    = log_every
        self.frame_counts = [0] * len(cam_ids)
        self.det_counts   = [0] * len(cam_ids)
        self.total_frames = 0
        self.debug_mode   = os.environ.get("RV_DEBUG_PROBE", "0") == "1"
        self.debug_budget = 60  # detailed lines for the first N detections, then quiet
        self.debug_frames = 0   # detailed header lines for the first N frames

        # ROI: per-camera polygons in NORMALIZED coords. Detections whose
        # bbox centre falls outside all polygons are dropped (treated as
        # not-a-jockey, e.g. spectator in stands).
        roi_path = Path(
            os.environ.get("RV_ROI_FILE",
                           str(REPO_ROOT / "configs" / "camera_roi_normalized.json"))
        )
        self.roi_polygons = _load_roi_polygons(roi_path)
        if self.roi_polygons:
            print(f"[probe] ROI loaded for {len(self.roi_polygons)} cameras from {roi_path.name}",
                  flush=True)
        # CSV dump of every classified detection — for offline analysis
        self.csv_path     = os.environ.get("RV_DUMP_CSV") or None
        self.csv_fp       = None
        if self.csv_path:
            self.csv_fp = open(self.csv_path, "w", buffering=1)
            self.csv_fp.write("frame,cam_id,x1,y1,w,h,yolo_conf,color,color_conf\n")

        # Frame saver (async snapshots when ≥N classified jockeys on a frame).
        # Disable by setting RV_SNAP_MIN=0.
        self.frame_saver: FrameSaver | None = None
        if _SNAP_MIN_COUNT > 0:
            exp_dir = Path(os.environ["RV_EXP_DIR"]) if os.environ.get("RV_EXP_DIR") else _make_exp_dir()
            self.frame_saver = FrameSaver(exp_dir, _SNAP_MIN_COUNT, _SNAP_RATE_SEC)

        # Audit logger — activated by RV_AUDIT=1. No-op otherwise.
        try:
            import sys
            sys.path.insert(0, str(REPO_ROOT))
            from api.audit_logger import AuditLogger
            self.audit = AuditLogger.get()
            if self.audit.enabled:
                self.audit.set_roi_polygons(self.roi_polygons)
                print(f"[probe] AUDIT enabled → {self.audit.audit_dir}", flush=True)
        except Exception as e:
            print(f"[probe] audit init failed: {e}", flush=True)
            self.audit = None

        # Standalone TimeTracker — horse-gated per-color first-arrival.
        try:
            from pipeline.time_tracker import TimeTracker
            self.tracker = TimeTracker(cam_order=list(self.cam_ids))
            print(f"[probe] TimeTracker active ({len(self.cam_ids)} cams, "
                  f"horse-gated per-color first-arrival)", flush=True)
        except Exception as e:
            print(f"[probe] tracker init failed: {e}", flush=True)
            self.tracker = None

        # PASS snapshot saver: whenever a new color registers on a cam,
        # grab the source video frame + draw bbox → output/pass_snaps/.
        self._pass_snap_dir = Path(os.environ.get(
            "RV_PASS_SNAP_DIR", str(REPO_ROOT / "output" / "pass_snaps")))
        self._pass_snap_q: "queue.Queue" = queue.Queue(maxsize=100)
        self._pass_snap_stop = threading.Event()
        self._pass_snap_thread = threading.Thread(
            target=self._pass_snap_worker, daemon=True,
            name="PassSnapWorker",
        )
        try:
            self._pass_snap_dir.mkdir(parents=True, exist_ok=True)
            self._pass_snap_thread.start()
            print(f"[probe] pass-snap saver → {self._pass_snap_dir}", flush=True)
        except Exception as e:
            print(f"[probe] pass-snap disabled: {e}", flush=True)

    @staticmethod
    def _extract_color(obj) -> tuple[int, float]:
        """Pull color label via obj.label — populated by SGIE custom parser's
        descString. Avoids iterating obj.classifier_items which deadlocks in
        pyservicemaker when classifier_meta is populated (DS 9.0)."""
        raw = getattr(obj, "label", "") or ""
        label = raw.strip().lower()
        cid = _COLOR_NAME_TO_ID.get(label)
        if cid is not None:
            return cid, 1.0
        return COLOR_UNKNOWN, 0.0

    def _pass_snap_worker(self):
        """Async: pop snapshot jobs, extract frame from source mp4, draw bbox."""
        try:
            import cv2
        except ImportError:
            print("[probe] cv2 missing — pass-snap disabled", flush=True)
            return
        while not self._pass_snap_stop.is_set():
            try:
                job = self._pass_snap_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                cam_id = job["cam_id"]
                uri    = job["uri"]
                frame_idx = job["frame_idx"]
                color  = job["color"]
                bbox   = job["bbox"]           # mux coords
                mux_w, mux_h = job["mux"]
                cls    = job["cls"]
                lg     = job["lg"]
                path   = uri.replace("file://", "")
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
                ok, frame = cap.read()
                cap.release()
                if not ok or frame is None:
                    continue
                fh, fw = frame.shape[:2]
                sx = fw / max(1, mux_w)
                sy = fh / max(1, mux_h)
                x1, y1, x2, y2 = bbox
                x1, x2 = int(x1 * sx), int(x2 * sx)
                y1, y2 = int(y1 * sy), int(y2 * sy)
                col_bgr = {"red":(40,40,255),"blue":(255,100,30),
                           "green":(50,220,50),"yellow":(30,220,255),
                           "purple":(200,50,180)}.get(color,(200,200,200))
                cv2.rectangle(frame, (x1,y1), (x2,y2), col_bgr, 3)
                cv2.putText(frame, f"{color} cls{cls:.2f} lgt{lg:.1f}",
                            (x1, max(30,y1-8)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, col_bgr, 2)
                cv2.putText(frame, f"{cam_id}  frame {frame_idx}",
                            (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (255,255,255), 2)
                cam_dir = self._pass_snap_dir / cam_id
                cam_dir.mkdir(exist_ok=True)
                out = cam_dir / f"{color}_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            except Exception as e:
                print(f"[pass-snap] {e}", flush=True)

    @staticmethod
    def _iou(a, b) -> float:
        """IoU between two (x1,y1,x2,y2) bboxes."""
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        aw = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
        bw = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
        u = aw + bw - inter
        return inter / u if u > 0 else 0.0

    def _has_horse_overlap(self, person_bbox, horses) -> bool:
        """Return True if person bbox overlaps any horse bbox (IoU ≥ 0.05
        OR person's bottom-center inside horse rect — rider-on-horse prior)."""
        px1, py1, px2, py2 = person_bbox
        pcx = 0.5 * (px1 + px2)
        pby = py2  # bottom of person
        for h in horses:
            hx1, hy1, hx2, hy2 = h
            # a) bottom-center inside horse
            if hx1 <= pcx <= hx2 and hy1 <= pby <= hy2:
                return True
            # b) general IoU fallback
            if self._iou(person_bbox, h) >= 0.05:
                return True
        return False

    def _passes_filters(self, x1, y1, x2, y2) -> bool:
        w = x2 - x1
        h = y2 - y1
        if h < self.MIN_BBOX_HEIGHT:
            return False
        if h > 0 and (w / h) < self.MIN_ASPECT_RATIO:
            return False
        if x1 < self.EDGE_MARGIN or x2 > self.mux_width - self.EDGE_MARGIN:
            return False
        return True

    def handle_metadata(self, batch_meta):
        try:
            self._handle_metadata_impl(batch_meta)
        except Exception as e:
            import traceback
            print(f"[probe] EXC: {e}", flush=True)
            print(traceback.format_exc(), flush=True)

    def _handle_metadata_impl(self, batch_meta):
        ts_us = int(time.time() * 1_000_000)
        wrote_any = False

        if self.debug_mode and self.debug_frames < 3:
            print(f"[probe.enter]", flush=True)

        # Restore original bbox after SGIE consumed tight version (pre-SGIE
        # probe shrinks rect_params; here we reverse the math to make
        # downstream OSD/SHM/audit see full-body bbox as before).
        # Float roundtrip in NvOSD_RectParams (gfloat) preserves exact bits.
        # Only run if SGIE attached AND tight preprocessing is enabled
        # (else pre-shrink probe didn't fire, reverse-math would corrupt bbox).
        if self.sgie_active and TIGHT_SGIE_ENABLED:
            for frame_meta in batch_meta.frame_items:
                for obj in frame_meta.object_items:
                    if obj.class_id != PERSON_CLASS_ID:
                        continue
                    rp = obj.rect_params
                    orig_w = rp.width  / _TIGHT_W_FRAC
                    orig_h = rp.height / _TIGHT_H_FRAC
                    rp.left   = rp.left - TIGHT_X_LO * orig_w
                    rp.top    = rp.top  - TIGHT_Y_LO * orig_h
                    rp.width  = orig_w
                    rp.height = orig_h

        for frame_meta in batch_meta.frame_items:
            pad = frame_meta.pad_index
            if pad < 0 or pad >= len(self.cam_ids):
                continue

            # In debug mode — SKIP OSD (known to hang with add_text + border_color)
            dm = None
            if not self.debug_mode:
                dm = batch_meta.acquire_display_meta()
                cam_label = osd.Text()
                cam_label.display_text = self.cam_ids[pad].encode("ascii")
                cam_label.x_offset     = 8
                cam_label.y_offset     = 8
                cam_label.font.name    = osd.FontFamily.Serif
                cam_label.font.size    = 6
                cam_label.font.color   = osd.Color(1.0, 1.0, 0.0, 1.0)
                cam_label.set_bg_color = True
                cam_label.bg_color     = osd.Color(0.0, 0.0, 0.0, 0.7)
                dm.add_text(cam_label)

                # IP (last octet) — green label right of cam-id
                ip_last = self.cam_ips[pad] if pad < len(self.cam_ips) else ""
                if ip_last:
                    ip_lbl = osd.Text()
                    ip_lbl.display_text = f".{ip_last}".encode("ascii")
                    ip_lbl.x_offset = 52
                    ip_lbl.y_offset = 8
                    ip_lbl.font.name = osd.FontFamily.Serif
                    ip_lbl.font.size = 6
                    ip_lbl.font.color = osd.Color(0.2, 1.0, 0.2, 1.0)
                    ip_lbl.set_bg_color = True
                    ip_lbl.bg_color = osd.Color(0.0, 0.0, 0.0, 0.7)
                    dm.add_text(ip_lbl)

                # ROI polygon — draw on a SEPARATE display_meta so it doesn't
                # evict text/rect items from the primary dm (~16 item limit).
                polys = self.roi_polygons.get(self.cam_ids[pad], [])
                if polys:
                    try:
                        dm_roi = batch_meta.acquire_display_meta()
                    except Exception:
                        dm_roi = None
                    if dm_roi is not None:
                        drawn = 0
                        for poly in polys:
                            if drawn >= 14:
                                break
                            if len(poly) < 2:
                                continue
                            for k in range(len(poly)):
                                if drawn >= 14:
                                    break
                                a = poly[k]
                                b = poly[(k + 1) % len(poly)]
                                line = osd.Line()
                                line.x1 = int(a[0] * self.mux_width)
                                line.y1 = int(a[1] * self.mux_height)
                                line.x2 = int(b[0] * self.mux_width)
                                line.y2 = int(b[1] * self.mux_height)
                                line.width = 2
                                line.color = osd.Color(0.0, 1.0, 0.0, 0.8)
                                try:
                                    dm_roi.add_line(line)
                                    drawn += 1
                                except Exception:
                                    pass
                        if drawn:
                            frame_meta.append(dm_roi)

            n_all_objs = 0
            n_persons  = 0
            n_horses   = 0
            n_passed   = 0
            n_with_cls = 0
            n_horse_gate_rej = 0

            classified_in_frame: list[tuple[float, str]] = []
            dets = []
            ts_now = time.time()

            # Single pass — with horse-emission disabled in C++, all objects
            # are persons. Keep horse-gate code dormant (empty list).
            horse_bboxes: list[tuple[float, float, float, float]] = []
            for obj in frame_meta.object_items:
                n_all_objs += 1
                if obj.class_id != PERSON_CLASS_ID:
                    continue
                n_persons += 1
                rp = obj.rect_params
                x1 = rp.left
                y1 = rp.top
                x2 = x1 + rp.width
                y2 = y1 + rp.height

                passed_filters = self._passes_filters(x1, y1, x2, y2)
                polys = self.roi_polygons.get(self.cam_ids[pad])
                inside_roi = True
                if polys:
                    cx_n = (x1 + x2) * 0.5 / self.mux_width
                    cy_n = (y1 + y2) * 0.5 / self.mux_height
                    inside_roi = any(_point_in_polygon(cx_n, cy_n, p) for p in polys)

                # Horse-gate only if horses detected (YOLO class 17 would populate
                # horse_bboxes). With person-only emission it stays empty → gate skipped.
                has_horse = True if not horse_bboxes else self._has_horse_overlap(
                    (x1, y1, x2, y2), horse_bboxes)
                if not has_horse:
                    n_horse_gate_rej += 1

                reject_reason = ""
                if not passed_filters:
                    reject_reason = "bbox_filter"
                elif not inside_roi:
                    reject_reason = "roi_outside"
                elif not has_horse:
                    reject_reason = "no_horse"

                # Audit: log even rejects (for diagnosis)
                if self.audit is not None and self.audit.enabled and reject_reason:
                    self.audit.log_detection(
                        cam_id=self.cam_ids[pad], ts=ts_now,
                        frame_idx=self.frame_counts[pad],
                        bbox=(x1, y1, x2, y2),
                        frame_w=self.mux_width, frame_h=self.mux_height,
                        color="", color_conf=0.0,
                        det_conf=float(obj.confidence),
                        passed_filters=passed_filters, inside_roi=inside_roi,
                        written_to_shm=False,
                        track_id=int(getattr(obj, "object_id", 0) or 0),
                        reject_reason=reject_reason,
                    )

                if not passed_filters or not inside_roi or not has_horse:
                    rp.border_width = 0
                    tp = getattr(obj, "text_params", None)
                    if tp is not None:
                        tp.display_text = b""
                    continue

                n_passed += 1

                # Extract color + confidence. Our custom parser encodes
                # label as "red|0.95" because pyservicemaker hides
                # attr.attributeConfidence.
                color_id, color_conf, color_logit = COLOR_UNKNOWN, 0.0, 0.0
                for cls_meta in obj.classifier_items:
                    n_lab = int(getattr(cls_meta, "n_labels", 0))
                    for j in range(n_lab):
                        raw = cls_meta.get_n_label(j) or ""
                        # Format from parser: "color|prob|logit"
                        parts = raw.split("|")
                        lbl = parts[0]
                        prob = 1.0
                        logit = 0.0
                        if len(parts) >= 2:
                            try: prob = float(parts[1])
                            except ValueError: prob = 0.0
                        if len(parts) >= 3:
                            try: logit = float(parts[2])
                            except ValueError: logit = 0.0
                        cid = _COLOR_NAME_TO_ID.get(lbl.strip().lower())
                        if cid is not None:
                            color_id, color_conf, color_logit = cid, prob, logit
                            break
                    if color_id != COLOR_UNKNOWN:
                        break

                verbose = self.debug_mode and self.debug_budget > 0
                if verbose:
                    print(f"  [det cam={self.cam_ids[pad]} conf={obj.confidence:.2f} "
                          f"bbox={int(x2-x1)}x{int(y2-y1)}] "
                          f"obj.label={getattr(obj, 'label', '')!r} "
                          f"resolved={_COLOR_ID_TO_NAME.get(color_id, '?')}",
                          flush=True)
                    self.debug_budget -= 1

                # Drop classifications for colors not active in today's race
                # (blue/purple would be spectators — no jockeys wear them today).
                if color_id != COLOR_UNKNOWN and color_id not in _ACTIVE_IDS:
                    color_id, color_conf = COLOR_UNKNOWN, 0.0

                if color_id != COLOR_UNKNOWN:
                    n_with_cls += 1
                    cx = float((x1 + x2) * 0.5)
                    classified_in_frame.append(
                        (cx, _COLOR_ID_TO_NAME[color_id],
                         float(color_conf), float(color_logit))
                    )

                # CSV dump: every PGIE detection that passed bbox filter,
                # whether classified or not
                if self.csv_fp is not None:
                    color_name = _COLOR_ID_TO_NAME.get(color_id, "UNK")
                    self.csv_fp.write(
                        f"{self.frame_counts[pad]},{self.cam_ids[pad]},"
                        f"{int(x1)},{int(y1)},{int(x2-x1)},{int(y2-y1)},"
                        f"{obj.confidence:.3f},{color_name},{color_conf:.3f}\n"
                    )

                track_id = getattr(obj, "object_id", 0) or 0

                # OSD: hide the default nvosd bbox (forced red) and draw our own
                # via display_meta Rect so it actually takes our color.
                rp.border_width = 0  # hide default
                if dm is not None and color_id != COLOR_UNKNOWN:
                    color_name = _COLOR_ID_TO_NAME.get(color_id, "?")
                    r, g, b = _COLOR_RGB_NORMALISED.get(color_name, (0.5, 0.5, 0.5))
                    box = osd.Rect()
                    box.left = int(x1)
                    box.top = int(y1)
                    box.width = int(x2 - x1)
                    box.height = int(y2 - y1)
                    box.border_width = 3
                    box.border_color = osd.Color(r, g, b, 1.0)
                    box.has_bg_color = False
                    try:
                        dm.add_rect(box)
                    except Exception:
                        pass

                    tp = getattr(obj, "text_params", None)
                    if tp is not None:
                        if color_id == COLOR_UNKNOWN:
                            tp.display_text = b""
                        else:
                            tid = int(getattr(obj, "object_id", 0) or 0)
                            tid_str = f" #{tid}" if 0 < tid < 1_000_000 else ""
                            label = (f"{color_name} cls{float(color_conf):.2f}"
                                     f" lgt{float(color_logit):.1f}{tid_str}")
                            tp.display_text = label.encode("ascii")
                            fp = getattr(tp, "font", None) or getattr(tp, "font_params", None)
                            if fp is not None:
                                try:
                                    fp.color = osd.Color(r, g, b, 1.0)
                                    fp.size = 6
                                except Exception:
                                    pass
                            try:
                                tp.set_bg_color = True
                                tp.bg_color = osd.Color(0, 0, 0, 0.7)
                            except Exception:
                                pass

                dets.append(make_detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    det_conf=obj.confidence,
                    color_id=color_id,
                    color_conf=color_conf,
                    track_id=track_id,
                ))

                # Audit: log passed detection
                if self.audit is not None and self.audit.enabled:
                    self.audit.log_detection(
                        cam_id=self.cam_ids[pad], ts=ts_now,
                        frame_idx=self.frame_counts[pad],
                        bbox=(x1, y1, x2, y2),
                        frame_w=self.mux_width, frame_h=self.mux_height,
                        color=_COLOR_ID_TO_NAME.get(color_id, ""),
                        color_conf=float(color_conf),
                        det_conf=float(obj.confidence),
                        passed_filters=True, inside_roi=True,
                        written_to_shm=True,
                        track_id=int(track_id),
                        reject_reason="",
                    )

                if len(dets) >= MAX_DETECTIONS:
                    break

            # Per-frame summary in debug mode
            if self.debug_mode and self.debug_frames < 20:
                print(f"[frame cam={self.cam_ids[pad]} idx={self.frame_counts[pad]}] "
                      f"all_objs={n_all_objs} persons={n_persons} horses={n_horses} "
                      f"horse_rej={n_horse_gate_rej} filter_pass={n_passed} "
                      f"classified={n_with_cls}", flush=True)
                self.debug_frames += 1

            # Snapshot trigger: ≥N classified jockeys on this camera frame
            if self.frame_saver is not None and classified_in_frame:
                self.frame_saver.maybe_trigger(
                    self.cam_ids[pad],
                    [(cx, c) for cx, c, _, _ in classified_in_frame],
                )

            # Feed TimeTracker with temporal stability check:
            #   a color must be classified on the SAME track_id MIN_CONSEC
            #   times in a row before it counts.
            if self.tracker is not None and dets:
                cam = self.cam_ids[pad]
                now = ts_us / 1_000_000.0
                stable_colors: list[tuple[float, str, float, float, tuple]] = []
                for d in dets:
                    cname = _COLOR_ID_TO_NAME.get(d.color_id)
                    if not cname:
                        continue
                    tid = int(d.track_id or 0)
                    if tid <= 0 or tid >= 1_000_000_000:
                        # no tracker → accept immediately (fallback)
                        stable_colors.append(
                            (d.center_x if hasattr(d,"center_x") else 0.5*(d.x1+d.x2),
                             cname, float(d.color_conf), 0.0,
                             (d.x1, d.y1, d.x2, d.y2)))
                        continue
                    key = (cam, tid)
                    st = self._stability.get(key)
                    if st is None or st["color"] != cname:
                        self._stability[key] = {"color": cname, "count": 1}
                        continue
                    st["count"] += 1
                    if st["count"] == self.MIN_CONSEC:
                        cx = 0.5 * (d.x1 + d.x2)
                        stable_colors.append(
                            (cx, cname, float(d.color_conf), 0.0,
                             (d.x1, d.y1, d.x2, d.y2)))

                if stable_colors:
                    # Rightmost first for same-frame ties
                    ordered = sorted(stable_colors, key=lambda t: -t[0])
                    new_arrivals = []
                    for rank_idx, (cx, c, cf, lg, bb) in enumerate(ordered):
                        res = self.tracker.ingest(now + rank_idx * 1e-6, cam, c)
                        if res:
                            new_arrivals.extend(res)
                            if hasattr(self, "_pass_snap_q"):
                                try:
                                    self._pass_snap_q.put_nowait({
                                        "cam_id": cam,
                                        "uri": (self.cam_uris[pad]
                                                if pad < len(self.cam_uris) else ""),
                                        "frame_idx": self.frame_counts[pad],
                                        "color": c,
                                        "bbox": bb,
                                        "mux": (self.mux_width, self.mux_height),
                                        "cls": cf,
                                        "lg": lg,
                                    })
                                except queue.Full:
                                    pass
                if new_arrivals:
                    for c in new_arrivals:
                        print(f"[PASS] {cam}  {c}  ts={now:.2f}", flush=True)
                    rk = self.tracker.get_ranking()
                    line = "  ".join(f"{r['rank']}:{r['color']}@{r['last_camera']}"
                                     for r in rk[:6])
                    print(f"[RANK] {line}", flush=True)

            # Global ranking overlay — large, top-center of every camera tile.
            # Who's #1 across the whole track.
            if dm is not None and self.tracker is not None:
                global_rk = self.tracker.get_ranking()
                if global_rk:
                    header = osd.Text()
                    header.display_text = b"GLOBAL"
                    header.x_offset = self.mux_width // 2 - 40
                    header.y_offset = 6
                    header.font.name = osd.FontFamily.Serif
                    header.font.size = 10
                    header.font.color = osd.Color(1, 1, 1, 1.0)
                    header.set_bg_color = True
                    header.bg_color = osd.Color(0, 0, 0, 0.85)
                    dm.add_text(header)
                    for idx, r in enumerate(global_rk[:5]):
                        line_txt = f"{r['rank']}. {r['color'].upper():<7} @{r['last_camera']}"
                        t = osd.Text()
                        t.display_text = line_txt.encode("ascii")
                        t.x_offset = self.mux_width // 2 - 110
                        t.y_offset = 28 + idx * 22
                        t.font.name = osd.FontFamily.Serif
                        t.font.size = 12
                        r_, g_, b_ = _COLOR_RGB_NORMALISED.get(r['color'], (1, 1, 1))
                        t.font.color = osd.Color(r_, g_, b_, 1.0)
                        t.set_bg_color = True
                        t.bg_color = osd.Color(0, 0, 0, 0.85)
                        dm.add_text(t)

                # Per-camera list — arrival order on this cam (top-right corner)
                cam_rk = self.tracker.camera_ranking(self.cam_ids[pad])
                if cam_rk:
                    hdr = osd.Text()
                    hdr.display_text = b"here"
                    hdr.x_offset = self.mux_width - 130
                    hdr.y_offset = 6
                    hdr.font.name = osd.FontFamily.Serif
                    hdr.font.size = 8
                    hdr.font.color = osd.Color(0.7, 0.7, 0.7, 1.0)
                    hdr.set_bg_color = True
                    hdr.bg_color = osd.Color(0, 0, 0, 0.75)
                    dm.add_text(hdr)
                    for idx, r in enumerate(cam_rk):
                        line_txt = f"{r['rank']}. {r['color'].upper()}"
                        txt = osd.Text()
                        txt.display_text = line_txt.encode("ascii")
                        txt.x_offset = self.mux_width - 130
                        txt.y_offset = 24 + idx * 18
                        txt.font.name = osd.FontFamily.Serif
                        txt.font.size = 8
                        r_, g_, b_ = _COLOR_RGB_NORMALISED.get(r['color'], (1, 1, 1))
                        txt.font.color = osd.Color(r_, g_, b_, 1.0)
                        txt.set_bg_color = True
                        txt.bg_color = osd.Color(0, 0, 0, 0.75)
                        dm.add_text(txt)

            # All texts for this frame are ready — now append the display_meta
            if dm is not None:
                frame_meta.append(dm)

            self.frame_counts[pad] += 1
            self.det_counts[pad]   += len(dets)
            self.total_frames      += 1

            src_frame_num = getattr(frame_meta, "frame_number", 0) or 0
            slot = make_camera_slot(
                cam_id=self.cam_ids[pad],
                frame_w=self.mux_width,
                frame_h=self.mux_height,
                timestamp_us=ts_us,
                detections=dets,
                source_frame_num=src_frame_num,
            )
            self.plugin.write_camera(self.shm_handle, pad, slot)
            wrote_any = True

            if self.total_frames % self.log_every == 0:
                self._log_snapshot()

        if wrote_any:
            self.plugin.commit(self.shm_handle)

    def _log_snapshot(self):
        # FPS since last snapshot (infer batch rate × batch-size ≈ per-cam fps)
        now = time.time()
        last = getattr(self, "_last_snap_t", now)
        last_n = getattr(self, "_last_snap_n", 0)
        dt = max(1e-6, now - last)
        batches_per_s = (self.total_frames - last_n) / dt
        self._last_snap_t = now
        self._last_snap_n = self.total_frames
        per_cam_fps = batches_per_s / max(1, len(self.cam_ids))

        live = [(self.cam_ids[i], self.det_counts[i], self.frame_counts[i])
                for i in range(len(self.cam_ids)) if self.det_counts[i] > 0]
        live.sort(key=lambda x: -x[1])
        head = ", ".join(f"{cid}:{d}/{f}" for cid, d, f in live[:6])
        print(f"[probe] frames={self.total_frames}  "
              f"batch_fps={batches_per_s:.1f}  per_cam_fps={per_cam_fps:.1f}  "
              f"active={len(live)}  top: {head}")

    def report(self):
        print("\n" + "=" * 60)
        print(f"Total frames processed: {self.total_frames}")
        print(f"Per-camera totals (cam: detections / frames):")
        for i, cid in enumerate(self.cam_ids):
            f = self.frame_counts[i]
            d = self.det_counts[i]
            avg = d / f if f else 0
            mark = "*" if d > 0 else " "
            print(f"  {mark} {cid}: {d:6d} / {f:5d}  (avg {avg:.2f}/frame)")


def build_pipeline(cameras: list[CameraEntry], nvinfer_config: Path,
                   plugin: RVPlugin,
                   sgie_config: Path | None = None,
                   tracker_config: Path | None = None,
                   mux_width: int = 800, mux_height: int = 800,
                   batched_push_timeout_us: int = 40000,
                   display: bool = False,
                   display_width: int = 1920,
                   display_height: int = 1080):
    n = len(cameras)
    if n == 0:
        raise ValueError("no cameras configured")

    cam_ids    = [c.cam_id for c in cameras]
    cam_uris   = [c.uri for c in cameras]
    shm_handle = plugin.create_shm(cam_ids)

    probe_op = DetectionProbe(plugin, shm_handle, cam_ids,
                              mux_width=mux_width, mux_height=mux_height,
                              cam_uris=cam_uris,
                              sgie_active=(sgie_config is not None))

    pipe = Pipeline("rv-pipeline")
    pipe.add("nvstreammux", "mux", {
        "batch-size":            n,
        "width":                 mux_width,
        "height":                mux_height,
        "batched-push-timeout":  batched_push_timeout_us,
        "live-source":           0,
    })

    for i, cam in enumerate(cameras):
        src_name = f"src_{i}"
        pipe.add("uridecodebin", src_name, {"uri": cam.uri})
        pipe.link((src_name, "mux"), ("", "sink_%u"))

    pipe.add("nvinfer", "infer", {
        "config-file-path": str(nvinfer_config),
        "unique-id":        1,
        "batch-size":       n,
    })

    # Sink (fakesink for headless, tiler+osd+eglsink for --display)
    if display:
        import math
        rows = int(math.sqrt(n))
        cols = math.ceil(n / rows)
        pipe.add("nvmultistreamtiler", "tiler", {
            "rows":   rows,  "columns": cols,
            "width":  display_width, "height": display_height,
        })
        pipe.add("nvosdbin",      "osd")
        # Realtime playback (sync=True) so labels stay readable.
        # Override with RV_DISPLAY_SYNC=0 for max-speed processing.
        _sync = os.environ.get("RV_DISPLAY_SYNC", "1") == "1"
        pipe.add("nveglglessink", "sink", {"sync": _sync})
    else:
        pipe.add("fakesink", "sink", {"sync": False, "async": False})

    # Build link chain (same ORDER and LAYOUT as Phase 5.3 — sgie added
    # AFTER sink; empirical: other orderings silently break classifier_meta)
    probe_attach_id = "infer"
    chain_tail = ["sink"] if not display else ["tiler", "osd", "sink"]

    if tracker_config is not None:
        from deepstream.config import TRACKER_LL_LIB_FILE
        pipe.add("nvtracker", "tracker", {
            "ll-config-file": str(tracker_config),
            "ll-lib-file":    TRACKER_LL_LIB_FILE,
            "tracker-width":  mux_width,
            "tracker-height": mux_height,
        })

    if sgie_config is not None:
        pipe.add("nvinfer", "sgie_color", {
            "config-file-path": str(sgie_config),
            "unique-id":        2,
        })
        probe_attach_id = "sgie_color"
    if tracker_config is not None:
        # Probe after tracker so we see both color (from sgie upstream) and
        # track_id. sgie-then-tracker ordering is enforced below.
        probe_attach_id = "tracker"

    # Link order: pgie → SGIE → tracker → sink.
    # Note: canonical DS order is pgie → tracker → sgie, but that breaks
    # classifier_meta with our combined plugin. sgie-before-tracker keeps
    # classification + tracking both working.
    chain = ["mux", "infer"]
    if sgie_config is not None:    chain.append("sgie_color")
    if tracker_config is not None: chain.append("tracker")
    chain += chain_tail
    pipe.link(*chain)

    # Pre-SGIE probe: shrink bbox to torso-only zone so SGIE crops a tight
    # region. Only attach if SGIE is actually present (otherwise bbox would
    # stay shrunk forever — DetectionProbe restore math would corrupt it).
    # Skip when RV_TIGHT_SGIE=0 (OSNet SGIE wants full bbox).
    if sgie_config is not None and TIGHT_SGIE_ENABLED:
        pipe.attach("infer", Probe("rv_tight_pre_sgie", TightBboxShrinkProbe()))
    pipe.attach(probe_attach_id, Probe("rv_probe", probe_op))

    return pipe, probe_op, shm_handle


def main(args):
    cameras = load_cameras(Path(args.cameras), limit=args.limit)
    print(f"[pipeline] cameras: {len(cameras)}")
    for c in cameras[:3]:
        print(f"  - {c.cam_id}: {c.uri[:80]}")
    if len(cameras) > 3:
        print(f"  ... and {len(cameras) - 3} more")

    plugin = RVPlugin.load()
    print(f"[pipeline] plugin loaded: {plugin.lib._name}")

    sgie = Path(args.sgie) if (args.sgie and args.sgie.strip()) else None
    pipe, probe_op, shm_handle = build_pipeline(
        cameras, Path(args.config), plugin,
        sgie_config=sgie,
        mux_width=args.mux_width, mux_height=args.mux_height,
    )

    print(f"[pipeline] starting ({len(cameras)} cameras, mux {args.mux_width}x{args.mux_height})")
    try:
        pipe.start().wait()
    except KeyboardInterrupt:
        print("\n[pipeline] interrupted")
    finally:
        probe_op.report()
        if getattr(probe_op, "frame_saver", None):
            probe_op.frame_saver.close()
        plugin.destroy_shm(shm_handle)


if __name__ == "__main__":
    import argparse
    from multiprocessing import Process

    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras",     default=str(DEFAULT_CAMERAS))
    ap.add_argument("--config",      default=str(DEFAULT_CONFIG))
    ap.add_argument("--sgie",        default=str(DEFAULT_SGIE),
                    help="path to SGIE color classifier config (empty string = disable)")
    ap.add_argument("--limit",       type=int, default=None,  help="max cameras to use")
    ap.add_argument("--mux-width",   type=int, default=800)
    ap.add_argument("--mux-height",  type=int, default=800)
    args = ap.parse_args()

    p = Process(target=main, args=(args,))
    try:
        p.start()
        p.join()
    except KeyboardInterrupt:
        p.terminate()
