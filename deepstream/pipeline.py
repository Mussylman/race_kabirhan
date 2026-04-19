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
import sys
import time
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

    # Detection filters (relaxed compared to old C++ defaults for jockeys)
    MIN_BBOX_HEIGHT  = 40
    MIN_ASPECT_RATIO = 0.2
    EDGE_MARGIN      = 5

    def __init__(self, plugin: RVPlugin, shm_handle, cam_ids: list[str],
                 mux_width: int, mux_height: int, log_every: int = 50):
        super().__init__()
        self.plugin       = plugin
        self.shm_handle   = shm_handle
        self.cam_ids      = cam_ids
        self.mux_width    = mux_width
        self.mux_height   = mux_height
        self.log_every    = log_every
        self.frame_counts = [0] * len(cam_ids)
        self.det_counts   = [0] * len(cam_ids)
        self.total_frames = 0
        self.debug_mode   = os.environ.get("RV_DEBUG_PROBE", "0") == "1"
        self.debug_budget = 60  # detailed lines for the first N detections, then quiet
        self.debug_frames = 0   # detailed header lines for the first N frames

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
        ts_us = int(time.time() * 1_000_000)
        wrote_any = False

        if self.debug_mode and self.debug_frames < 3:
            print(f"[probe.enter]", flush=True)

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
                cam_label.font.size    = 14
                cam_label.font.color   = osd.Color(1.0, 1.0, 0.0, 1.0)
                cam_label.set_bg_color = True
                cam_label.bg_color     = osd.Color(0.0, 0.0, 0.0, 0.7)
                dm.add_text(cam_label)

            n_all_objs = 0
            n_persons  = 0
            n_passed   = 0
            n_with_cls = 0

            dets = []
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
                if not self._passes_filters(x1, y1, x2, y2):
                    continue
                n_passed += 1

                # Extract color — prefer obj.label (set via descString), else
                # walk classifier_items as a fallback.
                color_id, color_conf = COLOR_UNKNOWN, 0.0
                obj_label = getattr(obj, "label", "") or ""
                if obj_label:
                    cid = _COLOR_NAME_TO_ID.get(obj_label.strip().lower())
                    if cid is not None:
                        color_id, color_conf = cid, 1.0
                if color_id == COLOR_UNKNOWN:
                    for cls_meta in obj.classifier_items:
                        n_lab = int(getattr(cls_meta, "n_labels", 0))
                        for j in range(n_lab):
                            raw = cls_meta.get_n_label(j) or ""
                            cid = _COLOR_NAME_TO_ID.get(raw.strip().lower())
                            if cid is not None:
                                color_id, color_conf = cid, 1.0
                                break
                        if color_id != COLOR_UNKNOWN:
                            break

                verbose = self.debug_mode and self.debug_budget > 0
                if verbose:
                    print(f"  [det cam={self.cam_ids[pad]} conf={obj.confidence:.2f} "
                          f"bbox={int(x2-x1)}x{int(y2-y1)}] "
                          f"obj.label={obj_label!r} "
                          f"resolved={_COLOR_ID_TO_NAME.get(color_id, '?')}",
                          flush=True)
                    self.debug_budget -= 1

                if color_id != COLOR_UNKNOWN:
                    n_with_cls += 1

                track_id = getattr(obj, "object_id", 0) or 0

                # OSD recolor + label: only when NOT in debug mode
                if dm is not None:
                    color_name = _COLOR_ID_TO_NAME.get(color_id, "?")
                    r, g, b = _COLOR_RGB_NORMALISED.get(color_name, (0.5, 0.5, 0.5))
                    rp.border_color = osd.Color(r, g, b, 1.0)
                    rp.border_width = 3

                    label_text = color_name
                    if track_id and track_id != 0xFFFFFFFFFFFFFFFF:
                        label_text += f" #{track_id}"
                    lbl = osd.Text()
                    lbl.display_text     = label_text.encode("ascii", "ignore")
                    lbl.x_offset         = int(x1)
                    lbl.y_offset         = max(int(y1) - 18, 0)
                    lbl.font.name        = osd.FontFamily.Serif
                    lbl.font.size        = 11
                    lbl.font.color       = osd.Color(1.0, 1.0, 1.0, 1.0)
                    lbl.set_bg_color     = True
                    lbl.bg_color         = osd.Color(0.0, 0.0, 0.0, 0.7)
                    dm.add_text(lbl)

                dets.append(make_detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    det_conf=obj.confidence,
                    color_id=color_id,
                    color_conf=color_conf,
                    track_id=track_id,
                ))
                if len(dets) >= MAX_DETECTIONS:
                    break

            # Per-frame summary in debug mode
            if self.debug_mode and self.debug_frames < 20:
                print(f"[frame cam={self.cam_ids[pad]} idx={self.frame_counts[pad]}] "
                      f"all_objs={n_all_objs} persons={n_persons} "
                      f"filter_pass={n_passed} classified={n_with_cls}", flush=True)
                self.debug_frames += 1

            # All texts for this frame are ready — now append the display_meta
            if dm is not None:
                frame_meta.append(dm)

            self.frame_counts[pad] += 1
            self.det_counts[pad]   += len(dets)
            self.total_frames      += 1

            slot = make_camera_slot(
                cam_id=self.cam_ids[pad],
                frame_w=self.mux_width,
                frame_h=self.mux_height,
                timestamp_us=ts_us,
                detections=dets,
            )
            self.plugin.write_camera(self.shm_handle, pad, slot)
            wrote_any = True

            if self.total_frames % self.log_every == 0:
                self._log_snapshot()

        if wrote_any:
            self.plugin.commit(self.shm_handle)

    def _log_snapshot(self):
        live = [(self.cam_ids[i], self.det_counts[i], self.frame_counts[i])
                for i in range(len(self.cam_ids)) if self.det_counts[i] > 0]
        live.sort(key=lambda x: -x[1])
        head = ", ".join(f"{cid}:{d}/{f}" for cid, d, f in live[:6])
        print(f"[probe] frames={self.total_frames}  active={len(live)}  top: {head}")

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
    shm_handle = plugin.create_shm(cam_ids)

    probe_op = DetectionProbe(plugin, shm_handle, cam_ids,
                              mux_width=mux_width, mux_height=mux_height)

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
        pipe.add("nveglglessink", "sink", {"sync": False})
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
