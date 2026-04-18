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

from pyservicemaker import BatchMetadataOperator, Pipeline, Probe

# Local plugin binding
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepstream.rv_plugin import (
    RVPlugin,
    COLOR_UNKNOWN,
    MAX_DETECTIONS,
    make_camera_slot,
    make_detection,
)

REPO_ROOT       = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG  = REPO_ROOT / "deepstream" / "configs" / "nvinfer_racevision.txt"
DEFAULT_CAMERAS = REPO_ROOT / "configs"   / "cameras_test_files.json"

PERSON_CLASS_ID = 0


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

        for frame_meta in batch_meta.frame_items:
            pad = frame_meta.pad_index
            if pad < 0 or pad >= len(self.cam_ids):
                continue

            dets = []
            for obj in frame_meta.object_items:
                if obj.class_id != PERSON_CLASS_ID:
                    continue
                rp = obj.rect_params
                x1 = rp.left
                y1 = rp.top
                x2 = x1 + rp.width
                y2 = y1 + rp.height
                if not self._passes_filters(x1, y1, x2, y2):
                    continue
                dets.append(make_detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    det_conf=obj.confidence,
                    color_id=COLOR_UNKNOWN,
                    color_conf=0.0,
                    track_id=getattr(obj, "object_id", 0) or 0,
                ))
                if len(dets) >= MAX_DETECTIONS:
                    break

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
                   mux_width: int = 800, mux_height: int = 800,
                   batched_push_timeout_us: int = 40000):
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
    pipe.add("fakesink", "sink", {"sync": False, "async": False})
    pipe.link("mux", "infer", "sink")
    pipe.attach("infer", Probe("rv_probe", probe_op))

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

    pipe, probe_op, shm_handle = build_pipeline(
        cameras, Path(args.config), plugin,
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
