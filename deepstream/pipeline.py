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

import sys
import time
from pathlib import Path
from typing import Optional

from pyservicemaker import BatchMetadataOperator, Pipeline, Probe

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepstream.config import (
    CameraEntry,
    DEFAULT_BATCHED_PUSH_TIMEOUT,
    DEFAULT_MUX_HEIGHT,
    DEFAULT_MUX_WIDTH,
)
from deepstream.rv_plugin import (
    RVPlugin,
    COLOR_UNKNOWN,
    MAX_DETECTIONS,
    make_camera_slot,
    make_detection,
)

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

    @staticmethod
    def _extract_color(obj) -> tuple[int, float]:
        """Pull color label from sgie classifier_meta.

        pyservicemaker's ClassifierMetadata only exposes label name strings
        (get_n_label), not per-label probabilities — so color_conf is set to
        1.0 when a label was emitted (i.e. it passed classifier-threshold)
        and 0.0 when no classification was attached to this object.
        """
        cls_items = getattr(obj, "classifier_items", None)
        if not cls_items:
            return COLOR_UNKNOWN, 0.0
        for cls_meta in cls_items:
            n = int(getattr(cls_meta, "n_labels", 0))
            for j in range(n):
                label = (cls_meta.get_n_label(j) or "").strip().lower()
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
                color_id, color_conf = self._extract_color(obj)
                dets.append(make_detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    det_conf=obj.confidence,
                    color_id=color_id,
                    color_conf=color_conf,
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
                   sgie_config: Path | None = None,
                   mux_width: int = DEFAULT_MUX_WIDTH,
                   mux_height: int = DEFAULT_MUX_HEIGHT,
                   batched_push_timeout_us: int = DEFAULT_BATCHED_PUSH_TIMEOUT):
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

    probe_attach_id = "infer"
    if sgie_config is not None:
        pipe.add("nvinfer", "sgie_color", {
            "config-file-path": str(sgie_config),
            "unique-id":        2,
        })
        pipe.link("mux", "infer", "sgie_color", "sink")
        probe_attach_id = "sgie_color"
    else:
        pipe.link("mux", "infer", "sink")

    pipe.attach(probe_attach_id, Probe("rv_probe", probe_op))

    return pipe, probe_op, shm_handle


if __name__ == "__main__":
    # Use deepstream/main.py as the production entry point.
    print("Use: python -m deepstream.main  (see deepstream/main.py)", file=sys.stderr)
    sys.exit(2)
