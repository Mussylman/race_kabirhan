"""GStreamer pipeline assembly + main() entry.

Holds the wiring code that puts nvstreammux + sources + nvinfer + sgie +
tracker + sink together and attaches probes. Extracted from pipeline.py
during the refactor to keep that file focused on the per-frame
DetectionProbe.

DetectionProbe is imported lazily inside build_pipeline() to break the
circular dependency: pipeline.py re-exports our build_pipeline/main for
backward compat (e.g. deepstream/main.py imports build_pipeline from
deepstream.pipeline).
"""
from __future__ import annotations

import os
from pathlib import Path

from pyservicemaker import Pipeline, Probe

from deepstream.rv_plugin import RVPlugin
from deepstream.config import CameraEntry, load_cameras
from deepstream.modules.probes import (
    TIGHT_SGIE_ENABLED,
    TightBboxShrinkProbe,
)


def build_pipeline(cameras: list[CameraEntry], nvinfer_config: Path,
                   plugin: RVPlugin,
                   sgie_config: Path | None = None,
                   tracker_config: Path | None = None,
                   mux_width: int = 800, mux_height: int = 800,
                   batched_push_timeout_us: int = 40000,
                   display: bool = False,
                   display_width: int = 1920,
                   display_height: int = 1080):
    # Deferred to break circular import: pipeline.py defines DetectionProbe
    # and also re-exports build_pipeline from this module.
    from deepstream.pipeline import DetectionProbe

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
