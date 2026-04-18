"""
deepstream/main.py — Entry point for the Python DeepStream pipeline.

Replaces the C++ `main.cpp` (~300 lines) with ~80 lines of Python.

Examples:
    # 25 RTSP cameras using configs/cameras_live.json
    python -m deepstream.main

    # 3 file sources for offline testing
    python -m deepstream.main --cameras configs/cameras_test_files.json --limit 3

    # disable color SGIE
    python -m deepstream.main --sgie ""

    # write logs and metrics CSV to /tmp/rv_logs
    python -m deepstream.main --log-dir /tmp/rv_logs --log-level DEBUG
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Process
from pathlib import Path

# Allow `python deepstream/main.py ...` and `python -m deepstream.main`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepstream import config as cfg
from deepstream import diag
from deepstream.pipeline import build_pipeline
from deepstream.rv_plugin import RVPlugin


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="deepstream.main",
                                 description="Race Vision Python pipeline")
    ap.add_argument("--cameras",    default=str(cfg.DEFAULT_CAMERAS))
    ap.add_argument("--config",     default=str(cfg.DEFAULT_PGIE),
                    help="primary nvinfer (YOLO) config")
    ap.add_argument("--sgie",       default=str(cfg.DEFAULT_SGIE),
                    help="secondary nvinfer (color) config; empty string disables")
    ap.add_argument("--plugin",     default=str(cfg.PLUGIN_LIB_PATH),
                    help="path to libnvdsinfer_racevision.so")
    ap.add_argument("--limit",      type=int, default=None,
                    help="cap the number of cameras (useful for smoke tests)")
    ap.add_argument("--mux-width",  type=int, default=cfg.DEFAULT_MUX_WIDTH)
    ap.add_argument("--mux-height", type=int, default=cfg.DEFAULT_MUX_HEIGHT)
    ap.add_argument("--log-dir",    default=None,
                    help="if set, also write rv_pipeline.log here")
    ap.add_argument("--log-level",  default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return ap.parse_args()


def run(args: argparse.Namespace) -> None:
    diag.setup_logging(level=args.log_level,
                       log_dir=Path(args.log_dir) if args.log_dir else None)
    log = diag.get_logger("rv.main")

    cameras = cfg.load_cameras(args.cameras, limit=args.limit)
    log.info("loaded %d cameras from %s", len(cameras), args.cameras)
    for c in cameras[:3]:
        log.info("  %s -> %s", c.cam_id, c.uri[:80])
    if len(cameras) > 3:
        log.info("  ... and %d more", len(cameras) - 3)

    plugin = RVPlugin.load(Path(args.plugin))
    log.info("plugin loaded: %s", args.plugin)

    sgie = Path(args.sgie) if (args.sgie and args.sgie.strip()) else None

    pipe, probe_op, shm_handle = build_pipeline(
        cameras            = cameras,
        nvinfer_config     = Path(args.config),
        plugin             = plugin,
        sgie_config        = sgie,
        mux_width          = args.mux_width,
        mux_height         = args.mux_height,
    )
    log.info("pipeline built (%d cameras, mux %dx%d, sgie=%s)",
             len(cameras), args.mux_width, args.mux_height,
             sgie.name if sgie else "off")

    try:
        pipe.start().wait()
    except KeyboardInterrupt:
        log.warning("interrupted by user")
    finally:
        probe_op.report()
        plugin.destroy_shm(shm_handle)
        log.info("shutdown complete")


def main():
    args = parse_args()
    # Run inside a child process so KeyboardInterrupt is delivered cleanly
    # (matches the pyservicemaker sample apps' pattern).
    p = Process(target=run, args=(args,))
    try:
        p.start()
        p.join()
    except KeyboardInterrupt:
        p.terminate()
        p.join()


if __name__ == "__main__":
    main()
