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
import os
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
    ap.add_argument("--tracker",    default="",
                    help="nvtracker low-level config (YAML); default disabled because "
                         "tracker + sgie combo drops classifier_meta in our setup")
    ap.add_argument("--plugin",     default=str(cfg.PLUGIN_LIB_PATH),
                    help="path to libnvdsinfer_racevision.so")
    ap.add_argument("--limit",      type=int, default=None,
                    help="cap the number of cameras (useful for smoke tests)")
    ap.add_argument("--camera",     default=None,
                    help="filter to a single camera id, e.g. cam-13")
    ap.add_argument("--mux-width",  type=int, default=cfg.DEFAULT_MUX_WIDTH)
    ap.add_argument("--mux-height", type=int, default=cfg.DEFAULT_MUX_HEIGHT)
    ap.add_argument("--display",    action="store_true",
                    help="show all cameras tiled in a GUI window with bbox/OSD")
    ap.add_argument("--display-width",  type=int, default=1920)
    ap.add_argument("--display-height", type=int, default=1080)
    ap.add_argument("--log-dir",    default=None,
                    help="if set, also write rv_pipeline.log here")
    ap.add_argument("--log-level",  default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return ap.parse_args()


def _dump_debug_snapshots(args: argparse.Namespace, run_dir: Path, log) -> None:
    """For each file:// camera, run tools/debug_detect_classify.py on its video
    and store images in run_dir/debug_<cam_id>/. Bounded to args.limit or
    args.camera to avoid spawning 25 subprocesses."""
    import subprocess
    from urllib.parse import urlparse

    cams = cfg.load_cameras(args.cameras, limit=args.limit)
    if args.camera:
        cams = [c for c in cams if c.cam_id == args.camera]
    # Safety: only auto-snapshot when the user is running a targeted diagnostic
    # (single camera, or --limit ≤ 3). Otherwise skip to not block a 25-cam run.
    if not args.camera and (args.limit is None or args.limit > 3):
        log.info("skipping per-camera debug snapshots (full run; use --camera or --limit ≤ 3)")
        return

    debug_script = Path(__file__).resolve().parent.parent / "tools" / "debug_detect_classify.py"
    resize = f"{args.mux_width}x{args.mux_height}"
    for c in cams:
        parsed = urlparse(c.uri)
        if parsed.scheme != "file":
            continue
        video_path = parsed.path
        if not Path(video_path).is_file():
            log.warning("debug snapshot: %s not found", video_path)
            continue
        sub_out = run_dir / f"debug_{c.cam_id}"
        log.info("debug snapshot (%s, resize=%s) -> %s", c.cam_id, resize, sub_out)
        r = subprocess.run(
            [sys.executable, str(debug_script), video_path,
             "--frames", "5", "--resize", resize,
             "--out-dir", str(sub_out)],
            capture_output=True, text=True,
        )
        (sub_out / "stdout.txt").write_text(r.stdout)
        if r.returncode != 0:
            (sub_out / "stderr.txt").write_text(r.stderr)
            log.warning("debug snapshot %s exited with code %d", c.cam_id, r.returncode)


def run(args: argparse.Namespace) -> None:
    # Auto-create runs/NN_main_<tag>/ unless user passed explicit --log-dir
    if args.log_dir is None:
        from tools.rv_run import new_run_dir, tee_stdout, write_meta
        tag_parts = []
        if args.camera: tag_parts.append(args.camera)
        if args.limit:  tag_parts.append(f"n{args.limit}")
        if args.display: tag_parts.append("disp")
        if os.environ.get("RV_DEBUG_PROBE") == "1": tag_parts.append("dbg")
        tag = "_".join(tag_parts) if tag_parts else "all"
        run_dir = new_run_dir(f"main_{tag}")
        print(f"[rv_run] writing artifacts to: {run_dir}", flush=True)
        tee_stdout(run_dir / "log.txt")
        write_meta(run_dir, {
            "kind":       "main_pipeline",
            "cameras":    args.cameras,
            "camera":     args.camera or "",
            "limit":      args.limit or "",
            "sgie":       args.sgie,
            "tracker":    args.tracker,
            "mux":        f"{args.mux_width}x{args.mux_height}",
            "display":    args.display,
            "debug_mode": os.environ.get("RV_DEBUG_PROBE", "0"),
        })
        args.log_dir = str(run_dir)

    diag.setup_logging(level=args.log_level,
                       log_dir=Path(args.log_dir))
    log = diag.get_logger("rv.main")

    # Dump per-camera debug snapshots (frames + crops) to run_dir/debug_<cam>/.
    # Only for file:// sources — RTSP would require pulling a separate capture.
    if args.log_dir and not args.display:
        _dump_debug_snapshots(args, Path(args.log_dir), log)

    cameras = cfg.load_cameras(args.cameras, limit=args.limit)
    if args.camera:
        cameras = [c for c in cameras if c.cam_id == args.camera]
        if not cameras:
            log.error("camera %s not found in %s", args.camera, args.cameras)
            sys.exit(1)
    log.info("loaded %d cameras from %s", len(cameras), args.cameras)
    for c in cameras[:3]:
        log.info("  %s -> %s", c.cam_id, c.uri[:80])
    if len(cameras) > 3:
        log.info("  ... and %d more", len(cameras) - 3)

    plugin = RVPlugin.load(Path(args.plugin))
    log.info("plugin loaded: %s", args.plugin)

    sgie    = Path(args.sgie)    if (args.sgie    and args.sgie.strip())    else None
    tracker = Path(args.tracker) if (args.tracker and args.tracker.strip()) else None

    pipe, probe_op, shm_handle = build_pipeline(
        cameras            = cameras,
        nvinfer_config     = Path(args.config),
        plugin             = plugin,
        sgie_config        = sgie,
        tracker_config     = tracker,
        mux_width          = args.mux_width,
        mux_height         = args.mux_height,
        display            = args.display,
        display_width      = args.display_width,
        display_height     = args.display_height,
    )
    log.info("pipeline built (%d cameras, mux %dx%d, sgie=%s, tracker=%s, display=%s)",
             len(cameras), args.mux_width, args.mux_height,
             sgie.name    if sgie    else "off",
             tracker.name if tracker else "off",
             args.display)

    try:
        pipe.start().wait()
    except KeyboardInterrupt:
        log.warning("interrupted by user")
    finally:
        probe_op.report()
        if getattr(probe_op, "frame_saver", None):
            probe_op.frame_saver.close()
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
