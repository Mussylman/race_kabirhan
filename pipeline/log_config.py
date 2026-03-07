"""
log_config.py — Structured logging for Race Vision.

Two separate log streams:
    1. SYSTEM log  — infrastructure messages (startup, connections, errors, GStreamer state)
    2. DETECTION log — model results only (detections, colors, positions, completions)

Console output is minimal: only detection events + periodic summary.
Full system log goes to /tmp/race_system.log for debugging.

Usage:
    from pipeline.log_config import setup_logging, sys_log, det_log

    setup_logging()  # call once at startup

    sys_log.info("Pipeline started")           # → file only
    det_log.info("DET cam-01 3 horses ...")     # → console + file
"""

import logging
import os

# Log file paths
SYSTEM_LOG_FILE = os.environ.get("RV_SYSTEM_LOG", "/tmp/race_system.log")
DETECTION_LOG_FILE = os.environ.get("RV_DETECTION_LOG", "/tmp/race_detections.log")

# Logger instances
sys_log = logging.getLogger("rv.system")
det_log = logging.getLogger("rv.detection")

# DeepStream C++ log (system-level, file only)
ds_log = logging.getLogger("rv.deepstream")

_configured = False


def setup_logging(console_level: int = logging.INFO, verbose: bool = False):
    """Configure all Race Vision loggers.

    Args:
        console_level: minimum level for console output.
        verbose: if True, system log also goes to console (for debugging).
    """
    global _configured
    if _configured:
        return
    _configured = True

    # Prevent propagation to root logger
    for logger in (sys_log, det_log, ds_log):
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

    # ── Formatters ──────────────────────────────────────────────────

    # System: full timestamp + level + source
    sys_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Detection: clean, scannable format
    det_fmt = logging.Formatter(
        "%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console: compact
    console_fmt = logging.Formatter(
        "%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── File handlers ───────────────────────────────────────────────

    # System log file (all system messages)
    sys_fh = logging.FileHandler(SYSTEM_LOG_FILE, mode="a")
    sys_fh.setLevel(logging.DEBUG)
    sys_fh.setFormatter(sys_fmt)
    sys_log.addHandler(sys_fh)

    # Detection log file
    det_fh = logging.FileHandler(DETECTION_LOG_FILE, mode="a")
    det_fh.setLevel(logging.DEBUG)
    det_fh.setFormatter(det_fmt)
    det_log.addHandler(det_fh)

    # DeepStream C++ → system log file
    ds_fh = logging.FileHandler(SYSTEM_LOG_FILE, mode="a")
    ds_fh.setLevel(logging.DEBUG)
    ds_fh.setFormatter(sys_fmt)
    ds_log.addHandler(ds_fh)

    # ── Console handlers ────────────────────────────────────────────

    # Detection log → always on console (this is what matters)
    det_ch = logging.StreamHandler()
    det_ch.setLevel(console_level)
    det_ch.setFormatter(console_fmt)
    det_log.addHandler(det_ch)

    # System log → console only if verbose or ERROR+
    sys_ch = logging.StreamHandler()
    sys_ch.setLevel(logging.DEBUG if verbose else logging.WARNING)
    sys_ch.setFormatter(console_fmt)
    sys_log.addHandler(sys_ch)

    # DeepStream C++ → console only if verbose
    if verbose:
        ds_ch = logging.StreamHandler()
        ds_ch.setLevel(logging.DEBUG)
        ds_ch.setFormatter(console_fmt)
        ds_log.addHandler(ds_ch)

    # ── Startup marker ──────────────────────────────────────────────

    sys_log.info("=" * 60)
    sys_log.info("Race Vision logging started")
    sys_log.info("  System log:    %s", SYSTEM_LOG_FILE)
    sys_log.info("  Detection log: %s", DETECTION_LOG_FILE)
    sys_log.info("=" * 60)

    det_log.info("--- Race Vision started ---")
    det_log.info("  Logs: system → %s | detections → %s", SYSTEM_LOG_FILE, DETECTION_LOG_FILE)


def setup_legacy_logging():
    """Configure the old-style loggers (pipeline.*) to route to system log file.

    This makes existing `log = logging.getLogger("pipeline.xxx")` statements
    write to the system log file instead of cluttering the console.
    """
    pipeline_logger = logging.getLogger("pipeline")
    pipeline_logger.setLevel(logging.DEBUG)
    pipeline_logger.propagate = False

    sys_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # File handler — same system log
    fh = logging.FileHandler(SYSTEM_LOG_FILE, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(sys_fmt)
    pipeline_logger.addHandler(fh)

    # Console — only warnings+
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(sys_fmt)
    pipeline_logger.addHandler(ch)
