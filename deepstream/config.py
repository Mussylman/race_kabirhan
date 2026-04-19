"""
deepstream/config.py — Centralised paths and runtime defaults.

Replaces the C++ PipelineConfig struct from the old `pipeline.h`. All
hard-coded paths and tuning knobs live here so changing them does not
require touching pipeline.py.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Paths ───────────────────────────────────────────────────────────
DEEPSTREAM_DIR     = REPO_ROOT / "deepstream"
CONFIGS_DIR        = DEEPSTREAM_DIR / "configs"
PLUGIN_LIB_PATH    = DEEPSTREAM_DIR / "build" / "libnvdsinfer_racevision.so"

DEFAULT_PGIE       = CONFIGS_DIR / "nvinfer_racevision.txt"
DEFAULT_SGIE       = CONFIGS_DIR / "sgie_color.txt"
DEFAULT_TRACKER    = CONFIGS_DIR / "tracker_iou.yml"
DEFAULT_CAMERAS    = REPO_ROOT / "configs" / "cameras_live.json"
TEST_CAMERAS_FILES = REPO_ROOT / "configs" / "cameras_test_files.json"

# nvtracker low-level library — shipped with DeepStream 9.0
TRACKER_LL_LIB_FILE = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"

# ── Pipeline tuning ────────────────────────────────────────────────
DEFAULT_MUX_WIDTH              = 1280
DEFAULT_MUX_HEIGHT             = 720
DEFAULT_BATCHED_PUSH_TIMEOUT   = 40_000   # microseconds


@dataclass
class CameraEntry:
    cam_id:      str
    uri:         str
    track_start: float = 0.0
    track_end:   float = 100.0


@dataclass
class PipelineSettings:
    """Everything pipeline.build_pipeline needs at construction time."""
    cameras:        list[CameraEntry]
    pgie_config:    Path
    sgie_config:    Optional[Path]                          = None
    plugin_lib:     Path                                    = PLUGIN_LIB_PATH
    mux_width:      int                                     = DEFAULT_MUX_WIDTH
    mux_height:     int                                     = DEFAULT_MUX_HEIGHT
    batch_timeout:  int                                     = DEFAULT_BATCHED_PUSH_TIMEOUT
    log_dir:        Optional[Path]                          = None
    extra:          dict                                    = field(default_factory=dict)


def load_cameras(config_path: Path | str, limit: Optional[int] = None) -> list[CameraEntry]:
    """Load cameras_*.json. Accepts both a list or {"analytics": [...]}."""
    config_path = Path(config_path)
    with open(config_path) as fp:
        cfg = json.load(fp)
    raw = cfg.get("analytics", cfg if isinstance(cfg, list) else [])

    cams: list[CameraEntry] = []
    for entry in raw:
        url = entry["url"]
        if not (url.startswith("rtsp://") or url.startswith("file://")
                or url.startswith("http://")):
            url = f"file://{os.path.abspath(url)}"
        cams.append(CameraEntry(
            cam_id      = entry["id"],
            uri         = url,
            track_start = float(entry.get("track_start", 0)),
            track_end   = float(entry.get("track_end", 100)),
        ))
    if limit is not None:
        cams = cams[:limit]
    return cams
