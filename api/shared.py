"""
Race Vision — Shared state, config, logging, and singletons.

This module is the foundation imported by all other api.* modules.
It imports NOTHING from the api/ package to prevent circular dependencies.
"""

import sys
import time
import logging
import threading
import numpy as np
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Pipeline modules (always available — no torch dependency)
from pipeline.camera_manager import CameraManager
from pipeline.track_topology import TrackTopology
from pipeline.detections import CameraDetections
from pipeline.fusion import FusionEngine
from pipeline.shm_reader import SharedMemoryReader

# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_RTSP_URL = "rtsp://admin:Qaz445566@192.168.18.59:554//stream"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000

BROADCAST_INTERVAL = 0.20  # 5 Hz — rankings, activation_map, camera_result
LIVE_DET_INTERVAL = 0.05  # 20 Hz — live detections (bboxes) — fast path
MJPEG_QUALITY = 75
MJPEG_FPS = 25
TRACK_LENGTH = 2500

# ============================================================
# COLOR → HORSE MAPPING (matches frontend SILK_COLORS)
# ============================================================

COLOR_TO_HORSE = {
    "red":    {"id": "horse-1", "number": 1, "name": "Red Runner",     "silkId": 1, "color": "#DC2626", "jockeyName": "Jockey 1"},
    "green":  {"id": "horse-2", "number": 2, "name": "Green Flash",    "silkId": 2, "color": "#16A34A", "jockeyName": "Jockey 2"},
    "yellow": {"id": "horse-3", "number": 3, "name": "Yellow Thunder", "silkId": 3, "color": "#FBBF24", "jockeyName": "Jockey 3"},
}

ALL_COLORS = list(COLOR_TO_HORSE.keys())

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("race_server")

# ============================================================
# HEAVY IMPORTS — only needed in non-DeepStream mode (require torch/ultralytics)
# Deferred to avoid import errors when running in lightweight DeepStream container
# ============================================================

TriggerLoop = None
AnalysisLoop = None
_RaceTracker = None
_draw = None
_COLORS_BGR = None
_REQUIRED_COLORS = None
_CameraStream = None
_MultiCameraReader = None

def _load_heavy_imports():
    """Load torch/ultralytics-dependent modules. Call only in non-DeepStream mode."""
    global TriggerLoop, AnalysisLoop
    global _RaceTracker, _draw, _COLORS_BGR, _REQUIRED_COLORS
    global _CameraStream, _MultiCameraReader
    from pipeline.trigger import TriggerLoop as _TL
    from pipeline.analyzer import AnalysisLoop as _AL
    TriggerLoop = _TL
    AnalysisLoop = _AL
    from tools.test_race_count import (
        RaceTracker, draw, COLORS_BGR, REQUIRED_COLORS
    )
    _RaceTracker = RaceTracker
    _draw = draw
    _COLORS_BGR = COLORS_BGR
    _REQUIRED_COLORS = REQUIRED_COLORS
    from tools.ffmpeg_reader import CameraStream, MultiCameraReader
    _CameraStream = CameraStream
    _MultiCameraReader = MultiCameraReader

# ============================================================
# SHARED STATE (thread-safe)
# ============================================================

class SharedState:
    """Thread-safe state shared between all pipeline threads and the server."""

    def __init__(self):
        self._lock = threading.Lock()

        # Per-camera latest frames (cam_id → numpy array)
        self._frames: dict[str, np.ndarray] = {}

        # Per-camera annotated frames for MJPEG display
        self._display_frames: dict[str, np.ndarray] = {}

        # Current rankings (formatted for frontend)
        self.rankings: list = []

        # Race state
        self.race_active: bool = False
        self.detection_fps: float = 0.0
        self.detection_count: int = 0

    def set_frame(self, cam_id: str, frame: np.ndarray):
        """Store latest frame from a camera."""
        with self._lock:
            self._frames[cam_id] = frame

    def get_frame(self, cam_id: str) -> Optional[np.ndarray]:
        """Get latest frame (copy) from a camera."""
        with self._lock:
            frame = self._frames.get(cam_id)
            return frame.copy() if frame is not None else None

    def set_display_frame(self, cam_id: str, frame: np.ndarray):
        """Store annotated frame for MJPEG display."""
        with self._lock:
            self._display_frames[cam_id] = frame

    def get_display_frame(self, cam_id: str) -> Optional[np.ndarray]:
        """Get annotated display frame (copy)."""
        with self._lock:
            frame = self._display_frames.get(cam_id)
            return frame.copy() if frame is not None else None

    def set_rankings(self, rankings: list):
        with self._lock:
            if rankings:
                self.rankings = rankings

    def get_rankings(self) -> list:
        with self._lock:
            return list(self.rankings)


state = SharedState()

# ============================================================
# MULTI-CAMERA FRAME STORE
# ============================================================

class FrameStore:
    """Stores latest frames from all cameras for pipeline consumption.

    Acts as the frame_source callable for TriggerLoop and AnalysisLoop.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frames: dict[str, np.ndarray] = {}
        self._timestamps: dict[str, float] = {}

    def put(self, cam_id: str, frame: np.ndarray):
        with self._lock:
            self._frames[cam_id] = frame
            self._timestamps[cam_id] = time.monotonic()

    def get(self, cam_id: str) -> Optional[np.ndarray]:
        with self._lock:
            frame = self._frames.get(cam_id)
            return frame.copy() if frame is not None else None

    def get_age(self, cam_id: str) -> float:
        """Seconds since last frame update."""
        with self._lock:
            ts = self._timestamps.get(cam_id)
            return time.monotonic() - ts if ts else float('inf')


frame_store = FrameStore()
