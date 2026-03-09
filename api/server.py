"""
Race Vision — FastAPI Backend Server (Multi-Camera)

Bridges the multi-camera detection pipeline with the React frontend
via WebSocket (ranking updates) and MJPEG (display camera streams).

Architecture (4 layers):
    Layer 0 — DECODE:
        25 analytics cameras → MultiCameraReader (ffmpeg decode)
        4 display cameras    → separate CameraStreams (MJPEG passthrough)

    Layer 1 — TRIGGER:
        TriggerLoop thread → YOLOv8n @ 640px on all 25 cameras
        → updates CameraManager.active_cameras

    Layer 2 — ANALYZE:
        AnalysisLoop thread → YOLOv8s @ 800px + ColorCNN on active cameras
        → per-camera detections → FusionEngine

    Layer 3 — FUSION:
        FusionEngine → global track positions → rankings
        → WebSocket broadcast

    Display:
        4 display CameraStreams → MJPEG /stream/cam{1-4}

Usage:
    # Multi-camera with RTSP config
    python -m api.server --config cameras.json

    # Local video files (simulate multi-camera)
    python -m api.server --video data/videos/exp10_cam1.mp4 data/videos/exp10_cam2.mp4

    # Single RTSP (legacy mode — still works)
    python -m api.server --url "rtsp://admin:pass@ip:554/stream"
"""

import os
import cv2
import sys
import time
import json
import signal
import asyncio
import logging
import argparse
import threading
import subprocess
import numpy as np
import urllib.request
from pathlib import Path
from typing import Optional
from collections import Counter

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

# Pipeline modules (always available — no torch dependency)
from pipeline.camera_manager import CameraManager
from pipeline.track_topology import TrackTopology
from pipeline.detections import CameraDetections
from pipeline.fusion import FusionEngine
from pipeline.shm_reader import SharedMemoryReader

# Heavy imports — only needed in non-DeepStream mode (require torch/ultralytics)
# Deferred to avoid import errors when running in lightweight DeepStream container
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
# CONFIGURATION
# ============================================================

DEFAULT_RTSP_URL = "rtsp://admin:Qaz445566@192.168.18.59:554//stream"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000

BROADCAST_INTERVAL = 0.20  # 5 Hz WebSocket broadcast
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

# ============================================================
# VIDEO FILE GRABBER (simulates multi-camera from video files)
# ============================================================

class VideoFileGrabber(threading.Thread):
    """Reads local video files and puts frames into FrameStore.

    Each video simulates a separate camera. Plays at native FPS.
    """

    def __init__(self, cam_video_map: dict[str, str]):
        """
        Args:
            cam_video_map: {cam_id: video_path}
        """
        super().__init__(daemon=True, name="VideoFileGrabber")
        self.cam_video_map = cam_video_map
        self.running = False

    def run(self):
        self.running = True

        while self.running:
            threads = []
            for cam_id, video_path in self.cam_video_map.items():
                t = threading.Thread(
                    target=self._play_video,
                    args=(cam_id, video_path),
                    daemon=True,
                )
                threads.append(t)
                t.start()

            # Wait for all to finish (video end)
            for t in threads:
                t.join()

            if self.running:
                log.info("All videos finished, looping...")

    def _play_video(self, cam_id: str, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log.error("Cannot open video: %s", video_path)
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        delay = 1.0 / fps
        name = Path(video_path).stem
        log.info("Playing %s as %s @ %.0f fps", name, cam_id, fps)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            frame_store.put(cam_id, frame)
            state.set_frame(cam_id, frame)
            time.sleep(delay)

        cap.release()
        log.info("%s (%s) ended", cam_id, name)

    def stop(self):
        self.running = False


# ============================================================
# RTSP MULTI-CAMERA GRABBER
# ============================================================

class RTSPGrabber:
    """Manages MultiCameraReader for RTSP analytics cameras."""

    def __init__(self, camera_manager: CameraManager, gpu: bool = False):
        self.camera_manager = camera_manager
        self._reader = _MultiCameraReader(gpu=gpu)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        # Add all analytics cameras to reader
        for cam in self.camera_manager.get_analytics_cameras():
            self._reader.add(
                cam.cam_id,
                cam.source,
                on_frame=self._on_frame,
                on_disconnect=self._on_disconnect,
                on_reconnect=self._on_reconnect,
            )

        self._reader.start_all()
        self._running = True
        log.info("RTSP grabber started (%d cameras)", len(self._reader.cameras))

    def _on_frame(self, cam_id: str, frame: np.ndarray):
        frame_store.put(cam_id, frame)
        state.set_frame(cam_id, frame)
        self.camera_manager.set_connected(cam_id, True)

    def _on_disconnect(self, cam_id: str, error: str):
        log.warning("RTSP %s disconnected: %s", cam_id, error)
        self.camera_manager.set_connected(cam_id, False)

    def _on_reconnect(self, cam_id: str, info: dict):
        log.info("RTSP %s connected: %dx%d %s",
                 cam_id, info['width'], info['height'], info['codec'].upper())
        self.camera_manager.set_connected(cam_id, True)

    def stop(self):
        self._running = False
        self._reader.stop_all()


# ============================================================
# LEGACY SINGLE-CAMERA MODE
# ============================================================

class LegacyDetectionLoop(threading.Thread):
    """Single-camera detection loop (backward-compatible).

    Used when --url or single --video is passed. Uses the original
    RaceTracker + 4-layer filtering from old server.py.
    """

    CONF_THRESHOLD = 0.75
    HSV_SKIP_CONF = 0.92
    MAX_SPEED_MPS = 120.0
    TEMPORAL_WINDOW = 5
    TEMPORAL_MIN = 2
    CAMERA_TRACK_M = 100.0
    DETECTION_INTERVAL = 0.10

    def __init__(self, cam_id: str = "cam-01"):
        super().__init__(daemon=True, name="LegacyDetection")
        self.cam_id = cam_id
        self.running = False
        self.tracker = None

        self._current_order: list = []
        self._smooth_x: dict = {}
        self._smooth_alpha = 0.12
        self._det_frames: dict = {}
        self._last_pos: dict = {}
        self._video_votes: list = []
        self._order_changes = 0
        self._total_frames = 0
        self._start_time = 0.0
        self._filter_stats = {'total': 0, 'f1_low_conf': 0, 'f2_hsv_mismatch': 0,
                              'f3_speed': 0, 'f4_temporal': 0, 'accepted': 0}

    def run(self):
        self.running = True
        self._start_time = time.time()

        output_dir = Path("results/race_server")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = _RaceTracker(output_dir, save_crops=False)
        log.info("Legacy detection pipeline ready")

        fps_counter = 0
        fps_timer = time.time()
        current_fps = 0.0

        while self.running:
            frame = frame_store.get(self.cam_id)
            if frame is None:
                time.sleep(0.05)
                continue

            if not state.race_active:
                rankings = self._build_rankings(frame.shape[1])
                state.set_rankings(rankings)
                state.set_display_frame(self.cam_id, frame.copy())
                time.sleep(0.1)
                continue

            t0 = time.time()
            jockeys, detections = self.tracker.update(frame)
            self._total_frames += 1
            frame_width = frame.shape[1]

            annotated = _draw(frame.copy(), jockeys, self.tracker)
            state.set_display_frame(self.cam_id, annotated)

            for j in jockeys:
                color = j['color']
                cx = float(j['center_x'])
                if color in self._smooth_x:
                    self._smooth_x[color] += self._smooth_alpha * (cx - self._smooth_x[color])
                else:
                    self._smooth_x[color] = cx

            # 4-layer filtering
            filtered_dets = []
            for det in detections:
                color = det['color']
                conf = float(det.get('conf', 0.0))
                hsv = det.get('hsv_guess', '')
                cx = float(det['center_x'])
                pos_m = (cx / max(frame_width, 1)) * self.CAMERA_TRACK_M
                self._filter_stats['total'] += 1

                if conf < self.CONF_THRESHOLD:
                    self._filter_stats['f1_low_conf'] += 1
                    continue
                if hsv and hsv != color and conf < self.HSV_SKIP_CONF:
                    self._filter_stats['f2_hsv_mismatch'] += 1
                    continue
                if color in self._last_pos:
                    last_pos_m, last_time = self._last_pos[color]
                    dt_sec = time.time() - last_time
                    if dt_sec > 0.01:
                        speed = abs(pos_m - last_pos_m) / dt_sec
                        if speed > self.MAX_SPEED_MPS:
                            self._filter_stats['f3_speed'] += 1
                            continue

                self._last_pos[color] = (pos_m, time.time())
                if color not in self._det_frames:
                    self._det_frames[color] = []
                self._det_frames[color].append(self._total_frames)
                if len(self._det_frames[color]) > self.TEMPORAL_WINDOW:
                    self._det_frames[color] = self._det_frames[color][-self.TEMPORAL_WINDOW:]

                recent_count = sum(
                    1 for f in self._det_frames[color]
                    if f > self._total_frames - self.TEMPORAL_WINDOW
                )
                if recent_count < self.TEMPORAL_MIN:
                    self._filter_stats['f4_temporal'] += 1
                    continue

                self._filter_stats['accepted'] += 1
                filtered_dets.append(det)

            filtered_colors = set(d['color'] for d in filtered_dets)
            if len(filtered_colors) >= 4:
                sorted_dets = sorted(filtered_dets, key=lambda d: -d['center_x'])
                seen = set()
                order = []
                for d in sorted_dets:
                    if d['color'] not in seen:
                        seen.add(d['color'])
                        order.append(d['color'])
                for c in set(ALL_COLORS) - seen:
                    order.append(c)
                self._video_votes.append(tuple(order))

            # Pick best order from votes
            if self._video_votes:
                vote_counts = Counter(self._video_votes)
                best_order, _ = vote_counts.most_common(1)[0]
                if list(best_order) != self._current_order:
                    self._current_order = list(best_order)
                    self._order_changes += 1

            rankings = self._build_rankings(frame_width)
            state.set_rankings(rankings)

            fps_counter += 1
            elapsed_fps = time.time() - fps_timer
            if elapsed_fps >= 1.0:
                current_fps = fps_counter / elapsed_fps
                fps_counter = 0
                fps_timer = time.time()
                state.detection_fps = current_fps

            dt = time.time() - t0
            sleep_time = self.DETECTION_INTERVAL - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _build_rankings(self, frame_width: int) -> list:
        if not self._current_order:
            return []

        rankings = []
        for pos, color_name in enumerate(self._current_order):
            horse_info = COLOR_TO_HORSE.get(color_name)
            if not horse_info:
                continue

            sx = self._smooth_x.get(color_name, frame_width * (1.0 - pos * 0.15))
            distance = (float(sx) / max(frame_width, 1)) * TRACK_LENGTH

            leader_x = self._smooth_x.get(self._current_order[0], float(frame_width))
            gap_px = float(leader_x) - float(sx)
            gap_seconds = abs(gap_px) / max(frame_width, 1) * 8.0

            rankings.append({
                "id": horse_info["id"],
                "number": int(horse_info["number"]),
                "name": horse_info["name"],
                "color": horse_info["color"],
                "jockeyName": horse_info["jockeyName"],
                "silkId": int(horse_info["silkId"]),
                "position": pos + 1,
                "distanceCovered": round(float(distance), 1),
                "currentLap": 1,
                "timeElapsed": 0,
                "speed": 0,
                "gapToLeader": round(float(gap_seconds), 2),
                "lastCameraId": "cam-1",
            })

        return rankings

    def stop(self):
        self.running = False
        if self.tracker:
            self.tracker.close()


# ============================================================
# MULTI-CAMERA PIPELINE ORCHESTRATOR
# ============================================================

class MultiCameraPipeline:
    """Orchestrates TriggerLoop + AnalysisLoop + FusionEngine for 25+ cameras."""

    def __init__(self, camera_manager: CameraManager, topology: TrackTopology):
        self.camera_manager = camera_manager
        self.topology = topology
        self.fusion = FusionEngine(topology, colors=ALL_COLORS)

        self._trigger: Optional[TriggerLoop] = None
        self._analyzer: Optional[AnalysisLoop] = None

    def start(self):
        # Start trigger loop (YOLOv8n on all analytics cameras)
        self._trigger = TriggerLoop(
            camera_manager=self.camera_manager,
            frame_source=frame_store.get,
            trigger_fps=3.0,
            fallback_pt="yolov8n.pt",
            imgsz=640,
            conf=0.25,
        )
        self._trigger.start()
        log.info("TriggerLoop started")

        # Start analysis loop (YOLOv8s + ColorCNN on active cameras)
        self._analyzer = AnalysisLoop(
            camera_manager=self.camera_manager,
            frame_source=frame_store.get,
            on_result=self._on_analysis_result,
            on_annotated_frame=lambda cam_id, frame: state.set_display_frame(cam_id, frame),
            analysis_fps=4.0,
            yolo_fallback="yolov8s.pt",
            classifier_fallback="models/color_classifier.pt",
            imgsz=800,
        )
        self._analyzer.start()
        log.info("AnalysisLoop started")

    def _on_analysis_result(self, cam_results: list[CameraDetections]):
        """Called by AnalysisLoop when new detections arrive."""
        if not state.race_active:
            return

        # Feed into fusion engine
        self.fusion.update(cam_results)

        # Build rankings from fusion output
        fusion_ranking = self.fusion.get_ranking()
        rankings = self._build_rankings(fusion_ranking)
        state.set_rankings(rankings)

    def _build_rankings(self, fusion_ranking: list[dict]) -> list:
        """Convert fusion ranking to frontend format."""
        rankings = []
        for entry in fusion_ranking:
            color = entry["color"]
            horse_info = COLOR_TO_HORSE.get(color)
            if not horse_info:
                continue

            distance = entry.get("position_m", 0)
            rank = entry.get("rank", 0)

            # Compute gap to leader
            if rankings:
                leader_dist = rankings[0]["distanceCovered"]
                gap = abs(leader_dist - distance) / max(TRACK_LENGTH, 1) * 60.0
            else:
                gap = 0.0

            rankings.append({
                "id": horse_info["id"],
                "number": int(horse_info["number"]),
                "name": horse_info["name"],
                "color": horse_info["color"],
                "jockeyName": horse_info["jockeyName"],
                "silkId": int(horse_info["silkId"]),
                "position": rank,
                "distanceCovered": round(float(distance), 1),
                "currentLap": 1,
                "timeElapsed": 0,
                "speed": round(entry.get("speed_mps", 0) * 3.6, 1),  # m/s → km/h
                "gapToLeader": round(float(gap), 2),
                "lastCameraId": entry.get("last_camera", ""),
            })

        return rankings

    def stop(self):
        if self._trigger:
            self._trigger.stop()
        if self._analyzer:
            self._analyzer.stop()

    def reset(self):
        """Reset for new race."""
        self.fusion.reset()
        if self._analyzer:
            self._analyzer.reset_votes()

    def get_stats(self) -> dict:
        stats = {"fusion": self.fusion.get_stats()}
        if self._trigger:
            stats["trigger"] = self._trigger.get_stats()
        if self._analyzer:
            stats["analyzer"] = self._analyzer.get_stats()
        return stats


# ============================================================
# DEEPSTREAM C++ SUBPROCESS MANAGER
# ============================================================

class DeepStreamSubprocess:
    """Manages the C++ DeepStream process as a subprocess."""

    def __init__(self, config_path: str,
                 yolo_engine: str = "/app/models/yolov8s_deepstream.engine",
                 color_engine: str = "/app/models/color_classifier.engine",
                 binary: str = "/app/bin/race_vision_deepstream",
                 file_mode: bool = False,
                 display: bool = False):
        self.config_path = config_path
        self.yolo_engine = yolo_engine
        self.color_engine = color_engine
        self.binary = binary
        self.file_mode = file_mode
        self.display = display
        self._proc: Optional[subprocess.Popen] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        cmd = [
            self.binary,
            "--config", self.config_path,
            "--yolo-engine", self.yolo_engine,
            "--color-engine", self.color_engine,
        ]
        if self.file_mode:
            cmd.append("--file-mode")
        if self.display:
            cmd.append("--display")
        log.info("Starting DeepStream C++: %s", " ".join(cmd))

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = (
            "/opt/nvidia/deepstream/deepstream/lib:"
            "/app/lib:"
            + env.get("LD_LIBRARY_PATH", "")
        )

        self._proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        self._running = True

        self._monitor_thread = threading.Thread(
            target=self._monitor, daemon=True, name="DeepStreamMonitor")
        self._monitor_thread.start()
        log.info("DeepStream C++ started (PID=%d)", self._proc.pid)

    def _monitor(self):
        """Forward C++ stdout/stderr to Python log and detect crashes."""
        for line in iter(self._proc.stdout.readline, b''):
            if not self._running:
                break
            text = line.decode('utf-8', errors='replace').rstrip()
            if text:
                log.info("[DeepStream C++] %s", text)

        retcode = self._proc.wait()
        if self._running:
            log.error("DeepStream C++ exited unexpectedly (code=%d)", retcode)

    def stop(self):
        self._running = False
        if self._proc and self._proc.poll() is None:
            log.info("Stopping DeepStream C++ (PID=%d)...", self._proc.pid)
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                log.warning("DeepStream C++ did not stop, killing...")
                self._proc.kill()
                self._proc.wait()
            log.info("DeepStream C++ stopped")

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc else None


# ============================================================
# DEEPSTREAM PIPELINE (replaces MultiCameraPipeline when --deepstream)
# ============================================================

class DeepStreamPipeline:
    """Reads detections from DeepStream C++ via shared memory.

    Replaces TriggerLoop + AnalysisLoop with a single SharedMemoryReader
    that receives pre-computed detections (YOLO + color) from the C++ pipeline.
    Keeps VoteEngine per camera and FusionEngine for ranking.
    """

    def __init__(self, camera_manager: CameraManager, topology: TrackTopology):
        self.camera_manager = camera_manager
        self.topology = topology
        self.fusion = FusionEngine(topology, colors=ALL_COLORS)

        self._reader = SharedMemoryReader(timeout_ms=200)
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Per-camera vote engines
        from pipeline.vote_engine import VoteEngine
        self._vote_engines: dict[str, VoteEngine] = {}

        # Frame skip: ~25fps from DeepStream → ~4fps effective
        self._cam_frame_count: dict[str, int] = {}
        self.frame_skip: int = 6

        # One-result-per-camera state
        self._cam_first_analysis: dict[str, float] = {}
        self._cam_all_visible_time: dict[str, float] = {}
        self._cam_completed: set[str] = set()
        self._pending_camera_results: list[dict] = []
        self.max_analysis_sec: float = 8.0
        self.grace_period_sec: float = 2.0

        # Stats
        self.frames_processed = 0
        self.cycles = 0
        self.last_cycle_time = 0.0
        self.current_fps = 0.0

        # Live detection status: {cam_id: n_detections} — updated every SHM read
        self.live_detections: dict[str, list] = {}  # {cam_id: [{color, conf}, ...]}

    def _get_vote_engine(self, cam_id: str):
        from pipeline.vote_engine import VoteEngine
        if cam_id not in self._vote_engines:
            self._vote_engines[cam_id] = VoteEngine(ALL_COLORS)
        return self._vote_engines[cam_id]

    def start(self):
        # Attach to shared memory (retry until DeepStream C++ is ready)
        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="DeepStreamReader"
        )
        self._thread.start()
        log.info("DeepStreamPipeline started (waiting for SHM)")

    def _reader_loop(self):
        # Retry attach until successful
        while self._running and not self._reader.is_attached:
            if self._reader.attach():
                log.info("DeepStreamPipeline attached to shared memory")
                break
            log.info("Waiting for DeepStream C++ process (SHM not found)...")
            time.sleep(2.0)

        fps_counter = 0
        fps_timer = time.monotonic()
        stale_counter = 0
        STALE_THRESHOLD = 30  # ~6 sec at 200ms timeout

        while self._running:
            t0 = time.monotonic()

            # Read detections from shared memory
            cam_results = self._reader.read()
            if cam_results is None:
                stale_counter += 1
                if self._reader.is_attached and stale_counter > STALE_THRESHOLD:
                    log.warning("SHM stale (%d timeouts), re-attaching...", stale_counter)
                    self._reader.detach()
                    stale_counter = 0
                if not self._reader.is_attached:
                    while self._running and not self._reader.is_attached:
                        if self._reader.attach():
                            log.info("Re-attached to SHM")
                            break
                        time.sleep(2.0)
                continue
            stale_counter = 0

            self.cycles += 1

            # Update live detection map with colors (regardless of race_active)
            live = {}
            for cam_det in cam_results:
                if cam_det.n_detections > 0:
                    live[cam_det.cam_id] = [
                        {
                            "color": d.get("color", "?"),
                            "conf": round(d.get("color_conf", 0) * 100),
                            "track_id": d.get("track_id", 0),
                        }
                        for d in cam_det.detections
                    ]
            self.live_detections = live

            if not state.race_active:
                continue

            # Process through vote engines with frame skip + one-result-per-camera
            for cam_det in cam_results:
                cam_id = cam_det.cam_id

                # Skip completed cameras
                if cam_id in self._cam_completed:
                    continue

                if cam_det.n_detections == 0:
                    continue

                # Frame skip: ~25fps → ~4fps effective
                self._cam_frame_count[cam_id] = self._cam_frame_count.get(cam_id, 0) + 1
                if self._cam_frame_count[cam_id] % self.frame_skip != 0:
                    continue

                engine = self._get_vote_engine(cam_id)
                assigned, weight = engine.submit_frame(cam_det.detections)
                self.frames_processed += 1

                # Track first analysis time
                now = time.monotonic()
                if cam_id not in self._cam_first_analysis:
                    self._cam_first_analysis[cam_id] = now

                # Track when all 5 colors become visible
                if weight > 0:
                    visible_colors = set(d['color'] for d in assigned)
                    if len(visible_colors) >= 5 and cam_id not in self._cam_all_visible_time:
                        self._cam_all_visible_time[cam_id] = now

                # Check completion conditions
                should_complete = False
                reason = ""

                if engine.is_result_ready():
                    # Condition 1: all positions filled + grace period
                    if cam_id in self._cam_all_visible_time:
                        elapsed_grace = now - self._cam_all_visible_time[cam_id]
                        if elapsed_grace >= self.grace_period_sec:
                            should_complete = True
                            reason = f"confident + grace {elapsed_grace:.1f}s"
                    # Condition 2: confident + enough frames (no need to wait)
                    if not should_complete and engine.vote_frames >= 8:
                        should_complete = True
                        reason = f"confident + {engine.vote_frames} frames"

                if not should_complete and engine.vote_frames >= 2:
                    # Condition 3: timeout with votes
                    elapsed = now - self._cam_first_analysis[cam_id]
                    if elapsed >= self.max_analysis_sec:
                        should_complete = True
                        reason = f"timeout ({elapsed:.1f}s)"

                if not should_complete and cam_id in self._cam_first_analysis:
                    # Condition 4: partial timeout — straggler case (few horses, no votes)
                    # Still send partial data to fusion so it knows horse positions
                    elapsed = now - self._cam_first_analysis[cam_id]
                    if elapsed >= self.max_analysis_sec:
                        should_complete = True
                        reason = f"partial timeout ({elapsed:.1f}s, {len(assigned)} horses)"

                if should_complete:
                    self._cam_completed.add(cam_id)
                    vote_result = engine.compute_result()
                    log.info("CAMERA COMPLETE  %s  order=[%s]  (%s, %d vote frames)",
                             cam_id, " > ".join(vote_result), reason, engine.vote_frames)

                    # Build single-camera CameraDetections and update fusion ONCE
                    voted = CameraDetections(cam_id, cam_det.frame_width, cam_det.frame_height)
                    voted.timestamp = cam_det.timestamp
                    for d in assigned:
                        voted.add(d)
                    self.fusion.update([voted])

                    # Build rankings and update state
                    fusion_ranking = self.fusion.get_ranking()
                    rankings = self._build_rankings(fusion_ranking)
                    state.set_rankings(rankings)

                    # Queue camera_result for broadcast
                    self._pending_camera_results.append({
                        "type": "camera_result",
                        "camera_id": cam_id,
                        "ranking": vote_result,
                        "vote_frames": engine.vote_frames,
                    })

            # FPS tracking
            fps_counter += 1
            elapsed_fps = time.monotonic() - fps_timer
            if elapsed_fps >= 1.0:
                self.current_fps = fps_counter / elapsed_fps
                fps_counter = 0
                fps_timer = time.monotonic()

            self.last_cycle_time = time.monotonic() - t0

    def _build_rankings(self, fusion_ranking: list[dict]) -> list:
        """Convert fusion ranking to frontend format (same as MultiCameraPipeline)."""
        rankings = []
        for entry in fusion_ranking:
            color = entry["color"]
            horse_info = COLOR_TO_HORSE.get(color)
            if not horse_info:
                continue

            distance = entry.get("position_m", 0)
            rank = entry.get("rank", 0)

            if rankings:
                leader_dist = rankings[0]["distanceCovered"]
                gap = abs(leader_dist - distance) / max(TRACK_LENGTH, 1) * 60.0
            else:
                gap = 0.0

            rankings.append({
                "id": horse_info["id"],
                "number": int(horse_info["number"]),
                "name": horse_info["name"],
                "color": horse_info["color"],
                "jockeyName": horse_info["jockeyName"],
                "silkId": int(horse_info["silkId"]),
                "position": rank,
                "distanceCovered": round(float(distance), 1),
                "currentLap": 1,
                "timeElapsed": 0,
                "speed": round(entry.get("speed_mps", 0) * 3.6, 1),
                "gapToLeader": round(float(gap), 2),
                "lastCameraId": entry.get("last_camera", ""),
            })

        return rankings

    def stop(self):
        self._running = False
        self._reader.detach()

    def reset(self):
        self.fusion.reset()
        for engine in self._vote_engines.values():
            engine.reset()
        self._cam_first_analysis.clear()
        self._cam_all_visible_time.clear()
        self._cam_completed.clear()
        self._cam_frame_count.clear()
        self._pending_camera_results.clear()

    def get_stats(self) -> dict:
        return {
            "deepstream": {
                "shm_attached": self._reader.is_attached,
                "shm_seq": self._reader.last_seq,
                "cycles": self.cycles,
                "frames_processed": self.frames_processed,
                "current_fps": round(self.current_fps, 1),
                "last_cycle_ms": round(self.last_cycle_time * 1000, 1),
            },
            "fusion": self.fusion.get_stats(),
        }


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="Race Vision Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ws_clients: set[WebSocket] = set()


# ============================================================
# GO2RTC STREAM MONITOR
# ============================================================

class Go2RTCMonitor:
    """Periodically polls go2rtc REST API and logs stream health.

    go2rtc /api/streams returns per-stream info:
    {
      "cam-01": {
        "producers": [{"url": "rtsp://...", "recv": <bytes>, ...}],
        "consumers": [...]
      }
    }
    A stream with no producers = RTSP source disconnected.
    """

    POLL_INTERVAL = 30  # seconds between polls
    LOG_INTERVAL = 60   # seconds between full status logs

    def __init__(self, go2rtc_url: str):
        self._url = go2rtc_url.rstrip("/")
        self._api_url = f"{self._url}/api/streams"
        self._running = False
        self._task: Optional[asyncio.Task] = None
        # Per-stream state: {stream_id: {"online": bool, "recv": int, "last_change": float}}
        self._streams: dict[str, dict] = {}
        self._last_log = 0.0
        self._lock = threading.Lock()

    def start(self):
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        log.info("Go2RTC monitor started (%s)", self._api_url)

    def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    def get_status(self) -> dict:
        """Return current stream health for /api/stats."""
        with self._lock:
            online = sum(1 for s in self._streams.values() if s.get("online"))
            offline = sum(1 for s in self._streams.values() if not s.get("online"))
            offline_list = [sid for sid, s in self._streams.items() if not s.get("online")]
            return {
                "total": len(self._streams),
                "online": online,
                "offline": offline,
                "offline_streams": offline_list,
            }

    async def _poll_loop(self):
        # Initial delay — let go2rtc boot up
        await asyncio.sleep(5)
        while self._running:
            try:
                await self._poll_once()
            except Exception as e:
                log.warning("Go2RTC monitor poll failed: %s", e)
            await asyncio.sleep(self.POLL_INTERVAL)

    async def _poll_once(self):
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, self._fetch_streams)
        if data is None:
            return

        now = time.time()
        changes = []

        with self._lock:
            for stream_id, info in data.items():
                producers = info.get("producers") or []
                has_producer = len(producers) > 0
                recv_bytes = sum(p.get("recv", 0) for p in producers)

                prev = self._streams.get(stream_id)
                was_online = prev["online"] if prev else None

                self._streams[stream_id] = {
                    "online": has_producer,
                    "recv": recv_bytes,
                    "consumers": len(info.get("consumers") or []),
                    "last_change": prev["last_change"] if prev else now,
                }

                # Detect state change
                if was_online is not None and has_producer != was_online:
                    self._streams[stream_id]["last_change"] = now
                    status = "ONLINE" if has_producer else "OFFLINE"
                    changes.append((stream_id, status))

        # Log state changes immediately
        for stream_id, status in changes:
            log.warning("go2rtc stream %s → %s", stream_id, status)

        # Periodic full status log
        if now - self._last_log >= self.LOG_INTERVAL:
            status = self.get_status()
            if status["offline"] > 0:
                log.warning("go2rtc: %d/%d online, offline: %s",
                            status["online"], status["total"],
                            ", ".join(status["offline_streams"]))
            else:
                log.info("go2rtc: %d/%d streams online", status["online"], status["total"])
            self._last_log = now

    def _fetch_streams(self) -> Optional[dict]:
        """Synchronous HTTP GET to go2rtc /api/streams."""
        try:
            req = urllib.request.Request(self._api_url, method="GET")
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return None


# Global pipeline references
_camera_manager: Optional[CameraManager] = None
_pipeline: Optional[MultiCameraPipeline] = None
_deepstream_subprocess: Optional[DeepStreamSubprocess] = None
_deepstream_pipeline: Optional[DeepStreamPipeline] = None
_legacy_detector: Optional[LegacyDetectionLoop] = None
_go2rtc_monitor: Optional[Go2RTCMonitor] = None

# ============================================================
# WEBSOCKET ENDPOINT
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.add(websocket)
    log.info(f"WebSocket client connected ({len(ws_clients)} total)")

    # Send horses_detected on connect
    horses_msg = {
        "type": "horses_detected",
        "horses": [
            {
                "id": info["id"],
                "number": info["number"],
                "name": info["name"],
                "color": info["color"],
                "jockeyName": info["jockeyName"],
                "silkId": info["silkId"],
            }
            for info in COLOR_TO_HORSE.values()
        ],
    }
    await websocket.send_json(horses_msg)

    # Send race_start if active
    if state.race_active:
        await websocket.send_json({
            "type": "race_start",
            "race": {
                "name": "Live Race",
                "totalLaps": 1,
                "trackLength": TRACK_LENGTH,
            },
        })

    # Send camera status if multi-camera
    if _camera_manager:
        await websocket.send_json({
            "type": "camera_status",
            **_camera_manager.get_status(),
        })

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg.get("type") == "get_state":
                rankings = state.get_rankings()
                resp = {
                    "type": "state",
                    "race": {
                        "name": "Live Race",
                        "totalLaps": 1,
                        "trackLength": TRACK_LENGTH,
                        "status": "active" if state.race_active else "pending",
                    },
                    "rankings": rankings,
                }
                if _camera_manager:
                    resp["cameras"] = _camera_manager.get_status()
                await websocket.send_json(resp)

            elif msg.get("type") == "start_race":
                state.race_active = True
                if _pipeline:
                    _pipeline.reset()
                if _deepstream_pipeline:
                    _deepstream_pipeline.reset()
                log.info("Race started (from operator)")
                await broadcast({
                    "type": "race_start",
                    "race": {
                        "name": "Live Race",
                        "totalLaps": 1,
                        "trackLength": TRACK_LENGTH,
                    },
                })

            elif msg.get("type") == "stop_race":
                state.race_active = False
                log.info("Race stopped (from operator)")
                await broadcast({"type": "race_stop"})

            elif msg.get("type") == "get_cameras":
                if _camera_manager:
                    await websocket.send_json({
                        "type": "camera_status",
                        **_camera_manager.get_status(),
                    })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error(f"WebSocket error: {e}")
    finally:
        ws_clients.discard(websocket)
        log.info(f"WebSocket client disconnected ({len(ws_clients)} total)")


async def broadcast(msg: dict):
    """Send to all connected WebSocket clients."""
    dead = set()
    for client in ws_clients:
        try:
            await client.send_json(msg)
        except Exception:
            dead.add(client)
    ws_clients.difference_update(dead)


# ============================================================
# REST ENDPOINTS
# ============================================================

from fastapi.responses import HTMLResponse
from starlette.responses import FileResponse


@app.get("/admin")
async def admin_panel():
    admin_path = Path(__file__).resolve().parent.parent / "admin" / "index.html"
    return FileResponse(admin_path, media_type="text/html")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Dashboard page with live camera streams and stats."""
    # Build camera grid
    cam_count = 4
    if _camera_manager:
        cams = _camera_manager.get_analytics_cameras()
        cam_count = len(cams) if cams else 4

    cam_cells = ""
    for i in range(1, cam_count + 1):
        cam_cells += f"""
        <div class="cam">
            <div class="cam-header">cam-{i:02d}</div>
            <img src="/stream/cam{i}" alt="Camera {i}">
        </div>"""

    return f"""<!DOCTYPE html>
<html><head>
<title>Race Vision</title>
<style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ background:#111; color:#eee; font-family:monospace; padding:16px; }}
    h1 {{ margin-bottom:12px; }}
    .info {{ background:#222; padding:12px; border-radius:8px; margin-bottom:16px; display:flex; gap:24px; flex-wrap:wrap; }}
    .info a {{ color:#4fc3f7; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(460px, 1fr)); gap:12px; }}
    .cam {{ background:#000; border-radius:8px; overflow:hidden; }}
    .cam img {{ width:100%; display:block; }}
    .cam-header {{ padding:6px 12px; background:#222; font-size:13px; }}
    #stats {{ background:#1a1a2e; padding:12px; border-radius:8px; margin-bottom:16px; white-space:pre; font-size:13px; }}
</style>
</head><body>
<h1>Race Vision — Multi-Camera Dashboard</h1>
<div class="info">
    <span>WebSocket: <a href="javascript:void(0)">ws://localhost:8000/ws</a></span>
    <span><a href="/api/stats">/api/stats</a></span>
    <span><a href="/api/cameras">/api/cameras</a></span>
</div>
<div id="stats">Loading stats...</div>
<div class="grid">{cam_cells}</div>
<script>
async function refreshStats() {{
    try {{
        const [stats, cams] = await Promise.all([
            fetch('/api/stats').then(r=>r.json()),
            fetch('/api/cameras').then(r=>r.json()),
        ]);
        const active = cams.active_analytics || 0;
        const total = cams.total_analytics || 0;
        const t = stats.trigger || {{}};
        const a = stats.analyzer || {{}};
        const f = stats.fusion || {{}};
        document.getElementById('stats').textContent =
            `Mode: ${{stats.mode}}  |  Cameras: ${{active}}/${{total}} active  |  ` +
            `Horses tracked: ${{f.horses_tracked || 0}}/5\\n` +
            `Trigger: ${{t.frames_processed||0}} frames, ${{t.last_batch_time_ms||0}}ms/batch  |  ` +
            `Analysis: ${{a.frames_processed||0}} frames @ ${{a.current_fps||0}} fps, ${{a.last_batch_time_ms||0}}ms/batch`;
    }} catch(e) {{}}
}}
refreshStats();
setInterval(refreshStats, 2000);
</script>
</body></html>"""


@app.get("/api/cameras")
async def get_cameras():
    """Camera status (activation map, connection status)."""
    if _camera_manager:
        return JSONResponse(_camera_manager.get_status())
    return JSONResponse({"total_analytics": 0, "cameras": []})


@app.get("/api/stats")
async def get_stats():
    """Pipeline performance stats."""
    if _deepstream_pipeline:
        stats = {"mode": "deepstream"}
        stats.update(_deepstream_pipeline.get_stats())
        if _deepstream_subprocess:
            stats["subprocess"] = {
                "pid": _deepstream_subprocess.pid,
                "running": _deepstream_subprocess.is_running,
            }
    elif _pipeline:
        stats = {"mode": "multi"}
        stats.update(_pipeline.get_stats())
    else:
        stats = {"mode": "legacy"}
    if _go2rtc_monitor:
        stats["go2rtc"] = _go2rtc_monitor.get_status()
    return JSONResponse(stats)


# ============================================================
# MJPEG STREAM ENDPOINT
# ============================================================

def mjpeg_generator(cam_id: str):
    """Yield MJPEG frames for a specific camera."""
    delay = 1.0 / MJPEG_FPS

    while True:
        # Try display frame first (annotated), then raw frame
        frame = state.get_display_frame(cam_id)
        if frame is None:
            frame = frame_store.get(cam_id)

        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            label = f"{cam_id} - waiting..."
            cv2.putText(frame, label, (80, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY])
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpeg.tobytes() +
            b"\r\n"
        )
        time.sleep(delay)


@app.get("/stream/cam{cam_id}")
async def mjpeg_stream(cam_id: int):
    """MJPEG video stream. cam_id is 1-based (cam1..cam4 for display cameras)."""
    # Map 1-based cam_id to cam_id string
    cam_id_str = f"cam-{cam_id:02d}"

    # For display cameras (ptz-1..4), check those first
    if _camera_manager:
        display_cams = _camera_manager.get_display_cameras()
        if cam_id <= len(display_cams):
            cam_id_str = display_cams[cam_id - 1].cam_id

    return StreamingResponse(
        mjpeg_generator(cam_id_str),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ============================================================
# BACKGROUND BROADCAST TASK
# ============================================================

async def ranking_broadcast_loop():
    """Periodically broadcast ranking updates to all WebSocket clients."""
    last_log_time = 0
    last_activation_broadcast = 0
    last_rankings_hash = ""

    while True:
        await asyncio.sleep(BROADCAST_INTERVAL)

        if not ws_clients:
            continue

        # Broadcast pending camera_result events from DeepStreamPipeline
        if _deepstream_pipeline and _deepstream_pipeline._pending_camera_results:
            pending = list(_deepstream_pipeline._pending_camera_results)
            _deepstream_pipeline._pending_camera_results.clear()
            for result_msg in pending:
                await broadcast(result_msg)

        # Broadcast rankings only when changed
        if state.race_active:
            rankings = state.get_rankings()
            if rankings:
                # Hash by positions to detect changes
                rankings_hash = "|".join(
                    f"{r.get('color','')}{r.get('position','')}{r.get('distanceCovered','')}"
                    for r in rankings
                )
                if rankings_hash != last_rankings_hash:
                    last_rankings_hash = rankings_hash
                    await broadcast({
                        "type": "ranking_update",
                        "rankings": rankings,
                    })

        # Broadcast live detections from DeepStream (which cameras see horses NOW)
        if _deepstream_pipeline and _deepstream_pipeline.live_detections:
            await broadcast({
                "type": "live_detections",
                "cameras": _deepstream_pipeline.live_detections,
            })

        # Broadcast activation map every 2 seconds
        now = time.time()
        if _camera_manager and now - last_activation_broadcast > 2.0:
            activation = _camera_manager.get_activation_map()
            await broadcast({
                "type": "activation_map",
                "cameras": activation,
            })
            last_activation_broadcast = now

        # Log every 5 seconds
        if state.race_active and now - last_log_time > 5.0:
            rankings = state.get_rankings()
            if rankings:
                names = [r.get("name", "?") for r in rankings[:3]]
                log.info("Broadcasting %d horses to %d clients (top 3: %s)",
                         len(rankings), len(ws_clients), names)
            if _pipeline:
                stats = _pipeline.get_stats()
                trigger = stats.get("trigger", {})
                analyzer = stats.get("analyzer", {})
                log.info("  Trigger: %d frames, Analysis: %d frames @ %.1f fps",
                         trigger.get("frames_processed", 0),
                         analyzer.get("frames_processed", 0),
                         analyzer.get("current_fps", 0))
            if _deepstream_pipeline:
                stats = _deepstream_pipeline.get_stats()
                ds = stats.get("deepstream", {})
                log.info("  DeepStream: seq=%d, %d frames @ %.1f fps, %.1fms/cycle",
                         ds.get("shm_seq", 0),
                         ds.get("frames_processed", 0),
                         ds.get("current_fps", 0),
                         ds.get("last_cycle_ms", 0))
            last_log_time = now


@app.on_event("startup")
async def startup():
    asyncio.create_task(ranking_broadcast_loop())
    if _go2rtc_monitor:
        _go2rtc_monitor.start()
    log.info(f"Race Vision backend running on http://{SERVER_HOST}:{SERVER_PORT}")
    log.info(f"  WebSocket: ws://localhost:{SERVER_PORT}/ws")
    log.info(f"  MJPEG:     http://localhost:{SERVER_PORT}/stream/cam1")
    if _camera_manager:
        status = _camera_manager.get_status()
        log.info(f"  Cameras:   {status['total_analytics']} analytics, {status['total_display']} display")


# ============================================================
# CAMERA CONFIG LOADER
# ============================================================

def load_camera_config(config_path: str) -> tuple[CameraManager, TrackTopology]:
    """Load camera configuration from JSON file.

    Expected format:
    {
        "track_length": 2500,
        "analytics": [
            {"id": "cam-01", "url": "rtsp://...", "track_start": 0, "track_end": 110},
            ...
        ],
        "display": [
            {"id": "ptz-1", "url": "rtsp://..."},
            ...
        ]
    }
    """
    with open(config_path) as f:
        config = json.load(f)

    track_length = config.get("track_length", TRACK_LENGTH)
    mgr = CameraManager(max_active=config.get("max_active", 8))
    topo = TrackTopology(track_length=track_length)

    for cam in config.get("analytics", []):
        mgr.add_analytics(
            cam["id"],
            cam["url"],
            track_start=cam.get("track_start", 0),
            track_end=cam.get("track_end", 100),
        )
        topo.add_camera(
            cam["id"],
            cam.get("track_start", 0),
            cam.get("track_end", 100),
        )

    for cam in config.get("display", []):
        mgr.add_display(cam["id"], cam["url"])

    return mgr, topo


# ============================================================
# MAIN
# ============================================================

_grabber = None  # VideoFileGrabber or RTSPGrabber


def _shutdown_handler(signum, frame):
    """Graceful shutdown on SIGTERM/SIGINT."""
    log.info("Received signal %d, shutting down...", signum)
    if _deepstream_subprocess:
        _deepstream_subprocess.stop()
    if _deepstream_pipeline:
        _deepstream_pipeline.stop()
    if _pipeline:
        _pipeline.stop()
    if _legacy_detector:
        _legacy_detector.stop()
    sys.exit(0)


def main():
    global _grabber, _pipeline, _deepstream_subprocess, _deepstream_pipeline
    global _legacy_detector, _camera_manager, _go2rtc_monitor

    parser = argparse.ArgumentParser(description="Race Vision Backend Server")
    parser.add_argument("--url", default=None, help="Single RTSP stream URL (legacy mode)")
    parser.add_argument("--video", nargs="+", default=None,
                        help="Local video file(s) — each simulates a camera")
    parser.add_argument("--config", default=None,
                        help="Camera config JSON file (multi-camera mode)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU decode")
    parser.add_argument("--host", default=SERVER_HOST, help="Server host")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="Server port")
    parser.add_argument("--auto-start", action="store_true",
                        help="Auto-start race on launch")
    parser.add_argument("--mode", choices=["auto", "multi", "legacy"], default="auto",
                        help="Pipeline mode: multi (trigger+analysis+fusion) or legacy (single detector)")
    parser.add_argument("--deepstream", action="store_true",
                        help="Use DeepStream C++ pipeline via shared memory")
    parser.add_argument("--ds-binary", default="/app/bin/race_vision_deepstream",
                        help="Path to DeepStream C++ binary")
    parser.add_argument("--yolo-engine", default="/app/models/yolov8s_deepstream.engine",
                        help="Path to YOLO TensorRT engine")
    parser.add_argument("--color-engine", default="/app/models/color_classifier.engine",
                        help="Path to color classifier TensorRT engine")
    parser.add_argument("--file-mode", action="store_true",
                        help="File playback mode (auto-detected if URLs start with file://)")
    parser.add_argument("--display", action="store_true",
                        help="Show video grid with OSD (requires X11 DISPLAY)")
    parser.add_argument("--go2rtc-url", default="http://localhost:1984",
                        help="go2rtc API URL for stream health monitoring")
    args = parser.parse_args()

    # Environment variable override (for Docker: DEEPSTREAM=1)
    if os.environ.get("DEEPSTREAM") == "1":
        args.deepstream = True

    # Signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    # go2rtc stream health monitor (started in FastAPI startup event)
    _go2rtc_monitor = Go2RTCMonitor(args.go2rtc_url)

    # Load heavy imports only when needed (non-DeepStream modes require torch)
    if not args.deepstream:
        log.info("Loading torch/ultralytics modules...")
        _load_heavy_imports()

    # ── Determine mode and sources ────────────────────────────────────

    use_multi = False

    if args.config:
        # Multi-camera from config file
        _camera_manager, topology = load_camera_config(args.config)
        if not args.deepstream:
            _grabber = RTSPGrabber(_camera_manager, gpu=args.gpu)
            _grabber.start()
        else:
            log.info("DeepStream mode — skipping RTSP grabber")
        use_multi = True
        log.info("Multi-camera mode from config: %s", args.config)

    elif args.video:
        if len(args.video) >= 2 or args.mode == "multi":
            # Multi-camera from video files
            _camera_manager = CameraManager(max_active=8)
            topology = TrackTopology(track_length=TRACK_LENGTH)

            cam_video_map = {}
            segment_len = TRACK_LENGTH / len(args.video)
            for i, vpath in enumerate(args.video):
                cam_id = f"cam-{i+1:02d}"
                start = i * segment_len
                end = start + segment_len + 10  # 10m overlap
                _camera_manager.add_analytics(cam_id, vpath, track_start=start, track_end=end)
                topology.add_camera(cam_id, start, end)
                cam_video_map[cam_id] = vpath

            _grabber = VideoFileGrabber(cam_video_map)
            _grabber.start()
            use_multi = True
            log.info("Multi-camera mode from %d videos", len(args.video))

        else:
            # Single video → legacy mode
            cam_id = "cam-01"
            cam_video_map = {cam_id: args.video[0]}
            _grabber = VideoFileGrabber(cam_video_map)
            _grabber.start()
            log.info("Legacy mode: single video %s", Path(args.video[0]).stem)

    else:
        # Single RTSP → legacy mode
        url = args.url or DEFAULT_RTSP_URL
        cam_id = "cam-01"

        def _rtsp_on_frame(cid, frame):
            frame_store.put(cam_id, frame)
            state.set_frame(cam_id, frame)

        cam = _CameraStream(
            cam_id, url, gpu=args.gpu,
            reconnect_delay=5.0,
            on_frame=_rtsp_on_frame,
        )
        cam.start()
        _grabber = cam
        log.info("Legacy mode: RTSP %s",
                 url.split('@')[-1] if '@' in url else url)

    # ── Start pipeline ────────────────────────────────────────────────

    if args.deepstream and use_multi:
        # Launch C++ DeepStream as subprocess (if binary exists)
        if os.path.isfile(args.ds_binary):
            # Auto-detect file mode from camera URLs
            file_mode = args.file_mode
            if not file_mode:
                try:
                    import json as _json
                    with open(args.config) as _f:
                        _cfg = _json.load(_f)
                    file_mode = any(
                        c.get("url", "").startswith("file://")
                        for c in _cfg.get("analytics", [])
                    )
                except Exception:
                    pass
            if file_mode:
                log.info("File mode detected — passing --file-mode to C++")
            _deepstream_subprocess = DeepStreamSubprocess(
                config_path=args.config,
                yolo_engine=args.yolo_engine,
                color_engine=args.color_engine,
                binary=args.ds_binary,
                file_mode=file_mode,
                display=args.display,
            )
            _deepstream_subprocess.start()
        else:
            log.info("DeepStream binary not found at %s — assuming external C++ process",
                     args.ds_binary)

        # Python SHM reader (retries until C++ creates SHM)
        _deepstream_pipeline = DeepStreamPipeline(_camera_manager, topology)
        _deepstream_pipeline.start()
        log.info("DeepStream pipeline started (C++ SHM → Python fusion)")
    elif use_multi and args.mode != "legacy":
        _pipeline = MultiCameraPipeline(_camera_manager, topology)
        _pipeline.start()
        log.info("Multi-camera pipeline started (trigger + analysis + fusion)")
    else:
        cam_id = "cam-01"
        _legacy_detector = LegacyDetectionLoop(cam_id=cam_id)
        _legacy_detector.start()
        log.info("Legacy detection loop started")

    if args.auto_start:
        state.race_active = True
        log.info("Race auto-started")

    # ── Run server ────────────────────────────────────────────────────

    import uvicorn
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        log.info("Shutting down...")
        if _go2rtc_monitor:
            _go2rtc_monitor.stop()
        if _grabber:
            if hasattr(_grabber, 'stop'):
                _grabber.stop()
        if _pipeline:
            _pipeline.stop()
        if _deepstream_pipeline:
            _deepstream_pipeline.stop()
        if _deepstream_subprocess:
            _deepstream_subprocess.stop()
        if _legacy_detector:
            _legacy_detector.stop()


if __name__ == "__main__":
    main()
