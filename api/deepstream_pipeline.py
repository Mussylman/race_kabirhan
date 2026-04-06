"""
Race Vision — DeepStream C++ subprocess and SHM-based pipeline.

Imports ONLY from api.shared (no other api.* modules).
"""

import os
import json
import time
import subprocess
import threading
from typing import Optional

from api.shared import (
    state, log,
    CameraManager, TrackTopology, CameraDetections,
    FusionEngine, SharedMemoryReader,
    COLOR_TO_HORSE, ALL_COLORS, TRACK_LENGTH,
)
from pipeline.log_utils import slog, throttle, agg


# ============================================================
# COLOR TRACKER — per-track EMA smoothing to filter noise
# ============================================================

class ColorTracker:
    """Tracks per-(cam, track_id) color probabilities via EMA.

    Solves: classifier sometimes flips yellow→blue for 1-2 frames.
    After 3+ frames of consistent color, the EMA stabilizes and
    transient misclassifications are overridden.
    """

    EMA_ALPHA = 0.3          # new observation weight (lower = smoother)
    MIN_CONF = 0.60          # below this → "unknown"
    COLOR_NAMES = ["blue", "green", "purple", "red", "yellow"]

    def __init__(self):
        # key: (cam_id, track_id) → {"probs": [5 floats], "frames": int}
        self._tracks: dict[tuple, dict] = {}

    def update(self, cam_id: str, track_id: int, color: str, conf: float) -> tuple[str, int]:
        """Feed a raw classification, return smoothed (color, conf%).

        Args:
            cam_id: camera identifier
            track_id: nvtracker object ID
            color: raw classifier output ("red", "blue", etc.)
            conf: raw confidence 0-100

        Returns:
            (smoothed_color, smoothed_conf_percent)
        """
        key = (cam_id, track_id)
        if key not in self._tracks:
            self._tracks[key] = {"probs": [0.0] * 5, "frames": 0}

        entry = self._tracks[key]
        alpha = self.EMA_ALPHA

        # Build one-hot from raw classification
        raw = [0.0] * 5
        if color in self.COLOR_NAMES:
            idx = self.COLOR_NAMES.index(color)
            raw[idx] = conf / 100.0

        # EMA update
        if entry["frames"] == 0:
            entry["probs"] = raw
        else:
            for i in range(5):
                entry["probs"][i] = alpha * raw[i] + (1 - alpha) * entry["probs"][i]
        entry["frames"] += 1

        # Read smoothed result
        best_idx = max(range(5), key=lambda i: entry["probs"][i])
        best_conf = entry["probs"][best_idx]

        if best_conf < self.MIN_CONF:
            return ("unknown", round(best_conf * 100))
        return (self.COLOR_NAMES[best_idx], round(best_conf * 100))

    def cleanup(self, max_age: int = 500):
        """Remove tracks not seen for max_age frames."""
        # Simple: just keep last 200 tracks
        if len(self._tracks) > 200:
            oldest = sorted(self._tracks.keys(),
                          key=lambda k: self._tracks[k]["frames"])
            for k in oldest[:len(self._tracks) - 100]:
                del self._tracks[k]


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
# DETECTION BUFFER — lock-free bridge between SHM Reader and Inference
# ============================================================

class DetectionBuffer:
    """Thread-safe single-slot buffer: SHM Reader writes, Inference reads.

    Stores only the latest frame of detections per camera.
    Reader overwrites, consumer reads a copy. No queue, no backpressure.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cam_detections: list = []   # latest list[CameraDetections]
        self._frame_id: int = 0
        self._live: dict = {}             # latest live_detections for broadcast

    def update(self, cam_results: list, live: dict):
        """Called by SHM Reader thread at ~1000hz."""
        with self._lock:
            self._cam_detections = cam_results
            self._frame_id += 1
            self._live = live

    def get(self) -> tuple:
        """Called by Inference thread. Returns (cam_results, frame_id, live)."""
        with self._lock:
            return self._cam_detections, self._frame_id, self._live


# ============================================================
# DEEPSTREAM PIPELINE (replaces MultiCameraPipeline when --deepstream)
# ============================================================

class DeepStreamPipeline:
    """Reads detections from DeepStream C++ via shared memory.

    Architecture: 3 independent threads/loops, 1 buffer
      SHM Reader (Thread, ~1000hz) → DetectionBuffer → Inference (Thread, ~4fps effective)
                                           ↓
                                   ranking_broadcast_loop (async, 10hz)

    Replaces TriggerLoop + AnalysisLoop with a single SharedMemoryReader
    that receives pre-computed detections (YOLO + color) from the C++ pipeline.
    Keeps VoteEngine per camera and FusionEngine for ranking.
    """

    def __init__(self, camera_manager: CameraManager, topology: TrackTopology):
        self.camera_manager = camera_manager
        self.topology = topology
        self.fusion = FusionEngine(topology, colors=ALL_COLORS)

        self._reader = SharedMemoryReader(timeout_ms=200)
        self._shm_thread: Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None
        self._running = False

        # Buffer between SHM Reader and Inference
        self._detection_buffer = DetectionBuffer()

        # Color tracker for EMA smoothing of per-track color classifications
        self._color_tracker = ColorTracker()

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
        self.shm_fps = 0.0  # SHM reader speed

        # Live detection status — updated by SHM Reader, read by broadcast
        self.live_detections: dict[str, list] = {}

        # Auto-direction detection: track movement of objects to determine
        # if horses run left-to-right (+center_x) or right-to-left (-center_x)
        # per camera. Once determined, we flip topology for inverted cameras.
        self._direction_votes: dict[str, list[float]] = {}  # cam_id → list of delta_x
        self._direction_decided: dict[str, bool] = {}  # cam_id → True if inverted
        self._direction_locked = False
        self._track_positions: dict[tuple[str, int], float] = {}  # (cam_id, track_id) → last center_x
        self.DIRECTION_VOTE_THRESHOLD = 30  # frames before deciding direction

        # Detection JSONL logger — saves every frame with detections for post-analysis
        self._jsonl_path = os.environ.get("DETECTION_LOG", "/tmp/race_analysis/detections.jsonl")
        os.makedirs(os.path.dirname(self._jsonl_path), exist_ok=True)
        self._jsonl_file = open(self._jsonl_path, "w")
        log.info("Detection JSONL logger: %s", self._jsonl_path)

    def _update_direction(self, cam_results: list):
        """Track object movement to auto-detect horse direction per camera.

        Compares center_x of same track_id between frames. If objects move
        right (increasing center_x), direction is normal. If left, inverted.
        Once enough votes collected, flips topology for inverted cameras.
        """
        if self._direction_locked:
            return

        for cam_det in cam_results:
            cam_id = cam_det.cam_id
            if cam_id in self._direction_decided:
                continue

            for det in cam_det.detections:
                track_id = det.get('track_id', 0)
                cx = det.get('center_x', 0)
                key = (cam_id, track_id)

                if key in self._track_positions:
                    delta = cx - self._track_positions[key]
                    # Only count significant movements (>5 pixels)
                    if abs(delta) > 5:
                        if cam_id not in self._direction_votes:
                            self._direction_votes[cam_id] = []
                        self._direction_votes[cam_id].append(delta)

                self._track_positions[key] = cx

            # Check if we have enough votes for this camera
            votes = self._direction_votes.get(cam_id, [])
            if len(votes) >= self.DIRECTION_VOTE_THRESHOLD:
                avg_delta = sum(votes) / len(votes)
                inverted = avg_delta < 0  # moving left = inverted
                self._direction_decided[cam_id] = inverted

                if inverted:
                    # Swap track_start and track_end in topology
                    seg = self.topology.get_segment(cam_id)
                    if seg and not seg.inverted:
                        seg.inverted = True
                        log.info("AUTO-DIRECTION: %s → INVERTED (avg_delta=%.1f, %d votes)",
                                 cam_id, avg_delta, len(votes))
                else:
                    log.info("AUTO-DIRECTION: %s → NORMAL (avg_delta=%.1f, %d votes)",
                             cam_id, avg_delta, len(votes))

        # Check if all active cameras have direction decided
        active_cams = set(c.cam_id for c in cam_results if c.n_detections > 0)
        if active_cams and active_cams.issubset(set(self._direction_decided.keys())):
            self._direction_locked = True
            n_inverted = sum(1 for v in self._direction_decided.values() if v)
            log.info("AUTO-DIRECTION: locked (%d cameras, %d inverted)",
                     len(self._direction_decided), n_inverted)

    def _get_vote_engine(self, cam_id: str):
        from pipeline.vote_engine import VoteEngine
        if cam_id not in self._vote_engines:
            self._vote_engines[cam_id] = VoteEngine(ALL_COLORS)
        return self._vote_engines[cam_id]

    def start(self):
        self._running = True

        # Thread 1: SHM Reader — reads at max speed, writes to DetectionBuffer
        self._shm_thread = threading.Thread(
            target=self._shm_reader_loop, daemon=True, name="SHMReader"
        )
        self._shm_thread.start()

        # Thread 2: Inference — reads DetectionBuffer, does voting/fusion
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="InferenceWorker"
        )
        self._inference_thread.start()

        log.info("DeepStreamPipeline started (SHMReader + InferenceWorker)")

    # ------------------------------------------------------------------
    # Thread 1: SHM Reader — fast, no processing, just reads and buffers
    # ------------------------------------------------------------------
    def _shm_reader_loop(self):
        """Reads SHM at max speed (~1000hz), writes to DetectionBuffer."""
        # Retry attach until successful
        while self._running and not self._reader.is_attached:
            if self._reader.attach():
                log.info("SHMReader: attached to shared memory")
                break
            log.info("SHMReader: waiting for DeepStream C++ (SHM not found)...")
            time.sleep(2.0)

        fps_counter = 0
        fps_timer = time.monotonic()
        stale_counter = 0
        STALE_THRESHOLD = 30  # ~6 sec at 200ms timeout

        while self._running:
            # Read detections from shared memory
            cam_results = self._reader.read()
            if cam_results is None:
                stale_counter += 1
                if self._reader.is_attached and stale_counter > STALE_THRESHOLD:
                    log.warning("SHMReader: stale (%d timeouts), re-attaching...", stale_counter)
                    self._reader.detach()
                    stale_counter = 0
                if not self._reader.is_attached:
                    while self._running and not self._reader.is_attached:
                        if self._reader.attach():
                            log.info("SHMReader: re-attached")
                            break
                        time.sleep(2.0)
                continue
            stale_counter = 0

            # Build live detection map (always, regardless of race_active)
            live = {}
            for cam_det in cam_results:
                if cam_det.n_detections > 0:
                    live[cam_det.cam_id] = {
                        "frame_w": cam_det.frame_width,
                        "frame_h": cam_det.frame_height,
                        "ts_capture": cam_det.timestamp,  # seconds since epoch from C++ SHM
                        "frame_seq": cam_det.frame_seq,   # SHM write_seq — cross-layer tracing key
                        "detections": [
                            {
                                "color": d.get("color", "?"),
                                "conf": round(d.get("conf", 0) * 100),
                                "track_id": d.get("track_id", 0),
                                "bbox": d.get("bbox", (0, 0, 0, 0)),
                            }
                            for d in cam_det.detections
                        ],
                    }
                    # LIVE_UPDATE log — throttled
                    if throttle.allow(f"LIVE_UPDATE:{cam_det.cam_id}", interval=2.0):
                        slog("LIVE_UPDATE", cam_det.cam_id, cam_det.frame_seq,
                             time.time(), dets=cam_det.n_detections)

                    # JSONL: every frame with detections — for post-run frame extraction
                    record = {
                        "ts_capture": cam_det.timestamp,
                        "cam_id":     cam_det.cam_id,
                        "frame_seq":  cam_det.frame_seq,
                        "frame_w":    cam_det.frame_width,
                        "frame_h":    cam_det.frame_height,
                        "detections": live[cam_det.cam_id]["detections"],
                    }
                    self._jsonl_file.write(json.dumps(record) + "\n")
                    self._jsonl_file.flush()

            # Auto-detect direction from object movement
            self._update_direction(cam_results)

            # Write to buffer — inference thread reads from here
            self._detection_buffer.update(cam_results, live)
            self.live_detections = live

            # SHM Reader FPS tracking
            fps_counter += 1
            elapsed_fps = time.monotonic() - fps_timer
            if elapsed_fps >= 1.0:
                self.shm_fps = fps_counter / elapsed_fps
                fps_counter = 0
                fps_timer = time.monotonic()

    # ------------------------------------------------------------------
    # Thread 2: Inference — voting, fusion, ranking (at its own pace)
    # ------------------------------------------------------------------
    def _inference_loop(self):
        """Reads DetectionBuffer, runs voting/fusion. Independent of SHM speed."""
        fps_counter = 0
        fps_timer = time.monotonic()
        last_frame_id = 0

        while self._running:
            cam_results, frame_id, _ = self._detection_buffer.get()

            # No new data — sleep briefly
            if frame_id == last_frame_id or not cam_results:
                time.sleep(0.001)  # 1ms poll — don't spin CPU
                continue
            last_frame_id = frame_id

            t0 = time.monotonic()
            self.cycles += 1

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

                # Smooth colors via EMA tracker (reduces transient misclassifications)
                smoothed_dets = []
                for d in cam_det.detections:
                    track_id = d.get('track_id', 0)
                    raw_color = d.get('color', 'unknown')
                    raw_conf = d.get('conf', 0) * 100 if d.get('conf', 0) <= 1.0 else d.get('conf', 0)
                    smooth_color, smooth_conf = self._color_tracker.update(
                        cam_id, track_id, raw_color, raw_conf)
                    sd = dict(d)
                    sd['color'] = smooth_color
                    sd['conf'] = smooth_conf / 100.0
                    smoothed_dets.append(sd)

                # Filter out "unknown" colors
                smoothed_dets = [d for d in smoothed_dets if d['color'] != 'unknown']
                if not smoothed_dets:
                    continue

                # If camera is inverted, flip center_x so VoteEngine's
                # "rightmost = 1st" logic still works correctly
                dets_for_vote = smoothed_dets
                if self._direction_decided.get(cam_id, False):
                    dets_for_vote = []
                    for d in smoothed_dets:
                        flipped = dict(d)
                        flipped['center_x'] = cam_det.frame_width - d['center_x']
                        dets_for_vote.append(flipped)
                assigned, weight = engine.submit_frame(dets_for_vote)
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
                    # Condition 4: partial timeout — straggler case
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

            # Inference FPS tracking
            fps_counter += 1
            elapsed_fps = time.monotonic() - fps_timer
            if elapsed_fps >= 1.0:
                self.current_fps = fps_counter / elapsed_fps
                fps_counter = 0
                fps_timer = time.monotonic()

            self.last_cycle_time = time.monotonic() - t0

    def stop(self):
        """Stop all pipeline threads."""
        self._running = False
        if self._shm_thread and self._shm_thread.is_alive():
            self._shm_thread.join(timeout=5)
        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=5)
        self._reader.detach()
        if self._jsonl_file:
            self._jsonl_file.close()
        log.info("DeepStreamPipeline stopped")

    def reset(self):
        """Reset state for a new race."""
        self.fusion.reset()
        self._vote_engines.clear()
        self._cam_frame_count.clear()
        self._cam_first_analysis.clear()
        self._cam_all_visible_time.clear()
        self._cam_completed.clear()
        self._pending_camera_results.clear()
        self.frames_processed = 0
        self.cycles = 0
        # Keep direction decisions — they don't change between races
        log.info("DeepStreamPipeline reset for new race")

    def get_stats(self) -> dict:
        """Return pipeline performance stats."""
        return {
            "deepstream": {
                "frames_processed": self.frames_processed,
                "cycles": self.cycles,
                "current_fps": round(self.current_fps, 1),
                "shm_fps": round(self.shm_fps, 1),
                "last_cycle_ms": round(self.last_cycle_time * 1000, 1),
                "shm_seq": self._reader.last_seq if hasattr(self._reader, 'last_seq') else 0,
                "cameras_completed": len(self._cam_completed),
                "vote_engines": len(self._vote_engines),
            },
            "fusion": self.fusion.get_stats(),
        }

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

            # Compute gap to leader (approximate seconds based on track position)
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
