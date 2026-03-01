"""
trigger.py — Lightweight YOLOv8n trigger for all 25 analytics cameras.

Runs at 2–5 fps per camera.  The trigger answers one question per camera:
"Are there horses visible?" (bool).  This is much cheaper than full
analysis (YOLOv8n @ 640px vs YOLOv8s @ 1280px).

Architecture:
    - Collects latest frames from all analytics cameras
    - Batches them through YOLOv8n
    - Updates CameraManager activation state

Usage:
    trigger = TriggerLoop(camera_manager, frame_source)
    trigger.start()
    ...
    trigger.stop()
"""

import time
import logging
import threading
from typing import Callable, Optional

import numpy as np

from .trt_inference import YOLODetector
from .camera_manager import CameraManager

log = logging.getLogger("pipeline.trigger")


# ── Detection filters (simpler than full analysis) ────────────────────

MIN_BBOX_HEIGHT_TRIGGER = 50     # pixels — less strict than analysis
MIN_ASPECT_RATIO_TRIGGER = 0.8   # h/w — less strict for trigger
MIN_DETECTIONS_ACTIVATE = 1      # need at least 1 person to activate


class TriggerLoop(threading.Thread):
    """Background thread that runs lightweight detection on all analytics cameras.

    Polls frames from a frame_source callable and updates CameraManager.
    """

    def __init__(
        self,
        camera_manager: CameraManager,
        frame_source: Callable[[str], Optional[np.ndarray]],
        *,
        trigger_fps: float = 3.0,
        engine_path: Optional[str] = None,
        fallback_pt: str = "yolov8n.pt",
        imgsz: int = 640,
        conf: float = 0.25,
        min_detections: int = MIN_DETECTIONS_ACTIVATE,
    ):
        """
        Args:
            camera_manager: CameraManager instance to update.
            frame_source: callable(cam_id) → numpy frame or None.
            trigger_fps: target FPS per camera for trigger checks.
            engine_path: path to TensorRT engine (optional).
            fallback_pt: path to YOLO .pt model (fallback).
            imgsz: inference image size.
            conf: confidence threshold.
            min_detections: minimum detections to activate a camera.
        """
        super().__init__(daemon=True, name="TriggerLoop")
        self.camera_manager = camera_manager
        self.frame_source = frame_source
        self.trigger_fps = trigger_fps
        self.min_detections = min_detections
        self.running = False

        # Stats
        self.frames_processed = 0
        self.trigger_count = 0
        self.last_batch_time = 0.0
        self.last_batch_size = 0

        # Detector (lazy init — created on first run)
        self._engine_path = engine_path
        self._fallback_pt = fallback_pt
        self._imgsz = imgsz
        self._conf = conf
        self._detector: Optional[YOLODetector] = None

    def run(self):
        self.running = True
        interval = 1.0 / max(self.trigger_fps, 0.1)

        # Lazy init detector
        self._detector = YOLODetector(
            engine_path=self._engine_path,
            fallback_pt=self._fallback_pt,
            imgsz=self._imgsz,
            conf=self._conf,
            device="cuda:0",
            half=True,
        )
        log.info("TriggerLoop started (%.1f fps, imgsz=%d)", self.trigger_fps, self._imgsz)

        while self.running:
            t0 = time.monotonic()
            self._trigger_step()
            elapsed = time.monotonic() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _trigger_step(self):
        """One trigger iteration: collect frames, batch detect, update cameras."""
        cameras = self.camera_manager.get_analytics_cameras()
        if not cameras:
            return

        # Collect latest frames
        cam_ids = []
        frames = []
        for cam in cameras:
            frame = self.frame_source(cam.cam_id)
            if frame is not None:
                cam_ids.append(cam.cam_id)
                frames.append(frame)

        if not frames:
            return

        # Batch detection
        t0 = time.monotonic()
        batch_results = self._detector.detect_batch(frames)
        self.last_batch_time = time.monotonic() - t0
        self.last_batch_size = len(frames)
        self.frames_processed += len(frames)

        # Count valid detections per camera and update activation
        trigger_results = {}
        for cam_id, dets in zip(cam_ids, batch_results):
            # Apply simple filters
            valid = 0
            for det in dets:
                x1, y1, x2, y2 = det['bbox']
                bh = y2 - y1
                bw = max(x2 - x1, 1)
                if bh >= MIN_BBOX_HEIGHT_TRIGGER and bh / bw >= MIN_ASPECT_RATIO_TRIGGER:
                    valid += 1

            trigger_results[cam_id] = valid

        # Log per-camera trigger results
        prev_active = self.camera_manager.get_activation_map()

        # Batch update camera manager
        self.camera_manager.update_trigger_results(trigger_results)

        new_active = self.camera_manager.get_activation_map()

        # Log activations/deactivations
        for cam_id in trigger_results:
            was = prev_active.get(cam_id, False)
            now = new_active.get(cam_id, False)
            count = trigger_results[cam_id]
            if now and not was:
                log.info("ACTIVATE  %s  (%d detections)", cam_id, count)
            elif not now and was:
                log.info("DEACTIVATE  %s  (cooldown expired)", cam_id)

        activated = sum(1 for v in trigger_results.values() if v >= self.min_detections)
        if activated > 0:
            self.trigger_count += 1
            if self.frames_processed % 30 == 0:  # log summary every ~10s
                active_list = [c for c, a in new_active.items() if a]
                log.info("Trigger: %d/%d cameras active [%s] batch=%.1fms",
                         len(active_list), len(trigger_results),
                         ", ".join(active_list), self.last_batch_time * 1000)

    def stop(self):
        self.running = False

    def get_stats(self) -> dict:
        return {
            "frames_processed": self.frames_processed,
            "trigger_count": self.trigger_count,
            "last_batch_time_ms": round(self.last_batch_time * 1000, 1),
            "last_batch_size": self.last_batch_size,
        }
