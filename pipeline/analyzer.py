"""
analyzer.py — Full analysis pipeline for active cameras.

Runs YOLOv8s @ 1280px + torso crop + SimpleColorCNN on cameras where
the trigger detected horses.  Produces per-camera detection results
that are fed into the fusion module.

Architecture:
    - Collects frames from active cameras (3-8 typically)
    - Batch YOLO detection (YOLOv8s)
    - Torso extraction + batch color classification
    - Per-camera voting via VoteEngine

Usage:
    analyzer = AnalysisLoop(camera_manager, frame_source, on_result=callback)
    analyzer.start()
    ...
    analyzer.stop()
"""

import cv2
import time
import logging
import threading
from typing import Callable, Optional

import numpy as np

from .trt_inference import YOLODetector, ColorClassifierInfer
from .camera_manager import CameraManager
from .vote_engine import VoteEngine

log = logging.getLogger("pipeline.analyzer")


# ── Parameters (same as test_race_count.py) ───────────────────────────

# Torso extraction (% of person bbox)
TORSO_TOP = 0.10
TORSO_BOTTOM = 0.40
TORSO_LEFT = 0.20
TORSO_RIGHT = 0.20

# Bbox filters
MIN_ASPECT_RATIO = 1.2
MIN_BBOX_HEIGHT = 100
EDGE_MARGIN = 10

# Classifier thresholds
MIN_COLOR_CONF = 0.60
MIN_REASSIGN_CONF = 0.20
MIN_CROP_PIXELS = 400
MAX_CROP_PIXELS = 15000

# Colors
ALL_COLORS = ["blue", "green", "purple", "red", "yellow"]

COLORS_BGR = {
    "blue": (255, 100, 0),
    "green": (0, 200, 0),
    "purple": (180, 0, 180),
    "red": (0, 0, 255),
    "yellow": (0, 230, 230),
    "unknown": (128, 128, 128),
}


def draw_detections(frame: np.ndarray, detections: list[dict], cam_id: str = "",
                    vote_result: list[str] = None) -> np.ndarray:
    """Draw bounding boxes, color labels, and ranking on frame."""
    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        color_name = det.get('color', 'unknown')
        conf = det.get('conf', 0)
        bgr = COLORS_BGR.get(color_name, (128, 128, 128))

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 3)

        # Label
        label = f"{color_name} {conf:.2f}"
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)

        # Torso region
        h, w = y2 - y1, x2 - x1
        ty1 = y1 + int(h * TORSO_TOP)
        ty2 = y1 + int(h * TORSO_BOTTOM)
        tx1 = x1 + int(w * TORSO_LEFT)
        tx2 = x2 - int(w * TORSO_RIGHT)
        cv2.rectangle(annotated, (tx1, ty1), (tx2, ty2), (0, 255, 255), 1)

    # Info panel
    fh, fw = annotated.shape[:2]
    cv2.rectangle(annotated, (5, 5), (340, 100), (0, 0, 0), -1)
    cv2.putText(annotated, f"{cam_id}  |  {len(detections)} detections",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    unique_colors = set(d.get('color') for d in detections)
    cv2.putText(annotated, f"Colors: {len(unique_colors)}/5  ({', '.join(sorted(unique_colors))})",
                (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Vote result
    if vote_result:
        cv2.putText(annotated, f"Order: {' > '.join(c[:3].upper() for c in vote_result)}",
                    (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # Side panel with ranking
        y = 130
        for i, color_name in enumerate(vote_result):
            bgr = COLORS_BGR.get(color_name, (128, 128, 128))
            cv2.putText(annotated, f"{i+1}. {color_name}", (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)
            y += 28

    return annotated


def extract_torso(frame: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
    """Extract torso region from person bounding box."""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = y2 - y1, x2 - x1
    ty1 = max(0, y1 + int(h * TORSO_TOP))
    ty2 = min(frame.shape[0], y1 + int(h * TORSO_BOTTOM))
    tx1 = max(0, x1 + int(w * TORSO_LEFT))
    tx2 = min(frame.shape[1], x2 - int(w * TORSO_RIGHT))
    if ty2 - ty1 < 10 or tx2 - tx1 < 10:
        return None
    return frame[ty1:ty2, tx1:tx2]


def analyze_hsv(crop_bgr: np.ndarray) -> tuple[str, int]:
    """Quick HSV analysis for cross-checking CNN."""
    if crop_bgr is None or crop_bgr.size < 100:
        return 'unknown', -1
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = (s > 30) & (v > 40)
    if mask.sum() < 20:
        return 'unknown', -1
    h_filtered = h[mask]
    hist = cv2.calcHist([h_filtered.reshape(-1, 1)], [0], None, [18], [0, 180])
    dominant_hue = int(hist.argmax() * 10 + 5)
    if dominant_hue < 10 or dominant_hue > 170:
        return 'red', dominant_hue
    elif 20 <= dominant_hue < 40:
        return 'yellow', dominant_hue
    elif 40 <= dominant_hue < 85:
        return 'green', dominant_hue
    elif 85 <= dominant_hue < 130:
        return 'blue', dominant_hue
    elif 130 <= dominant_hue <= 170:
        return 'purple', dominant_hue
    return 'unknown', dominant_hue


# ── Per-camera detection result ──────────────────────────────────────

class CameraDetections:
    """Results from analyzing one frame of one camera."""

    def __init__(self, cam_id: str, frame_width: int, frame_height: int):
        self.cam_id = cam_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.detections: list[dict] = []
        self.timestamp: float = time.time()

    def add(self, det: dict):
        self.detections.append(det)

    @property
    def colors(self) -> list[str]:
        return [d['color'] for d in self.detections]

    @property
    def n_detections(self) -> int:
        return len(self.detections)


# ── Analysis Loop ────────────────────────────────────────────────────

class AnalysisLoop(threading.Thread):
    """Background thread that runs full analysis on active cameras."""

    def __init__(
        self,
        camera_manager: CameraManager,
        frame_source: Callable[[str], Optional[np.ndarray]],
        on_result: Optional[Callable[[list[CameraDetections]], None]] = None,
        on_annotated_frame: Optional[Callable[[str, np.ndarray], None]] = None,
        *,
        analysis_fps: float = 12.0,
        yolo_engine: Optional[str] = None,
        yolo_fallback: str = "yolov8s.pt",
        classifier_engine: Optional[str] = None,
        classifier_fallback: str = "models/color_classifier.pt",
        imgsz: int = 1280,
        det_conf: float = 0.35,
        det_iou: float = 0.3,
    ):
        super().__init__(daemon=True, name="AnalysisLoop")
        self.camera_manager = camera_manager
        self.frame_source = frame_source
        self.on_result = on_result
        self.on_annotated_frame = on_annotated_frame
        self.analysis_fps = analysis_fps
        self.running = False

        # Stats
        self.frames_processed = 0
        self.detections_total = 0
        self.last_batch_time = 0.0
        self.current_fps = 0.0

        # Models (lazy init)
        self._yolo_engine = yolo_engine
        self._yolo_fallback = yolo_fallback
        self._classifier_engine = classifier_engine
        self._classifier_fallback = classifier_fallback
        self._imgsz = imgsz
        self._det_conf = det_conf
        self._det_iou = det_iou

        self._detector: Optional[YOLODetector] = None
        self._classifier: Optional[ColorClassifierInfer] = None

        # Per-camera vote engines
        self._vote_engines: dict[str, VoteEngine] = {}

    def _get_vote_engine(self, cam_id: str) -> VoteEngine:
        if cam_id not in self._vote_engines:
            self._vote_engines[cam_id] = VoteEngine(ALL_COLORS)
        return self._vote_engines[cam_id]

    def run(self):
        self.running = True

        # Lazy init models
        self._detector = YOLODetector(
            engine_path=self._yolo_engine,
            fallback_pt=self._yolo_fallback,
            imgsz=self._imgsz,
            conf=self._det_conf,
            iou=self._det_iou,
        )
        self._classifier = ColorClassifierInfer(
            engine_path=self._classifier_engine,
            fallback_pt=self._classifier_fallback,
        )
        log.info("AnalysisLoop started (%.1f fps, imgsz=%d)", self.analysis_fps, self._imgsz)

        interval = 1.0 / max(self.analysis_fps, 0.1)
        fps_counter = 0
        fps_timer = time.monotonic()

        while self.running:
            t0 = time.monotonic()
            self._analysis_step()

            # FPS tracking
            fps_counter += 1
            elapsed_fps = time.monotonic() - fps_timer
            if elapsed_fps >= 1.0:
                self.current_fps = fps_counter / elapsed_fps
                fps_counter = 0
                fps_timer = time.monotonic()

            elapsed = time.monotonic() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _analysis_step(self):
        """One analysis iteration: detect + classify on active cameras."""
        active_cams = self.camera_manager.get_active_cameras()
        if not active_cams:
            return

        # Collect frames from active cameras
        cam_ids = []
        frames = []
        for cam in active_cams:
            frame = self.frame_source(cam.cam_id)
            if frame is not None:
                cam_ids.append(cam.cam_id)
                frames.append(frame)

        if not frames:
            return

        # Step 1: Batch YOLO detection
        t0 = time.monotonic()
        batch_dets = self._detector.detect_batch(frames)

        # Step 2: For each camera, extract torsos and batch classify
        all_results: list[CameraDetections] = []

        for cam_id, frame, raw_dets in zip(cam_ids, frames, batch_dets):
            cam_info = self.camera_manager.get_camera(cam_id)
            fh, fw = frame.shape[:2]
            cam_result = CameraDetections(cam_id, fw, fh)

            # Filter detections and extract torsos
            crops = []
            valid_dets = []

            for det in raw_dets:
                x1, y1, x2, y2 = det['bbox']
                bw, bh = x2 - x1, y2 - y1

                # Edge filter
                if x1 <= EDGE_MARGIN or x2 >= fw - EDGE_MARGIN:
                    continue

                # Aspect ratio + height filter
                if bh < MIN_BBOX_HEIGHT:
                    continue
                if bh / max(bw, 1) < MIN_ASPECT_RATIO:
                    continue

                # Extract torso
                torso = extract_torso(frame, det['bbox'])
                if torso is None:
                    continue

                crop_h, crop_w = torso.shape[:2]
                crop_pixels = crop_h * crop_w
                if crop_pixels < MIN_CROP_PIXELS or crop_pixels > MAX_CROP_PIXELS:
                    continue

                crops.append(torso)
                valid_dets.append(det)

            # Batch classify all torsos
            if crops:
                classifications = self._classifier.classify_batch(crops)
            else:
                classifications = []

            # Build detection list
            frame_dets = []
            for det, (color, conf, prob_dict) in zip(valid_dets, classifications):
                if conf < MIN_REASSIGN_CONF:
                    continue

                hsv_guess, hsv_hue = analyze_hsv(
                    crops[valid_dets.index(det)]
                )

                frame_det = {
                    'bbox': det['bbox'],
                    'center_x': det['center_x'],
                    'det_conf': det['conf'],
                    'color': color,
                    'conf': conf,
                    'prob_dict': prob_dict,
                    'hsv_guess': hsv_guess,
                    'hsv_hue': hsv_hue,
                    'cam_id': cam_id,
                }
                frame_dets.append(frame_det)

            # Submit to per-camera vote engine
            engine = self._get_vote_engine(cam_id)
            assigned, weight = engine.submit_frame(frame_dets)

            for d in assigned:
                cam_result.add(d)

            all_results.append(cam_result)
            self.detections_total += cam_result.n_detections

            # Draw annotations on frame
            if self.on_annotated_frame:
                vote_result = engine.compute_result()
                annotated = draw_detections(frame, assigned, cam_id, vote_result)
                self.on_annotated_frame(cam_id, annotated)

        self.last_batch_time = time.monotonic() - t0
        self.frames_processed += len(frames)

        # Deliver results to fusion
        if self.on_result and all_results:
            self.on_result(all_results)

    def get_vote_result(self, cam_id: str) -> list[str]:
        """Get current vote result for a specific camera."""
        engine = self._vote_engines.get(cam_id)
        return engine.compute_result() if engine else []

    def reset_votes(self, cam_id: Optional[str] = None):
        """Reset vote engines."""
        if cam_id:
            engine = self._vote_engines.get(cam_id)
            if engine:
                engine.reset()
        else:
            for engine in self._vote_engines.values():
                engine.reset()

    def stop(self):
        self.running = False

    def get_stats(self) -> dict:
        return {
            "frames_processed": self.frames_processed,
            "detections_total": self.detections_total,
            "current_fps": round(self.current_fps, 1),
            "last_batch_time_ms": round(self.last_batch_time * 1000, 1),
            "active_vote_engines": len(self._vote_engines),
        }
