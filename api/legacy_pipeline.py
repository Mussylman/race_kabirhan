"""
Race Vision — Legacy single-camera and multi-camera (torch-based) pipelines.

Imports ONLY from api.shared (no other api.* modules).
"""

import time
import threading
from typing import Optional
from collections import Counter

from api.shared import (
    state, frame_store, log,
    CameraManager, TrackTopology, CameraDetections,
    FusionEngine,
    TriggerLoop, AnalysisLoop,
    _RaceTracker, _draw,
    COLOR_TO_HORSE, ALL_COLORS, TRACK_LENGTH,
)

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
        from pathlib import Path
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
