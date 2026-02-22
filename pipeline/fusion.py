"""
fusion.py — Multi-camera detection fusion and global ranking.

Merges per-camera detection results into a single global view:
    1. Map pixel_x → global track position via TrackTopology
    2. Merge same-color detections from overlapping cameras
    3. Temporal smoothing (EMA) for stable positions
    4. Produce final ranking (1st..5th by global position)

Usage:
    fusion = FusionEngine(track_topology, colors=["blue","green","purple","red","yellow"])
    fusion.update(camera_detections_list)
    ranking = fusion.get_ranking()   # [{"color": "red", "position_m": 1800, ...}, ...]
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

from .track_topology import TrackTopology
from .analyzer import CameraDetections

log = logging.getLogger("pipeline.fusion")


@dataclass
class HorseState:
    """Smoothed state for one horse (identified by color)."""
    color: str
    position_m: float = 0.0         # global track position (EMA-smoothed)
    raw_position_m: float = 0.0     # latest raw position
    speed_mps: float = 0.0          # estimated speed (meters/sec)
    last_seen_time: float = 0.0
    last_camera: str = ""
    observation_count: int = 0
    rank: int = 0                   # 1-based position in race


class FusionEngine:
    """Merges detections from multiple cameras into global ranking."""

    def __init__(
        self,
        topology: TrackTopology,
        colors: Optional[list[str]] = None,
        *,
        ema_alpha: float = 0.15,
        stale_timeout: float = 5.0,
        merge_distance_m: float = 15.0,
    ):
        self.topology = topology
        self.colors = colors or ["blue", "green", "purple", "red", "yellow"]
        self.ema_alpha = ema_alpha
        self.stale_timeout = stale_timeout
        self.merge_distance_m = merge_distance_m

        self._lock = threading.Lock()
        self._horses: dict[str, HorseState] = {}
        self._ranking: list[HorseState] = []
        self._update_count = 0

        # Initialize horse states
        for color in self.colors:
            self._horses[color] = HorseState(color=color)

    def update(self, cam_results: list[CameraDetections]):
        """Process a batch of per-camera detection results.

        Called by AnalysisLoop whenever new detections are available.
        """
        now = time.time()

        with self._lock:
            # Collect all observations per color
            color_observations: dict[str, list[tuple[str, float]]] = {
                c: [] for c in self.colors
            }

            for cam_result in cam_results:
                for det in cam_result.detections:
                    color = det.get('color')
                    if color not in color_observations:
                        continue
                    cam_id = det.get('cam_id', cam_result.cam_id)
                    pixel_x = det.get('center_x', 0)
                    color_observations[color].append((cam_id, pixel_x))

            # Update each horse
            for color, observations in color_observations.items():
                horse = self._horses[color]

                if not observations:
                    continue

                # Merge observations (weighted by FOV position)
                raw_pos = self.topology.merge_positions(observations)
                if raw_pos is None:
                    continue

                horse.raw_position_m = raw_pos
                horse.last_seen_time = now
                horse.last_camera = observations[-1][0]
                horse.observation_count += 1

                # EMA smoothing
                if horse.observation_count <= 1:
                    horse.position_m = raw_pos
                else:
                    dt = max(now - horse.last_seen_time, 0.01)
                    # Adaptive alpha: faster updates for larger changes
                    delta = abs(raw_pos - horse.position_m)
                    alpha = min(self.ema_alpha * (1 + delta / 50.0), 0.8)
                    old_pos = horse.position_m
                    horse.position_m += alpha * (raw_pos - horse.position_m)
                    horse.speed_mps = abs(horse.position_m - old_pos) / dt

            # Compute ranking (higher position_m = further ahead = lower rank number)
            visible = [h for h in self._horses.values()
                       if now - h.last_seen_time < self.stale_timeout or h.observation_count > 0]

            visible.sort(key=lambda h: -h.position_m)
            for i, horse in enumerate(visible):
                horse.rank = i + 1

            self._ranking = list(visible)
            self._update_count += 1

    def get_ranking(self) -> list[dict]:
        """Return current ranking as list of dicts (for WebSocket/API)."""
        with self._lock:
            now = time.time()
            return [
                {
                    "color": h.color,
                    "rank": h.rank,
                    "position_m": round(h.position_m, 1),
                    "speed_mps": round(h.speed_mps, 1),
                    "last_camera": h.last_camera,
                    "stale": (now - h.last_seen_time) > self.stale_timeout
                    if h.observation_count > 0 else True,
                    "observations": h.observation_count,
                }
                for h in self._ranking
            ]

    def get_horse_positions(self) -> dict[str, float]:
        """Return {color: position_m} for all horses."""
        with self._lock:
            return {
                h.color: h.position_m
                for h in self._horses.values()
            }

    def reset(self):
        """Reset all horse states (new race)."""
        with self._lock:
            for horse in self._horses.values():
                horse.position_m = 0.0
                horse.raw_position_m = 0.0
                horse.speed_mps = 0.0
                horse.last_seen_time = 0.0
                horse.last_camera = ""
                horse.observation_count = 0
                horse.rank = 0
            self._ranking.clear()
            self._update_count = 0

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "update_count": self._update_count,
                "horses_tracked": sum(
                    1 for h in self._horses.values() if h.observation_count > 0
                ),
                "ranking_size": len(self._ranking),
            }
