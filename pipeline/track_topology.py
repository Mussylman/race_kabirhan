"""
track_topology.py — Maps camera pixel coordinates to global track position.

The track is a 2500m oval. 25 cameras are placed around it, each covering
a segment.  This module converts (camera_id, pixel_x) → global_position_m.

Key concepts:
    - Each camera has a [track_start, track_end] range in meters
    - pixel_x=0 (left edge) → track_start, pixel_x=frame_width → track_end
    - Cameras overlap by ~10m at boundaries (for handoff)
    - When horses are seen by 2 cameras simultaneously, we average positions

Usage:
    topo = TrackTopology(track_length=2500)
    topo.add_camera("cam-01", track_start=0, track_end=110, frame_width=1920)
    topo.add_camera("cam-02", track_start=100, track_end=210, frame_width=1920)

    pos = topo.pixel_to_track("cam-01", pixel_x=960)  # → 55.0m
    pos = topo.pixel_to_track("cam-02", pixel_x=0)    # → 100.0m
"""

import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("pipeline.track_topology")


@dataclass
class CameraSegment:
    """A camera's coverage on the track."""
    cam_id: str
    track_start_m: float    # position at left edge of frame
    track_end_m: float      # position at right edge of frame
    frame_width: int = 1920
    inverted: bool = False  # True if horses move right-to-left (decreasing track_m)

    @property
    def track_length_m(self) -> float:
        return abs(self.track_end_m - self.track_start_m)

    @property
    def meters_per_pixel(self) -> float:
        return self.track_length_m / max(self.frame_width, 1)


class TrackTopology:
    """Converts (camera, pixel_x) → global track position in meters."""

    def __init__(self, track_length: float = 2500.0, overlap_m: float = 10.0):
        self.track_length = track_length
        self.overlap_m = overlap_m
        self._segments: dict[str, CameraSegment] = {}

    def add_camera(
        self,
        cam_id: str,
        track_start: float,
        track_end: float,
        frame_width: int = 1920,
        inverted: bool = False,
    ):
        """Register a camera's track segment."""
        seg = CameraSegment(
            cam_id=cam_id,
            track_start_m=track_start,
            track_end_m=track_end,
            frame_width=frame_width,
            inverted=inverted,
        )
        self._segments[cam_id] = seg
        log.debug("Camera %s: %.0f–%.0fm (%.2f m/px)",
                  cam_id, track_start, track_end, seg.meters_per_pixel)

    def auto_distribute(self, n_cameras: int, frame_width: int = 1920):
        """Evenly distribute N cameras around the track with overlap.

        Creates cameras named "cam-01" .. "cam-NN", each covering
        track_length/n_cameras + overlap_m.
        """
        segment_len = self.track_length / n_cameras

        for i in range(n_cameras):
            cam_id = f"cam-{i+1:02d}"
            start = i * segment_len
            end = start + segment_len + self.overlap_m
            # Wrap around for the last camera
            if end > self.track_length:
                end = self.track_length
            self.add_camera(cam_id, start, end, frame_width)

    # ── Conversion ────────────────────────────────────────────────────

    def pixel_to_track(self, cam_id: str, pixel_x: float) -> Optional[float]:
        """Convert pixel X coordinate to global track position (meters).

        Returns None if camera not registered.
        """
        seg = self._segments.get(cam_id)
        if seg is None:
            return None

        # Normalize pixel to [0, 1]
        t = pixel_x / max(seg.frame_width, 1)
        t = max(0.0, min(1.0, t))

        if seg.inverted:
            t = 1.0 - t

        return seg.track_start_m + t * seg.track_length_m

    def track_to_pixel(self, cam_id: str, track_m: float) -> Optional[float]:
        """Convert global track position to pixel X (inverse of pixel_to_track)."""
        seg = self._segments.get(cam_id)
        if seg is None:
            return None

        if seg.track_length_m == 0:
            return 0.0

        t = (track_m - seg.track_start_m) / seg.track_length_m
        if seg.inverted:
            t = 1.0 - t

        return t * seg.frame_width

    # ── Multi-camera merge ────────────────────────────────────────────

    def merge_positions(
        self,
        observations: list[tuple[str, float]],
    ) -> Optional[float]:
        """Merge position observations from multiple cameras.

        Args:
            observations: list of (cam_id, pixel_x) pairs for the same horse.

        Returns:
            Weighted average global position (meters), or None if empty.
            Weight favors observations from the center of camera FOV.
        """
        if not observations:
            return None

        total_weight = 0.0
        weighted_pos = 0.0

        for cam_id, pixel_x in observations:
            pos = self.pixel_to_track(cam_id, pixel_x)
            if pos is None:
                continue

            seg = self._segments[cam_id]
            # Weight: higher at center of FOV, lower at edges
            t = pixel_x / max(seg.frame_width, 1)
            # Parabolic weight: max at center (t=0.5), zero at edges
            weight = max(0.01, 1.0 - (2.0 * t - 1.0) ** 2)

            weighted_pos += pos * weight
            total_weight += weight

        return weighted_pos / total_weight if total_weight > 0 else None

    # ── Overlap detection ─────────────────────────────────────────────

    def get_overlapping_cameras(self, cam_id: str) -> list[str]:
        """Return IDs of cameras whose track segments overlap with cam_id."""
        seg = self._segments.get(cam_id)
        if seg is None:
            return []

        overlaps = []
        for other_id, other in self._segments.items():
            if other_id == cam_id:
                continue
            # Overlap if ranges intersect
            if (seg.track_start_m < other.track_end_m and
                    seg.track_end_m > other.track_start_m):
                overlaps.append(other_id)
        return overlaps

    def is_in_overlap_zone(self, cam_id: str, pixel_x: float) -> bool:
        """Check if a pixel position is in the overlap zone with an adjacent camera."""
        track_m = self.pixel_to_track(cam_id, pixel_x)
        if track_m is None:
            return False

        seg = self._segments[cam_id]
        # Check if position is within overlap_m of either edge
        return (track_m - seg.track_start_m < self.overlap_m or
                seg.track_end_m - track_m < self.overlap_m)

    # ── Queries ───────────────────────────────────────────────────────

    def get_segment(self, cam_id: str) -> Optional[CameraSegment]:
        return self._segments.get(cam_id)

    def get_cameras_at(self, track_m: float) -> list[str]:
        """Return cameras that cover a given track position."""
        result = []
        for cam_id, seg in self._segments.items():
            if seg.track_start_m <= track_m <= seg.track_end_m:
                result.append(cam_id)
        return result

    def camera_center(self, cam_id: str) -> Optional[float]:
        """Midpoint of a camera's track coverage (meters)."""
        seg = self._segments.get(cam_id)
        if seg is None:
            return None
        return 0.5 * (seg.track_start_m + seg.track_end_m)

    def distance_along(self, cam_a: str, cam_b: str,
                       direction: str = "forward") -> Optional[float]:
        """Signed distance along the track from cam_a's center to cam_b's center.

        Args:
            cam_a, cam_b: camera ids
            direction: "forward"  → positive if cam_b is ahead of cam_a
                       "shortest" → always positive, wraps around track
                       "signed"   → simple (cam_b - cam_a), can be negative

        Returns None if either camera is unknown.

        Used by IdentityResolver's spatio-temporal mask:
            dt = now - last_seen.ts
            dx = topo.distance_along(last_seen.cam, current_cam, "forward")
            reachable = 0 <= dx <= v_max * (dt + slack)
        """
        a = self.camera_center(cam_a)
        b = self.camera_center(cam_b)
        if a is None or b is None:
            return None
        raw = b - a
        if direction == "signed":
            return raw
        if direction == "shortest":
            L = self.track_length
            if L <= 0:
                return abs(raw)
            d = raw % L
            return min(d, L - d)
        # forward (default): positive ahead, wraps around for an oval track
        L = self.track_length
        if L <= 0:
            return raw
        return raw % L

    @property
    def cameras(self) -> dict[str, CameraSegment]:
        return dict(self._segments)
