"""
camera_manager.py — Manages 25 analytics + 4 display cameras.

Responsibilities:
    - Track which cameras are active (horses detected by trigger)
    - Provide camera lists for trigger loop (all 25) and analysis loop (active only)
    - Enforce max active cameras (GPU budget limit)
    - Cooldown: keep camera active for N seconds after last trigger

Usage:
    mgr = CameraManager()
    mgr.add_analytics("cam-01", "rtsp://...", track_start=0, track_end=100)
    mgr.add_display("ptz-1", "rtsp://...")

    # Called by trigger loop:
    mgr.activate("cam-03")      # horses detected
    mgr.deactivate("cam-03")    # no horses

    # Called by analysis loop:
    active = mgr.get_active_cameras()  # list of CameraInfo
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("pipeline.camera_manager")


@dataclass
class CameraInfo:
    """Metadata for a single camera."""
    cam_id: str
    source: str                     # RTSP URL or video path
    role: str = "analytics"         # "analytics" or "display"
    track_start_m: float = 0.0      # track position at left edge (meters)
    track_end_m: float = 100.0      # track position at right edge (meters)
    frame_width: int = 1920
    frame_height: int = 1080

    # Runtime state
    active: bool = False
    last_trigger_time: float = 0.0
    last_detection_count: int = 0   # detections in last trigger frame
    connected: bool = False
    fps: float = 0.0


class CameraManager:
    """Thread-safe manager for all cameras."""

    def __init__(
        self,
        max_active: int = 8,
        cooldown_sec: float = 3.0,
    ):
        self.max_active = max_active
        self.cooldown_sec = cooldown_sec
        self._lock = threading.Lock()
        self._cameras: dict[str, CameraInfo] = {}

    # ── Camera registration ───────────────────────────────────────────

    def add_analytics(
        self,
        cam_id: str,
        source: str,
        track_start: float = 0.0,
        track_end: float = 100.0,
        width: int = 1920,
        height: int = 1080,
    ) -> CameraInfo:
        """Register an analytics camera."""
        cam = CameraInfo(
            cam_id=cam_id,
            source=source,
            role="analytics",
            track_start_m=track_start,
            track_end_m=track_end,
            frame_width=width,
            frame_height=height,
        )
        with self._lock:
            self._cameras[cam_id] = cam
        return cam

    def add_display(
        self,
        cam_id: str,
        source: str,
        width: int = 1920,
        height: int = 1080,
    ) -> CameraInfo:
        """Register a display-only camera (no analysis)."""
        cam = CameraInfo(
            cam_id=cam_id,
            source=source,
            role="display",
            frame_width=width,
            frame_height=height,
        )
        with self._lock:
            self._cameras[cam_id] = cam
        return cam

    # ── Activation (called by trigger) ────────────────────────────────

    def activate(self, cam_id: str, detection_count: int = 1):
        """Mark camera as active (horses detected)."""
        with self._lock:
            cam = self._cameras.get(cam_id)
            if cam and cam.role == "analytics":
                cam.active = True
                cam.last_trigger_time = time.monotonic()
                cam.last_detection_count = detection_count

    def deactivate(self, cam_id: str):
        """Mark camera as inactive (no horses). Respects cooldown."""
        with self._lock:
            cam = self._cameras.get(cam_id)
            if cam and cam.role == "analytics":
                elapsed = time.monotonic() - cam.last_trigger_time
                if elapsed >= self.cooldown_sec:
                    cam.active = False

    def update_trigger_results(self, results: dict[str, int]):
        """Batch update from trigger loop.

        Args:
            results: {cam_id: detection_count} — 0 means no horses.
        """
        now = time.monotonic()
        with self._lock:
            for cam_id, count in results.items():
                cam = self._cameras.get(cam_id)
                if not cam or cam.role != "analytics":
                    continue

                if count > 0:
                    cam.active = True
                    cam.last_trigger_time = now
                    cam.last_detection_count = count
                else:
                    # Apply cooldown
                    elapsed = now - cam.last_trigger_time
                    if elapsed >= self.cooldown_sec:
                        cam.active = False

            # Enforce max active: keep the ones with most recent triggers
            self._enforce_max_active()

    def _enforce_max_active(self):
        """If more than max_active cameras, deactivate oldest ones."""
        active = [c for c in self._cameras.values()
                  if c.role == "analytics" and c.active]
        if len(active) <= self.max_active:
            return

        # Sort by trigger time descending (keep most recent)
        active.sort(key=lambda c: c.last_trigger_time, reverse=True)
        for cam in active[self.max_active:]:
            cam.active = False
            log.debug("Deactivated %s (max_active=%d exceeded)", cam.cam_id, self.max_active)

    # ── Queries ───────────────────────────────────────────────────────

    def get_analytics_cameras(self) -> list[CameraInfo]:
        """All analytics cameras (for trigger loop)."""
        with self._lock:
            return [c for c in self._cameras.values() if c.role == "analytics"]

    def get_active_cameras(self) -> list[CameraInfo]:
        """Currently active analytics cameras (for analysis loop)."""
        with self._lock:
            return [c for c in self._cameras.values()
                    if c.role == "analytics" and c.active]

    def get_display_cameras(self) -> list[CameraInfo]:
        """Display-only cameras."""
        with self._lock:
            return [c for c in self._cameras.values() if c.role == "display"]

    def get_camera(self, cam_id: str) -> Optional[CameraInfo]:
        with self._lock:
            return self._cameras.get(cam_id)

    def get_all_cameras(self) -> list[CameraInfo]:
        with self._lock:
            return list(self._cameras.values())

    def get_activation_map(self) -> dict[str, bool]:
        """Return {cam_id: active} for all analytics cameras."""
        with self._lock:
            return {
                c.cam_id: c.active
                for c in self._cameras.values()
                if c.role == "analytics"
            }

    def get_status(self) -> dict:
        """Return full status dict for API/WebSocket."""
        with self._lock:
            analytics = [c for c in self._cameras.values() if c.role == "analytics"]
            active = [c for c in analytics if c.active]
            display = [c for c in self._cameras.values() if c.role == "display"]
            return {
                "total_analytics": len(analytics),
                "active_analytics": len(active),
                "total_display": len(display),
                "cameras": [
                    {
                        "cam_id": c.cam_id,
                        "role": c.role,
                        "active": c.active,
                        "connected": c.connected,
                        "track_segment": f"{c.track_start_m:.0f}-{c.track_end_m:.0f}m"
                        if c.role == "analytics" else None,
                    }
                    for c in self._cameras.values()
                ],
            }

    def set_connected(self, cam_id: str, connected: bool):
        """Update connection status (called by frame grabbers)."""
        with self._lock:
            cam = self._cameras.get(cam_id)
            if cam:
                cam.connected = connected
