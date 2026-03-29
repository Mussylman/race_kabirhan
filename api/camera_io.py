"""
Race Vision — Camera I/O: video file grabber, RTSP grabber, go2rtc monitor.

Imports ONLY from api.shared (no other api.* modules).
"""

import time
import json
import asyncio
import threading
import numpy as np
import urllib.request
from pathlib import Path
from typing import Optional

from api.shared import (
    state, frame_store, log,
    CameraManager,
    _MultiCameraReader,
)

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
        import cv2
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
        import cv2
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
