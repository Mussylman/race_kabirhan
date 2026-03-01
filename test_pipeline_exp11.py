"""
Test trigger + analyzer pipeline on exp11 videos.
Reads 4 video files, feeds frames to trigger → analyzer, logs everything.
"""

import sys
import os
import time
import logging
import cv2
import threading
from pathlib import Path

# Add project root
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.camera_manager import CameraManager
from pipeline.trigger import TriggerLoop
from pipeline.analyzer import AnalysisLoop

# ── Logging ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_exp11")

# ── Video frame source ───────────────────────────────────────────────

class VideoFrameSource:
    """Reads frames from video files, simulating live cameras."""

    def __init__(self, video_map: dict[str, str], fps: float = 10.0):
        """
        Args:
            video_map: {cam_id: video_path}
            fps: playback speed
        """
        self.caps: dict[str, cv2.VideoCapture] = {}
        self.frames: dict[str, any] = {}
        self.lock = threading.Lock()
        self.fps = fps
        self.running = False
        self.finished_count = 0
        self.total = len(video_map)

        for cam_id, path in video_map.items():
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                self.caps[cam_id] = cap
                log.info("Opened %s: %s (%.0f frames, %.1f fps)",
                         cam_id, path,
                         cap.get(cv2.CAP_PROP_FRAME_COUNT),
                         cap.get(cv2.CAP_PROP_FPS))
            else:
                log.error("Cannot open %s: %s", cam_id, path)

    def get_frame(self, cam_id: str):
        """Called by trigger/analyzer to get latest frame."""
        with self.lock:
            return self.frames.get(cam_id)

    def start(self):
        """Start reading frames in background threads."""
        self.running = True
        for cam_id in self.caps:
            t = threading.Thread(target=self._read_loop, args=(cam_id,), daemon=True)
            t.start()

    def _read_loop(self, cam_id: str):
        cap = self.caps[cam_id]
        interval = 1.0 / self.fps
        frame_num = 0

        while self.running and cap.isOpened():
            t0 = time.monotonic()
            ret, frame = cap.read()
            if not ret:
                log.info("%s: video ended at frame %d", cam_id, frame_num)
                with self.lock:
                    self.finished_count += 1
                break

            with self.lock:
                self.frames[cam_id] = frame
            frame_num += 1

            elapsed = time.monotonic() - t0
            sleep = interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

        cap.release()

    def all_finished(self) -> bool:
        return self.finished_count >= self.total

    def stop(self):
        self.running = False
        for cap in self.caps.values():
            cap.release()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    video_dir = Path("data/videos")

    # exp11 videos — 4 cameras
    video_map = {
        "exp11-cam1": str(video_dir / "exp11_cam1.mp4"),
        "exp11-cam2": str(video_dir / "exp11_cam2.mp4"),
        "exp11-cam3": str(video_dir / "exp11_cam3.mp4"),
        "exp11-cam4": str(video_dir / "exp11_cam4.mp4"),
    }

    # Check files exist
    for cam_id, path in video_map.items():
        if not Path(path).exists():
            log.error("Video not found: %s", path)
            return

    # ── Setup ────────────────────────────────────────────────────────

    camera_mgr = CameraManager(max_active=4, cooldown_sec=2.0)

    # Register cameras
    for i, cam_id in enumerate(video_map):
        camera_mgr.add_analytics(
            cam_id=cam_id,
            source=video_map[cam_id],
            track_start=i * 100,
            track_end=(i + 1) * 100,
        )

    # Frame source (10 fps playback)
    frame_source = VideoFrameSource(video_map, fps=10.0)

    # Results callback
    def on_analysis_result(results):
        for r in results:
            if r.n_detections > 0:
                log.info("RESULT  %s: %d detections → %s",
                         r.cam_id, r.n_detections, r.colors)

    # ── Trigger (YOLOv8n @ 640) ──────────────────────────────────────

    trigger = TriggerLoop(
        camera_manager=camera_mgr,
        frame_source=frame_source.get_frame,
        trigger_fps=3.0,
        fallback_pt="yolov8n.pt",
        imgsz=640,
        conf=0.25,
    )

    # ── Analyzer (YOLOv8s @ 1280 + color classifier) ─────────────────

    analyzer = AnalysisLoop(
        camera_manager=camera_mgr,
        frame_source=frame_source.get_frame,
        on_result=on_analysis_result,
        analysis_fps=5.0,
        yolo_fallback="yolov8s.pt",
        classifier_fallback="models/color_classifier.pt",
        imgsz=1280,
        det_conf=0.35,
    )

    # ── Run ───────────────────────────────────────────────────────────

    log.info("=" * 60)
    log.info("Starting pipeline test on exp11 (4 cameras)")
    log.info("  Trigger: YOLOv8n @ 640px, 3 fps")
    log.info("  Analyzer: YOLOv8s @ 1280px, 5 fps (active only)")
    log.info("=" * 60)

    frame_source.start()
    time.sleep(0.5)  # let frames buffer

    trigger.start()
    analyzer.start()

    try:
        while not frame_source.all_finished():
            time.sleep(1.0)

            # Print status every second
            active = camera_mgr.get_active_cameras()
            active_ids = [c.cam_id for c in active]
            completed = camera_mgr.get_completed_cameras()
            t_stats = trigger.get_stats()
            a_stats = analyzer.get_stats()

            log.info("STATUS  active=%d/%d [%s]  completed=%d [%s]  trigger=%d  analyzer=%d (%.1f fps)",
                     len(active_ids), len(video_map), ", ".join(active_ids) or "none",
                     len(completed), ", ".join(sorted(completed)) or "none",
                     t_stats["frames_processed"],
                     a_stats["frames_processed"], a_stats["current_fps"])

        # Wait a bit for final processing
        log.info("Videos finished, waiting for pipeline to drain...")
        time.sleep(3.0)

    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        trigger.stop()
        analyzer.stop()
        frame_source.stop()

    # ── Summary ───────────────────────────────────────────────────────

    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("  Trigger: %s", trigger.get_stats())
    log.info("  Analyzer: %s", analyzer.get_stats())
    log.info("  Activation map: %s", camera_mgr.get_activation_map())
    log.info("  Completed: %s", camera_mgr.get_completed_cameras())

    # Print vote results per camera
    for cam_id in video_map:
        result = analyzer.get_vote_result(cam_id)
        if result:
            log.info("  %s result: %s", cam_id, " > ".join(result))

    log.info("=" * 60)


if __name__ == "__main__":
    main()
