"""Lightweight detection data classes (no torch dependency).

Shared between analyzer.py (PyTorch mode) and shm_reader.py (DeepStream mode).
"""

import time


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
