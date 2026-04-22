"""
frame_server.py — on-demand extraction of frames and detection crops from
source MP4 recordings. Results are cached on disk under
  $ARTIFACTS_DIR/frames/<rec>/<cam>/<seq>.jpg
  $ARTIFACTS_DIR/crops/<rec>/<cam>/<seq>_<det>.jpg

Strategy:
  - Use OpenCV VideoCapture with frame-seek (CAP_PROP_POS_FRAMES).
  - Cache JPEGs after first read; subsequent requests are served from disk.
  - Annotated frames (with bbox) are produced at request time with
    plain cv2.rectangle; no video re-encoding.

The server is stateless — all inputs come via explicit paths passed in
by the router. Thread-safe: each request opens its own VideoCapture.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("api.analytics.frame_server")

COLOR_MAP_BGR: dict[str, tuple[int, int, int]] = {
    "red":    (0, 0, 255),
    "blue":   (255, 0, 0),
    "green":  (0, 200, 0),
    "yellow": (0, 255, 255),
    "purple": (200, 0, 200),
    "unknown": (128, 128, 128),
}


class FrameServer:
    def __init__(self, artifacts_dir: str | Path):
        self.root = Path(artifacts_dir)
        self.frames_dir = self.root / "frames"
        self.crops_dir = self.root / "crops"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_frame(self, rec_id: str, cam_id: str, mp4_path: str,
                  frame_seq: int,
                  annotate_with: Optional[list[dict]] = None,
                  quality: int = 85) -> Optional[Path]:
        """Extract frame as JPEG. Returns cached path or None on failure.

        annotate_with: optional list of detection dicts for bbox overlay
                       (keys: bbox_x1, bbox_y1, bbox_x2, bbox_y2, color, conf)
        """
        cache_key = "annot" if annotate_with else "raw"
        cache_path = (self.frames_dir / rec_id / cam_id /
                      f"{frame_seq}_{cache_key}.jpg")
        if cache_path.exists():
            return cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        frame = self._read_frame(mp4_path, frame_seq)
        if frame is None:
            return None
        if annotate_with:
            frame = self._annotate(frame, annotate_with)
        cv2.imwrite(str(cache_path),
                    frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return cache_path

    def get_crop(self, rec_id: str, cam_id: str, mp4_path: str,
                 frame_seq: int, det_id: int,
                 bbox: tuple[float, float, float, float],
                 pad: float = 0.10,
                 quality: int = 90) -> Optional[Path]:
        """Extract a crop around bbox (with optional padding). Cached."""
        cache_path = (self.crops_dir / rec_id / cam_id /
                      f"{frame_seq}_{det_id}.jpg")
        if cache_path.exists():
            return cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        frame = self._read_frame(mp4_path, frame_seq)
        if frame is None:
            return None
        crop = self._crop_with_pad(frame, bbox, pad)
        if crop is None or crop.size == 0:
            return None
        cv2.imwrite(str(cache_path),
                    crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return cache_path

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _read_frame(self, mp4_path: str, frame_seq: int) -> Optional[np.ndarray]:
        if not mp4_path or not Path(mp4_path).exists():
            log.warning("mp4 not found: %s", mp4_path)
            return None
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            return None
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_seq))
            ok, frame = cap.read()
            return frame if ok else None
        finally:
            cap.release()

    @staticmethod
    def _crop_with_pad(frame: np.ndarray,
                       bbox: tuple[float, float, float, float],
                       pad: float) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            return None
        px = bw * pad
        py = bh * pad
        x1p = int(max(0, x1 - px))
        y1p = int(max(0, y1 - py))
        x2p = int(min(w, x2 + px))
        y2p = int(min(h, y2 + py))
        if x2p <= x1p or y2p <= y1p:
            return None
        return frame[y1p:y2p, x1p:x2p].copy()

    @staticmethod
    def _annotate(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        out = frame.copy()
        for det in detections:
            x1 = int(det.get("bbox_x1", 0))
            y1 = int(det.get("bbox_y1", 0))
            x2 = int(det.get("bbox_x2", 0))
            y2 = int(det.get("bbox_y2", 0))
            color = det.get("color") or "unknown"
            bgr = COLOR_MAP_BGR.get(color, COLOR_MAP_BGR["unknown"])
            cv2.rectangle(out, (x1, y1), (x2, y2), bgr, 2)
            conf = det.get("conf", 0)
            label = f"{color} {conf:.0f}%" if conf else color
            tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), bgr, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
        return out
