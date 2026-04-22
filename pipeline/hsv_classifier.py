"""
hsv_classifier.py — deterministic silk color classifier via HSV space.

Replaces the CNN color classifier (color_classifier_v3.engine) for cases
where a light, robust, lighting-invariant signal is enough. Runs on CPU
in ~0.1 ms per crop.

Design:
  1. Extract silk region from bbox crop (upper-central).
  2. Mask pixels by saturation + value (drop shadows, background, highlights).
  3. Compute median Hue of masked pixels.
  4. Circular distance to each color centroid → nearest wins.
  5. Confidence = 1 - dist/tolerance, clipped to [0, 1].

Inputs: BGR crop (uint8, H×W×3) already cropped to YOLO bbox.
Output: (color: str, confidence: float) or (None, 0.0) if mask is too small.

Why not CNN:
  - HSV decouples color (H) from brightness (V) → shadow/sun-invariant.
  - Saturation mask drops background, shadows, highlights automatically.
  - Deterministic, debuggable (can visualize H histogram).
  - No GPU, no training, no brittleness to silk angle.
  - Pre-race calibration of centroids (5 min) adapts to actual silks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

log = logging.getLogger("pipeline.hsv_classifier")

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "hsv_centroids.yml"


class HSVClassifier:
    def __init__(self, config_path: str | Path = _DEFAULT_CONFIG_PATH):
        self.config_path = Path(config_path)
        with open(self.config_path, "r") as f:
            cfg = yaml.safe_load(f)

        m = cfg["mask"]
        self.min_saturation: int = int(m["min_saturation"])
        self.min_value: int = int(m["min_value"])
        self.max_value: int = int(m["max_value"])
        self.min_pixels: int = int(m["min_pixels"])

        r = cfg["region"]
        self.y_start: float = float(r["y_start"])
        self.y_end: float = float(r["y_end"])
        self.x_start: float = float(r["x_start"])
        self.x_end: float = float(r["x_end"])

        self.tolerance_h: float = float(cfg["tolerance_h"])
        self.max_distance_h: float = float(cfg["max_distance_h"])

        # Flatten centroids into parallel arrays for vectorized distance.
        colors_cfg = cfg["colors"]
        self.color_names: list[str] = []
        self.centroid_hues: list[float] = []
        self.centroid_color_idx: list[int] = []
        for idx, (name, data) in enumerate(colors_cfg.items()):
            self.color_names.append(name)
            for h in data["hue_centroids"]:
                self.centroid_hues.append(float(h))
                self.centroid_color_idx.append(idx)
        self._centroids_np = np.array(self.centroid_hues, dtype=np.float32)
        self._color_idx_np = np.array(self.centroid_color_idx, dtype=np.int32)

        log.info(
            "HSVClassifier loaded: %d colors, %d centroids (tol=%.1f°, max=%.1f°)",
            len(self.color_names), len(self.centroid_hues),
            self.tolerance_h, self.max_distance_h,
        )

    def classify(self, bgr_crop: np.ndarray) -> tuple[Optional[str], float]:
        """Classify a single BGR crop (uint8, H×W×3). Returns (color, conf)."""
        if bgr_crop is None or bgr_crop.size == 0:
            return None, 0.0
        h_med, n_pixels = self._extract_median_hue(bgr_crop)
        if h_med is None or n_pixels < self.min_pixels:
            return None, 0.0
        return self._match_centroid(h_med)

    def classify_batch(self, bgr_crops: list[np.ndarray]) -> list[tuple[Optional[str], float]]:
        """Classify a list of crops. Simple Python loop — ~0.1ms each so no
        advantage to vectorising across crops (region sizes vary)."""
        return [self.classify(c) for c in bgr_crops]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _extract_median_hue(self, bgr: np.ndarray) -> tuple[Optional[float], int]:
        h, w = bgr.shape[:2]
        y0 = int(h * self.y_start)
        y1 = int(h * self.y_end)
        x0 = int(w * self.x_start)
        x1 = int(w * self.x_end)
        if y1 <= y0 or x1 <= x0:
            return None, 0
        region = bgr[y0:y1, x0:x1]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        mask = (sat >= self.min_saturation) & (val >= self.min_value) & (val <= self.max_value)
        n = int(mask.sum())
        if n < self.min_pixels:
            return None, n
        h_pixels = hsv[:, :, 0][mask]
        # Circular median: rotate distribution so variance is minimised, then median.
        # For H in [0, 180), simple median works unless distribution wraps around 0.
        # Quick check: if pixels split across [0, 20] and [160, 180], use circular.
        low_count = int((h_pixels < 20).sum())
        high_count = int((h_pixels > 160).sum())
        total = len(h_pixels)
        if low_count > 0.2 * total and high_count > 0.2 * total:
            # Wraparound case — rotate by 90, median, rotate back
            rotated = (h_pixels.astype(np.int32) + 90) % 180
            med_rotated = float(np.median(rotated))
            med = (med_rotated - 90) % 180
            return med, n
        return float(np.median(h_pixels)), n

    def _match_centroid(self, h_med: float) -> tuple[Optional[str], float]:
        # Circular distance in [0, 90]
        diff = np.abs(self._centroids_np - h_med)
        circ_dist = np.minimum(diff, 180.0 - diff)
        best_idx = int(np.argmin(circ_dist))
        best_dist = float(circ_dist[best_idx])
        if best_dist > self.max_distance_h:
            return None, 0.0
        color = self.color_names[int(self._color_idx_np[best_idx])]
        confidence = max(0.0, 1.0 - best_dist / self.tolerance_h)
        confidence = min(1.0, confidence)
        return color, float(confidence)
