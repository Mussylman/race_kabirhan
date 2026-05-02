#!/usr/bin/env python3
"""tight_crop_utils.py — torso-only cropping for DINOv2 SGIE.

Принцип:
  - Y range: 10-50% (от верха bbox = торс с силком, без головы и
    без лошади)
  - X range: 20-80% (центральные 60%, без соседей-жокеев)
  - Fallback на full bbox если tight не имеет смысла (degenerate)

Note: наши crops_index.json bbox в xyxy формате, не xywh.
Поддерживаем оба через явные функции.
"""
from __future__ import annotations
import numpy as np

TIGHT_X_LO, TIGHT_X_HI = 0.10, 0.70   # центр сдвинут влево
TIGHT_Y_LO, TIGHT_Y_HI = 0.00, 0.55   # верх до конца + низ чуть ниже


def tight_crop_xyxy(image: np.ndarray, bbox_xyxy):
    """Tight torso crop. bbox = [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    w = x2 - x1
    h = y2 - y1
    x_start = x1 + int(w * TIGHT_X_LO)
    x_end   = x1 + int(w * TIGHT_X_HI)
    y_start = y1 + int(h * TIGHT_Y_LO)
    y_end   = y1 + int(h * TIGHT_Y_HI)
    H_img, W_img = image.shape[:2]
    x_start = max(0, x_start); y_start = max(0, y_start)
    x_end   = min(W_img, x_end); y_end = min(H_img, y_end)
    if x_end <= x_start or y_end <= y_start:
        # Fallback to original bbox if tight is degenerate (very small bbox)
        return image[max(0,y1):min(H_img,y2), max(0,x1):min(W_img,x2)]
    return image[y_start:y_end, x_start:x_end]


def tight_crop_xywh(image: np.ndarray, bbox_xywh):
    """User's spec signature. bbox = [x, y, w, h]."""
    x, y, w, h = [int(v) for v in bbox_xywh]
    return tight_crop_xyxy(image, [x, y, x+w, y+h])


def tight_crop_from_pre_cropped(crop_image: np.ndarray):
    """Когда мы уже имеем cropped (full bbox) изображение, можно
    извлечь tight как % от его размеров. Для color_dataset crops
    у которых нет original raw_frame + bbox.
    """
    h, w = crop_image.shape[:2]
    x_start = int(w * TIGHT_X_LO); x_end = int(w * TIGHT_X_HI)
    y_start = int(h * TIGHT_Y_LO); y_end = int(h * TIGHT_Y_HI)
    if x_end <= x_start or y_end <= y_start:
        return crop_image  # fallback
    return crop_image[y_start:y_end, x_start:x_end]
