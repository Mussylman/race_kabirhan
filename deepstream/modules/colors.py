"""Color constants and mappings for the DeepStream pipeline.

Extracted from deepstream/pipeline.py during the pipeline refactor — keeps
all color-related state in one place so the probe/builder can stay focused
on streaming concerns.

Names match labels_color.txt and deepstream/src/config.h ColorId enum.
"""
from __future__ import annotations

import os


# Color label → SHM color id (matches deepstream/src/config.h ColorId enum
# and deepstream/configs/labels_color.txt order).
_COLOR_NAME_TO_ID = {
    "blue":   0,
    "green":  1,
    "purple": 2,
    "red":    3,
    "yellow": 4,
}
_COLOR_ID_TO_NAME = {v: k for k, v in _COLOR_NAME_TO_ID.items()}

# Race-specific: today only these colors exist on the track; everything else
# is a spectator / false positive. Override with env RV_ACTIVE_COLORS.
_ACTIVE_COLORS = set(
    os.environ.get("RV_ACTIVE_COLORS", "blue,green,red,yellow").split(",")
)
_ACTIVE_IDS = {cid for name, cid in _COLOR_NAME_TO_ID.items() if name in _ACTIVE_COLORS}

# RGB (0-1) for OSD border_color — bright so it shows well on video
_COLOR_RGB_NORMALISED = {
    "blue":   (0.1, 0.4, 1.0),
    "green":  (0.2, 0.9, 0.2),
    "purple": (0.8, 0.2, 0.8),
    "red":    (1.0, 0.1, 0.1),
    "yellow": (1.0, 0.9, 0.1),
    "?":      (0.6, 0.6, 0.6),
}
