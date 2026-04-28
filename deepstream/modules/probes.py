"""Standalone probes used in the DeepStream pipeline.

Currently hosts the DINOv2-era tight-bbox preprocessing probe and its
configuration constants. DetectionProbe (the main per-frame probe) still
lives in deepstream/pipeline.py until a later refactor step.
"""
from __future__ import annotations

import os

from pyservicemaker import BatchMetadataOperator


# COCO person class id, also our PGIE primary class. Defined here so probes/
# don't depend on pipeline.py at import time. pipeline.py keeps its own copy
# for DetectionProbe; the values must stay in sync (both 0).
PERSON_CLASS_ID = 0


# Tight crop preprocessing for DINOv2 SGIE — отрезаем голову/ноги/лошадь/
# соседей, оставляем только торс с силком. Применяется через bbox-rewrite
# в pre-SGIE probe; post-SGIE probe восстанавливает original bbox через
# обратную математику (no shared state, no race conditions).
#  X: 10-70% (центр сдвинут влево, отрезает соседей)
#  Y: 0-55% (от верха до середины, отрезает ноги/лошадь)
TIGHT_X_LO, TIGHT_X_HI = 0.10, 0.70   # → ширина = 0.60 от original
TIGHT_Y_LO, TIGHT_Y_HI = 0.00, 0.55   # → высота = 0.55 от original
_TIGHT_W_FRAC = TIGHT_X_HI - TIGHT_X_LO   # 0.60
_TIGHT_H_FRAC = TIGHT_Y_HI - TIGHT_Y_LO   # 0.55

# Tight bbox preprocessing was added for DINOv2 (which needed torso-only crops
# to avoid head/legs/horse contamination of the embedding). OSNet was trained
# on full-body person crops at 256x128 stretch and works on full bbox directly.
# Disable via RV_TIGHT_SGIE=0 when running OSNet SGIE; default 1 keeps the
# behaviour for DINOv2 fallback. Gates BOTH the pre-SGIE shrink probe AND the
# post-SGIE reverse-math in DetectionProbe — they must move together.
TIGHT_SGIE_ENABLED = os.environ.get("RV_TIGHT_SGIE", "1") == "1"


class TightBboxShrinkProbe(BatchMetadataOperator):
    """Pre-SGIE probe: shrink rect_params per-obj to torso-only zone.
    SGIE (process-mode=2) crop'ит NvBufSurface по obj.rect_params, поэтому
    модификация bbox здесь = SGIE подаст в TRT engine только torso region.
    DetectionProbe потом восстанавливает original bbox через reverse math."""

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            for obj in frame_meta.object_items:
                if obj.class_id != PERSON_CLASS_ID:
                    continue
                rp = obj.rect_params
                orig_w = rp.width
                orig_h = rp.height
                rp.left   = rp.left + TIGHT_X_LO * orig_w
                rp.top    = rp.top  + TIGHT_Y_LO * orig_h
                rp.width  = _TIGHT_W_FRAC * orig_w
                rp.height = _TIGHT_H_FRAC * orig_h
