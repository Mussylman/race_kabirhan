# DeepStream Detection Quality Audit Report

**Date:** 2026-04-15
**Scope:** Full code audit — why DeepStream pipeline has worse detection recall than Python inference
**Repository:** mussylman/race_kabirhan

---

## Executive Summary

The audit identified **7 critical/high issues** and **6 moderate/low issues** that collectively explain why DeepStream detection quality is worse than Python inference. The **#1 root cause** is `model-color-format=1` (BGR) in the nvinfer config — the YOLO model receives BGR input instead of the RGB it was trained on. Combined with double NMS suppression and a 3-color vs 5-color mismatch in the Python pipeline, these bugs cascade to produce significantly degraded recall and broken downstream logic.

---

## CRITICAL Issues

### C1. model-color-format=1 delivers BGR to a model trained on RGB

**Priority: CRITICAL — Single biggest root cause of detection quality loss**

| File | Line | Value |
|------|------|-------|
| `deepstream/configs/nvinfer_jockey.txt` | 35 | `model-color-format=1` |
| `deepstream/configs/archive/nvinfer_yolov8s.txt` | 36 | `model-color-format=1` |
| `deepstream/configs/archive/nvinfer_jockey_1cam.txt` | 35 | `model-color-format=1` |
| `deepstream/configs/archive/nvinfer_jockey_5cam.txt` | 36 | `model-color-format=1` |

**The bug:** In DeepStream nvinfer:
- `model-color-format=0` = **RGB**
- `model-color-format=1` = **BGR**

The archive config `nvinfer_yolov8s.txt:35` has a **wrong comment**: `# 0=BGR, 1=RGB — YOLOv8 expects RGB`. This comment is **backwards** from NVIDIA's actual documentation. All subsequent configs copied this wrong value.

**Impact:** YOLO (Ultralytics) is trained with RGB input. Setting `model-color-format=1` delivers BGR, swapping the Red and Blue channels. The model still detects some objects (edge/shape features are color-agnostic) but recall drops substantially — especially for smaller or distant jockeys where color-dependent features matter.

**Python reference:** `pipeline/trt_inference.py:123` documents frames as BGR (OpenCV), and Ultralytics internally converts BGR→RGB before inference. DeepStream should deliver RGB (`model-color-format=0`).

**Fix:**
```diff
- model-color-format=1
+ model-color-format=0
```

---

### C2. Double NMS: parser NMS (0.45) + DeepStream cluster-mode=2 NMS

**Priority: CRITICAL — Suppresses valid closely-spaced jockeys**

| File | Line | Setting |
|------|------|---------|
| `deepstream/src/yolo_parser.cpp` | 25, 161 | `NMS_IOU_THRESH = 0.45f`, `nms(raw_dets, NMS_IOU_THRESH)` |
| `deepstream/configs/nvinfer_jockey.txt` | 38 | `cluster-mode=2` (NMS) |

**The bug:** The custom parser applies greedy NMS with IoU=0.45 at `yolo_parser.cpp:161`. Then nvinfer's `cluster-mode=2` applies **a second NMS pass** with its default threshold. Detections pass through NMS twice.

**Impact:** In crowded horse race scenes with closely bunched jockeys, the second NMS pass can suppress valid detections that survived the first pass. Python uses a single NMS pass (ultralytics, IoU=0.5).

**Fix:** Change `cluster-mode=2` to `cluster-mode=4` (no clustering) in nvinfer config, since the parser already handles NMS:
```diff
- cluster-mode=2
+ cluster-mode=4
```

---

### C3. COLOR_TO_HORSE has only 3 colors, but DeepStream detects 5

**Priority: CRITICAL — 40% of detections silently discarded**

| File | Line | Issue |
|------|------|-------|
| `api/shared.py` | 44-48 | `COLOR_TO_HORSE` only maps `red`, `green`, `yellow` |
| `api/shared.py` | 50 | `ALL_COLORS = list(COLOR_TO_HORSE.keys())` = 3 colors |
| `deepstream/src/config.h` | 28-34 | C++ enum: `blue=0, green=1, purple=2, red=3, yellow=4` |
| `deepstream/src/color_infer.h` | 104 | `MODEL_NUM_CLASSES = 5` |

**The bug:** DeepStream's C++ pipeline classifies into 5 colors (blue, green, purple, red, yellow) and writes all 5 to shared memory. But the Python side (`api/shared.py:50`) only knows 3 colors. Any detection classified as `blue` or `purple` is silently dropped at:
- `api/deepstream_pipeline.py:516-518`: `COLOR_TO_HORSE.get(color)` returns `None` → excluded from rankings
- `pipeline/fusion.py:93-95`: observations for unknown colors are discarded
- `pipeline/vote_engine.py:42`: VoteEngine initialized with only 3 colors

**Fix:** Add all 5 colors to `COLOR_TO_HORSE` in `api/shared.py`:
```python
COLOR_TO_HORSE = {
    "blue":   {"id": "horse-1", "number": 1, ...},
    "green":  {"id": "horse-2", "number": 2, ...},
    "purple": {"id": "horse-3", "number": 3, ...},
    "red":    {"id": "horse-4", "number": 4, ...},
    "yellow": {"id": "horse-5", "number": 5, ...},
}
```

---

### C4. `visible_colors >= 5` check is dead code with only 3 colors

**Priority: CRITICAL — Completion logic never uses "confident + grace" path**

| File | Line | Issue |
|------|------|-------|
| `api/deepstream_pipeline.py` | 442 | `if len(visible_colors) >= 5` — impossible with 3 colors |
| `api/deepstream_pipeline.py` | 282 | `VoteEngine(ALL_COLORS)` — only 3 colors |
| `pipeline/vote_engine.py` | 220 | `len(self.position_votes) >= self.n_colors` — only needs 3 |

**The bug:** The "all visible" check at line 442 requires `>= 5` unique colors to be seen in a single frame. But `ALL_COLORS` contains only 3, so `visible_colors` can never reach 5. This means:
- Completion Condition 1 (confident + grace period) **never fires**
- The system always falls through to Condition 2 (8 vote frames) or Condition 3/4 (timeout)
- The grace period logic is completely bypassed

**Fix:** Change the hardcoded 5 to `len(ALL_COLORS)`:
```diff
- if len(visible_colors) >= 5 and cam_id not in self._cam_all_visible_time:
+ if len(visible_colors) >= len(ALL_COLORS) and cam_id not in self._cam_all_visible_time:
```

---

## HIGH Issues

### H1. Torso crop mismatch: DeepStream uses full bbox, Python uses torso sub-crop

**Priority: HIGH — Color classification feeds completely different image regions**

| File | Line | Value |
|------|------|-------|
| `deepstream/src/color_infer.h` | 77-80 | `TORSO_TOP=0.0, TORSO_BOTTOM=1.0, TORSO_LEFT=0.0, TORSO_RIGHT=0.0` |
| `pipeline/analyzer.py` | 41-44 | `TORSO_TOP=0.10, TORSO_BOTTOM=0.40, TORSO_LEFT=0.20, TORSO_RIGHT=0.20` |

**The bug:** DeepStream feeds the **entire bounding box** (100% of the person) to the color classifier, including horse legs, ground, and background. Python feeds only the **torso sub-region** (10-40% height, 20% side margins) — just the jockey's silk uniform.

**Impact:** The color classifier was trained on torso crops. Feeding the full bbox includes horse body, legs, ground texture — noise that degrades color classification accuracy. This directly reduces the quality of `color` and `color_conf` in the SHM output.

**Fix:** Update `color_infer.h` to match Python:
```cpp
static constexpr float TORSO_TOP    = 0.10f;
static constexpr float TORSO_BOTTOM = 0.40f;
static constexpr float TORSO_LEFT   = 0.20f;
static constexpr float TORSO_RIGHT  = 0.20f;
```

---

### H2. ColorTracker (EMA smoother) defined but never used

**Priority: HIGH — Transient color misclassifications pass through unfiltered**

| File | Line | Issue |
|------|------|-------|
| `api/deepstream_pipeline.py` | 27-91 | `ColorTracker` class fully implemented |
| `api/deepstream_pipeline.py` | 230+ | `DeepStreamPipeline.__init__` — never instantiates `ColorTracker` |

**The bug:** The `ColorTracker` class implements EMA smoothing to filter 1-2 frame color flips (e.g., yellow→blue→yellow). It was written specifically for this problem. But it is **never instantiated or called** anywhere. The pipeline in `pipeline.cpp` has its own C++ `ColorSmoother` (pipeline.h:164-170), but the Python-side one is dead code.

**Fix:** Either wire `ColorTracker` into the SHM reader path, or verify the C++ `color_smooth_` is working correctly.

---

### H3. EMA speed calculation bug: `last_seen_time` updated before `dt` computed

**Priority: HIGH — Speed values are always wrong**

| File | Line | Issue |
|------|------|-------|
| `pipeline/fusion.py` | 142 | `horse.last_seen_time = now` — set BEFORE dt is used |
| `pipeline/fusion.py` | 152 | `dt = max(now - horse.last_seen_time, 0.01)` — always 0.01 |

**The bug:** Line 142 sets `horse.last_seen_time = now`. Line 152 computes `dt = max(now - horse.last_seen_time, 0.01)`. Since `last_seen_time` was just set to `now`, the subtraction is 0, and `dt` is always clamped to 0.01 (10ms). All speed calculations use this hardcoded 10ms instead of the actual inter-observation interval.

**Fix:** Move `horse.last_seen_time = now` to AFTER the EMA block:
```python
old_last_seen = horse.last_seen_time
horse.raw_position_m = raw_pos
horse.last_camera = cam_id_str
horse.observation_count += 1
horse.missing_frames = 0
horse.track_confidence = 1.0

# EMA smoothing
if horse.observation_count <= 1:
    horse.position_m = raw_pos
else:
    dt = max(now - old_last_seen, 0.01)
    ...

horse.last_seen_time = now  # move here
```

---

## MODERATE Issues

### M1. NMS IoU threshold mismatch (0.45 vs 0.5)

| Component | Threshold |
|-----------|-----------|
| Python (ultralytics) | `iou=0.5` (`trt_inference.py:84`) |
| DeepStream parser | `NMS_IOU_THRESH=0.45` (`yolo_parser.cpp:25`) |

Lower IoU in DeepStream means more aggressive suppression of overlapping detections.

**Fix:** Change to 0.5 in `yolo_parser.cpp:25`:
```cpp
static constexpr float NMS_IOU_THRESH = 0.50f;
```

---

### M2. Confidence threshold mismatch (0.15 vs 0.35)

| Component | Threshold |
|-----------|-----------|
| Python YOLO detector | `conf=0.35` (`trt_inference.py:83`) |
| DeepStream nvinfer | `pre-cluster-threshold=0.15` (`nvinfer_jockey.txt:47`) |
| DeepStream C++ default | `det_conf=0.35` (`pipeline.h:34`) |

The 0.15 threshold in DeepStream produces many more low-confidence detections that may pollute downstream voting. However, this alone wouldn't reduce recall — it increases false positives.

---

### M3. Letterbox padding value: black (0) vs gray (114)

| Component | Pad Value |
|-----------|-----------|
| Python (ultralytics) | `114` (gray) |
| DeepStream streammux | `0` (black, default for `enable-padding=TRUE`) |

**Impact:** The model was trained with gray padding (114/255 ≈ 0.447 after normalization). DeepStream uses black padding (0.0 after normalization). Detections near the padded borders may be slightly affected.

---

### M4. Color classifier input resolution: 64x64 (C++) vs 128x128 (Python v2)

| File | Line | Resolution |
|------|------|-----------|
| `deepstream/src/color_infer.cpp` | 25 | `CROP_SIZE = 64` |
| `pipeline/trt_inference.py` | 186 | `INPUT_SIZE = 128` (overridable from checkpoint) |
| `tools/export_trt.py` | 158 | `torch.randn(1, 3, 64, 64)` — export at 64x64 |

If the Python runtime loads a v2 model with `img_size=128`, the Python and DeepStream classifiers use different input resolutions. The TRT engine was exported at 64x64, so DeepStream is internally consistent — but Python may use 128x128 with a v2 model.

---

### M5. nvinfer `interval=1` skips every other frame — halves detection rate

| File | Line | Value |
|------|------|-------|
| `deepstream/configs/nvinfer_jockey.txt` | 20 | `interval=1` |

`interval=1` means nvinfer runs inference on every **other** frame (skip 1, process 1). At 25fps per camera, only ~12.5fps get inference. Setting `interval=0` runs inference on every frame. Combined with Python-side frame_skip=6 (`api/deepstream_pipeline.py:249`), the effective detection rate is ~2fps per camera.

---

### M6. Frame skip drops 83% of remaining DeepStream detections

| File | Line | Value |
|------|------|-------|
| `api/deepstream_pipeline.py` | 249 | `self.frame_skip = 6` |

At ~12.5fps from DeepStream (after `interval=1`), only every 6th frame is processed (≈2fps effective). Combined with single-slot DetectionBuffer (line 187-211), most SHM frames are never seen by the voting engine.

---

### M7. Analysis pipeline aspect ratio filter too strict (MIN_ASPECT_RATIO=1.2)

| File | Line | Value |
|------|------|-------|
| `deepstream/src/analysis_pipeline.h` | 110 | `MIN_ASPECT_RATIO = 1.2f` |

The filter computes `bh / max(bw, 1.0f) < MIN_ASPECT_RATIO`. With 1.2, any detection wider than tall (h/w < 1.2) is rejected. Jockeys in certain poses (leaning, crouching) can have bounding boxes that are wider than tall.

---

### M8. Edge filter in analysis pipeline uses wrong coordinate space

| File | Line | Issue |
|------|------|-------|
| `analysis_pipeline.cpp` | 354 | `x2 >= fw - EDGE_MARGIN` where `fw` is original frame width (1920) but `x2` is in mux coords (0-800) |

The edge margin check compares mux-space coordinates (0-800) against original frame dimensions (e.g., 1920). The condition `x2 >= 1910` can never be true when x2 max is 800. The edge filter is **effectively disabled** despite being set to 10.

---

### M9. Speed unit inconsistency between pipelines

| Pipeline | Unit | File:Line |
|----------|------|-----------|
| DeepStream | m/s (raw) | `api/deepstream_pipeline.py:541` |
| Legacy | km/h (×3.6) | `api/legacy_pipeline.py:307` |

Frontend receives different speed units depending on which pipeline is active.

---

## Git History Analysis

### Timeline of Key Changes

| Commit | Date | Change | Impact |
|--------|------|--------|--------|
| `41f8251` | — | YOLOv8 parser created, DS 8.0 upgrade | **First introduction of model-color-format=1** |
| `ebec177` | — | Upgrade to YOLOv11s | Parser NOT updated (still works — same output format) |
| `04626c4` | — | C++ color classifier with 5-class model → 5-slot SHM | SHM writes 5 colors |
| `21dcaf7` | — | Fix batch_id surface indexing | Fixed surface access bug |
| `bf001c7` | — | "bbox letterbox fix" | Fixed some letterbox issues but NOT model-color-format |
| `7278d8a` | — | Fix truncated DeepStreamPipeline methods | Python side still 3 colors |

### Regression Origin

The `model-color-format=1` bug was introduced in commit `41f8251` ("Fix DeepStream pipeline: YOLOv8 parser, DS 8.0 upgrade") when the config was first created with the wrong comment `# 0=BGR, 1=RGB`. This was never caught because:
1. Detection still "works" with swapped channels (reduced recall, not zero)
2. The comment made it look intentional and correct
3. No A/B comparison was done between Python and DeepStream on the same frames

The 3-color vs 5-color mismatch emerged when `04626c4` added the 5-class C++ color classifier but `api/shared.py` was never updated from 3 colors.

---

## Summary: Priority Fix Order

| # | Fix | Impact | Effort |
|---|-----|--------|--------|
| 1 | `model-color-format=0` in nvinfer configs | **Restores RGB input — single biggest quality fix** | 1 line |
| 2 | `cluster-mode=4` in nvinfer configs | **Removes double NMS suppression** | 1 line |
| 3 | Add all 5 colors to `COLOR_TO_HORSE` | **Stops dropping 40% of classifications** | 5 lines |
| 4 | Fix `visible_colors >= 5` to `>= len(ALL_COLORS)` | **Enables confident completion logic** | 1 line |
| 5 | Set torso crop params in `color_infer.h` | **Matches Python torso extraction** | 4 lines |
| 6 | Fix `last_seen_time` ordering in `fusion.py` | **Correct speed calculation** | 3 lines |
| 7 | Wire `ColorTracker` or verify C++ smoothing | **Reduces color classification noise** | ~20 lines |
| 8 | Match NMS IoU to 0.5 in parser | Minor consistency | 1 line |

**Fixes 1-4 are the critical path.** Applying them should bring DeepStream detection quality to parity with Python inference.

---

## Appendix: Additional Pipeline Issues

### A1. Drop warnings silently suppressed

`pipeline.cpp:393-399`: Warnings containing "drop" are silently hidden. This masks legitimate buffer-drop warnings from any pipeline element, not just the display queue.

### A2. Missing nvinfer config files for dual pipeline

The dual pipeline references nonexistent configs:
- `configs/nvinfer_yolov8n_trigger.txt` (trigger)
- `configs/nvinfer_yolov8s_analysis.txt` (analysis)
- `configs/nvinfer_yolov8s.txt` (fallback)

Only `configs/nvinfer_jockey.txt` exists. Dual pipeline mode will fail at startup.

### A3. `tools/export_trt.py` still references YOLOv8 models

The export script was never updated after the YOLOv11 migration. Running it would produce YOLOv8 engines, not YOLOv11. The current TRT engine was likely built from the ONNX file directly by nvinfer's auto-build.

### A4. Analysis pipeline has no nvtracker

`analysis_pipeline.cpp` goes `streammux → nvinfer → probe → fakesink` with no tracker. Track IDs are always 0 in SHM for analysis pipeline results. Per-track color smoothing cannot work without track persistence.

### A5. Dead `max_batch` config in analysis pipeline

`AnalysisConfig.max_batch` (default 8) is stored but never applied to streammux or nvinfer. The analysis streammux allocates batch-size=25 even though max 8 cameras are active, wasting GPU memory.
