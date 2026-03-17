# Jockey Timeline Analysis - 15 Camera Test

**Analysis Date:** 2026-03-06
**Log File:** `/tmp/deepstream_eval.log`
**Camera Configuration:** `cameras_test_15.json` (15 cameras)
**Total Batches Processed:** ~6160

---

## 📊 Executive Summary

### Detection Results
- **Jockeys Detected:** 2 out of 3 expected
  - ✅ Yellow jockey: 11 track IDs, 247 total detections
  - ✅ Green jockey: 11 track IDs, 152 total detections
  - ❌ **Red/Pink jockey: NOT DETECTED** (0 detections)

- **Cameras with Activity:** Only 5 out of 15 cameras detected jockeys
  - Active: cam-01, cam-05, cam-06, cam-08, cam-13
  - Silent: cam-02, cam-03, cam-04, cam-07, cam-09, cam-10, cam-11, cam-12, cam-14, cam-15

- **Track ID Fragmentation:** 22 unique track IDs for 2 jockeys (expected: ~3)
  - Suggests frequent tracker ID reassignment
  - Average 11 track IDs per jockey (high fragmentation)

---

## 🎨 Jockey Color Breakdown

### Yellow Jockey (Detected)
**Total:** 11 track IDs, 247 detections across 4 cameras

| Track ID | Cameras | First Batch | Last Batch | Duration (batches) | Detections |
|----------|---------|-------------|------------|-------------------|------------|
| 6  | cam-05 | 779  | 803  | 25  | 7  |
| 11 | cam-05, cam-06 | 1005 | 1050 | 45  | 46 |
| 16 | cam-06 | 1305 | 1350 | 46  | 46 |
| 18 | cam-06 | 1455 | 1456 | 2   | 2  |
| 19 | cam-06 | 1571 | 1650 | 80  | 40 |
| 33 | cam-06 | 2806 | 2850 | 45  | 42 |
| 34 | cam-06 | 3042 | 3044 | 3   | 2  |
| 43 | cam-06 | 4469 | 4500 | 32  | 20 |
| 44 | cam-06 | 4709 | 4750 | 42  | 42 |
| 52 | cam-08 | 5661 | 5690 | 30  | 16 |
| 55 | cam-13 | 6015 | 6022 | 8   | 6  |

**Pattern:** Yellow jockey appears primarily on cam-06 (235 detections), with sporadic appearances on cam-05, cam-08, and cam-13.

---

### Green Jockey (Detected)
**Total:** 11 track IDs, 152 detections across 5 cameras

| Track ID | Cameras | First Batch | Last Batch | Duration (batches) | Detections |
|----------|---------|-------------|------------|-------------------|------------|
| 1  | cam-06 | 17   | 24   | 8   | 5  |
| 2  | cam-05 | 73   | 709  | 637 | 13 |
| 12 | cam-05, cam-06 | 1042 | 1050 | 9   | 7  |
| 14 | cam-05 | 1053 | 1070 | 18  | 12 |
| 25 | cam-06 | 2006 | 2074 | 69  | 13 |
| 29 | cam-01 | 2089 | 2104 | 16  | 15 |
| 32 | cam-06 | 2409 | 2450 | 42  | 22 |
| 38 | cam-06 | 3560 | 3594 | 35  | 21 |
| 54 | cam-13 | 6009 | 6026 | 18  | 8  |
| 56 | cam-13 | 6023 | 6052 | 30  | 20 |
| 57 | cam-06 | 6160 | 6160 | 1   | 1  |

**Pattern:** Green jockey has longest single appearance on cam-05 (track ID #2: batches 73-709). Also frequent on cam-06 (64 detections).

---

### Red/Pink Jockey (NOT DETECTED) ❌
**Total:** 0 track IDs, 0 detections

**Possible Reasons:**
1. **Color classification error:** Red jockey may be misclassified as purple or another color
2. **Low YOLO confidence:** Red jockey not detected by YOLO model at 35% threshold
3. **Camera coverage gaps:** Red jockey only appears on cameras that had no detections (cam-02 through cam-15)
4. **Actual absence:** Red jockey may not be present in these video files

---

## 📹 Camera-by-Camera Breakdown

### Cameras with Detections

| Camera | Yellow Detections | Green Detections | Total | Track IDs |
|--------|------------------|-----------------|-------|-----------|
| **cam-06** | 235 (8 track IDs) | 64 (6 track IDs) | **299** | 14 |
| cam-05 | 12 (2 track IDs) | 30 (3 track IDs) | 42 | 5 |
| cam-13 | 6 (1 track ID) | 28 (2 track IDs) | 34 | 3 |
| cam-08 | 16 (1 track ID) | 0 | 16 | 1 |
| cam-01 | 0 | 15 (1 track ID) | 15 | 1 |

**Analysis:**
- **cam-06 is the primary detection zone** (299 detections = 75% of all detections)
- cam-05 and cam-13 have moderate activity
- cam-01 and cam-08 have minimal activity

### Cameras with NO Detections
cam-02, cam-03, cam-04, cam-07, cam-09, cam-10, cam-11, cam-12, cam-14, cam-15

**Expected track configuration for these cameras:**
```json
cam-02: track_start=150,  track_end=333
cam-03: track_start=300,  track_end=500
cam-04: track_start=450,  track_end=666
cam-07: track_start=950,  track_end=1166
cam-09: track_start=1300, track_end=1500
cam-10: track_start=1450, track_end=1666
cam-11: track_start=1600, track_end=1833
cam-12: track_start=1800, track_end=2000
cam-14: track_start=2100, track_end=2333
cam-15: track_start=2300, track_end=2500
```

**Possible reasons for no detections:**
1. **Obstacles blocking view:** User mentioned some cameras have obstacles
2. **Jockeys not yet reached these track positions:** Race may not have progressed to these sections
3. **Camera positioning issues:** Cameras may not cover the track area where jockeys are
4. **Video content:** These cameras may show different parts of the track without horses

---

## 🔍 Key Issues Identified

### 1. Missing Red/Pink Jockey ❌
**Priority:** HIGH
**Impact:** 33% of jockeys not detected

**Recommended Actions:**
- Check if "purple" color detections exist (red may be misclassified)
- Lower YOLO confidence threshold temporarily to see if red jockey appears
- Manually inspect videos from cam-02, cam-03, cam-04 where red jockey might appear
- Retrain color classifier with more red/pink jockey samples

### 2. Track ID Fragmentation 🔀
**Priority:** MEDIUM
**Impact:** 22 track IDs for 2 jockeys (5.5x more than expected)

**Recommended Actions:**
- Review nvtracker configuration (`tracker_config.txt`)
- Consider increasing IOU threshold for tracker
- Implement track ID merging based on color + temporal proximity
- Use more robust tracker (DeepSORT instead of IOU tracker)

### 3. Limited Camera Coverage 📹
**Priority:** MEDIUM
**Impact:** 10 out of 15 cameras have zero detections

**Recommended Actions:**
- Verify camera `track_start` and `track_end` configuration matches actual race progression
- Check if videos from silent cameras actually contain jockeys
- Identify cameras with obstacles (user mentioned this) and mark them as inactive
- Run test on full race duration to see if jockeys eventually appear on all cameras

---

## 📈 Comparison with Expected Configuration

### Expected Behavior (from `cameras_test_15.json`)
```
Track length: 2500 meters
Cameras: 15 (sequential coverage)
Expected progression: cam-01 → cam-02 → ... → cam-15
```

### Actual Behavior
```
Active cameras: 5 (cam-01, cam-05, cam-06, cam-08, cam-13)
Progression pattern: Irregular, concentrated on cam-06
Track coverage: Sparse (only ~20% of cameras active)
```

**Discrepancy Analysis:**
The actual detection pattern does NOT match the expected sequential progression through all 15 cameras. This suggests either:
1. The videos are from different race segments (not continuous)
2. Camera positions don't align with `track_start/track_end` values
3. Jockeys haven't completed full track (race stopped early)
4. Some cameras genuinely have no horses due to obstacles/positioning

---

## 🎯 Next Steps

### Immediate Actions
1. ✅ **Timeline analysis complete** - This document
2. 🔴 **Investigate missing red/pink jockey:**
   - Check for "purple" or "blue" misclassifications
   - Run color classifier on sample frames from cam-02, cam-03
3. 🟡 **Validate camera coverage:**
   - Manually inspect videos from cameras with zero detections
   - Confirm which cameras actually show jockeys

### Short-term Improvements
1. **Improve tracker stability:**
   - Tune IOU tracker parameters
   - Consider switching to DeepSORT tracker
2. **Retrain color classifier:**
   - Add more red/pink jockey samples
   - Balance dataset (currently yellow and green dominate)
3. **Update track configuration:**
   - Align `track_start/track_end` with actual camera positions
   - Mark obstacle-blocked cameras as inactive

### Long-term Goals
1. **Run 25-camera full test:**
   - Process all 25 cameras to get complete race coverage
   - Validate entire track progression
2. **Model quality improvements:**
   - Increase YOLO confidence (currently averaging 35%)
   - Reduce false positives
   - Improve red/pink jockey detection rate

---

## 📄 Files Generated

- **Timeline JSON:** `/tmp/deepstream_eval_timeline.json`
- **Analysis Script:** `tools/track_jockey_timeline.py`
- **Log File:** `/tmp/deepstream_eval.log` (318 KB)

---

## 🤔 Questions for User

1. **Red/Pink Jockey:**
   - Is the red/pink jockey definitely in these video files?
   - Which cameras would you expect to see the red/pink jockey on?

2. **Camera Obstacles:**
   - Which specific cameras have obstacles blocking the view?
   - Should we exclude these cameras from the configuration?

3. **Track Configuration:**
   - Is the `track_start/track_end` configuration accurate for the Yaris track?
   - Should we adjust based on actual detection patterns?

4. **Race Completion:**
   - Do these videos show a complete race from start to finish?
   - Or are they segments from different time periods?

---

**Generated by:** `tools/track_jockey_timeline.py`
**Analysis Date:** 2026-03-06 14:31 UTC
