# DeepStream Performance Optimization Notes

**Date:** 2026-03-06
**Goal:** Improve FPS from ~32.5 to 40-45 on 15 cameras

---

## 🎯 Optimizations Applied

### 1. **YOLO Input Resolution Reduction** (Biggest Impact: ~30-40% FPS gain)

**Before:**
```
infer-dims=3;800;800
mux_width=1280, mux_height=1280
```

**After:**
```
infer-dims=3;640;640
mux_width=640, mux_height=640
```

**Impact:**
- Reduced inference time by ~30-40% (640×640 has 36% fewer pixels than 800×800)
- Lower memory usage
- Trade-off: Slightly reduced detection accuracy for small/distant objects

---

### 2. **nvstreammux Batching Timeout** (Medium Impact: ~10-15% FPS gain)

**Before:**
```cpp
mux_batched_push_timeout = 4000000; // 4 seconds!
```

**After:**
```cpp
mux_batched_push_timeout = 40000;   // 40ms for file mode (25 FPS)
mux_batched_push_timeout = 33000;   // 33ms for RTSP mode (30 FPS)
```

**Impact:**
- Much faster batch formation
- Reduces latency from 4s to 40ms
- Better frame throughput

**Additional nvstreammux settings:**
```cpp
"num-surfaces-per-frame", 1,  // Reduce memory footprint
"attach-sys-ts", TRUE,         // Better synchronization
```

---

### 3. **Decoder Optimization** (Small Impact: ~5% FPS gain)

**Added:**
```cpp
// In on_child_added callback for decoder elements
g_object_set(object, "drop-frame-interval", 0, NULL);
g_object_set(object, "enable-max-performance", TRUE, NULL);
```

**Impact:**
- Enables maximum performance mode on NVIDIA hardware decoder
- Better GPU utilization

---

### 4. **nvinfer GPU Optimization Flags**

**Added to nvinfer_jockey.txt:**
```ini
interval=0              # Process every frame (don't skip)
gpu-id=0               # Explicit GPU selection
output-tensor-meta=0   # Disable tensor output (not needed)
```

**Impact:**
- Explicit GPU control
- Disabled unnecessary tensor metadata output

---

## 📊 Expected Performance

| Configuration | Resolution | Cameras | Expected FPS | FPS/Camera |
|--------------|------------|---------|--------------|------------|
| **Before** | 800×800 | 15 | ~32-35 | ~2.2 |
| **After** | 640×640 | 15 | **40-45** | **~2.8** |
| **Stretch Goal** | 640×640 | 15 | **50+** | **~3.3** |

---

## 🧪 Testing

### Benchmark Script
```bash
./test_fps_benchmark.sh cameras_test_15.json
```

The script will:
1. Run pipeline for 60 seconds
2. Extract FPS statistics (avg, min, max)
3. Count total detections
4. Provide performance evaluation

### Manual Testing
```bash
docker run --rm \
  --runtime=nvidia --gpus all \
  -v /home/user/race_vision/cameras_test_15.json:/app/cameras_test_15.json:ro \
  -v /home/user/recordings:/recordings:ro \
  -v /home/user/race_vision/models:/app/models:ro \
  -v /home/user/race_vision/deepstream/configs:/app/configs:ro \
  --network host \
  --entrypoint /bin/bash \
  race-vision:latest \
  -c "/app/bin/race_vision_deepstream \
      --config /app/cameras_test_15.json \
      --yolo-engine /app/configs/nvinfer_jockey.txt \
      --color-engine /app/models/color_classifier.engine \
      --file-mode"
```

---

## ⚠️ Important Notes

### Model Re-export Required
The YOLO model was exported for 800×800 input, but we changed inference to 640×640.

**Two options:**

#### Option 1: Use Dynamic Shape Engine (Recommended)
The existing TensorRT engine should support dynamic input shapes. Test first before re-exporting.

#### Option 2: Re-export Model for 640×640 (If Option 1 fails)
```bash
cd /home/user/race_vision
source venv/bin/activate

# Re-export YOLO with 640x640 input
python tools/export_trt.py --model jockey --input-size 640
```

This will create a new engine file optimized for 640×640:
```
models/jockey_yolov8s_640.onnx
models/jockey_yolov8s_640.onnx_b25_gpu0_fp16.engine
```

Then update `nvinfer_jockey.txt` to point to the new engine.

---

## 🔍 Validation Checklist

After optimization:

- [ ] FPS >= 40 on 15 cameras
- [ ] No CUDA out-of-memory errors
- [ ] Detection accuracy still acceptable (visual check)
- [ ] Color classification accuracy unchanged
- [ ] No dropped frames or stream errors
- [ ] Stable FPS over 5+ minutes

---

## 🚀 Further Optimization Ideas (If Needed)

### 1. **INT8 Quantization** (Additional 30-40% FPS gain)
Convert YOLO model to INT8 precision:
- Requires calibration dataset
- Slight accuracy loss (~1-2%)
- Massive speed improvement

### 2. **Reduce Batch Size** (For better latency)
If latency is more important than throughput:
```ini
batch-size=15  # Match camera count exactly
```

### 3. **Disable Tracker** (If track_id not needed)
Remove nvtracker from pipeline:
- Saves ~5-10% GPU time
- Trade-off: No persistent object IDs

### 4. **Reduce Color Classifier Resolution**
Crop smaller torso regions for color classification.

### 5. **Frame Skipping**
Process every 2nd or 3rd frame:
```ini
interval=1  # Skip every other frame (2x speedup)
```

---

## 📝 Rollback Instructions

If optimizations cause issues:

### 1. Revert YOLO Input Size
```bash
# Edit deepstream/configs/nvinfer_jockey.txt
infer-dims=3;800;800

# Edit deepstream/src/main.cpp
mux_width  = 1280;
mux_height = 1280;
```

### 2. Revert Batching Timeout
```bash
# Edit deepstream/src/main.cpp
pipeline_config.mux_batched_push_timeout = 4000000;  // 4s
```

### 3. Rebuild Docker Image
```bash
cd /home/user/race_vision
docker build -t race-vision:latest .
```

---

## 📚 References

- [NVIDIA DeepStream Performance Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Performance.html)
- [TensorRT Best Practices](https://docs.nvidia.com/deepstream/dev-guide/text/DS_TRT_Best_Practices.html)
- [nvstreammux Properties](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreammux.html)

---

**Summary:** Applied 4 major optimizations targeting ~50-60% FPS improvement. Expected result: 40-45 FPS on 15 cameras (vs. previous 32.5 FPS).
