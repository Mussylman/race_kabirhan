# Race Vision - Model Documentation

This document specifies all model files, their paths, configurations, and usage for the Race Vision DeepStream pipeline.

**Last Updated:** 2026-03-05
**Training Date:** 2026-03-04

---

## 📁 Model Files Location

All model files are stored in: `/home/user/race_vision/models/`

```
models/
├── jockey_yolov8s.onnx                           # 43 MB - Custom YOLOv8s ONNX
├── jockey_yolov8s.onnx_b25_gpu0_fp16.engine     # 26 MB - Pre-built TensorRT engine
├── color_classifier.engine                       # 1.9 MB - Color CNN TensorRT engine
└── yolov8s.onnx                                  # 88 MB - Standard YOLOv8s (80 classes, not used)
```

---

## 🎯 1. Jockey Detector (YOLO)

### Model Information
- **Name:** Custom YOLOv8s Jockey Detector
- **Type:** Single-class object detection
- **Architecture:** YOLOv8s (small)
- **Training Date:** 2026-03-04
- **Classes:** 1 (jockey)
- **Input Size:** 800×800 pixels
- **Output Shape:** [5, 13125]
  - 5 channels: 4 bbox coordinates + 1 class score
  - 13125 predictions per image

### Files
```
ONNX Model:     /home/user/race_vision/models/jockey_yolov8s.onnx
TensorRT Engine: /home/user/race_vision/models/jockey_yolov8s.onnx_b25_gpu0_fp16.engine
```

### TensorRT Engine Details
- **Precision:** FP16
- **Batch Size:** Dynamic (1-25)
- **GPU:** GPU 0
- **Built for:** NVIDIA RTX series (Compute Capability 8.6+)

### Configuration File
**Path:** `/home/user/race_vision/deepstream/configs/nvinfer_jockey.txt`

Key parameters:
```ini
[property]
gpu-id=0
onnx-file=/app/models/jockey_yolov8s.onnx
model-engine-file=/app/models/jockey_yolov8s.onnx_b25_gpu0_fp16.engine

batch-size=25
network-mode=2          # FP16
infer-dims=3;800;800    # 800×800 input
num-detected-classes=1  # Single class: jockey

# Preprocessing
maintain-aspect-ratio=1
symmetric-padding=1
net-scale-factor=0.00392157
model-color-format=1

# NMS
cluster-mode=2

# Custom parser
parse-bbox-func-name=NvDsInferParseYoloV8
custom-lib-path=/app/lib/libnvdsinfer_yolov8_parser.so
output-blob-names=output0

[class-attrs-all]
pre-cluster-threshold=0.35

[class-attrs-0]
pre-cluster-threshold=0.35
```

### Performance
- **Expected Confidence:** 80-95%
- **FPS (15 cameras):** ~32-35 FPS
- **FPS (10 cameras):** ~45-50 FPS

---

## 🎨 2. Color Classifier (CNN)

### Model Information
- **Name:** Jockey Color Classifier
- **Type:** Multi-class classification
- **Architecture:** Custom CNN
- **Training Date:** 2026-03-04
- **Classes:** 5 colors
  - 0: Blue
  - 1: Green
  - 2: Purple
  - 3: Red
  - 4: Yellow
- **Input Size:** Cropped jockey bbox from YOLO
- **Output:** [5] class probabilities

### File
```
TensorRT Engine: /home/user/race_vision/models/color_classifier.engine
```

### TensorRT Engine Details
- **Precision:** FP16
- **Batch Size:** Dynamic (1-25)
- **GPU:** GPU 0
- **Input:** RGB image of detected jockey bbox

### Configuration
The color classifier is loaded directly as a TensorRT engine file. Configuration is handled in the C++ pipeline code.

**Command-line argument:**
```bash
--color-engine /app/models/color_classifier.engine
```

### Performance
- **Expected Confidence:** 60-99%
- **Inference Time:** ~1-2ms per detection (after YOLO)

---

## 🔧 3. DeepStream Configuration

### nvinfer_jockey.txt
**Location:** `/home/user/race_vision/deepstream/configs/nvinfer_jockey.txt`

This file configures the YOLO model for DeepStream's nvinfer plugin.

**Docker container path:** `/app/configs/nvinfer_jockey.txt`

**Usage:**
```bash
--yolo-engine /app/configs/nvinfer_jockey.txt
```

### Important Notes

1. **Model Path Inside Docker Container:**
   - Host: `/home/user/race_vision/models/`
   - Container: `/app/models/`
   - Mounted as: `-v /home/user/race_vision/models:/app/models:ro`

2. **Config Path Inside Docker Container:**
   - Host: `/home/user/race_vision/deepstream/configs/`
   - Container: `/app/configs/`
   - Mounted as: `-v /home/user/race_vision/deepstream/configs:/app/configs:ro`

3. **Always Use Pre-built TensorRT Engine:**
   - The `.engine` file is optimized for your specific GPU
   - Building engine at runtime adds 5-10 minutes delay
   - Engine file must match GPU architecture

4. **Dynamic Batch Support:**
   - Engine supports batch sizes from 1 to 25
   - Automatically adjusts to number of active cameras
   - No need to rebuild engine when changing camera count

---

## 🚀 Quick Start

### Running 15-Camera Test

Use the provided launch script:
```bash
cd /home/user/race_vision
./test_deepstream_15cam.sh
```

Or manually:
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

## 🔄 Rebuilding TensorRT Engines

If you need to rebuild engines (e.g., for a different GPU):

### YOLO Engine
```bash
cd /home/user/race_vision
source venv/bin/activate
python tools/export_trt.py --model jockey
```

### Color Classifier Engine
```bash
cd /home/user/race_vision
source venv/bin/activate
python tools/export_trt.py --model classifier
```

**Note:** Rebuilding takes 5-10 minutes per model. Only needed when:
- Switching to a different GPU model
- Upgrading TensorRT version
- Changing batch size limits
- Updating model architecture

---

## 📊 Model Training Information

### Training Dataset
- **Date:** 2026-03-04
- **Source:** Recorded race footage from Yaris track
- **YOLO Dataset:** Jockey bounding boxes (1 class)
- **Classifier Dataset:** Cropped jockeys with color labels (5 classes)

### Training Parameters
- **YOLO:**
  - Base model: YOLOv8s pretrained on COCO
  - Fine-tuned on jockey detection
  - Training epochs: ~100-150 (exact value from training logs)
  - Input size: 800×800

- **Color Classifier:**
  - Custom CNN architecture
  - Input size: Variable (cropped bbox)
  - Training epochs: ~50-100 (exact value from training logs)
  - Data augmentation: Applied during training

---

## ⚠️ Troubleshooting

### Error: "deserialize engine from file failed"
**Solution:** Check that engine file exists and path is correct:
```bash
ls -lh /home/user/race_vision/models/jockey_yolov8s.onnx_b25_gpu0_fp16.engine
```

### Error: "Could not deserialize engine"
**Solution:** Engine was built for different GPU. Rebuild with:
```bash
python tools/export_trt.py --model jockey
```

### Error: "CUDA out of memory"
**Solution:** Reduce batch size in `nvinfer_jockey.txt`:
```ini
batch-size=15  # Reduce from 25
```

### Low FPS / Performance Issues
**Checklist:**
- [ ] Verify GPU is being used: `nvidia-smi`
- [ ] Check engine uses FP16, not FP32
- [ ] Ensure pre-built engine exists (not building at runtime)
- [ ] Verify batch size matches or exceeds camera count
- [ ] Check Docker has GPU access: `--runtime=nvidia --gpus all`

---

## 📝 Version History

### 2026-03-05
- Documented current model architecture
- Added 15-camera test script
- Clarified TensorRT engine paths

### 2026-03-04
- Trained custom jockey YOLO model (1 class)
- Trained color classifier (5 classes)
- Built TensorRT engines for both models
- Tested on 10 cameras: ~45-50 FPS
- Tested on 15 cameras: ~32-35 FPS

---

## 📞 Contact

For model retraining or updates, refer to:
- Training scripts: `/home/user/race_vision/tools/train_jockey.py`
- Export scripts: `/home/user/race_vision/tools/export_trt.py`
- Dataset preparation: `/home/user/race_vision/tools/prepare_training_data.py`
