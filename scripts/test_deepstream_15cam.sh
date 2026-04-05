#!/bin/bash
# ==============================================================================
# Race Vision - 15-Camera DeepStream Test Script
# ==============================================================================
# This script launches DeepStream pipeline with the custom jockey detector
# and color classification models, processing 15 video files simultaneously.
#
# Date: 2026-03-05
# Performance: ~32.5 FPS on 15 cameras (RTX 4090 / similar)
# ==============================================================================

set -e  # Exit on error

# Colors for terminal output
BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
CYAN="\033[36m"
RESET="\033[0m"

echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${CYAN}${BOLD}  Race Vision - 15 Camera DeepStream Test${RESET}"
echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""

# ==============================================================================
# MODEL PATHS - Specify models trained on 2026-03-04
# ==============================================================================
# YOLO Model: Custom YOLOv8s trained on jockey detection (1 class)
#   - ONNX: /home/user/race_vision/models/jockey_yolov8s.onnx (43MB)
#   - Engine: /home/user/race_vision/models/jockey_yolov8s.onnx_b25_gpu0_fp16.engine (26MB)
#   - Input: 800x800, Output: [5, 13125]
#   - Dynamic batch: 1-25 cameras supported
#
# Color Classifier: CNN trained on jockey color detection (5 classes)
#   - Engine: /home/user/race_vision/models/color_classifier.engine (1.9MB)
#   - Classes: blue, green, purple, red, yellow
# ==============================================================================

CAMERA_CONFIG="/home/user/race_vision/configs/cameras_test_15.json"
YOLO_CONFIG="/home/user/race_vision/deepstream/configs/nvinfer_jockey.txt"
COLOR_ENGINE="/home/user/race_vision/models/color_classifier.engine"
RECORDINGS_DIR="/home/user/recordings"
MODELS_DIR="/home/user/race_vision/models"
CONFIGS_DIR="/home/user/race_vision/deepstream/configs"
DOCKER_IMAGE="race-vision:latest"

# ==============================================================================
# Pre-flight Checks
# ==============================================================================
echo -e "${YELLOW}🔍 Pre-flight checks...${RESET}"

# Check camera config
if [ ! -f "$CAMERA_CONFIG" ]; then
    echo -e "${YELLOW}❌ Camera config not found: $CAMERA_CONFIG${RESET}"
    exit 1
fi
echo -e "${GREEN}✓${RESET} Camera config: $CAMERA_CONFIG"

# Check YOLO config
if [ ! -f "$YOLO_CONFIG" ]; then
    echo -e "${YELLOW}❌ YOLO config not found: $YOLO_CONFIG${RESET}"
    exit 1
fi
echo -e "${GREEN}✓${RESET} YOLO config: $YOLO_CONFIG"

# Check color engine
if [ ! -f "$COLOR_ENGINE" ]; then
    echo -e "${YELLOW}❌ Color classifier engine not found: $COLOR_ENGINE${RESET}"
    exit 1
fi
echo -e "${GREEN}✓${RESET} Color classifier: $COLOR_ENGINE"

# Check recordings directory
if [ ! -d "$RECORDINGS_DIR" ]; then
    echo -e "${YELLOW}❌ Recordings directory not found: $RECORDINGS_DIR${RESET}"
    exit 1
fi
echo -e "${GREEN}✓${RESET} Recordings directory: $RECORDINGS_DIR"

# Check Docker image
if ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
    echo -e "${YELLOW}❌ Docker image not found: $DOCKER_IMAGE${RESET}"
    exit 1
fi
echo -e "${GREEN}✓${RESET} Docker image: $DOCKER_IMAGE"

# Check NVIDIA GPU
if ! nvidia-smi >/dev/null 2>&1; then
    echo -e "${YELLOW}❌ NVIDIA GPU not detected${RESET}"
    exit 1
fi
echo -e "${GREEN}✓${RESET} NVIDIA GPU detected"

echo ""
echo -e "${GREEN}${BOLD}✓ All pre-flight checks passed${RESET}"
echo ""

# ==============================================================================
# Launch DeepStream Pipeline
# ==============================================================================
echo -e "${CYAN}${BOLD}🚀 Launching DeepStream pipeline...${RESET}"
echo -e "${YELLOW}📊 Configuration:${RESET}"
echo -e "   • Cameras: 15"
echo -e "   • YOLO Model: Custom jockey detector (1 class)"
echo -e "   • Color CNN: 5 classes (blue/green/purple/red/yellow)"
echo -e "   • Batch size: Dynamic (1-25)"
echo -e "   • Expected FPS: ~30-35"
echo ""
echo -e "${YELLOW}💡 Tip: Press Ctrl+C to stop${RESET}"
echo ""
sleep 2

# Run Docker container with proper volume mounts
docker run --rm \
  --runtime=nvidia \
  --gpus all \
  -v "$CAMERA_CONFIG:/app/cameras_test_15.json:ro" \
  -v "$RECORDINGS_DIR:/recordings:ro" \
  -v "$MODELS_DIR:/app/models:ro" \
  -v "$CONFIGS_DIR:/app/configs:ro" \
  --network host \
  --entrypoint /bin/bash \
  "$DOCKER_IMAGE" \
  -c "/app/bin/race_vision_deepstream \
      --config /app/cameras_test_15.json \
      --yolo-engine /app/configs/nvinfer_jockey.txt \
      --color-engine /app/models/color_classifier.engine \
      --file-mode"

# ==============================================================================
# Cleanup
# ==============================================================================
echo ""
echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${GREEN}${BOLD}✓ Test completed${RESET}"
echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
