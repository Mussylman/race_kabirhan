#!/bin/bash
# Run all services: go2rtc + DeepStream + Python backend + Frontend
# Usage: ./scripts/run_all.sh [--5cam|--25cam]
# Stop: Ctrl+C (kills all)

set -e
cd /home/user/race_vision

CONFIG="cameras_5cam.json"
if [ "$1" = "--25cam" ]; then
    CONFIG="configs/cameras_all_25.json"
fi

cleanup() {
    echo "[run_all] Stopping all..."
    kill $PID_GO2RTC $PID_DS $PID_BACKEND $PID_FRONTEND 2>/dev/null
    wait 2>/dev/null
    echo "[run_all] Done"
}
trap cleanup EXIT INT TERM

# 1. go2rtc
echo "[run_all] Starting go2rtc..."
/tmp/go2rtc -config configs/go2rtc_files.yaml &
PID_GO2RTC=$!
sleep 1

# 2. DeepStream headless
echo "[run_all] Starting DeepStream (headless)..."
LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH \
./deepstream/build/race_vision_deepstream \
  --config $CONFIG \
  --yolo-engine deepstream/configs/nvinfer_jockey.txt \
  --color-engine models/color_classifier_v3.engine \
  --mux-width 2880 --mux-height 1620 \
  --file-mode &
PID_DS=$!
sleep 3

# 3. Python backend
echo "[run_all] Starting Python backend..."
.venv/bin/python -m api.server \
  --deepstream --auto-start \
  --config $CONFIG &
PID_BACKEND=$!
sleep 2

# 4. Frontend
echo "[run_all] Starting frontend..."
cd Kabirhan-Frontend && npm run dev &
PID_FRONTEND=$!
cd ..

echo ""
echo "========================================"
echo "  All services running!"
echo "  go2rtc:    http://localhost:1984"
echo "  Backend:   http://localhost:8000"
echo "  Frontend:  http://localhost:5173"
echo "  Press Ctrl+C to stop all"
echo "========================================"

wait
