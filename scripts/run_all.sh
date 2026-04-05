#!/bin/bash
# Run all services: go2rtc (Docker) + DeepStream + Python backend + Frontend
#
# Usage:
#   ./scripts/run_all.sh                         # запись 1 (162028)
#   ./scripts/run_all.sh --rec2                  # запись 2 (164835)
#   ./scripts/run_all.sh --go2rtc <yaml> --cameras <json>  # custom
#
# Stop: Ctrl+C (kills all)

set -e
cd /home/user/race_vision

# Defaults — recording 1
GO2RTC_YAML="configs/go2rtc_recording1.yaml"
CAMERAS_JSON="configs/cameras_recording1_rtsp.json"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --rec2)
            GO2RTC_YAML="configs/go2rtc_recording2.yaml"
            CAMERAS_JSON="configs/cameras_recording2_rtsp.json"
            shift ;;
        --go2rtc)
            GO2RTC_YAML="$2"; shift 2 ;;
        --cameras)
            CAMERAS_JSON="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "  go2rtc:  $GO2RTC_YAML"
echo "  cameras: $CAMERAS_JSON"
echo "========================================"

cleanup() {
    echo ""
    echo "[run_all] Stopping all..."
    kill $PID_DS $PID_BACKEND $PID_FRONTEND 2>/dev/null
    docker rm -f rv-go2rtc 2>/dev/null
    wait 2>/dev/null
    echo "[run_all] Done"
}
trap cleanup EXIT INT TERM

# 1. go2rtc (Docker)
echo "[run_all] Starting go2rtc (Docker)..."
docker rm -f rv-go2rtc 2>/dev/null
docker run -d --name rv-go2rtc --network host \
  -v "$(pwd)/${GO2RTC_YAML}:/config/go2rtc.yaml:ro" \
  -v /home/user/recordings:/home/user/recordings:ro \
  alexxit/go2rtc
sleep 3

# Verify go2rtc is running
if ! docker ps | grep -q rv-go2rtc; then
    echo "[run_all] ERROR: go2rtc failed to start"
    docker logs rv-go2rtc
    exit 1
fi
echo "[run_all] go2rtc OK (RTSP :8554, API :1984)"

# 2. DeepStream headless (reads RTSP from go2rtc)
echo "[run_all] Starting DeepStream..."
LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH \
./deepstream/build/race_vision_deepstream \
  --config "$CAMERAS_JSON" \
  --yolo-engine deepstream/configs/nvinfer_jockey.txt \
  --color-engine models/color_classifier_v3.engine \
  --mux-width 1520 --mux-height 1520 &
PID_DS=$!
sleep 5

# 3. Python backend
echo "[run_all] Starting Python backend..."
.venv/bin/python -m api.server \
  --deepstream --auto-start \
  --config "$CAMERAS_JSON" &
PID_BACKEND=$!
sleep 3

# 4. Frontend
echo "[run_all] Starting frontend..."
cd Kabirhan-Frontend && npx vite --host 0.0.0.0 &
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
