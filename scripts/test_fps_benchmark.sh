#!/bin/bash
# FPS Benchmark Test for Race Vision DeepStream Pipeline
# Tests performance with different configurations

set -e

echo "════════════════════════════════════════════════════════"
echo "  Race Vision - FPS Benchmark Test"
echo "════════════════════════════════════════════════════════"
echo ""

# Test configuration
CONFIG_FILE="${1:-configs/cameras_test_15.json}"
YOLO_ENGINE="/app/configs/nvinfer_jockey.txt"
COLOR_ENGINE="/app/models/color_classifier.engine"
TEST_DURATION=60  # seconds

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  YOLO engine: $YOLO_ENGINE"
echo "  Color engine: $COLOR_ENGINE"
echo "  Test duration: ${TEST_DURATION}s"
echo ""

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Count cameras in config
NUM_CAMERAS=$(jq '.analytics | length' "$CONFIG_FILE")
echo "Number of cameras: $NUM_CAMERAS"
echo ""

echo "Starting benchmark in 3 seconds..."
sleep 3

# Run the pipeline with timeout
echo "════════════════════════════════════════════════════════"
echo "  Starting DeepStream pipeline..."
echo "════════════════════════════════════════════════════════"
echo ""

timeout ${TEST_DURATION}s docker run --rm \
  --runtime=nvidia --gpus all \
  -v "/home/user/race_vision/${CONFIG_FILE}:/app/${CONFIG_FILE}:ro" \
  -v /home/user/recordings:/recordings:ro \
  -v /home/user/race_vision/models:/app/models:ro \
  -v /home/user/race_vision/deepstream/configs:/app/configs:ro \
  --network host \
  --entrypoint /bin/bash \
  race-vision:latest \
  -c "/app/bin/race_vision_deepstream \
      --config /app/${CONFIG_FILE} \
      --yolo-engine ${YOLO_ENGINE} \
      --color-engine ${COLOR_ENGINE} \
      --file-mode" \
  2>&1 | tee /tmp/deepstream_benchmark.log || true

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Benchmark Results"
echo "════════════════════════════════════════════════════════"
echo ""

# Extract FPS statistics from log
if [ -f /tmp/deepstream_benchmark.log ]; then
    # Find all FPS values
    FPS_VALUES=$(grep -oP '⚡ \K[0-9.]+(?= FPS)' /tmp/deepstream_benchmark.log || true)

    if [ -n "$FPS_VALUES" ]; then
        # Calculate average, min, max FPS using awk
        FPS_STATS=$(echo "$FPS_VALUES" | awk '
        BEGIN { max=0; min=999999; sum=0; count=0 }
        {
            if ($1 > max) max=$1;
            if ($1 < min) min=$1;
            sum+=$1;
            count++;
        }
        END {
            if (count > 0) {
                avg = sum/count;
                printf "Average FPS: %.2f\nMin FPS: %.2f\nMax FPS: %.2f\nSamples: %d\n", avg, min, max, count;
            } else {
                print "No FPS data found";
            }
        }')

        echo "$FPS_STATS"

        # Extract average for comparison
        AVG_FPS=$(echo "$FPS_STATS" | grep "Average FPS" | awk '{print $3}')

        echo ""
        echo "Performance Summary:"
        echo "  Cameras: $NUM_CAMERAS"
        echo "  Average FPS: $AVG_FPS"
        echo "  FPS per camera: $(echo "scale=2; $AVG_FPS / $NUM_CAMERAS" | bc)"

        # Performance evaluation
        if (( $(echo "$AVG_FPS >= 45" | bc -l) )); then
            echo "  Status: ✅ EXCELLENT (target reached!)"
        elif (( $(echo "$AVG_FPS >= 40" | bc -l) )); then
            echo "  Status: ✅ GOOD (close to target)"
        elif (( $(echo "$AVG_FPS >= 35" | bc -l) )); then
            echo "  Status: ⚠️  ACCEPTABLE (improvement needed)"
        else
            echo "  Status: ❌ POOR (optimization required)"
        fi
    else
        echo "❌ No FPS data found in log"
    fi

    # Count total detections
    TOTAL_BATCHES=$(grep -c "BATCH #" /tmp/deepstream_benchmark.log || echo "0")
    echo ""
    echo "  Total batches processed: $TOTAL_BATCHES"

    # Extract detection counts
    DETECTION_LINES=$(grep -oP '🎯 \K[0-9]+(?= detections)' /tmp/deepstream_benchmark.log || true)
    if [ -n "$DETECTION_LINES" ]; then
        TOTAL_DETS=$(echo "$DETECTION_LINES" | awk '{sum+=$1} END {print sum}')
        echo "  Total detections: $TOTAL_DETS"
    fi
else
    echo "❌ Log file not found"
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo ""
echo "Benchmark complete! Log saved to: /tmp/deepstream_benchmark.log"
