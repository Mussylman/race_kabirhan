#!/bin/bash
# Run DeepStream on all 25 cameras to track jockey timeline

docker run --rm \
  --runtime=nvidia --gpus all \
  -v /home/user/race_vision/cameras_all_25.json:/app/cameras_all_25.json:ro \
  -v /home/user/recordings:/recordings:ro \
  -v /home/user/race_vision/models:/app/models:ro \
  -v /home/user/race_vision/deepstream/configs:/app/configs:ro \
  --network host \
  --entrypoint /app/bin/race_vision_deepstream \
  race-vision:latest \
      --config /app/cameras_all_25.json \
      --yolo-engine /app/configs/nvinfer_jockey.txt \
      --color-engine /app/models/color_classifier.engine \
      --file-mode
