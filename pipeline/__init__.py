"""
pipeline — Multi-camera horse race analysis pipeline.

Layers:
    0. DECODE   — 25 analytics + 4 display RTSP streams
    1. TRIGGER  — YOLOv8n lightweight detection on all 25 cameras
    2. ANALYZE  — YOLOv8s + color classifier on active cameras only
    3. FUSION   — merge per-camera results → global ranking

Modules:
    camera_manager   — camera state (active/inactive), activation logic
    track_topology   — pixel_x → global track position (0–2500m)
    trigger          — batch YOLOv8n trigger for 25 cameras
    analyzer         — batch YOLOv8s + color classifier for active cameras
    fusion           — merge detections, resolve overlaps, temporal smoothing
    vote_engine      — position-based weighted voting (extracted from test_race_count.py)
    trt_inference    — TensorRT inference wrapper with PyTorch fallback
"""
