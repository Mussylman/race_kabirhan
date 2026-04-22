"""
Analytics service — post-hoc browsing, debugging, and cross-camera labeling
of detections from recorded sessions.

Components:
  - indexer.py        — JSONL → SQLite (fast time/camera lookups)
  - frame_server.py   — mp4 frame extraction via ffmpeg, JPEG on demand
  - pairs_store.py    — persistent store for labeled ReID pairs
  - active_learning.py — priority queue, uncertainty scoring, retrain trigger
  - export.py         — Market-1501 / Triplets-JSONL / COCO dataset writers
  - labeler_session.py — per-user labeling session state
  - router.py         — FastAPI router (/analytics/*)

Data layout (outside the repo — configured via ARTIFACTS_DIR env):
  $ARTIFACTS_DIR/
    analytics.db              — SQLite index
    crops/<rec>/<cam>/*.jpg   — on-demand cropped detections
    frames/<rec>/<cam>/*.jpg  — full annotated frames (cache)
    pairs/<rec>/pairs.jsonl   — ReID pair labels
    dataset_reid/<rec>/       — exported Market-1501 tree
"""
