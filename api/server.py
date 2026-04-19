"""
Race Vision — FastAPI Backend Server (Multi-Camera)

Bridges the multi-camera detection pipeline with the React frontend
via WebSocket (ranking updates) and MJPEG (display camera streams).

Architecture (4 layers):
    Layer 0 — DECODE:
        25 analytics cameras → MultiCameraReader (ffmpeg decode)
        4 display cameras    → separate CameraStreams (MJPEG passthrough)

    Layer 1 — TRIGGER:
        TriggerLoop thread → YOLOv8n @ 640px on all 25 cameras
        → updates CameraManager.active_cameras

    Layer 2 — ANALYZE:
        AnalysisLoop thread → YOLOv8s @ 800px + ColorCNN on active cameras
        → per-camera detections → FusionEngine

    Layer 3 — FUSION:
        FusionEngine → global track positions → rankings
        → WebSocket broadcast

    Display:
        4 display CameraStreams → MJPEG /stream/cam{1-4}

Usage:
    # Multi-camera with RTSP config
    python -m api.server --config cameras.json

    # Local video files (simulate multi-camera)
    python -m api.server --video data/videos/exp10_cam1.mp4 data/videos/exp10_cam2.mp4

    # Single RTSP (legacy mode — still works)
    python -m api.server --url "rtsp://admin:pass@ip:554/stream"
"""

import os
import cv2
import sys
import time
import json
import signal
import asyncio
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from starlette.responses import FileResponse

# Shared state, config, logging, singletons
from api.shared import (
    state, frame_store, log,
    _load_heavy_imports, _CameraStream,
    CameraManager, TrackTopology,
    COLOR_TO_HORSE, ALL_COLORS,
    DEFAULT_RTSP_URL, SERVER_HOST, SERVER_PORT,
    BROADCAST_INTERVAL, LIVE_DET_INTERVAL, MJPEG_QUALITY, MJPEG_FPS, TRACK_LENGTH,
)

# Camera I/O
from api.camera_io import VideoFileGrabber, RTSPGrabber, Go2RTCMonitor, Go2RTCStreamKeeper

# DeepStream pipeline
from api.deepstream_pipeline import DeepStreamSubprocess, DeepStreamPipeline

# Legacy / torch-based pipelines
from api.legacy_pipeline import LegacyDetectionLoop, MultiCameraPipeline

from pipeline.log_utils import slog, throttle, agg

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="Race Vision Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ws_clients: set[WebSocket] = set()

# Global pipeline references
_camera_manager: Optional[CameraManager] = None
_pipeline: Optional[MultiCameraPipeline] = None
_deepstream_subprocess: Optional[DeepStreamSubprocess] = None
_deepstream_pipeline: Optional[DeepStreamPipeline] = None
_legacy_detector: Optional[LegacyDetectionLoop] = None
_go2rtc_monitor: Optional[Go2RTCMonitor] = None
_go2rtc_keeper: Optional[Go2RTCStreamKeeper] = None

# ============================================================
# WEBSOCKET ENDPOINT
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.add(websocket)
    log.info(f"WebSocket client connected ({len(ws_clients)} total)")

    # Send horses_detected on connect
    horses_msg = {
        "type": "horses_detected",
        "horses": [
            {
                "id": info["id"],
                "number": info["number"],
                "name": info["name"],
                "color": info["color"],
                "jockeyName": info["jockeyName"],
                "silkId": info["silkId"],
            }
            for info in COLOR_TO_HORSE.values()
        ],
    }
    await websocket.send_json(horses_msg)

    # Send race_start if active
    if state.race_active:
        await websocket.send_json({
            "type": "race_start",
            "race": {
                "name": "Live Race",
                "totalLaps": 1,
                "trackLength": TRACK_LENGTH,
            },
        })

    # Send camera status if multi-camera
    if _camera_manager:
        await websocket.send_json({
            "type": "camera_status",
            **_camera_manager.get_status(),
        })

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg.get("type") == "get_state":
                rankings = state.get_rankings()
                resp = {
                    "type": "state",
                    "race": {
                        "name": "Live Race",
                        "totalLaps": 1,
                        "trackLength": TRACK_LENGTH,
                        "status": "active" if state.race_active else "pending",
                    },
                    "rankings": rankings,
                }
                if _camera_manager:
                    resp["cameras"] = _camera_manager.get_status()
                await websocket.send_json(resp)

            elif msg.get("type") == "start_race":
                state.race_active = True
                if _pipeline:
                    _pipeline.reset()
                if _deepstream_pipeline:
                    _deepstream_pipeline.reset()
                log.info("Race started (from operator)")
                await broadcast({
                    "type": "race_start",
                    "race": {
                        "name": "Live Race",
                        "totalLaps": 1,
                        "trackLength": TRACK_LENGTH,
                    },
                })

            elif msg.get("type") == "stop_race":
                state.race_active = False
                log.info("Race stopped (from operator)")
                await broadcast({"type": "race_stop"})

            elif msg.get("type") == "get_cameras":
                if _camera_manager:
                    await websocket.send_json({
                        "type": "camera_status",
                        **_camera_manager.get_status(),
                    })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error(f"WebSocket error: {e}")
    finally:
        ws_clients.discard(websocket)
        log.info(f"WebSocket client disconnected ({len(ws_clients)} total)")


async def broadcast(msg: dict):
    """Send to all connected WebSocket clients."""
    dead = set()
    for client in ws_clients:
        try:
            await client.send_json(msg)
        except Exception:
            dead.add(client)
    ws_clients.difference_update(dead)


# ============================================================
# REST ENDPOINTS
# ============================================================

@app.get("/admin")
async def admin_panel():
    admin_path = Path(__file__).resolve().parent.parent / "admin" / "index.html"
    return FileResponse(admin_path, media_type="text/html")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Dashboard page with live camera streams and stats."""
    # Build camera grid
    cam_count = 4
    if _camera_manager:
        cams = _camera_manager.get_analytics_cameras()
        cam_count = len(cams) if cams else 4

    cam_cells = ""
    for i in range(1, cam_count + 1):
        cam_cells += f"""
        <div class="cam">
            <div class="cam-header">cam-{i:02d}</div>
            <img src="/stream/cam{i}" alt="Camera {i}">
        </div>"""

    return f"""<!DOCTYPE html>
<html><head>
<title>Race Vision</title>
<style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ background:#111; color:#eee; font-family:monospace; padding:16px; }}
    h1 {{ margin-bottom:12px; }}
    .info {{ background:#222; padding:12px; border-radius:8px; margin-bottom:16px; display:flex; gap:24px; flex-wrap:wrap; }}
    .info a {{ color:#4fc3f7; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(460px, 1fr)); gap:12px; }}
    .cam {{ background:#000; border-radius:8px; overflow:hidden; }}
    .cam img {{ width:100%; display:block; }}
    .cam-header {{ padding:6px 12px; background:#222; font-size:13px; }}
    #stats {{ background:#1a1a2e; padding:12px; border-radius:8px; margin-bottom:16px; white-space:pre; font-size:13px; }}
</style>
</head><body>
<h1>Race Vision — Multi-Camera Dashboard</h1>
<div class="info">
    <span>WebSocket: <a href="javascript:void(0)">ws://localhost:8000/ws</a></span>
    <span><a href="/api/stats">/api/stats</a></span>
    <span><a href="/api/cameras">/api/cameras</a></span>
</div>
<div id="stats">Loading stats...</div>
<div class="grid">{cam_cells}</div>
<script>
async function refreshStats() {{
    try {{
        const [stats, cams] = await Promise.all([
            fetch('/api/stats').then(r=>r.json()),
            fetch('/api/cameras').then(r=>r.json()),
        ]);
        const active = cams.active_analytics || 0;
        const total = cams.total_analytics || 0;
        const t = stats.trigger || {{}};
        const a = stats.analyzer || {{}};
        const f = stats.fusion || {{}};
        document.getElementById('stats').textContent =
            `Mode: ${{stats.mode}}  |  Cameras: ${{active}}/${{total}} active  |  ` +
            `Horses tracked: ${{f.horses_tracked || 0}}/5\\n` +
            `Trigger: ${{t.frames_processed||0}} frames, ${{t.last_batch_time_ms||0}}ms/batch  |  ` +
            `Analysis: ${{a.frames_processed||0}} frames @ ${{a.current_fps||0}} fps, ${{a.last_batch_time_ms||0}}ms/batch`;
    }} catch(e) {{}}
}}
refreshStats();
setInterval(refreshStats, 2000);
</script>
</body></html>"""


@app.get("/api/cameras")
async def get_cameras():
    """Camera status (activation map, connection status)."""
    if _camera_manager:
        return JSONResponse(_camera_manager.get_status())
    return JSONResponse({"total_analytics": 0, "cameras": []})


@app.get("/api/stats")
async def get_stats():
    """Pipeline performance stats."""
    if _deepstream_pipeline:
        stats = {"mode": "deepstream"}
        stats.update(_deepstream_pipeline.get_stats())
        if _deepstream_subprocess:
            stats["subprocess"] = {
                "pid": _deepstream_subprocess.pid,
                "running": _deepstream_subprocess.is_running,
            }
    elif _pipeline:
        stats = {"mode": "multi"}
        stats.update(_pipeline.get_stats())
    else:
        stats = {"mode": "legacy"}
    if _go2rtc_monitor:
        stats["go2rtc"] = _go2rtc_monitor.get_status()
    return JSONResponse(stats)


# ============================================================
# MJPEG STREAM ENDPOINT
# ============================================================

def mjpeg_generator(cam_id: str):
    """Yield MJPEG frames for a specific camera."""
    delay = 1.0 / MJPEG_FPS

    while True:
        # Try display frame first (annotated), then raw frame
        frame = state.get_display_frame(cam_id)
        if frame is None:
            frame = frame_store.get(cam_id)

        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            label = f"{cam_id} - waiting..."
            cv2.putText(frame, label, (80, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY])
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpeg.tobytes() +
            b"\r\n"
        )
        time.sleep(delay)


@app.get("/stream/cam{cam_id}")
async def mjpeg_stream(cam_id: int):
    """MJPEG video stream. cam_id is 1-based (cam1..cam4 for display cameras)."""
    # Map 1-based cam_id to cam_id string
    cam_id_str = f"cam-{cam_id:02d}"

    # For display cameras (ptz-1..4), check those first
    if _camera_manager:
        display_cams = _camera_manager.get_display_cameras()
        if cam_id <= len(display_cams):
            cam_id_str = display_cams[cam_id - 1].cam_id

    return StreamingResponse(
        mjpeg_generator(cam_id_str),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ============================================================
# BACKGROUND BROADCAST TASKS
# ============================================================

async def live_detection_broadcast_loop():
    """Fast-path: broadcast live detections as soon as SHM reader has new data.

    Event-driven — wakes up when DeepStreamPipeline signals new detections,
    then broadcasts immediately. Each camera's results are sent independently
    without waiting for other cameras. Max rate: LIVE_DET_INTERVAL (20 Hz).
    """
    while True:
        if not _deepstream_pipeline:
            await asyncio.sleep(1.0)
            continue

        # Wait for SHM reader to signal new detections (instead of sleeping 200ms)
        loop = asyncio.get_event_loop()
        got_data = await loop.run_in_executor(
            None, _deepstream_pipeline.live_detections_event.wait, LIVE_DET_INTERVAL
        )

        if got_data:
            _deepstream_pipeline.live_detections_event.clear()

        if not ws_clients:
            continue

        live = _deepstream_pipeline.live_detections
        if not live:
            continue

        ts_send = time.time()
        msg = {
            "type": "live_detections",
            "ts_server_send": ts_send,
            "cameras": live,
        }
        await broadcast(msg)

        # Structured WS_SEND log — throttled per camera
        for cam_id, cam_data in live.items():
            ts_cap = cam_data.get("ts_capture", 0)
            age_ms = (ts_send - ts_cap) * 1000 if ts_cap else -1
            dets = cam_data.get("detections", [])
            frame_seq = cam_data.get("frame_seq", 0)
            agg.record_ws_send(cam_id)
            if throttle.allow(f"WS_SEND:{cam_id}", interval=2.0):
                slog("WS_SEND", cam_id, frame_seq, ts_send,
                     clients=len(ws_clients), dets=len(dets), age_ms=age_ms)


async def ranking_broadcast_loop():
    """Slow-path: rankings, camera_results, activation_map, logging."""
    last_log_time = 0
    last_activation_broadcast = 0
    last_rankings_hash = ""

    while True:
        await asyncio.sleep(BROADCAST_INTERVAL)

        if not ws_clients:
            continue

        # Broadcast pending camera_result events from DeepStreamPipeline
        if _deepstream_pipeline and _deepstream_pipeline._pending_camera_results:
            pending = list(_deepstream_pipeline._pending_camera_results)
            _deepstream_pipeline._pending_camera_results.clear()
            for result_msg in pending:
                await broadcast(result_msg)

        # Broadcast rankings only when changed
        if state.race_active:
            rankings = state.get_rankings()
            if rankings:
                # Hash by positions to detect changes
                rankings_hash = "|".join(
                    f"{r.get('color','')}{r.get('position','')}{r.get('distanceCovered','')}"
                    for r in rankings
                )
                if rankings_hash != last_rankings_hash:
                    last_rankings_hash = rankings_hash
                    await broadcast({
                        "type": "ranking_update",
                        "rankings": rankings,
                    })

        # Broadcast activation map every 2 seconds
        now = time.time()
        if _camera_manager and now - last_activation_broadcast > 2.0:
            activation = _camera_manager.get_activation_map()
            await broadcast({
                "type": "activation_map",
                "cameras": activation,
            })
            last_activation_broadcast = now

        # Log every 5 seconds
        if state.race_active and now - last_log_time > 5.0:
            rankings = state.get_rankings()
            if rankings:
                names = [r.get("name", "?") for r in rankings[:3]]
                log.info("Broadcasting %d horses to %d clients (top 3: %s)",
                         len(rankings), len(ws_clients), names)
            if _pipeline:
                stats = _pipeline.get_stats()
                trigger = stats.get("trigger", {})
                analyzer = stats.get("analyzer", {})
                log.info("  Trigger: %d frames, Analysis: %d frames @ %.1f fps",
                         trigger.get("frames_processed", 0),
                         analyzer.get("frames_processed", 0),
                         analyzer.get("current_fps", 0))
            if _deepstream_pipeline:
                stats = _deepstream_pipeline.get_stats()
                ds = stats.get("deepstream", {})
                log.info("  DeepStream: seq=%d, %d frames, SHM=%.0fhz Infer=%.1ffps, %.1fms/cycle",
                         ds.get("shm_seq", 0),
                         ds.get("frames_processed", 0),
                         ds.get("shm_fps", 0),
                         ds.get("current_fps", 0),
                         ds.get("last_cycle_ms", 0))
            last_log_time = now


@app.on_event("startup")
async def startup():
    asyncio.create_task(ranking_broadcast_loop())
    asyncio.create_task(live_detection_broadcast_loop())
    if _go2rtc_monitor:
        _go2rtc_monitor.start()
    if _go2rtc_keeper:
        _go2rtc_keeper.start()
    log.info(f"Race Vision backend running on http://{SERVER_HOST}:{SERVER_PORT}")
    log.info(f"  WebSocket: ws://localhost:{SERVER_PORT}/ws")
    log.info(f"  MJPEG:     http://localhost:{SERVER_PORT}/stream/cam1")
    if _camera_manager:
        status = _camera_manager.get_status()
        log.info(f"  Cameras:   {status['total_analytics']} analytics, {status['total_display']} display")


# ============================================================
# CAMERA CONFIG LOADER
# ============================================================

def load_camera_config(config_path: str) -> tuple[CameraManager, TrackTopology]:
    """Load camera configuration from JSON file.

    Expected format:
    {
        "track_length": 2500,
        "analytics": [
            {"id": "cam-01", "url": "rtsp://...", "track_start": 0, "track_end": 110},
            ...
        ],
        "display": [
            {"id": "ptz-1", "url": "rtsp://..."},
            ...
        ]
    }
    """
    with open(config_path) as f:
        config = json.load(f)

    track_length = config.get("track_length", TRACK_LENGTH)
    mgr = CameraManager(max_active=config.get("max_active", 8))
    topo = TrackTopology(track_length=track_length)

    # RV_INVERT_TRACK=1: jockeys move right-to-left on screen; leftmost =
    # leader. Default inverted for Ipodrome feeds.
    invert_default = os.environ.get("RV_INVERT_TRACK", "1") == "1"

    for cam in config.get("analytics", []):
        mgr.add_analytics(
            cam["id"],
            cam["url"],
            track_start=cam.get("track_start", 0),
            track_end=cam.get("track_end", 100),
        )
        topo.add_camera(
            cam["id"],
            cam.get("track_start", 0),
            cam.get("track_end", 100),
            inverted=cam.get("inverted", invert_default),
        )

    for cam in config.get("display", []):
        mgr.add_display(cam["id"], cam["url"])

    return mgr, topo


# ============================================================
# MAIN
# ============================================================

_grabber = None  # VideoFileGrabber or RTSPGrabber


def _shutdown_handler(signum, frame):
    """Graceful shutdown on SIGTERM/SIGINT."""
    log.info("Received signal %d, shutting down...", signum)
    if _deepstream_subprocess:
        _deepstream_subprocess.stop()
    if _deepstream_pipeline:
        _deepstream_pipeline.stop()
    if _pipeline:
        _pipeline.stop()
    if _legacy_detector:
        _legacy_detector.stop()
    sys.exit(0)


def main():
    global _grabber, _pipeline, _deepstream_subprocess, _deepstream_pipeline
    global _legacy_detector, _camera_manager, _go2rtc_monitor, _go2rtc_keeper

    parser = argparse.ArgumentParser(description="Race Vision Backend Server")
    parser.add_argument("--url", default=None, help="Single RTSP stream URL (legacy mode)")
    parser.add_argument("--video", nargs="+", default=None,
                        help="Local video file(s) — each simulates a camera")
    parser.add_argument("--config", default=None,
                        help="Camera config JSON file (multi-camera mode)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU decode")
    parser.add_argument("--host", default=SERVER_HOST, help="Server host")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="Server port")
    parser.add_argument("--auto-start", action="store_true",
                        help="Auto-start race on launch")
    parser.add_argument("--mode", choices=["auto", "multi", "legacy"], default="auto",
                        help="Pipeline mode: multi (trigger+analysis+fusion) or legacy (single detector)")
    parser.add_argument("--deepstream", action="store_true",
                        help="Use DeepStream C++ pipeline via shared memory")
    parser.add_argument("--ds-binary", default="/app/bin/race_vision_deepstream",
                        help="Path to DeepStream C++ binary")
    parser.add_argument("--yolo-engine", default="/app/models/yolov8s_deepstream.engine",
                        help="Path to YOLO TensorRT engine")
    parser.add_argument("--color-engine", default="/app/models/color_classifier.engine",
                        help="Path to color classifier TensorRT engine")
    parser.add_argument("--file-mode", action="store_true",
                        help="File playback mode (auto-detected if URLs start with file://)")
    parser.add_argument("--display", action="store_true",
                        help="Show video grid with OSD (requires X11 DISPLAY)")
    parser.add_argument("--go2rtc-url", default="http://localhost:1984",
                        help="go2rtc API URL for stream health monitoring")
    args = parser.parse_args()

    # Environment variable override (for Docker: DEEPSTREAM=1)
    if os.environ.get("DEEPSTREAM") == "1":
        args.deepstream = True

    # Signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    # go2rtc stream health monitor (started in FastAPI startup event)
    _go2rtc_monitor = Go2RTCMonitor(args.go2rtc_url)

    # go2rtc stream keeper — держит все RTSP потоки активными независимо
    # от того, открыт ли сайт. Когда пользователь откроет страницу,
    # видео появится мгновенно.
    _go2rtc_keeper = Go2RTCStreamKeeper(args.go2rtc_url)

    # Load heavy imports only when needed (non-DeepStream modes require torch)
    if not args.deepstream:
        log.info("Loading torch/ultralytics modules...")
        _load_heavy_imports()

    # ── Determine mode and sources ────────────────────────────────────

    use_multi = False

    if args.config:
        # Multi-camera from config file
        _camera_manager, topology = load_camera_config(args.config)
        if not args.deepstream:
            _grabber = RTSPGrabber(_camera_manager, gpu=args.gpu)
            _grabber.start()
        else:
            log.info("DeepStream mode — skipping RTSP grabber")
        use_multi = True
        log.info("Multi-camera mode from config: %s", args.config)

    elif args.video:
        if len(args.video) >= 2 or args.mode == "multi":
            # Multi-camera from video files
            _camera_manager = CameraManager(max_active=8)
            topology = TrackTopology(track_length=TRACK_LENGTH)

            cam_video_map = {}
            segment_len = TRACK_LENGTH / len(args.video)
            for i, vpath in enumerate(args.video):
                cam_id = f"cam-{i+1:02d}"
                start = i * segment_len
                end = start + segment_len + 10  # 10m overlap
                _camera_manager.add_analytics(cam_id, vpath, track_start=start, track_end=end)
                topology.add_camera(cam_id, start, end)
                cam_video_map[cam_id] = vpath

            _grabber = VideoFileGrabber(cam_video_map)
            _grabber.start()
            use_multi = True
            log.info("Multi-camera mode from %d videos", len(args.video))

        else:
            # Single video → legacy mode
            cam_id = "cam-01"
            cam_video_map = {cam_id: args.video[0]}
            _grabber = VideoFileGrabber(cam_video_map)
            _grabber.start()
            log.info("Legacy mode: single video %s", Path(args.video[0]).stem)

    else:
        # Single RTSP → legacy mode
        from api.shared import _CameraStream
        url = args.url or DEFAULT_RTSP_URL
        cam_id = "cam-01"

        def _rtsp_on_frame(cid, frame):
            frame_store.put(cam_id, frame)
            state.set_frame(cam_id, frame)

        cam = _CameraStream(
            cam_id, url, gpu=args.gpu,
            reconnect_delay=5.0,
            on_frame=_rtsp_on_frame,
        )
        cam.start()
        _grabber = cam
        log.info("Legacy mode: RTSP %s",
                 url.split('@')[-1] if '@' in url else url)

    # ── Start pipeline ────────────────────────────────────────────────

    if args.deepstream and use_multi:
        # Launch C++ DeepStream as subprocess (if binary exists)
        if os.path.isfile(args.ds_binary):
            # Auto-detect file mode from camera URLs
            file_mode = args.file_mode
            if not file_mode:
                try:
                    import json as _json
                    with open(args.config) as _f:
                        _cfg = _json.load(_f)
                    file_mode = any(
                        c.get("url", "").startswith("file://")
                        for c in _cfg.get("analytics", [])
                    )
                except Exception:
                    pass
            if file_mode:
                log.info("File mode detected — passing --file-mode to C++")
            _deepstream_subprocess = DeepStreamSubprocess(
                config_path=args.config,
                yolo_engine=args.yolo_engine,
                color_engine=args.color_engine,
                binary=args.ds_binary,
                file_mode=file_mode,
                display=args.display,
            )
            _deepstream_subprocess.start()
        else:
            log.info("DeepStream binary not found at %s — assuming external C++ process",
                     args.ds_binary)

        # Python SHM reader (retries until C++ creates SHM)
        _deepstream_pipeline = DeepStreamPipeline(_camera_manager, topology)
        _deepstream_pipeline.start()
        log.info("DeepStream pipeline started (C++ SHM → Python fusion)")
    elif use_multi and args.mode != "legacy":
        _pipeline = MultiCameraPipeline(_camera_manager, topology)
        _pipeline.start()
        log.info("Multi-camera pipeline started (trigger + analysis + fusion)")
    else:
        cam_id = "cam-01"
        _legacy_detector = LegacyDetectionLoop(cam_id=cam_id)
        _legacy_detector.start()
        log.info("Legacy detection loop started")

    if args.auto_start:
        state.race_active = True
        log.info("Race auto-started")

    # ── Run server ────────────────────────────────────────────────────

    import uvicorn
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        log.info("Shutting down...")
        if _go2rtc_keeper:
            _go2rtc_keeper.stop()
        if _go2rtc_monitor:
            _go2rtc_monitor.stop()
        if _grabber:
            if hasattr(_grabber, 'stop'):
                _grabber.stop()
        if _pipeline:
            _pipeline.stop()
        if _deepstream_pipeline:
            _deepstream_pipeline.stop()
        if _deepstream_subprocess:
            _deepstream_subprocess.stop()
        if _legacy_detector:
            _legacy_detector.stop()


if __name__ == "__main__":
    main()
