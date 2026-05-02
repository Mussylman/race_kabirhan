"""tools/replay_realtime_ws.py — Realtime WebSocket replay of an audit run.

Standalone FastAPI server on port 8000 with a /ws endpoint mimicking
api/server.py. Replays detections.jsonl + pipeline log [PASS]/[RANK]
events at content-time pacing. Frontend (Kabirhan-Frontend on :5173)
connects to ws://localhost:8000/ws and receives the same payloads as
production — no frontend changes needed.

Usage:
    python -m tools.replay_realtime_ws \
        --audit /tmp/demo_180010_real_audit \
        --pipeline-log /tmp/demo_180010_real_log/run_stdout.log \
        [--cameras-config configs/cameras_demo_180010_real.json] \
        [--speed 1.0] [--loop] [--port 8000]

NOTE on schema: WS payload structure is duplicated from api/server.py:
  - 'ranking_update'  built per api/deepstream_pipeline.py::_build_rankings
  - 'live_detections' built per api/server.py::live_detection_broadcast_loop
If api.server schema changes, sync this file by hand.
"""
import argparse
import asyncio
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from api.shared import COLOR_TO_HORSE, TRACK_LENGTH


# ── audit / log parsing ────────────────────────────────────────────────

RE_RANK = re.compile(r'^\[RANK\]\s+(.+)$')
RE_PASS = re.compile(r'^\[PASS\]\s+(\S+)\s+(\S+)\s+ts=([0-9.]+)')
RE_RANK_ITEM = re.compile(r'(\d+):(\w+)@(\S+)')


def load_audit(audit_dir: Path):
    """Parse detections.jsonl and per-cam wall<->content anchors."""
    by_key = defaultdict(list)
    cam_t0, cam_t_last = {}, {}
    cam_fi_first, cam_fi_last = {}, {}
    cam_frame_w, cam_frame_h = {}, {}
    for line in open(audit_dir / 'detections.jsonl'):
        d = json.loads(line)
        if not d.get('passed_filters') or not d.get('inside_roi'):
            continue
        cam = d['cam']
        ts = float(d.get('ts', 0.0))
        fi = int(d.get('frame_idx', 0))
        by_key[(cam, fi)].append(d)
        cam_frame_w.setdefault(cam, int(d.get('frame_w', 1280)))
        cam_frame_h.setdefault(cam, int(d.get('frame_h', 720)))
        if ts > 0:
            if cam not in cam_t0 or ts < cam_t0[cam]:
                cam_t0[cam] = ts
                cam_fi_first[cam] = fi
            if cam not in cam_t_last or ts > cam_t_last[cam]:
                cam_t_last[cam] = ts
                cam_fi_last[cam] = fi
    return by_key, cam_t0, cam_t_last, cam_fi_first, cam_fi_last, cam_frame_w, cam_frame_h


def make_wall_to_content(cam_t0, cam_t_last, cam_fi_first, cam_fi_last, fps):
    """Per-cam linear projection wall_ts -> demo content seconds."""
    rate = {}
    for c in cam_t0:
        sp = cam_fi_last[c] - cam_fi_first[c]
        rate[c] = (cam_t_last[c] - cam_t0[c]) / sp if sp > 0 else None

    def w2c(cam, wall_ts):
        if cam not in cam_t0 or rate.get(cam) in (None, 0):
            return None
        return (cam_fi_first[cam] + (wall_ts - cam_t0[cam]) / rate[cam]) / fps

    return w2c


def parse_pipeline_log(log_path: Path, w2c):
    """Build (content_t, [(rank, color, last_cam), ...]) snapshots from [RANK] lines."""
    rank_events = []
    last_pass_ct = None
    for line in open(log_path, errors='replace'):
        line = line.rstrip()
        m = RE_PASS.match(line)
        if m:
            cam, color, ts_wall = m.group(1), m.group(2), float(m.group(3))
            ct = w2c(cam, ts_wall)
            if ct is None:
                continue
            last_pass_ct = ct
            continue
        m = RE_RANK.match(line)
        if m and last_pass_ct is not None:
            items = [(int(r), c, lc) for r, c, lc in RE_RANK_ITEM.findall(m.group(1))]
            rank_events.append((last_pass_ct, items))
    rank_events.sort(key=lambda x: x[0])
    return rank_events


# ── ranking payload builder (mirrors api/deepstream_pipeline._build_rankings) ──

def load_cam_track_starts(cameras_config_path: Path) -> dict:
    cfg = json.loads(cameras_config_path.read_text())
    return {c['id']: float(c.get('track_start', 0.0)) for c in cfg.get('analytics', [])}


def build_rankings(rank_items, prev_distances, cam_track_starts, content_t, prev_content_t):
    """Build a frontend-shaped rankings list from a [RANK] snapshot.

    Differs from api/deepstream_pipeline._build_rankings only in that we sort
    detected horses by explicit RANK position (not distance) — replay has
    authoritative rank from tracker output instead of fusion estimate.
    """
    detected = {}
    for rank, color, last_cam in rank_items:
        dist = cam_track_starts.get(last_cam, 0.0)
        detected[color] = {"distance": dist, "rank": rank, "last_camera": last_cam}

    rankings = []
    rank_counter = 1
    sorted_detected = sorted(detected.items(), key=lambda x: x[1]["rank"])
    for color, data in sorted_detected:
        info = COLOR_TO_HORSE[color]
        prev_dist = prev_distances.get(color, data["distance"])
        dt = max(0.001, content_t - prev_content_t)
        speed_kmh = max(0.0, (data["distance"] - prev_dist) / dt * 3.6)
        rankings.append({
            "id": info["id"], "number": int(info["number"]), "name": info["name"],
            "color": info["color"], "jockeyName": info["jockeyName"],
            "silkId": int(info["silkId"]),
            "position": rank_counter,
            "distanceCovered": round(float(data["distance"]), 1),
            "currentLap": 1, "timeElapsed": round(content_t, 1),
            "speed": round(speed_kmh, 1), "gapToLeader": 0.0,
            "lastCameraId": data["last_camera"],
        })
        rank_counter += 1

    for color, info in COLOR_TO_HORSE.items():
        if color in detected:
            continue
        last = prev_distances.get(color, 0.0)
        rankings.append({
            "id": info["id"], "number": int(info["number"]), "name": info["name"],
            "color": info["color"], "jockeyName": info["jockeyName"],
            "silkId": int(info["silkId"]),
            "position": rank_counter,
            "distanceCovered": round(float(last), 1),
            "currentLap": 1, "timeElapsed": round(content_t, 1),
            "speed": 0.0, "gapToLeader": 0.0,
            "lastCameraId": "",
        })
        rank_counter += 1

    if rankings:
        leader_dist = rankings[0]["distanceCovered"]
        for r in rankings:
            gap = abs(leader_dist - r["distanceCovered"]) / max(TRACK_LENGTH, 1) * 60.0
            r["gapToLeader"] = round(gap, 2)

    return rankings


# ── replay server ──────────────────────────────────────────────────────

class ReplayServer:
    def __init__(self, audit_dir, pipeline_log, cameras_config, speed=1.0,
                 loop_replay=False, fps=25.0):
        self.speed = speed
        self.loop_replay = loop_replay
        self.fps = fps

        (self.by_key, cam_t0, cam_t_last, cam_fi_first, cam_fi_last,
         self.cam_frame_w, self.cam_frame_h) = load_audit(Path(audit_dir))
        w2c = make_wall_to_content(cam_t0, cam_t_last, cam_fi_first, cam_fi_last, fps)
        self.cam_track_starts = load_cam_track_starts(Path(cameras_config))
        self.rank_events = parse_pipeline_log(Path(pipeline_log), w2c)

        # Detection events: per (cam, frame_idx), grouped, content_t = fi/fps
        self.det_events = []
        for (cam, fi), dets in self.by_key.items():
            ct = fi / fps
            payload_dets = []
            for d in dets:
                bbox = d.get('bbox', [0, 0, 0, 0])
                conf = int(round(float(d.get('color_conf', 0.0)) * 100))
                # DeepStream's no-tracker sentinel is 2^64-1 — clamp to 0
                # so frontend hides the #N track badge.
                tid = int(d.get('track_id', 0) or 0)
                if tid > 10_000_000_000:
                    tid = 0
                payload_dets.append({
                    "color": d.get('color', '') or 'unknown',
                    "conf": conf,
                    "track_id": tid,
                    "bbox": [float(x) for x in bbox],
                })
            self.det_events.append((ct, cam, fi, payload_dets))
        self.det_events.sort(key=lambda x: x[0])

        self.total_duration = max(
            max((ct for ct, *_ in self.rank_events), default=0.0),
            max((ct for ct, *_ in self.det_events), default=0.0),
        )

        self.ws_clients: set[WebSocket] = set()
        self._reset_state()

    def _reset_state(self):
        self._prev_distances = {c: 0.0 for c in COLOR_TO_HORSE}
        self._prev_content_t = 0.0

    async def broadcast(self, msg: dict):
        dead = set()
        for client in list(self.ws_clients):
            try:
                await client.send_json(msg)
            except Exception:
                dead.add(client)
        self.ws_clients.difference_update(dead)

    async def replay_once(self):
        timeline = [(ct, 'rank', items) for ct, items in self.rank_events]
        timeline += [(ct, 'det', (cam, fi, dets)) for ct, cam, fi, dets in self.det_events]
        timeline.sort(key=lambda x: x[0])

        # Initial empty rankings — frontend has known state from t=0
        await self.broadcast({
            "type": "ranking_update",
            "rankings": build_rankings([], self._prev_distances,
                                       self.cam_track_starts, 0.0, 0.0),
        })

        start_wall = time.monotonic()
        for content_t, kind, payload in timeline:
            target_wall = start_wall + (content_t / self.speed)
            delay = target_wall - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

            if kind == 'rank':
                rankings = build_rankings(payload, self._prev_distances,
                                          self.cam_track_starts,
                                          content_t, self._prev_content_t)
                # Update state for next iteration's speed estimate
                color_by_id = {info['id']: c for c, info in COLOR_TO_HORSE.items()}
                for r in rankings:
                    color = color_by_id.get(r['id'])
                    if color:
                        self._prev_distances[color] = r['distanceCovered']
                self._prev_content_t = content_t
                await self.broadcast({"type": "ranking_update", "rankings": rankings})
            else:
                cam, fi, dets = payload
                ts_now = time.time()
                await self.broadcast({
                    "type": "live_detections",
                    "ts_server_send": ts_now,
                    "cameras": {
                        cam: {
                            "frame_w": self.cam_frame_w.get(cam, 1280),
                            "frame_h": self.cam_frame_h.get(cam, 720),
                            "ts_capture": ts_now,  # use NOW so STALE check passes
                            "frame_seq": fi,
                            "detections": dets,
                        },
                    },
                })

    async def replay_loop(self):
        while True:
            try:
                await self.replay_once()
                self._reset_state()
                if not self.loop_replay:
                    print("[replay] timeline complete, idle")
                    while True:
                        await asyncio.sleep(60)
                else:
                    print("[replay] loop iteration complete, restarting in 2s")
                    await asyncio.sleep(2.0)
            except Exception as e:
                print(f"[replay] error: {e}")
                await asyncio.sleep(2.0)


# ── FastAPI app ────────────────────────────────────────────────────────

def make_app(server: ReplayServer) -> FastAPI:
    app = FastAPI(title="Race Vision Replay")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        server.ws_clients.add(ws)
        print(f"[replay] client connected ({len(server.ws_clients)} total)")
        try:
            while True:
                msg = await ws.receive_text()
                try:
                    data = json.loads(msg)
                    if data.get('type') == 'ping':
                        await ws.send_json({"type": "pong"})
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            server.ws_clients.discard(ws)
            print(f"[replay] client disconnected ({len(server.ws_clients)} total)")

    @app.on_event("startup")
    async def startup():
        asyncio.create_task(server.replay_loop())
        print(f"[replay] timeline: {len(server.rank_events)} RANK + "
              f"{len(server.det_events)} det events, "
              f"duration={server.total_duration:.1f}s, "
              f"speed={server.speed}x, loop={server.loop_replay}")

    @app.get("/api/stats")
    async def stats():
        return {"replay_active": True, "ws_clients": len(server.ws_clients),
                "duration_s": server.total_duration,
                "rank_events": len(server.rank_events),
                "det_events": len(server.det_events)}

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--audit', required=True, help='Audit dir with detections.jsonl')
    p.add_argument('--pipeline-log', required=True,
                   help='Pipeline stdout log with [PASS]/[RANK] lines')
    p.add_argument('--cameras-config',
                   default='configs/cameras_demo_180010_real.json',
                   help='Camera config (track_start used for distanceCovered estimate)')
    p.add_argument('--speed', type=float, default=1.0)
    p.add_argument('--loop', action='store_true')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--fps', type=float, default=25.0)
    args = p.parse_args()

    server = ReplayServer(args.audit, args.pipeline_log, args.cameras_config,
                          speed=args.speed, loop_replay=args.loop, fps=args.fps)

    import uvicorn
    app = make_app(server)
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
