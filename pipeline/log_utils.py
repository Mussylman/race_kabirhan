"""
log_utils.py — Structured logging helpers for Race Vision backend.

Provides:
    slog(stage, cam_id, frame_seq, ts, **fields)
        Emits one structured log line: STAGE  cam=X  frame=N  ts=T.TTT  ...

    ThrottleMap
        Per-key rate limiter: skip logging if called too frequently.

    PerCameraAggregator
        Accumulates per-camera counters over 1-second windows, then emits STATS_1S.

Environment flags (read once at import):
    LOG_LEVEL     DEBUG / INFO / WARNING       (default INFO)
    LOG_TIMING    true / false                 (default false)
        When true: throttle disabled for SHM_READ, WS_SEND, LIVE_UPDATE
    LOG_GEOMETRY  true / false                 (default false)
        When true: bbox coords included in SHM_READ
    LOG_FUSION    true / false                 (default false)
        When true: accepted FUSION_UPDATE logged (not just rejects)
    LOG_STATS     true / false                 (default true)
        When false: STATS_1S suppressed
"""

import os
import time
import logging
from typing import Any

# ── Env flags ─────────────────────────────────────────────────────────

LOG_TIMING   = os.environ.get("LOG_TIMING",   "false").lower() == "true"
LOG_GEOMETRY = os.environ.get("LOG_GEOMETRY", "false").lower() == "true"
LOG_FUSION   = os.environ.get("LOG_FUSION",   "false").lower() == "true"
LOG_STATS    = os.environ.get("LOG_STATS",    "true").lower()  == "true"

_log = logging.getLogger("rv.trace")

# Configure rv.trace to follow LOG_LEVEL env (independent of root logger)
_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
_log.setLevel(getattr(logging, _level_str, logging.INFO))
# Propagate to root so basicConfig handler picks it up
_log.propagate = True


# ── Structured log emitter ─────────────────────────────────────────────

def slog(stage: str, cam_id: str, frame_seq: int, ts: float, **fields: Any) -> None:
    """Emit one structured log line.

    Level: INFO (always visible), unless LOG_LEVEL=WARNING disables it.
    Example output:
        SHM_READ  cam=cam-05  frame=14823  ts=1743417600.142  dets=3  age_ms=2.1
    """
    parts = [
        f"cam={cam_id}",
        f"frame={frame_seq}",
        f"ts={ts:.3f}",
    ]
    for k, v in fields.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.1f}")
        else:
            parts.append(f"{k}={v}")

    _log.info("%-14s  %s", stage, "  ".join(parts))


# ── ThrottleMap ────────────────────────────────────────────────────────

class ThrottleMap:
    """Per-key rate limiter for log calls.

    Usage:
        throttle = ThrottleMap()

        # Returns True if the call should be logged (not throttled).
        if throttle.allow("SHM_READ:cam-05", interval=2.0):
            slog("SHM_READ", ...)
    """

    def __init__(self):
        self._last: dict[str, float] = {}

    def allow(self, key: str, interval: float) -> bool:
        """Return True and record timestamp if enough time has passed."""
        now = time.monotonic()
        last = self._last.get(key, 0.0)
        if now - last >= interval:
            self._last[key] = now
            return True
        return False

    def reset(self, key: str) -> None:
        self._last.pop(key, None)


# ── PerCameraAggregator ────────────────────────────────────────────────

class PerCameraAggregator:
    """Accumulates per-camera counters over 1-second windows.

    Call record_*() methods freely (every frame).
    The aggregator internally checks if 1 second has elapsed and emits
    a STATS_1S log line, then resets counters.

    Usage:
        agg = PerCameraAggregator()

        # In SHM reader:
        agg.record_shm(cam_id, age_ms=2.1)

        # In broadcast loop:
        agg.record_ws_send(cam_id, age_ms=3.0)
        agg.flush_if_due()   # call once per broadcast cycle
    """

    _WINDOW = 1.0  # seconds

    def __init__(self):
        self._cams: dict[str, dict] = {}

    def _get(self, cam_id: str) -> dict:
        if cam_id not in self._cams:
            self._cams[cam_id] = {
                "window_start": time.monotonic(),
                "shm_count":     0,
                "ws_count":      0,
                "age_sum":       0.0,
                "age_max":       0.0,
                "fusion_ok":     0,
                "fusion_reject": 0,
            }
        return self._cams[cam_id]

    def record_shm(self, cam_id: str, age_ms: float = 0.0) -> None:
        c = self._get(cam_id)
        c["shm_count"]  += 1
        c["age_sum"]    += age_ms
        c["age_max"]     = max(c["age_max"], age_ms)

    def record_ws_send(self, cam_id: str) -> None:
        c = self._get(cam_id)
        c["ws_count"] += 1

    def record_fusion(self, cam_id: str, accepted: bool) -> None:
        c = self._get(cam_id)
        if accepted:
            c["fusion_ok"] += 1
        else:
            c["fusion_reject"] += 1

    def flush_if_due(self) -> None:
        """Emit STATS_1S for any camera whose window has elapsed."""
        if not LOG_STATS:
            return
        now = time.monotonic()
        for cam_id, c in list(self._cams.items()):
            elapsed = now - c["window_start"]
            if elapsed < self._WINDOW:
                continue

            shm_fps  = c["shm_count"]  / elapsed
            ws_fps   = c["ws_count"]   / elapsed
            age_avg  = (c["age_sum"] / c["shm_count"]) if c["shm_count"] else 0.0

            _log.info(
                "%-14s  cam=%s  ts=%.3f  shm_fps=%.1f  ws_fps=%.1f"
                "  age_avg_ms=%.1f  age_max_ms=%.1f"
                "  fusion_ok=%d  fusion_reject=%d",
                "STATS_1S", cam_id, now,
                shm_fps, ws_fps,
                age_avg, c["age_max"],
                c["fusion_ok"], c["fusion_reject"],
            )

            # Reset window
            self._cams[cam_id] = {
                "window_start": now,
                "shm_count":     0,
                "ws_count":      0,
                "age_sum":       0.0,
                "age_max":       0.0,
                "fusion_ok":     c["fusion_ok"],   # carry over for trend
                "fusion_reject": c["fusion_reject"],
            }
            self._cams[cam_id]["fusion_ok"]     = 0
            self._cams[cam_id]["fusion_reject"]  = 0


# ── Module-level singletons (shared across pipeline modules) ───────────

#: Global throttle instance — import and use anywhere in pipeline.*
throttle = ThrottleMap()

#: Global per-camera aggregator
agg = PerCameraAggregator()
