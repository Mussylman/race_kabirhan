"""Time-based race tracker.

Simpler alternative to FusionEngine: rank jockeys by which physical camera
they were last confirmed at, tiebreak by timestamp of passing.

Event model:
    ingest(ts, cam_id, color) — one detection event
Confirmation:
    A jockey is "confirmed passed" a camera after N detections of that color
    on that camera within a rolling window of W seconds.
    FIRST registration (color not yet in system) additionally requires
    ≥pack_min_colors distinct classified colors active on that same
    camera within the same window — solves "which camera does the race
    start from?" (a lone false classification on a spectator won't count).
Invariants:
    - Backward moves (confirmed at cam_idx < current) are IGNORED.
    - Missing sightings keep the jockey at their last confirmed camera
      (no timeout reset).
Ranking:
    sort by cam_idx DESC, then by confirmed-pass ts ASC.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class _State:
    cam_idx: int
    pass_ts: float        # wall-clock ts when pass was confirmed
    last_seen_ts: float   # most recent sighting (may be same cam)


class TimeTracker:
    def __init__(
        self,
        cam_order: list[str],
        topology=None,
        confirm_frames: int = 3,
        confirm_window_sec: float = 1.0,
        pack_min_colors: int = 2,
    ):
        self.cam_order = list(cam_order)
        self.cam_idx = {cid: i for i, cid in enumerate(self.cam_order)}
        self.topology = topology
        self.confirm_frames = confirm_frames
        self.confirm_window = confirm_window_sec
        self.pack_min_colors = pack_min_colors

        # Rolling buffer of recent ts per (cam_id, color)
        self._buffers: dict[tuple[str, str], list[float]] = {}
        # Confirmed per-color state
        self._state: dict[str, _State] = {}

    # ------------------------------------------------------------------
    def ingest(self, ts: float, cam_id: str, color: str) -> bool:
        """Record one detection. Returns True iff at least one NEW
        forward pass was confirmed on this call (own color or a packmate)."""
        cam_idx = self.cam_idx.get(cam_id)
        if cam_idx is None or not color:
            return False

        # Touch last_seen for UI liveness, even if not confirming
        cur = self._state.get(color)
        if cur is not None:
            cur.last_seen_ts = ts

        key = (cam_id, color)
        buf = self._buffers.setdefault(key, [])
        cutoff = ts - self.confirm_window
        while buf and buf[0] < cutoff:
            buf.pop(0)
        buf.append(ts)

        # Sweep: any buffer on this cam with enough recent entries is a
        # candidate. Gather candidates, check pack condition, then commit.
        candidates = []
        for (c_id, c_color), tss in self._buffers.items():
            if c_id != cam_id or len(tss) < self.confirm_frames:
                continue
            # prune stale and re-check
            while tss and tss[0] < cutoff:
                tss.pop(0)
            if len(tss) < self.confirm_frames:
                continue
            cur_c = self._state.get(c_color)
            # only forward moves count as candidates
            if cur_c is not None and cam_idx <= cur_c.cam_idx:
                continue
            candidates.append((c_color, tss[0]))

        if not candidates:
            return False

        # Pack check: first-time registrations need ≥N distinct active
        # colors on this cam within the window (pool = candidates + already
        # registered colors with recent activity here).
        first_time_candidates = [
            (c, t) for c, t in candidates if c not in self._state
        ]
        if first_time_candidates and self.pack_min_colors > 1:
            active_colors = set()
            # still-buffered (detected recently, not yet confirmed/cleared)
            for (c_id, c_color), tss in self._buffers.items():
                if c_id == cam_id and tss and tss[-1] >= cutoff:
                    active_colors.add(c_color)
            # already-registered AT this cam with recent sighting
            for c_color, st in self._state.items():
                if st.cam_idx == cam_idx and st.last_seen_ts >= cutoff:
                    active_colors.add(c_color)
            if len(active_colors) < self.pack_min_colors:
                return False

        # Commit all candidates atomically
        for c_color, pass_ts in candidates:
            self._state[c_color] = _State(
                cam_idx=cam_idx, pass_ts=pass_ts, last_seen_ts=ts,
            )
            self._buffers[(cam_id, c_color)].clear()
        return True

    # ------------------------------------------------------------------
    def get_ranking(self) -> list[dict]:
        """Ranking in `_build_rankings`-compatible format.

        Returns list of dicts with: color, position_m, rank, speed_mps,
        last_camera, pass_ts, last_seen_ts.
        """
        items = sorted(
            self._state.items(),
            key=lambda kv: (-kv[1].cam_idx, kv[1].pass_ts),
        )
        out = []
        for rank, (color, st) in enumerate(items, 1):
            cam_id = self.cam_order[st.cam_idx]
            position_m = self._cam_center_m(cam_id)
            out.append({
                "color": color,
                "position_m": position_m,
                "rank": rank,
                "speed_mps": 0.0,         # not estimated in time-tracker
                "last_camera": cam_id,
                "pass_ts": st.pass_ts,
                "last_seen_ts": st.last_seen_ts,
            })
        return out

    # ------------------------------------------------------------------
    def _cam_center_m(self, cam_id: str) -> float:
        """Use topology's track_start..track_end midpoint for position."""
        if self.topology is None:
            return float(self.cam_idx[cam_id])
        seg = getattr(self.topology, "_segments", {}).get(cam_id)
        if seg is None:
            return float(self.cam_idx[cam_id])
        return 0.5 * (seg.track_start_m + seg.track_end_m)

    # ------------------------------------------------------------------
    def reset(self):
        self._buffers.clear()
        self._state.clear()

    def snapshot(self) -> dict:
        """Debug snapshot."""
        return {
            color: {
                "cam_id": self.cam_order[st.cam_idx],
                "cam_idx": st.cam_idx,
                "pass_ts": st.pass_ts,
                "last_seen_ts": st.last_seen_ts,
            }
            for color, st in self._state.items()
        }
