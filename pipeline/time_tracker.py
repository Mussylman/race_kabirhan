"""Per-color first-arrival tracker (no pack, no sum).

For each color: remember the FIRST camera where it was registered AND the
FIRST arrival timestamp on every later camera. Registration happens once
per (cam_id, color).

Ranking:
  — Sort colors by last cam_idx DESC (who's furthest along).
  — Tiebreak: earliest pass_ts on that cam (who arrived first).
"""

from __future__ import annotations


class TimeTracker:
    def __init__(self, cam_order: list[str], topology=None):
        self.cam_order = list(cam_order)
        self.cam_idx = {cid: i for i, cid in enumerate(self.cam_order)}
        self.topology = topology
        # color -> {cam_id, cam_idx, pass_ts, history: [(cam_id, ts), ...]}
        self._state: dict[str, dict] = {}
        # (cam_id, color) seen set — never re-register same cam
        self._seen: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------
    def ingest(self, ts: float, cam_id: str, color: str) -> list[str]:
        """Record one jockey sighting (already horse-gated by caller).
        Returns [color] if it advanced forward, else []."""
        cam_idx = self.cam_idx.get(cam_id)
        if cam_idx is None or not color:
            return []

        key = (cam_id, color)
        if key in self._seen:
            return []  # already registered at this cam

        cur = self._state.get(color)
        if cur is not None and cam_idx <= cur["cam_idx"]:
            return []  # backward or same — ignore

        self._seen.add(key)
        self._state[color] = {
            "cam_id": cam_id,
            "cam_idx": cam_idx,
            "pass_ts": ts,
        }
        return [color]

    # ------------------------------------------------------------------
    def get_ranking(self) -> list[dict]:
        items = sorted(
            self._state.items(),
            key=lambda kv: (-kv[1]["cam_idx"], kv[1]["pass_ts"]),
        )
        out = []
        for rank, (color, st) in enumerate(items, 1):
            out.append({
                "rank": rank,
                "color": color,
                "last_camera": st["cam_id"],
                "cam_idx": st["cam_idx"],
                "pass_ts": st["pass_ts"],
                "position_m": self._cam_center_m(st["cam_id"]),
                "speed_mps": 0.0,
                "last_seen_ts": st["pass_ts"],
                "color_conf": 1.0,
                "center_x": 0.0,
                "color_logit": 0.0,
            })
        return out

    def camera_ranking(self, cam_id: str) -> list[dict]:
        """All colors registered at cam_id, sorted by arrival ts."""
        rk = []
        for color, st in self._state.items():
            if st["cam_id"] == cam_id:
                rk.append((st["pass_ts"], color))
        rk.sort()
        return [
            {"rank": i + 1, "color": c, "pass_ts": ts, "color_conf": 1.0,
             "color_logit": 0.0, "center_x": 0.0}
            for i, (ts, c) in enumerate(rk)
        ]

    def _cam_center_m(self, cam_id: str) -> float:
        if self.topology is None:
            return float(self.cam_idx.get(cam_id, 0))
        seg = getattr(self.topology, "_segments", {}).get(cam_id)
        if seg is None:
            return float(self.cam_idx.get(cam_id, 0))
        return 0.5 * (seg.track_start_m + seg.track_end_m)

    def reset(self):
        self._state.clear()
        self._seen.clear()

    # Back-compat shims
    def ingest_frame(self, ts: float, cam_id: str,
                     detections: list) -> list[str]:
        """Not used with horse-gated flow — routed through ingest()."""
        return []

    def frozen_cameras(self) -> list[str]:
        return sorted({st["cam_id"] for st in self._state.values()})
