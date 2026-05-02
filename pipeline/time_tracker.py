"""TimeTracker v2 — per-camera order-of-appearance + global ranking.

One color = one jockey. Each camera maintains its own list of jockeys in
the order they first appeared in its FoV (frame-time). Global ranking on
every ingest = jockeys active on the most-recently-updated camera (sorted
by first_seen_ts ASC) followed by "laggards" (every other color seen so
far) which keep their relative order from the previous global ranking.

Replaces the prior CNN-era forward-only-cam_idx implementation. v2 has
no monotonic-progression assumption: backward camera updates are accepted,
the same (cam, color) pair refreshes state on every ingest, and the only
"forward-only" guard left is a per-(cam, color) backward-pixel-motion
sanity filter that drops obvious classifier flicker (bbox jumping right→
left across a single camera).

Spec: /tmp/timetracker_v2_spec.md (final, 2026-05-02).
"""

from __future__ import annotations

import os
import threading
from typing import Optional


# ── Tunables ────────────────────────────────────────────────────────────

# Sliding window in seconds: a jockey on a camera is "active" iff its
# last_seen_ts is within this window of the most recent ingest on that cam.
# Default 8.0 covers a typical FoV traversal at ~15 m/s through a 110m
# segment.
ACTIVE_WINDOW_SEC = float(os.environ.get("RV_ACTIVE_WINDOW_SEC", "8.0"))

# Backward-motion sanity margin in mux pixels. If a sighting on the same
# (cam, color) drops bbox_x by more than this many pixels relative to the
# previous sighting, the new sighting is treated as a classifier flicker
# (right→left jump) and ignored. ~4% of 1280px mux width by default.
SANITY_X_MARGIN = int(os.environ.get("RV_BACKWARD_MARGIN_PX", "50"))


class TimeTracker:
    """Per-camera order-of-appearance ranking tracker (v2).

    Thread-safety: ingest() and read methods are guarded by an internal
    Lock — ingest fires from the DeepStream probe thread, get_ranking()
    fires from the API broadcast loop.

    State:
        _cam_state[cam_id][color] = JockeyOnCam dict
        _last_ranking             = list[str]  (cached after last ingest)
        _last_update_cam_id       = str       (cam from most recent ingest)
    """

    def __init__(self,
                 cam_order: Optional[list[str]] = None,
                 topology=None):
        # Both args are kept for legacy call-site compatibility but are
        # NOT used by v2. cam_order's linear ordering is the very thing
        # we are dropping in this rewrite. topology was only needed for
        # _cam_center_m which is now an optional helper using cam_idx
        # if topology is provided, else returns the cam's index in the
        # order it was first seen.
        self.cam_order = list(cam_order) if cam_order else []
        self.cam_idx = {cid: i for i, cid in enumerate(self.cam_order)}
        self.topology = topology

        self._lock = threading.Lock()
        # _cam_state: cam_id -> {color -> JockeyOnCam dict}
        self._cam_state: dict[str, dict[str, dict]] = {}
        # Last computed ranking (list of color names, leader first)
        self._last_ranking: list[str] = []
        # Cam from most recent ingest — its active set drives the ranking
        self._last_update_cam_id: Optional[str] = None

    # ── Hot path ────────────────────────────────────────────────────────

    def ingest(self,
               ts: float,
               cam_id: str,
               color: str,
               bbox_x: Optional[float] = None) -> dict:
        """Record one sighting. Returns a dict describing what changed.

        Args:
            ts: wall-clock timestamp (seconds).
            cam_id: camera identifier (e.g. "cam-24").
            color: jockey color (e.g. "red"); falsy → not accepted.
            bbox_x: x-center of the bbox in mux pixels. None disables the
                backward-motion sanity filter (legacy callers without
                bbox info).

        Returns:
            {
              "accepted": bool,
              "filter_reason": str,        # "" if accepted, else reason
              "is_first_on_cam": bool,     # True iff first sighting of
                                           # this color on this cam_id —
                                           # the canonical moment for
                                           # [PASS] log
              "ranking_changed": bool,
              "committed_colors": [color]  # legacy compat shim — colors
                                           # that registered for the first
                                           # time on cam_id; equals
                                           # [color] iff is_first_on_cam,
                                           # else []
            }
        """
        # Cheap reject: empty color or empty cam_id
        if not color or not cam_id:
            return self._reject("empty_color_or_cam")

        with self._lock:
            cam_dict = self._cam_state.setdefault(cam_id, {})
            prev = cam_dict.get(color)

            # Backward-motion sanity filter (only if bbox_x provided AND
            # we have a prev sighting on the same cam+color)
            if (prev is not None
                    and bbox_x is not None
                    and bbox_x < prev["last_x"] - SANITY_X_MARGIN):
                return self._reject("backward_motion")

            # Update / insert the per-cam-per-color record
            if prev is None:
                cam_dict[color] = {
                    "color": color,
                    "first_seen_ts": ts,
                    "last_seen_ts":  ts,
                    "last_x":        bbox_x if bbox_x is not None else 0.0,
                    "n_sightings":   1,
                }
                is_first_on_cam = True
            else:
                prev["last_seen_ts"] = ts
                if bbox_x is not None:
                    prev["last_x"] = bbox_x
                prev["n_sightings"] += 1
                is_first_on_cam = False

            # Track which cam drove the most recent update (used by global
            # ranking algorithm as the "active set" source).
            self._last_update_cam_id = cam_id
            # Also track in cam_idx for legacy ranking metadata. New cams
            # get appended at the end; existing cams keep their index.
            if cam_id not in self.cam_idx:
                self.cam_idx[cam_id] = len(self.cam_order)
                self.cam_order.append(cam_id)

            # Recompute global ranking
            new_ranking = self._compute_ranking_locked(ts)
            ranking_changed = new_ranking != self._last_ranking
            self._last_ranking = new_ranking

            return {
                "accepted":         True,
                "filter_reason":    "",
                "is_first_on_cam":  is_first_on_cam,
                "ranking_changed":  ranking_changed,
                "committed_colors": [color] if is_first_on_cam else [],
            }

    # ── Read path ───────────────────────────────────────────────────────

    def get_ranking_simple(self) -> list[str]:
        """v2 clean API: return current global ranking as list of color
        names, leader first. Empty if nobody has been seen."""
        with self._lock:
            return list(self._last_ranking)

    def get_ranking(self) -> list[dict]:
        """Legacy-compat API: same data as get_ranking_simple but each
        entry is a dict with the fields the old call-sites expect.

        Returned dict per color:
            rank, color, last_camera, cam_idx, pass_ts, position_m,
            speed_mps, last_seen_ts, color_conf, center_x, color_logit
        """
        with self._lock:
            return self._build_legacy_ranking_locked()

    def camera_ranking(self, cam_id: str) -> list[dict]:
        """All jockeys currently registered at cam_id, sorted by
        first_seen_ts ASC. Used by per-camera OSD overlay."""
        with self._lock:
            cam_dict = self._cam_state.get(cam_id, {})
            ordered = sorted(cam_dict.values(),
                             key=lambda j: j["first_seen_ts"])
            return [
                {
                    "rank":         i + 1,
                    "color":        j["color"],
                    "pass_ts":      j["first_seen_ts"],
                    "color_conf":   1.0,
                    "color_logit":  0.0,
                    "center_x":     j["last_x"],
                }
                for i, j in enumerate(ordered)
            ]

    # ── Lifecycle ───────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all per-camera state and ranking. New race."""
        with self._lock:
            self._cam_state.clear()
            self._last_ranking = []
            self._last_update_cam_id = None

    # ── Legacy back-compat shims (no callers in current code) ───────────

    def ingest_frame(self, ts: float, cam_id: str,
                     detections: list) -> list[str]:
        """Not used by current pipelines (per-detection ingest() preferred).
        Kept as a no-op so legacy imports don't crash."""
        return []

    def frozen_cameras(self) -> list[str]:
        """Legacy method: in v1 returned cams that had at least one
        registered color. In v2 we keep the same semantic so any caller
        gets a sensible result."""
        with self._lock:
            return sorted(cam for cam, d in self._cam_state.items() if d)

    # ── Internal ────────────────────────────────────────────────────────

    @staticmethod
    def _reject(reason: str) -> dict:
        return {
            "accepted":         False,
            "filter_reason":    reason,
            "is_first_on_cam":  False,
            "ranking_changed":  False,
            "committed_colors": [],
        }

    def _compute_ranking_locked(self, now_ts: float) -> list[str]:
        """Build the global ranking. Must be called with self._lock held.

        Algorithm (per spec section 4):
          A) active_on_cam = colors on _last_update_cam_id with
             last_seen_ts within ACTIVE_WINDOW_SEC; sorted by first_seen_ts
             ASC (earlier appearance = leading position).
          B) laggards = every color ever seen on any camera, minus the
             active_on_cam set; sorted by their position in the previous
             ranking (preserves stability across updates).
          C) global = active_on_cam ++ laggards.
        """
        cam_id = self._last_update_cam_id
        if cam_id is None:
            return []

        cam_dict = self._cam_state.get(cam_id, {})
        active = [
            j for j in cam_dict.values()
            if (now_ts - j["last_seen_ts"]) <= ACTIVE_WINDOW_SEC
        ]
        active.sort(key=lambda j: j["first_seen_ts"])
        active_colors_in_order = [j["color"] for j in active]
        active_set = set(active_colors_in_order)

        # All ever-seen colors across all cameras
        all_seen: set[str] = set()
        for cd in self._cam_state.values():
            all_seen.update(cd.keys())

        laggards_set = all_seen - active_set
        # Preserve relative order from prev ranking
        laggards_ordered = [c for c in self._last_ranking if c in laggards_set]
        # Any laggard not in prev ranking (rare: race just started edge
        # case) → append by latest last_seen_ts DESC across all cams
        unseen_laggards = laggards_set - set(laggards_ordered)
        if unseen_laggards:
            def _max_last_seen(color: str) -> float:
                ts = 0.0
                for cd in self._cam_state.values():
                    j = cd.get(color)
                    if j and j["last_seen_ts"] > ts:
                        ts = j["last_seen_ts"]
                return ts
            laggards_ordered.extend(
                sorted(unseen_laggards, key=lambda c: -_max_last_seen(c))
            )

        return active_colors_in_order + laggards_ordered

    def _build_legacy_ranking_locked(self) -> list[dict]:
        """Build the legacy list-of-dicts ranking. Caller holds the lock.

        Each color's last_camera = the camera it was most-recently sighted
        on. pass_ts = first_seen_ts on that camera. position_m = topology
        midpoint if topology was provided, else cam_idx.
        """
        out = []
        for rank, color in enumerate(self._last_ranking, 1):
            # Find the most recent sighting of this color across all cams
            latest_cam: Optional[str] = None
            latest_ts = -1.0
            latest_x = 0.0
            first_seen_on_latest_cam = 0.0
            for cam_id, cam_dict in self._cam_state.items():
                j = cam_dict.get(color)
                if j is None:
                    continue
                if j["last_seen_ts"] > latest_ts:
                    latest_ts = j["last_seen_ts"]
                    latest_cam = cam_id
                    latest_x = j["last_x"]
                    first_seen_on_latest_cam = j["first_seen_ts"]
            if latest_cam is None:
                # Should never happen — _last_ranking only contains colors
                # that exist somewhere in _cam_state — but be defensive.
                continue
            out.append({
                "rank":         rank,
                "color":        color,
                "last_camera":  latest_cam,
                "cam_idx":      self.cam_idx.get(latest_cam, 0),
                "pass_ts":      first_seen_on_latest_cam,
                "position_m":   self._cam_center_m(latest_cam),
                "speed_mps":    0.0,
                "last_seen_ts": latest_ts,
                "color_conf":   1.0,
                "center_x":     latest_x,
                "color_logit":  0.0,
            })
        return out

    def _cam_center_m(self, cam_id: str) -> float:
        """Use topology's track_start..track_end midpoint when provided
        (legacy compat); otherwise fall back to insertion index."""
        if self.topology is None:
            return float(self.cam_idx.get(cam_id, 0))
        seg = getattr(self.topology, "_segments", {}).get(cam_id)
        if seg is None:
            return float(self.cam_idx.get(cam_id, 0))
        return 0.5 * (seg.track_start_m + seg.track_end_m)
