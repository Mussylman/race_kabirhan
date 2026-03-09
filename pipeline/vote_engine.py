"""
vote_engine.py — Position-based weighted voting for finish order.

Extracted from tools/test_race_count.py.  The algorithm is:

1. Each frame: detect jockeys → sort by X → assign positions 1st..Nth
2. Enforce unique colors per frame (reassign duplicates via softmax fallback)
3. Vote per position, weighted by frame completeness (5/5 visible → weight 5)
4. Final result: 3-pass assignment (strict → relaxed → fallback)

Usage:
    engine = VoteEngine(colors=["blue","green","purple","red","yellow"])
    engine.submit_frame(detections)   # list of {color, conf, center_x, prob_dict}
    result = engine.compute_result()  # ["red", "blue", "green", ...]
    engine.reset()                    # clear for next race / video
"""

from collections import Counter, defaultdict
from typing import Optional


# ── Default parameters (can be overridden in constructor) ──────────────

DEFAULT_MIN_JOCKEYS_FOR_VOTE = 2
DEFAULT_MIN_VOTES_PER_POS = 5
DEFAULT_MIN_REASSIGN_CONF = 0.20
DEFAULT_VOTE_WEIGHTS = {5: 5, 4: 4, 3: 3, 2: 1}


class VoteEngine:
    """Accumulates position votes across frames and resolves finish order."""

    def __init__(
        self,
        colors: list[str],
        *,
        min_jockeys_for_vote: int = DEFAULT_MIN_JOCKEYS_FOR_VOTE,
        min_votes_per_pos: int = DEFAULT_MIN_VOTES_PER_POS,
        min_reassign_conf: float = DEFAULT_MIN_REASSIGN_CONF,
        vote_weights: Optional[dict[int, int]] = None,
    ):
        self.colors = list(colors)
        self.n_colors = len(colors)
        self.min_jockeys_for_vote = min_jockeys_for_vote
        self.min_votes_per_pos = min_votes_per_pos
        self.min_reassign_conf = min_reassign_conf
        self.vote_weights = vote_weights or dict(DEFAULT_VOTE_WEIGHTS)

        # position_votes[pos_index] = Counter({color: weighted_count})
        self.position_votes: dict[int, Counter] = defaultdict(Counter)
        self.vote_frames: int = 0
        self.total_votes: int = 0

    # ── Per-frame input ────────────────────────────────────────────────

    def submit_frame(self, detections: list[dict]) -> tuple[list[dict], int]:
        """Process one frame's detections and accumulate votes.

        Args:
            detections: list of dicts, each with at least:
                - center_x (float): horizontal pixel position
                - color (str): predicted color
                - conf (float): classifier confidence
                - prob_dict (dict): {color_name: probability} from softmax

        Returns:
            (assigned_detections, vote_weight) — detections after uniqueness
            enforcement (sorted by -center_x), and the weight applied (0 if
            frame was not voted on).
        """
        if not detections:
            return [], 0

        # Sort by X (rightmost = 1st place)
        dets = sorted(detections, key=lambda d: -d['center_x'])

        # Enforce unique colors
        dets = self._enforce_unique(dets)

        n_unique = len(dets)
        used_in_vote = n_unique >= self.min_jockeys_for_vote
        vote_weight = self.vote_weights.get(n_unique, 1) if used_in_vote else 0

        if used_in_vote:
            self.vote_frames += 1
            for pos, det in enumerate(dets):
                self.position_votes[pos][det['color']] += vote_weight
                self.total_votes += vote_weight

        return dets, vote_weight

    # ── Uniqueness enforcement ─────────────────────────────────────────

    def _enforce_unique(self, detections: list[dict]) -> list[dict]:
        """Assign unique colors using full softmax probabilities.

        Instead of dropping duplicates, reassign them to their next-best
        unused color.  This prevents losing a 'green' jockey that got
        classified as 'blue' when a stronger 'blue' exists.
        """
        if len(detections) <= 1:
            return detections

        # Sort by top-1 confidence descending — strongest gets first pick
        dets = sorted(detections, key=lambda d: -d['conf'])

        used_colors: set[str] = set()
        assigned: list[dict] = []

        for det in dets:
            sorted_colors = sorted(
                det.get('prob_dict', {}).items(), key=lambda x: -x[1]
            )

            for color, prob in sorted_colors:
                if color not in used_colors and prob >= self.min_reassign_conf:
                    used_colors.add(color)
                    new_det = dict(det)
                    new_det['color'] = color
                    new_det['conf'] = prob
                    assigned.append(new_det)
                    break
            # If no unused color found with enough confidence → skip

        # Sort back by center_x descending (rightmost = 1st place)
        return sorted(assigned, key=lambda d: -d['center_x'])

    # ── Result computation ─────────────────────────────────────────────

    def compute_result(self) -> list[str]:
        """Compute finish order from accumulated position votes.

        Three passes:
            1. Strict: positions with >= min_votes_per_pos for an unused color
            2. Relaxed: positions with >= 2 votes for an unused color
            3. Fallback: place remaining colors in their best open position
        """
        if not self.position_votes:
            return []

        all_colors = set(self.colors)
        max_pos = max(self.position_votes.keys()) + 1

        result: list[Optional[str]] = [None] * max_pos
        used_colors: set[str] = set()

        # Pass 1: strict threshold
        for pos in range(max_pos):
            votes = self.position_votes.get(pos, Counter())
            for color, count in votes.most_common():
                if count < self.min_votes_per_pos:
                    break
                if color not in used_colors:
                    used_colors.add(color)
                    result[pos] = color
                    break

        # Pass 2: fill gaps with lower threshold (>= 2 votes)
        for pos in range(max_pos):
            if result[pos] is not None:
                continue
            votes = self.position_votes.get(pos, Counter())
            for color, count in votes.most_common():
                if count < 2:
                    break
                if color not in used_colors:
                    used_colors.add(color)
                    result[pos] = color
                    break

        # Pass 3: fallback — place remaining colors in best open position
        missing = all_colors - used_colors
        open_positions = [p for p in range(max_pos) if result[p] is None]

        if len(missing) == 1 and open_positions:
            missing_color = missing.pop()
            best_pos = max(
                open_positions,
                key=lambda p: self.position_votes.get(p, Counter()).get(missing_color, 0),
            )
            result[best_pos] = missing_color
            used_colors.add(missing_color)
        elif missing and open_positions:
            for color in sorted(missing):
                if not open_positions:
                    break
                best_pos = max(
                    open_positions,
                    key=lambda p: self.position_votes.get(p, Counter()).get(color, 0),
                )
                votes_here = self.position_votes.get(best_pos, Counter()).get(color, 0)
                if votes_here > 0:
                    result[best_pos] = color
                    used_colors.add(color)
                    open_positions.remove(best_pos)

        return [c for c in result if c is not None]

    # ── Readiness check ────────────────────────────────────────────────

    def is_result_ready(self, min_frames: int = 3) -> bool:
        """Check if we have enough votes for a confident result.

        Ready when:
            - At least min_frames voted frames collected
            - All 5 positions have a clear winner (>= min_votes_per_pos)
        """
        if self.vote_frames < min_frames:
            return False

        # Check that each position up to n_colors has a winner
        for pos in range(min(self.n_colors, max(self.position_votes.keys()) + 1 if self.position_votes else 0)):
            votes = self.position_votes.get(pos, Counter())
            if not votes:
                return False
            top_count = votes.most_common(1)[0][1]
            if top_count < self.min_votes_per_pos:
                return False

        return len(self.position_votes) >= self.n_colors

    # ── Utilities ──────────────────────────────────────────────────────

    def reset(self):
        """Clear all votes (for next race / video)."""
        self.position_votes.clear()
        self.vote_frames = 0
        self.total_votes = 0

    def get_vote_table(self) -> list[dict]:
        """Return vote table for diagnostics.

        Returns list of dicts:
            [{"position": 1, "votes": {"red": 50, "blue": 10, ...}, "winner": "red"}, ...]
        """
        table = []
        for pos in sorted(self.position_votes.keys()):
            votes = dict(self.position_votes[pos])
            top = self.position_votes[pos].most_common(1)
            winner = top[0][0] if top else None
            table.append({
                "position": pos + 1,
                "votes": votes,
                "winner": winner,
            })
        return table
