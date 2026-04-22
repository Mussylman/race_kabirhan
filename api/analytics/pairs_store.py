"""
pairs_store.py — persistent store for ReID cross-camera pair labels.

Each pair:
  - id          : stable integer (assigned at creation)
  - rec_id      : recording this pair is from
  - a_cam, a_seq, a_det_id, a_bbox, a_color, a_conf   — left side
  - b_cam, b_seq, b_det_id, b_bbox, b_color, b_conf   — right side
  - auto_conf   : automatic label confidence from Phase 1+2 [0..1]
  - model_sim   : ReID model similarity at last check [0..1] (nullable)
  - uncertainty : current active-learning priority score [0..1, higher=more useful]
  - human_label : "same" | "different" | "skip" | None (unlabeled)
  - labeled_at  : unix ts or None
  - labeler     : free-form tag for who labeled (default "operator")
  - source      : "auto_phase1" | "auto_phase2" | "manual_seed"

Backed by a SQLite table in the same DB as the detection index. Exposed
via simple CRUD methods; the labeler UI calls `next_to_label()` which
implements the priority queue (uncertainty + unlabeled + recent retrain
first).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("api.analytics.pairs_store")

SCHEMA = """
CREATE TABLE IF NOT EXISTS reid_pairs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    rec_id        TEXT NOT NULL,
    a_cam         TEXT NOT NULL,
    a_seq         INTEGER NOT NULL,
    a_det_id      INTEGER,
    a_bbox        TEXT,        -- JSON [x1,y1,x2,y2]
    a_color       TEXT,
    a_conf        REAL,
    b_cam         TEXT NOT NULL,
    b_seq         INTEGER NOT NULL,
    b_det_id      INTEGER,
    b_bbox        TEXT,
    b_color       TEXT,
    b_conf        REAL,
    auto_conf     REAL,
    model_sim     REAL,
    uncertainty   REAL,
    human_label   TEXT,        -- same/different/skip/null
    labeled_at    INTEGER,
    labeler       TEXT,
    source        TEXT,
    created_at    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pairs_rec ON reid_pairs(rec_id);
CREATE INDEX IF NOT EXISTS idx_pairs_unlabeled
    ON reid_pairs(rec_id, human_label, uncertainty);
CREATE INDEX IF NOT EXISTS idx_pairs_labeled ON reid_pairs(rec_id, human_label);
"""


class PairsStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def add_pair(self, rec_id: str,
                 a: dict, b: dict,
                 auto_conf: float,
                 source: str = "auto_phase1") -> int:
        """Create a new pair. `a` and `b` must each have keys:
        cam, seq, det_id, bbox (list of 4 floats), color, conf.
        """
        uncertainty = self._initial_uncertainty(auto_conf)
        cur = self.conn.execute(
            "INSERT INTO reid_pairs "
            "(rec_id, a_cam, a_seq, a_det_id, a_bbox, a_color, a_conf, "
            " b_cam, b_seq, b_det_id, b_bbox, b_color, b_conf, "
            " auto_conf, uncertainty, source, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%s','now'))",
            (rec_id,
             a["cam"], int(a["seq"]), a.get("det_id"),
             json.dumps(a["bbox"]), a.get("color"), float(a.get("conf", 0)),
             b["cam"], int(b["seq"]), b.get("det_id"),
             json.dumps(b["bbox"]), b.get("color"), float(b.get("conf", 0)),
             float(auto_conf), float(uncertainty), source),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def add_pairs_bulk(self, pairs: list[dict]) -> int:
        """Bulk import from Phase 1+2 auto-labeler output. Returns count inserted."""
        rows = []
        for p in pairs:
            uc = self._initial_uncertainty(p["auto_conf"])
            rows.append((
                p["rec_id"],
                p["a"]["cam"], int(p["a"]["seq"]), p["a"].get("det_id"),
                json.dumps(p["a"]["bbox"]), p["a"].get("color"),
                float(p["a"].get("conf", 0)),
                p["b"]["cam"], int(p["b"]["seq"]), p["b"].get("det_id"),
                json.dumps(p["b"]["bbox"]), p["b"].get("color"),
                float(p["b"].get("conf", 0)),
                float(p["auto_conf"]), float(uc),
                p.get("source", "auto_phase1"),
            ))
        self.conn.executemany(
            "INSERT INTO reid_pairs "
            "(rec_id, a_cam, a_seq, a_det_id, a_bbox, a_color, a_conf, "
            " b_cam, b_seq, b_det_id, b_bbox, b_color, b_conf, "
            " auto_conf, uncertainty, source, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%s','now'))",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def label(self, pair_id: int, human_label: str,
              labeler: str = "operator") -> bool:
        """Record a human label. Returns True if the pair was found."""
        if human_label not in ("same", "different", "skip"):
            raise ValueError(f"invalid label: {human_label}")
        cur = self.conn.execute(
            "UPDATE reid_pairs SET human_label=?, labeled_at=?, labeler=? "
            "WHERE id=?",
            (human_label, int(time.time()), labeler, int(pair_id)),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def update_model_scores(self, updates: list[tuple[int, float]]):
        """After retrain, update model_sim and derived uncertainty for a batch of ids.

        updates: [(pair_id, new_model_sim), ...]
        """
        rows = [(float(sim),
                 self._uncertainty_from_sim(float(sim)),
                 int(pid)) for pid, sim in updates]
        self.conn.executemany(
            "UPDATE reid_pairs SET model_sim=?, uncertainty=? WHERE id=?",
            rows,
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Read — priority queue for labeler
    # ------------------------------------------------------------------
    def next_to_label(self, rec_id: str, strategy: str = "uncertainty",
                      exclude_ids: Optional[list[int]] = None) -> Optional[dict]:
        """Return the next pair the labeler should see, or None if no pairs left.

        strategy:
          - "uncertainty": highest uncertainty first (default; active learning)
          - "random": random unlabeled
          - "auto_conf_middle": pairs with auto_conf in [0.4, 0.7] (Phase 2 fallback)
        """
        excl = tuple(exclude_ids or [])
        excl_sql = ""
        params: list = [rec_id]
        if excl:
            excl_sql = " AND id NOT IN (" + ",".join("?" * len(excl)) + ")"
            params.extend(excl)

        if strategy == "uncertainty":
            sql = (
                "SELECT * FROM reid_pairs WHERE rec_id=? AND human_label IS NULL"
                + excl_sql
                + " ORDER BY uncertainty DESC, id ASC LIMIT 1"
            )
        elif strategy == "random":
            sql = (
                "SELECT * FROM reid_pairs WHERE rec_id=? AND human_label IS NULL"
                + excl_sql
                + " ORDER BY RANDOM() LIMIT 1"
            )
        elif strategy == "auto_conf_middle":
            sql = (
                "SELECT * FROM reid_pairs WHERE rec_id=? AND human_label IS NULL "
                "AND auto_conf BETWEEN 0.4 AND 0.7"
                + excl_sql
                + " ORDER BY ABS(auto_conf - 0.5) ASC LIMIT 1"
            )
        else:
            raise ValueError(f"unknown strategy: {strategy}")

        row = self.conn.execute(sql, tuple(params)).fetchone()
        return self._row_to_dict(row) if row else None

    def get_pair(self, pair_id: int) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM reid_pairs WHERE id=?", (int(pair_id),)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def stats(self, rec_id: str) -> dict:
        row = self.conn.execute(
            "SELECT "
            " COUNT(*) AS total, "
            " SUM(CASE WHEN human_label='same' THEN 1 ELSE 0 END) AS same_cnt, "
            " SUM(CASE WHEN human_label='different' THEN 1 ELSE 0 END) AS diff_cnt, "
            " SUM(CASE WHEN human_label='skip' THEN 1 ELSE 0 END) AS skip_cnt, "
            " SUM(CASE WHEN human_label IS NULL THEN 1 ELSE 0 END) AS unlabeled "
            "FROM reid_pairs WHERE rec_id=?",
            (rec_id,),
        ).fetchone()
        return dict(row) if row else {}

    def list_labeled(self, rec_id: str,
                     label: Optional[str] = None) -> list[dict]:
        if label:
            rows = self.conn.execute(
                "SELECT * FROM reid_pairs WHERE rec_id=? AND human_label=? "
                "ORDER BY labeled_at",
                (rec_id, label),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM reid_pairs WHERE rec_id=? "
                "AND human_label IS NOT NULL ORDER BY labeled_at",
                (rec_id,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Active-learning helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _initial_uncertainty(auto_conf: float) -> float:
        """Peaks at auto_conf=0.5 (maximum ambiguity)."""
        return max(0.0, 1.0 - abs(auto_conf - 0.5) * 2.0)

    @staticmethod
    def _uncertainty_from_sim(model_sim: float,
                              threshold: float = 0.6) -> float:
        """After a retrain, pair is most useful when model similarity is
        near the decision threshold."""
        d = abs(model_sim - threshold)
        # Scale so d=0 → uncertainty=1, d=threshold → 0
        return max(0.0, 1.0 - d / max(threshold, 1 - threshold))

    @staticmethod
    def _row_to_dict(row) -> dict:
        if row is None:
            return None
        d = dict(row)
        for key in ("a_bbox", "b_bbox"):
            if d.get(key):
                try:
                    d[key] = json.loads(d[key])
                except (TypeError, json.JSONDecodeError):
                    pass
        return d

    def close(self):
        self.conn.close()
