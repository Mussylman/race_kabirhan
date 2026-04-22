"""
indexer.py — builds a SQLite index over JSONL detection records so the
analytics UI can do fast time/camera/color lookups without scanning files.

Schema:
  recordings(rec_id, path, n_cameras, n_frames, duration_s, created_at)
  cameras(rec_id, cam_id, mp4_path, jsonl_path, n_frames, n_detections)
  detections(id, rec_id, cam_id, frame_seq, ts_capture, video_sec,
             frame_w, frame_h, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
             color, conf, track_id)

Indexes:
  idx_det_rec_cam_ts ON (rec_id, cam_id, ts_capture)
  idx_det_rec_color  ON (rec_id, color)
  idx_det_track      ON (rec_id, cam_id, track_id)

Re-indexing is idempotent: JSONL files are fingerprinted by (size, mtime),
and only changed files are re-parsed.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Iterable

log = logging.getLogger("api.analytics.indexer")

SCHEMA = """
CREATE TABLE IF NOT EXISTS recordings (
    rec_id       TEXT PRIMARY KEY,
    jsonl_dir    TEXT NOT NULL,
    mp4_dir      TEXT,
    n_cameras    INTEGER,
    n_frames     INTEGER,
    duration_s   REAL,
    created_at   INTEGER
);

CREATE TABLE IF NOT EXISTS cameras (
    rec_id         TEXT NOT NULL,
    cam_id         TEXT NOT NULL,
    mp4_path       TEXT,
    jsonl_path     TEXT NOT NULL,
    jsonl_size     INTEGER,
    jsonl_mtime    INTEGER,
    n_frames       INTEGER,
    n_detections   INTEGER,
    PRIMARY KEY (rec_id, cam_id)
);

CREATE TABLE IF NOT EXISTS detections (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    rec_id       TEXT    NOT NULL,
    cam_id       TEXT    NOT NULL,
    frame_seq    INTEGER NOT NULL,
    ts_capture   REAL,
    video_sec    REAL,
    frame_w      INTEGER,
    frame_h      INTEGER,
    bbox_x1      REAL,
    bbox_y1      REAL,
    bbox_x2      REAL,
    bbox_y2      REAL,
    color        TEXT,
    conf         REAL,
    track_id     INTEGER
);

CREATE INDEX IF NOT EXISTS idx_det_rec_cam_ts
    ON detections(rec_id, cam_id, ts_capture);
CREATE INDEX IF NOT EXISTS idx_det_rec_color
    ON detections(rec_id, color);
CREATE INDEX IF NOT EXISTS idx_det_rec_cam_track
    ON detections(rec_id, cam_id, track_id);
"""


class AnalyticsIndex:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def index_recording(self,
                        rec_id: str,
                        jsonl_dir: str | Path,
                        mp4_dir: str | Path | None = None) -> dict:
        """Index (or re-index) one recording. Returns summary stats."""
        jsonl_dir = Path(jsonl_dir)
        mp4_dir = Path(mp4_dir) if mp4_dir else None
        if not jsonl_dir.is_dir():
            raise FileNotFoundError(f"JSONL dir not found: {jsonl_dir}")

        self.conn.execute(
            "INSERT OR REPLACE INTO recordings (rec_id, jsonl_dir, mp4_dir, created_at) "
            "VALUES (?, ?, ?, strftime('%s','now'))",
            (rec_id, str(jsonl_dir), str(mp4_dir) if mp4_dir else None),
        )

        total_dets = 0
        total_frames = 0
        cam_count = 0
        for jsonl_file in sorted(jsonl_dir.glob("*.jsonl")):
            cam_id = self._cam_id_from_path(jsonl_file)
            stat = jsonl_file.stat()
            existing = self.conn.execute(
                "SELECT jsonl_size, jsonl_mtime FROM cameras WHERE rec_id=? AND cam_id=?",
                (rec_id, cam_id),
            ).fetchone()
            if existing and existing["jsonl_size"] == stat.st_size \
                    and existing["jsonl_mtime"] == int(stat.st_mtime):
                log.debug("Skip %s (unchanged)", jsonl_file.name)
                cam_count += 1
                continue

            # Purge stale detection rows for this cam.
            self.conn.execute(
                "DELETE FROM detections WHERE rec_id=? AND cam_id=?",
                (rec_id, cam_id),
            )

            n_frames, n_dets = self._ingest_jsonl(rec_id, cam_id, jsonl_file)
            total_frames += n_frames
            total_dets += n_dets
            cam_count += 1

            mp4_path = None
            if mp4_dir:
                for ext in (".mp4", ".MP4", ".mov"):
                    p = mp4_dir / f"{cam_id}{ext}"
                    if p.exists():
                        mp4_path = str(p)
                        break

            self.conn.execute(
                "INSERT OR REPLACE INTO cameras "
                "(rec_id, cam_id, mp4_path, jsonl_path, jsonl_size, jsonl_mtime, n_frames, n_detections) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (rec_id, cam_id, mp4_path, str(jsonl_file),
                 stat.st_size, int(stat.st_mtime), n_frames, n_dets),
            )

        self.conn.execute(
            "UPDATE recordings SET n_cameras=?, n_frames=? WHERE rec_id=?",
            (cam_count, total_frames, rec_id),
        )
        self.conn.commit()
        log.info("Indexed %s: %d cams, %d frames, %d detections",
                 rec_id, cam_count, total_frames, total_dets)
        return {"rec_id": rec_id, "cameras": cam_count,
                "frames": total_frames, "detections": total_dets}

    def _ingest_jsonl(self, rec_id: str, cam_id: str,
                      jsonl_file: Path) -> tuple[int, int]:
        n_frames = 0
        rows: list[tuple] = []
        with open(jsonl_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_frames += 1
                frame_seq = int(rec.get("frame_seq", 0))
                ts_capture = float(rec.get("ts_capture", 0) or 0)
                video_sec = float(rec.get("video_sec", 0) or 0)
                frame_w = int(rec.get("frame_w", 0) or 0)
                frame_h = int(rec.get("frame_h", 0) or 0)
                for det in rec.get("detections", []):
                    bbox = det.get("bbox") or [0, 0, 0, 0]
                    if len(bbox) < 4:
                        continue
                    rows.append((
                        rec_id, cam_id, frame_seq, ts_capture, video_sec,
                        frame_w, frame_h,
                        float(bbox[0]), float(bbox[1]),
                        float(bbox[2]), float(bbox[3]),
                        det.get("color"),
                        float(det.get("conf", 0) or 0),
                        int(det.get("track_id") or 0) or None,
                    ))
        if rows:
            self.conn.executemany(
                "INSERT INTO detections "
                "(rec_id, cam_id, frame_seq, ts_capture, video_sec, "
                " frame_w, frame_h, bbox_x1, bbox_y1, bbox_x2, bbox_y2, "
                " color, conf, track_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return n_frames, len(rows)

    @staticmethod
    def _cam_id_from_path(p: Path) -> str:
        """Derive cam_id from JSONL filename. Accepts 'cam-05.jsonl',
        'cam_05.jsonl', 'frames_cam-05.jsonl'."""
        stem = p.stem
        for prefix in ("frames_", "det_", "detections_"):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
        stem = stem.replace("cam_", "cam-")
        if not stem.startswith("cam-"):
            stem = f"cam-{stem}"
        return stem

    # ------------------------------------------------------------------
    # Queries used by router
    # ------------------------------------------------------------------
    def list_recordings(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT rec_id, jsonl_dir, mp4_dir, n_cameras, n_frames, created_at "
            "FROM recordings ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def list_cameras(self, rec_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT cam_id, mp4_path, n_frames, n_detections "
            "FROM cameras WHERE rec_id=? ORDER BY cam_id",
            (rec_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def timeline(self, rec_id: str, bucket_s: float = 1.0) -> dict:
        """Return {cam_id: [{t: bucket_sec, colors: {red: n, ...}, dets: N}, ...]}."""
        rows = self.conn.execute(
            "SELECT cam_id, CAST(video_sec / ? AS INTEGER) AS bucket, "
            "       color, COUNT(*) AS n "
            "FROM detections WHERE rec_id=? "
            "GROUP BY cam_id, bucket, color "
            "ORDER BY cam_id, bucket",
            (bucket_s, rec_id),
        ).fetchall()
        out: dict[str, list[dict]] = {}
        for r in rows:
            cam = r["cam_id"]
            t = float(r["bucket"]) * bucket_s
            colors = {r["color"] or "unknown": int(r["n"])}
            if not out.get(cam) or out[cam][-1]["t"] != t:
                out.setdefault(cam, []).append({"t": t, "colors": colors,
                                                "dets": int(r["n"])})
            else:
                out[cam][-1]["colors"].update(colors)
                out[cam][-1]["dets"] += int(r["n"])
        return out

    def frame_detections(self, rec_id: str, cam_id: str,
                         frame_seq: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM detections "
            "WHERE rec_id=? AND cam_id=? AND frame_seq=?",
            (rec_id, cam_id, frame_seq),
        ).fetchall()
        return [dict(r) for r in rows]

    def detections_in_range(self, rec_id: str, cam_id: str,
                            t_start: float, t_end: float) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM detections WHERE rec_id=? AND cam_id=? "
            "AND video_sec BETWEEN ? AND ? "
            "ORDER BY video_sec, frame_seq",
            (rec_id, cam_id, t_start, t_end),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()
