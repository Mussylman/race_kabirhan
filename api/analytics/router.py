"""
router.py — FastAPI router for the analytics service.

Mount in api/server.py with:
    from api.analytics.router import build_analytics_router
    app.include_router(build_analytics_router())

Endpoints (all under /analytics):
  POST   /recordings/index             — build/refresh index for a recording
  GET    /recordings                   — list recordings
  GET    /recordings/{rec}/cameras     — list cameras
  GET    /recordings/{rec}/timeline    — timeline buckets per camera
  GET    /recordings/{rec}/detections  — filtered detections
  GET    /frame/{rec}/{cam}/{seq}      — annotated frame JPEG
  GET    /crop/{rec}/{cam}/{seq}/{det} — detection crop JPEG
  GET    /pairs/stats                  — labeling stats
  GET    /pairs/next                   — next pair for labeler
  POST   /pairs/{id}/label             — record a label
  GET    /pairs/labeled                — list labeled pairs (for export)
  POST   /pairs/bulk                   — import auto-generated pairs

Dependencies (lazy-initialised singletons keyed by artifacts dir):
  AnalyticsIndex, FrameServer, PairsStore
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from api.analytics.export import DatasetExporter
from api.analytics.frame_server import FrameServer
from api.analytics.indexer import AnalyticsIndex
from api.analytics.pairs_store import PairsStore

log = logging.getLogger("api.analytics.router")

_DEFAULT_ARTIFACTS = os.environ.get(
    "ARTIFACTS_DIR",
    "/home/user/race_kabirhan_artifacts",
)


# ----------------------------------------------------------------------
# Singletons (one per router instance)
# ----------------------------------------------------------------------
class _AnalyticsContext:
    def __init__(self, artifacts_dir: str | Path):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.artifacts_dir / "analytics.db"
        self.index = AnalyticsIndex(self.db_path)
        self.frame_server = FrameServer(self.artifacts_dir)
        self.pairs = PairsStore(self.db_path)
        self.exporter = DatasetExporter(
            self.index, self.pairs, self.frame_server, self.artifacts_dir
        )


# ----------------------------------------------------------------------
# Request/response models
# ----------------------------------------------------------------------
class IndexRequest(BaseModel):
    rec_id: str
    jsonl_dir: str
    mp4_dir: Optional[str] = None


class LabelRequest(BaseModel):
    label: str  # "same" | "different" | "skip"
    labeler: Optional[str] = "operator"


class PairSide(BaseModel):
    cam: str
    seq: int
    det_id: Optional[int] = None
    bbox: list[float]
    color: Optional[str] = None
    conf: Optional[float] = 0.0


class BulkPairsRequest(BaseModel):
    rec_id: str
    pairs: list[dict]  # each: {a: PairSide, b: PairSide, auto_conf, source?}


# ----------------------------------------------------------------------
# Router factory
# ----------------------------------------------------------------------
def build_analytics_router(artifacts_dir: str | Path = _DEFAULT_ARTIFACTS) -> APIRouter:
    ctx = _AnalyticsContext(artifacts_dir)
    router = APIRouter(prefix="/analytics", tags=["analytics"])

    # ── Recordings ────────────────────────────────────────────────────
    @router.post("/recordings/index")
    def index_recording(req: IndexRequest):
        try:
            stats = ctx.index.index_recording(req.rec_id, req.jsonl_dir, req.mp4_dir)
            return stats
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))

    @router.get("/recordings")
    def list_recordings():
        return ctx.index.list_recordings()

    @router.get("/recordings/{rec_id}/cameras")
    def list_cameras(rec_id: str):
        cams = ctx.index.list_cameras(rec_id)
        if not cams:
            raise HTTPException(404, f"no cameras for rec {rec_id}")
        return cams

    @router.get("/recordings/{rec_id}/timeline")
    def timeline(rec_id: str,
                 bucket_s: float = Query(1.0, ge=0.1, le=30.0)):
        data = ctx.index.timeline(rec_id, bucket_s)
        if not data:
            return {}
        return data

    @router.get("/recordings/{rec_id}/detections")
    def detections(rec_id: str,
                   cam: str,
                   t_start: float = 0.0,
                   t_end: float = 9999.0):
        return ctx.index.detections_in_range(rec_id, cam, t_start, t_end)

    # ── Frames & crops ────────────────────────────────────────────────
    @router.get("/frame/{rec_id}/{cam_id}/{seq}")
    def get_frame(rec_id: str, cam_id: str, seq: int,
                  annotate: bool = True):
        cams = {c["cam_id"]: c for c in ctx.index.list_cameras(rec_id)}
        if cam_id not in cams:
            raise HTTPException(404, f"cam {cam_id} not in rec {rec_id}")
        mp4 = cams[cam_id].get("mp4_path")
        if not mp4:
            raise HTTPException(404, f"no mp4 for {cam_id}")
        dets = ctx.index.frame_detections(rec_id, cam_id, seq) if annotate else None
        path = ctx.frame_server.get_frame(rec_id, cam_id, mp4, seq, dets)
        if not path:
            raise HTTPException(500, "could not extract frame")
        return FileResponse(path, media_type="image/jpeg")

    @router.get("/crop/{rec_id}/{cam_id}/{seq}/{det_id}")
    def get_crop(rec_id: str, cam_id: str, seq: int, det_id: int):
        cams = {c["cam_id"]: c for c in ctx.index.list_cameras(rec_id)}
        if cam_id not in cams:
            raise HTTPException(404, f"cam {cam_id} not in rec {rec_id}")
        mp4 = cams[cam_id].get("mp4_path")
        if not mp4:
            raise HTTPException(404, f"no mp4 for {cam_id}")
        dets = ctx.index.frame_detections(rec_id, cam_id, seq)
        if det_id < 0 or det_id >= len(dets):
            raise HTTPException(404, f"det {det_id} out of range")
        d = dets[det_id]
        bbox = (d["bbox_x1"], d["bbox_y1"], d["bbox_x2"], d["bbox_y2"])
        path = ctx.frame_server.get_crop(rec_id, cam_id, mp4, seq, det_id, bbox)
        if not path:
            raise HTTPException(500, "could not extract crop")
        return FileResponse(path, media_type="image/jpeg")

    # ── Pairs / labeler ───────────────────────────────────────────────
    @router.get("/pairs/stats")
    def pairs_stats(rec_id: str):
        return ctx.pairs.stats(rec_id)

    @router.get("/pairs/next")
    def pairs_next(rec_id: str,
                   strategy: str = Query("uncertainty",
                                         regex="^(uncertainty|random|auto_conf_middle)$")):
        p = ctx.pairs.next_to_label(rec_id, strategy)
        if not p:
            return JSONResponse({"done": True})
        return p

    @router.post("/pairs/{pair_id}/label")
    def pairs_label(pair_id: int, req: LabelRequest):
        ok = ctx.pairs.label(pair_id, req.label, req.labeler or "operator")
        if not ok:
            raise HTTPException(404, f"pair {pair_id} not found")
        return {"ok": True, "pair_id": pair_id}

    @router.get("/pairs/labeled")
    def pairs_labeled(rec_id: str, label: Optional[str] = None):
        return ctx.pairs.list_labeled(rec_id, label)

    @router.post("/pairs/bulk")
    def pairs_bulk(req: BulkPairsRequest):
        n = ctx.pairs.add_pairs_bulk([
            {**p, "rec_id": req.rec_id} for p in req.pairs
        ])
        return {"inserted": n}

    # ── Export ────────────────────────────────────────────────────────
    @router.post("/export/market1501")
    def export_market1501(rec_id: str):
        path = ctx.exporter.export_market1501(rec_id)
        return {"path": str(path)}

    @router.post("/export/triplets")
    def export_triplets(rec_id: str,
                        negatives_per_anchor: int = 4):
        path = ctx.exporter.export_triplets(rec_id,
                                            negatives_per_anchor=negatives_per_anchor)
        return {"path": str(path)}

    @router.post("/export/coco")
    def export_coco(rec_id: str):
        path = ctx.exporter.export_coco(rec_id)
        return {"path": str(path)}

    return router
