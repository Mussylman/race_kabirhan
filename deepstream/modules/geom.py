"""ROI / polygon geometry utilities.

Pure functions extracted from deepstream/pipeline.py during the refactor.
No side effects, no module-level state — safe to import from anywhere.
"""
from __future__ import annotations

import json
from pathlib import Path


def _load_roi_polygons(path: Path) -> dict[str, list[list[tuple[float, float]]]]:
    """Load normalized ROI polygons from JSON. Returns
    {cam_id: [polygon1, polygon2, ...]} where each polygon is a list of
    (x_norm, y_norm) pairs in [0, 1]."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    out: dict[str, list[list[tuple[float, float]]]] = {}
    for cam_id, polys in data.items():
        cam_polys = []
        for p in polys:
            pts = [(float(pt["x"]), float(pt["y"])) for pt in p]
            if len(pts) >= 3:
                cam_polys.append(pts)
        if cam_polys:
            out[cam_id] = cam_polys
    return out


def _point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside
