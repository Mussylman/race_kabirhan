"""
export.py — write labeled ReID pairs to three dataset formats:

  1. Market-1501 style directory tree
       dataset_reid/<rec>/
         bounding_box_train/<pid>_c<cam>s1_<seq>_00.jpg
         bounding_box_test/
         query/
         splits.json

     Consumable by torchreid's ImageDataset and many stock ReID training
     scripts without conversion.

  2. Triplets JSONL
       pairs/<rec>/triplets.jsonl
         {"anchor": "...jpg", "positive": "...jpg",
          "negatives": ["...jpg", ...], "jockey_id": N,
          "source": "manual"|"auto_phase1"}

     For custom training loops that need explicit triplets.

  3. COCO-extended JSON
       pairs/<rec>/coco.json
         {images, annotations (with jockey_id + color), categories}

     Import into FiftyOne / CVAT for further annotation.

The writer is driven by data from PairsStore (only pairs labeled "same"
become positive edges; pairs labeled "different" seed hard-negatives).
Pair → identity mapping is computed by connected-components over
"same" edges.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional

from api.analytics.frame_server import FrameServer
from api.analytics.indexer import AnalyticsIndex
from api.analytics.pairs_store import PairsStore

log = logging.getLogger("api.analytics.export")


# ----------------------------------------------------------------------
# Union-find over pair endpoints: pairs labelled "same" cluster the
# endpoints (cam,seq,det_id) into jockey identities.
# ----------------------------------------------------------------------
class _UnionFind:
    def __init__(self):
        self.parent: dict = {}

    def _find(self, x):
        while self.parent.get(x, x) != x:
            self.parent[x] = self.parent.get(self.parent[x], self.parent[x])
            x = self.parent[x]
        self.parent.setdefault(x, x)
        return x

    def union(self, a, b):
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self.parent[ra] = rb

    def group_of(self, x) -> int:
        return id(self._find(x))


def _endpoint_key(cam: str, seq: int, det_id) -> tuple:
    return (cam, int(seq), int(det_id) if det_id is not None else -1)


class DatasetExporter:
    def __init__(self,
                 index: AnalyticsIndex,
                 pairs: PairsStore,
                 frame_server: FrameServer,
                 artifacts_dir: str | Path):
        self.index = index
        self.pairs = pairs
        self.fs = frame_server
        self.artifacts = Path(artifacts_dir)

    # ------------------------------------------------------------------
    def _assign_jockey_ids(self, rec_id: str) -> dict[tuple, int]:
        """Cluster endpoints using 'same' labels. Returns map
        (cam, seq, det_id) → integer jockey_id (1-based)."""
        same_pairs = self.pairs.list_labeled(rec_id, label="same")
        uf = _UnionFind()
        for p in same_pairs:
            a = _endpoint_key(p["a_cam"], p["a_seq"], p.get("a_det_id"))
            b = _endpoint_key(p["b_cam"], p["b_seq"], p.get("b_det_id"))
            uf.union(a, b)
        # Also include singletons from "different" labels so their endpoints
        # get ids too (one distinct identity each, not merged).
        diff_pairs = self.pairs.list_labeled(rec_id, label="different")
        for p in diff_pairs:
            a = _endpoint_key(p["a_cam"], p["a_seq"], p.get("a_det_id"))
            b = _endpoint_key(p["b_cam"], p["b_seq"], p.get("b_det_id"))
            uf._find(a)
            uf._find(b)
        # Build group → small id
        group_to_id: dict = {}
        result: dict[tuple, int] = {}
        for key in list(uf.parent.keys()):
            g = uf.group_of(key)
            if g not in group_to_id:
                group_to_id[g] = len(group_to_id) + 1
            result[key] = group_to_id[g]
        return result

    def _materialise_crop(self, rec_id: str, cam: str, seq: int,
                          det_id: int, bbox: list[float]) -> Optional[Path]:
        cams = {c["cam_id"]: c for c in self.index.list_cameras(rec_id)}
        cam_row = cams.get(cam)
        if not cam_row or not cam_row.get("mp4_path"):
            return None
        return self.fs.get_crop(rec_id, cam, cam_row["mp4_path"],
                                seq, det_id, tuple(bbox))

    # ------------------------------------------------------------------
    # Market-1501 exporter
    # ------------------------------------------------------------------
    def export_market1501(self, rec_id: str,
                          out_dir: Optional[str | Path] = None,
                          val_cams: Optional[set[str]] = None) -> Path:
        """Build a Market-1501 style tree.

        Filename format: <pid>_c<NN>s1_<seq>_00.jpg where NN is the camera
        number parsed from "cam-NN". pid is zero-padded jockey id.

        val_cams: set of cam_ids to put into bounding_box_test/query.
                  If None, use a deterministic even-odd split on cams.
        """
        out = Path(out_dir or self.artifacts / "dataset_reid" / rec_id)
        if out.exists():
            shutil.rmtree(out)
        (out / "bounding_box_train").mkdir(parents=True)
        (out / "bounding_box_test").mkdir(parents=True)
        (out / "query").mkdir(parents=True)

        jid_map = self._assign_jockey_ids(rec_id)
        if not jid_map:
            log.warning("No labeled pairs for %s — nothing to export", rec_id)
            return out

        all_cams = {c["cam_id"] for c in self.index.list_cameras(rec_id)}
        if val_cams is None:
            cams_sorted = sorted(all_cams)
            val_cams = set(cams_sorted[::3])  # every 3rd cam → val

        queries_per_jid: dict[int, bool] = defaultdict(bool)
        written = 0
        for (cam, seq, det_id), jid in jid_map.items():
            dets = self.index.frame_detections(rec_id, cam, seq)
            if not dets or det_id < 0 or det_id >= len(dets):
                continue
            d = dets[det_id]
            bbox = [d["bbox_x1"], d["bbox_y1"], d["bbox_x2"], d["bbox_y2"]]
            crop_path = self._materialise_crop(rec_id, cam, seq, det_id, bbox)
            if not crop_path or not crop_path.exists():
                continue

            try:
                cam_num = int(cam.replace("cam-", ""))
            except ValueError:
                cam_num = 1
            fname = f"{jid:04d}_c{cam_num:02d}s1_{seq:06d}_00.jpg"

            if cam in val_cams:
                # First val-split sample for a jid → query; rest → gallery
                if not queries_per_jid[jid]:
                    dest = out / "query" / fname
                    queries_per_jid[jid] = True
                else:
                    dest = out / "bounding_box_test" / fname
            else:
                dest = out / "bounding_box_train" / fname
            shutil.copyfile(crop_path, dest)
            written += 1

        (out / "splits.json").write_text(json.dumps({
            "rec_id": rec_id,
            "val_cams": sorted(val_cams),
            "n_identities": len(set(jid_map.values())),
            "n_images": written,
        }, indent=2))
        log.info("Exported Market-1501: %d images, %d ids → %s",
                 written, len(set(jid_map.values())), out)
        return out

    # ------------------------------------------------------------------
    # Triplets JSONL
    # ------------------------------------------------------------------
    def export_triplets(self, rec_id: str,
                        out_path: Optional[str | Path] = None,
                        negatives_per_anchor: int = 4) -> Path:
        out = Path(out_path or
                   self.artifacts / "pairs" / rec_id / "triplets.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)

        same_pairs = self.pairs.list_labeled(rec_id, label="same")
        # Build jid → [endpoints]
        jid_map = self._assign_jockey_ids(rec_id)
        by_jid: dict[int, list[tuple]] = defaultdict(list)
        for ep, jid in jid_map.items():
            by_jid[jid].append(ep)

        n_triplets = 0
        with open(out, "w") as f:
            for p in same_pairs:
                a = _endpoint_key(p["a_cam"], p["a_seq"], p.get("a_det_id"))
                b = _endpoint_key(p["b_cam"], p["b_seq"], p.get("b_det_id"))
                anchor_jid = jid_map.get(a)
                if anchor_jid is None:
                    continue
                # Negatives: random endpoints from other jids
                negs: list = []
                for other_jid, eps in by_jid.items():
                    if other_jid == anchor_jid:
                        continue
                    if eps:
                        negs.append(eps[0])
                    if len(negs) >= negatives_per_anchor:
                        break
                rec = {
                    "anchor": list(a),
                    "positive": list(b),
                    "negatives": [list(n) for n in negs],
                    "jockey_id": anchor_jid,
                    "source": p.get("source", "manual"),
                }
                f.write(json.dumps(rec) + "\n")
                n_triplets += 1
        log.info("Exported triplets: %d → %s", n_triplets, out)
        return out

    # ------------------------------------------------------------------
    # COCO-extended
    # ------------------------------------------------------------------
    def export_coco(self, rec_id: str,
                    out_path: Optional[str | Path] = None) -> Path:
        out = Path(out_path or self.artifacts / "pairs" / rec_id / "coco.json")
        out.parent.mkdir(parents=True, exist_ok=True)

        jid_map = self._assign_jockey_ids(rec_id)
        images: list = []
        annotations: list = []
        seen_images: dict[tuple, int] = {}
        next_image_id = 1

        for (cam, seq, det_id), jid in jid_map.items():
            dets = self.index.frame_detections(rec_id, cam, seq)
            if not dets or det_id < 0 or det_id >= len(dets):
                continue
            d = dets[det_id]
            img_key = (cam, seq)
            if img_key not in seen_images:
                images.append({
                    "id": next_image_id,
                    "file_name": f"{cam}/{seq}.jpg",
                    "cam_id": cam, "frame_seq": seq,
                    "width": d.get("frame_w", 0),
                    "height": d.get("frame_h", 0),
                })
                seen_images[img_key] = next_image_id
                next_image_id += 1
            x1, y1, x2, y2 = d["bbox_x1"], d["bbox_y1"], d["bbox_x2"], d["bbox_y2"]
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": seen_images[img_key],
                "jockey_id": jid,
                "color": d.get("color"),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
                "category_id": 0,
            })
        coco = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 0, "name": "jockey"}],
            "info": {"rec_id": rec_id,
                     "n_identities": len(set(jid_map.values()))},
        }
        out.write_text(json.dumps(coco, indent=2))
        log.info("Exported COCO: %d imgs, %d anns → %s",
                 len(images), len(annotations), out)
        return out
