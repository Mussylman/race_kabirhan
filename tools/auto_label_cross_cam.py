#!/usr/bin/env python3
"""
auto_label_cross_cam.py — generate cross-camera ReID pair candidates
(Phase 1 + 2 of the labelling pipeline).

Idea:
  Horses run through cameras in a known order. Between consecutive cameras
  the pack is usually contiguous (~30-50m spread). Within a camera's last
  few seconds of detections, sort horses by X-pixel — that's the pack order.
  Within the next camera's first few seconds, do the same. Match position
  k in camera A to position k in camera B.

  Phase 2: validate with color. If both have a confident color and they
  match, upweight confidence; if they disagree, downweight.

Output: upload pairs into PairsStore via the analytics router:
  POST /analytics/pairs/bulk  {rec_id, pairs: [...]}

Or write a JSON file for manual import.

Usage:
  python tools/auto_label_cross_cam.py \\
      --rec-id rec2_yaris_20260303_164835 \\
      --jsonl-dir /tmp/logs2/frames \\
      --cam-order "cam-13,cam-08,cam-14,cam-11,cam-15,cam-16,cam-17,cam-18,cam-19,cam-20,cam-21,cam-22,cam-24,cam-01,cam-02,cam-05,cam-06,cam-03" \\
      --api http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

log = logging.getLogger("tools.auto_label_cross_cam")

# Phase 1+2 parameters
EXIT_WINDOW_S = 2.0     # last N seconds on cam_A
ENTRY_WINDOW_S = 2.0    # first N seconds on cam_B
MIN_HANDOFF_S = 2.0     # min gap (pack travel time between cameras)
MAX_HANDOFF_S = 30.0    # max gap (slow pack or large inter-cam distance)
MIN_DETECTIONS_PER_WINDOW = 3


def load_jsonl(path: Path) -> list[dict]:
    recs: list[dict] = []
    if not path.exists():
        return recs
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def detections_in_window(records: list[dict],
                         t_start: float, t_end: float) -> list[dict]:
    """Flatten per-frame records → per-detection rows in a time window.

    Each output dict: cam_id, frame_seq, video_sec, ts_capture, bbox,
                      color, conf, track_id, det_index_in_frame, x_center
    """
    out: list[dict] = []
    for rec in records:
        t = rec.get("video_sec")
        if t is None:
            t = rec.get("ts_capture", 0)
        if t is None:
            continue
        if t < t_start or t > t_end:
            continue
        for idx, det in enumerate(rec.get("detections", [])):
            bbox = det.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x_center = (bbox[0] + bbox[2]) / 2.0
            out.append({
                "cam_id": rec.get("cam_id"),
                "frame_seq": rec.get("frame_seq", 0),
                "video_sec": t,
                "ts_capture": rec.get("ts_capture", 0),
                "bbox": bbox,
                "color": det.get("color"),
                "conf": float(det.get("conf", 0)),
                "track_id": det.get("track_id"),
                "det_index_in_frame": idx,
                "x_center": x_center,
            })
    return out


def last_video_sec(records: list[dict]) -> Optional[float]:
    ts = [r.get("video_sec") for r in records
          if r.get("video_sec") is not None]
    return max(ts) if ts else None


def first_video_sec(records: list[dict]) -> Optional[float]:
    ts = [r.get("video_sec") for r in records
          if r.get("video_sec") is not None]
    return min(ts) if ts else None


def pack_positions(dets: list[dict]) -> list[list[dict]]:
    """For each frame in the window, sort detections by x_center and
    return per-position lists (position 0 = rightmost = lead in default
    orientation; flip externally if track is inverted).

    Returns: [ [det_pos0_frame0, det_pos0_frame1, ...],
               [det_pos1_frame0, det_pos1_frame1, ...], ... ]
    """
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for d in dets:
        by_frame[d["frame_seq"]].append(d)
    positions: dict[int, list[dict]] = defaultdict(list)
    for frame_seq, frame_dets in sorted(by_frame.items()):
        # Rightmost = leader. If track is inverted, caller flips.
        ordered = sorted(frame_dets, key=lambda d: -d["x_center"])
        for pos, d in enumerate(ordered):
            positions[pos].append(d)
    max_pos = max(positions.keys(), default=-1) + 1
    return [positions[p] for p in range(max_pos)]


def representative_det(pos_dets: list[dict]) -> Optional[dict]:
    """Pick the best representative detection for a pack position — highest
    conf, mid-window in time."""
    if not pos_dets:
        return None
    # Prefer mid-window, high conf
    mid_idx = len(pos_dets) // 2
    mid = pos_dets[mid_idx]
    best = max(pos_dets,
               key=lambda d: d["conf"] * (1 - abs(d["video_sec"] - mid["video_sec"]) / 2))
    return best


def color_agreement(a: dict, b: dict,
                    min_conf: float = 0.85) -> str:
    """Returns 'agree', 'disagree', or 'unknown' (at least one low-conf)."""
    ca, cb = a.get("color"), b.get("color")
    cona = a.get("conf", 0) / 100.0 if (a.get("conf", 0) > 1) else a.get("conf", 0)
    conb = b.get("conf", 0) / 100.0 if (b.get("conf", 0) > 1) else b.get("conf", 0)
    if cona < min_conf or conb < min_conf:
        return "unknown"
    if ca and cb:
        return "agree" if ca == cb else "disagree"
    return "unknown"


def build_pairs_for_handoff(records_by_cam: dict[str, list[dict]],
                            cam_a: str, cam_b: str,
                            rec_id: str) -> list[dict]:
    """Emit candidate pairs for one camera→camera handoff."""
    rec_a = records_by_cam.get(cam_a, [])
    rec_b = records_by_cam.get(cam_b, [])
    if not rec_a or not rec_b:
        return []

    t_exit_a = last_video_sec(rec_a)
    t_entry_b = first_video_sec(rec_b)
    if t_exit_a is None or t_entry_b is None:
        return []

    gap = t_entry_b - t_exit_a
    if not (MIN_HANDOFF_S <= gap <= MAX_HANDOFF_S):
        # Likely the recordings overlap differently or pack didn't actually go
        # from A to B directly. Still attempt matching but flag as low conf.
        log.debug("cam %s→%s gap=%.1fs outside expected range", cam_a, cam_b, gap)

    dets_a = detections_in_window(rec_a,
                                  t_exit_a - EXIT_WINDOW_S, t_exit_a)
    dets_b = detections_in_window(rec_b,
                                  t_entry_b, t_entry_b + ENTRY_WINDOW_S)
    if len(dets_a) < MIN_DETECTIONS_PER_WINDOW or \
       len(dets_b) < MIN_DETECTIONS_PER_WINDOW:
        return []

    pos_a = pack_positions(dets_a)
    pos_b = pack_positions(dets_b)
    n = min(len(pos_a), len(pos_b))
    if n == 0:
        return []

    pairs: list[dict] = []
    for k in range(n):
        a_rep = representative_det(pos_a[k])
        b_rep = representative_det(pos_b[k])
        if a_rep is None or b_rep is None:
            continue

        base_conf = 0.60  # Phase 1 trust level (time-order only)
        agree = color_agreement(a_rep, b_rep)
        if agree == "agree":
            auto_conf = 0.95
            source = "auto_phase2_color_agree"
        elif agree == "disagree":
            auto_conf = 0.25   # likely different jockeys swapped
            source = "auto_phase2_color_disagree"
        else:
            auto_conf = base_conf
            source = "auto_phase1"

        pairs.append({
            "a": {
                "cam": a_rep["cam_id"],
                "seq": a_rep["frame_seq"],
                "det_id": a_rep["det_index_in_frame"],
                "bbox": a_rep["bbox"],
                "color": a_rep.get("color"),
                "conf": a_rep.get("conf", 0),
            },
            "b": {
                "cam": b_rep["cam_id"],
                "seq": b_rep["frame_seq"],
                "det_id": b_rep["det_index_in_frame"],
                "bbox": b_rep["bbox"],
                "color": b_rep.get("color"),
                "conf": b_rep.get("conf", 0),
            },
            "auto_conf": auto_conf,
            "source": source,
            "rec_id": rec_id,
            "meta": {
                "cam_a": cam_a, "cam_b": cam_b,
                "pack_position": k,
                "gap_s": gap,
                "agree": agree,
            },
        })
    return pairs


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--rec-id", required=True)
    p.add_argument("--jsonl-dir", required=True, type=Path)
    p.add_argument("--cam-order", required=True,
                   help="Comma-separated camera order along the track, "
                        "e.g. cam-13,cam-08,cam-14,...")
    p.add_argument("--output", type=Path,
                   help="Write pairs to this JSON (default: print to stdout)")
    p.add_argument("--api", type=str,
                   help="If set, POST pairs to this API base URL "
                        "(e.g. http://localhost:8000)")
    args = p.parse_args()

    cam_order = [c.strip() for c in args.cam_order.split(",") if c.strip()]
    if len(cam_order) < 2:
        raise SystemExit("Need at least 2 cameras in --cam-order")

    # Load all JSONL files once.
    records_by_cam: dict[str, list[dict]] = {}
    for cam in cam_order:
        candidates = [
            args.jsonl_dir / f"{cam}.jsonl",
            args.jsonl_dir / f"frames_{cam}.jsonl",
            args.jsonl_dir / f"{cam.replace('-', '_')}.jsonl",
        ]
        loaded = 0
        for path in candidates:
            if path.exists():
                records_by_cam[cam] = load_jsonl(path)
                loaded = len(records_by_cam[cam])
                break
        log.info("%s: %d records", cam, loaded)

    all_pairs: list[dict] = []
    for a, b in zip(cam_order[:-1], cam_order[1:]):
        pairs = build_pairs_for_handoff(records_by_cam, a, b, args.rec_id)
        all_pairs.extend(pairs)
        log.info("Handoff %s→%s: %d pairs", a, b, len(pairs))

    # Summarise by source
    by_source: dict[str, int] = defaultdict(int)
    for p_ in all_pairs:
        by_source[p_["source"]] += 1
    log.info("Totals: %d pairs — %s",
             len(all_pairs), dict(by_source))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "rec_id": args.rec_id,
            "pairs": all_pairs,
        }, indent=2))
        log.info("Wrote %s", args.output)

    if args.api:
        import urllib.request
        body = json.dumps({"rec_id": args.rec_id,
                           "pairs": all_pairs}).encode()
        req = urllib.request.Request(
            args.api.rstrip("/") + "/analytics/pairs/bulk",
            data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            log.info("API response: %s", resp.read().decode())


if __name__ == "__main__":
    main()
