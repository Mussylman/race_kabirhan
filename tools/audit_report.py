#!/usr/bin/env python3
"""Post-process Race Vision audit output.

Reads:
  output/audit/detections.jsonl
  output/audit/tracker_events.jsonl

Produces:
  output/audit/detections.json          (same records, merged into array)
  output/audit/tracker_events.json
  output/audit/stats_per_camera.json
  output/audit/timeline_per_color.json
  output/audit/suspicious.json
  output/audit/summary.txt              (human-readable)

Usage:
  python tools/audit_report.py                # defaults to output/audit/
  python tools/audit_report.py /path/to/audit
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def per_camera_stats(detections: list[dict]) -> dict:
    stats: dict = defaultdict(lambda: {
        "total": 0,
        "bbox_filter_rejected": 0,
        "roi_rejected": 0,
        "passed": 0,
        "written_to_shm": 0,
        "by_color": defaultdict(int),
        "first_seen_ts": None,
        "last_seen_ts": None,
        "unique_track_ids": set(),
    })
    for d in detections:
        cam = d["cam"]
        s = stats[cam]
        s["total"] += 1
        if d.get("reject_reason") == "bbox_filter":
            s["bbox_filter_rejected"] += 1
        elif d.get("reject_reason") == "roi_outside":
            s["roi_rejected"] += 1
        else:
            s["passed"] += 1
            if d.get("written_to_shm"):
                s["written_to_shm"] += 1
        if d.get("color"):
            s["by_color"][d["color"]] += 1
        ts = d["ts"]
        if s["first_seen_ts"] is None or ts < s["first_seen_ts"]:
            s["first_seen_ts"] = ts
        if s["last_seen_ts"] is None or ts > s["last_seen_ts"]:
            s["last_seen_ts"] = ts
        if d.get("track_id"):
            s["unique_track_ids"].add(d["track_id"])

    # Finalize: set → list for JSON
    out = {}
    for cam, s in stats.items():
        out[cam] = {
            **{k: v for k, v in s.items() if k not in ("by_color", "unique_track_ids")},
            "by_color": dict(s["by_color"]),
            "unique_track_ids": sorted(s["unique_track_ids"]),
            "n_unique_tracks": len(s["unique_track_ids"]),
        }
    return out


def timeline_per_color(tracker_events: list[dict]) -> dict:
    tl: dict = defaultdict(list)
    for e in tracker_events:
        color = e.get("color") or ""
        if not color:
            continue
        tl[color].append({
            "time": round(e["ts"], 3),
            "cam": e.get("cam"),
            "cam_idx": e.get("cam_idx"),
            "accepted": e.get("accepted"),
            "reason": e.get("reason"),
        })
    # Sort each color's list
    for c in tl:
        tl[c].sort(key=lambda x: x["time"])
    return dict(tl)


def detect_suspicious(detections: list[dict], tracker_events: list[dict]) -> list[dict]:
    """Flag likely false positives / pipeline issues."""
    sus = []

    # 1) Detection passed filters AND inside_roi reported TRUE, but bbox_y2
    #    is extremely low (≤0.2 of frame height) = at the top of the frame,
    #    likely stands / outside track.
    for d in detections:
        if not (d.get("inside_roi") and d.get("passed_filters")):
            continue
        h = d["frame_wh"][1]
        y2 = d["bbox"][3]
        if h > 0 and (y2 / h) < 0.25:
            sus.append({
                "kind": "bbox_high_in_frame",
                "cam": d["cam"], "ts": d["ts"],
                "bbox_y2_norm": round(y2 / h, 3),
                "hint": "likely stands — ROI polygon may be too generous",
            })

    # 2) Tracker events — suspicious forward jumps (>3 cams in <5s)
    #    Using cam_idx in the ranked list (cam_order).
    last_by_color: dict = {}
    for e in sorted(tracker_events, key=lambda x: x["ts"]):
        if not e.get("accepted"):
            continue
        color = e.get("color") or ""
        cam_idx = e.get("cam_idx")
        if cam_idx is None:
            continue
        prev = last_by_color.get(color)
        if prev is not None:
            dt = e["ts"] - prev["ts"]
            dj = cam_idx - prev["cam_idx"]
            if dj >= 3 and dt < 5.0:
                sus.append({
                    "kind": "fast_forward_jump",
                    "color": color, "cams": [prev["cam_idx"], cam_idx],
                    "dt_s": round(dt, 2), "d_cam_idx": dj,
                    "hint": "jockey teleported too fast — probable spurious classification",
                })
        last_by_color[color] = e

    return sus


def write_summary(audit_dir: Path, stats: dict, timeline: dict,
                  sus: list, n_dets: int, n_events: int):
    lines = []
    lines.append(f"Race Vision audit report — {audit_dir}")
    lines.append("=" * 60)
    lines.append(f"Total detections logged : {n_dets}")
    lines.append(f"Total tracker events    : {n_events}")
    lines.append(f"Suspicious flags        : {len(sus)}")
    lines.append("")
    lines.append("Per-camera summary:")
    lines.append(f"{'cam':<10}{'total':>8}{'filt-rej':>10}{'roi-rej':>10}{'pass':>8}{'shm':>8}  colors")
    for cam in sorted(stats.keys()):
        s = stats[cam]
        colors = ",".join(f"{k}:{v}" for k, v in sorted(s["by_color"].items()))
        lines.append(f"{cam:<10}{s['total']:>8}{s['bbox_filter_rejected']:>10}"
                     f"{s['roi_rejected']:>10}{s['passed']:>8}{s['written_to_shm']:>8}  {colors}")
    lines.append("")
    lines.append("Tracker events by color:")
    for color, evs in sorted(timeline.items()):
        accepted = sum(1 for e in evs if e["accepted"])
        lines.append(f"  {color:<8} {accepted}/{len(evs)} accepted  "
                     f"(passes: " + " → ".join(
                         e["cam"] for e in evs if e["accepted"]) + ")")
    if sus:
        lines.append("")
        lines.append("Suspicious events:")
        for s in sus[:30]:
            lines.append(f"  [{s['kind']}] " + json.dumps(
                {k: v for k, v in s.items() if k != 'kind'}, ensure_ascii=False))
        if len(sus) > 30:
            lines.append(f"  ... and {len(sus) - 30} more")
    (audit_dir / "summary.txt").write_text("\n".join(lines) + "\n")


def main(argv):
    audit_dir = Path(argv[1] if len(argv) > 1 else "output/audit")
    if not audit_dir.is_dir():
        print(f"audit dir not found: {audit_dir}", file=sys.stderr)
        return 1

    detections = load_jsonl(audit_dir / "detections.jsonl")
    events = load_jsonl(audit_dir / "tracker_events.jsonl")
    print(f"loaded {len(detections)} detections, {len(events)} tracker events")

    stats = per_camera_stats(detections)
    timeline = timeline_per_color(events)
    sus = detect_suspicious(detections, events)

    (audit_dir / "detections.json").write_text(
        json.dumps(detections, ensure_ascii=False, indent=2))
    (audit_dir / "tracker_events.json").write_text(
        json.dumps(events, ensure_ascii=False, indent=2))
    (audit_dir / "stats_per_camera.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2))
    (audit_dir / "timeline_per_color.json").write_text(
        json.dumps(timeline, ensure_ascii=False, indent=2))
    (audit_dir / "suspicious.json").write_text(
        json.dumps(sus, ensure_ascii=False, indent=2))

    write_summary(audit_dir, stats, timeline, sus, len(detections), len(events))
    print(f"wrote reports under {audit_dir}/")
    print(f"open {audit_dir}/summary.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
