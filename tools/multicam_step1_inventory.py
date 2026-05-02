#!/usr/bin/env python3
"""multicam_step1_inventory.py — ШАГ 1 BIS: full inventory.

Сканирует ВСЕ kamera_*.mp4 в 5 сессиях (find), извлекает metadata
через ffprobe, сохраняет полный inventory.json.

Включает _r1_ recovery файлы (помечает в metadata).
"""
from __future__ import annotations
import json, os, re, subprocess, sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VIDEOS_ROOT = REPO / "data" / "videos" / "test_full_loop"
OUT_DIR = Path("/home/ipodrom/race_vision_bench/dinov2_prod/multicam_collection")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SESSIONS = [
    "yaris_20260421_135242",
    "yaris_20260421_140345",
    "yaris_20260421_174240",
    "yaris_20260421_175124",
    "yaris_20260421_180010",
]

CAM_RX = re.compile(r"^kamera_(\d+)(?:_(r\d+))?_")


def ffprobe(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration,size",
        "-of", "json", str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    j = json.loads(r.stdout)
    s = j["streams"][0]
    fmt = j.get("format", {})
    rfr = s.get("r_frame_rate", "0/1").split("/")
    fps = float(rfr[0]) / float(rfr[1]) if len(rfr) == 2 and float(rfr[1]) else 0.0
    nb = int(s.get("nb_frames", 0)) if s.get("nb_frames", "").isdigit() else None
    dur = float(s.get("duration") or fmt.get("duration") or 0)
    if nb is None and fps and dur:
        nb = int(round(dur * fps))
    return {
        "width":         int(s["width"]),
        "height":        int(s["height"]),
        "fps":           round(fps, 3),
        "n_frames":      nb or 0,
        "duration_sec":  round(dur, 2),
        "size_mb":       round(int(fmt.get("size", os.path.getsize(path))) / 1024 / 1024, 1),
    }


def main():
    inventory = []
    failures = []
    for session in SESSIONS:
        sess_dir = VIDEOS_ROOT / session
        if not sess_dir.is_dir():
            failures.append(f"missing dir: {session}")
            continue
        # find sorted by name (gives кам-01, кам-02, ...)
        files = sorted(sess_dir.glob("kamera_*.mp4"))
        for f in files:
            m = CAM_RX.match(f.name)
            if not m:
                failures.append(f"unparseable name: {f.name}")
                continue
            cam_num = m.group(1).zfill(2)
            recovery = m.group(2)  # 'r1' или None
            try:
                meta = ffprobe(f)
            except Exception as e:
                failures.append(f"{f.name}: ffprobe {e}")
                continue
            res = f"{meta['width']}x{meta['height']}"
            res_short = ("1080p" if meta["height"] == 1080 else
                         "720p"  if meta["height"] == 720  else
                         f"{meta['height']}p")
            inventory.append({
                "session":      session,
                "session_short": session.split("_")[-1],
                "camera":       f"cam-{cam_num}",
                "camera_num":   cam_num,
                "is_recovery":  recovery is not None,
                "recovery_tag": recovery,
                "video_path":   str(f),
                "video_name":   f.name,
                "resolution":   res,
                "resolution_short": res_short,
                **meta,
            })

    inv_path = OUT_DIR / "inventory.json"
    inv_path.write_text(json.dumps(inventory, indent=2))

    # Summary table per session
    print("=" * 92)
    print("MULTICAM FULL INVENTORY (5 sessions × ALL kamera files)")
    print("=" * 92)
    by_session = defaultdict(list)
    for e in inventory:
        by_session[e["session"]].append(e)

    print(f"  {'session_short':<14s} {'#vids':>6s} {'#cams':>6s} {'recovery':>9s} "
          f"{'resolution(s)':<14s} {'duration_total':>14s} {'size_total':>10s}")
    print("  " + "-" * 90)
    for sess in SESSIONS:
        ents = by_session.get(sess, [])
        if not ents:
            print(f"  {sess.split('_')[-1]:<14s}  (no files found)")
            continue
        n_vids = len(ents)
        cams = sorted({e["camera"] for e in ents})
        n_recovery = sum(1 for e in ents if e["is_recovery"])
        ress = sorted({e["resolution_short"] for e in ents})
        dur_total = sum(e["duration_sec"] for e in ents)
        size_total = sum(e["size_mb"] for e in ents)
        ress_str = ",".join(ress)
        print(f"  {sess.split('_')[-1]:<14s} {n_vids:>6d} {len(cams):>6d} "
              f"{n_recovery:>9d} {ress_str:<14s} "
              f"{dur_total:>11.0f}s   {size_total:>8.0f}M")

    print()
    print("=== camera coverage per session (camera presence matrix) ===")
    all_cams = sorted({e["camera"] for e in inventory})
    print(f"  {'session':<14s} {'cams covered':<5s}")
    for sess in SESSIONS:
        ents = by_session.get(sess, [])
        present = {e["camera"] for e in ents if not e["is_recovery"]}
        recovery = {e["camera"] for e in ents if e["is_recovery"]}
        line_cells = []
        for c in all_cams:
            num = c.split("-")[1]
            if c in present and c in recovery: line_cells.append(f"{num}+r")
            elif c in present:                 line_cells.append(num)
            elif c in recovery:                line_cells.append(f"({num}r)")
            else:                              line_cells.append(" ·")
        print(f"  {sess.split('_')[-1]:<14s} " + " ".join(line_cells))
    print()

    # Resolution breakdown
    print("=== resolution counts ===")
    res_count = defaultdict(int); res_size = defaultdict(float); res_dur = defaultdict(float)
    for e in inventory:
        res_count[e["resolution_short"]] += 1
        res_size[e["resolution_short"]] += e["size_mb"]
        res_dur[e["resolution_short"]] += e["duration_sec"]
    for r in sorted(res_count.keys()):
        print(f"  {r}: {res_count[r]} files  {res_dur[r]:.0f}s total  {res_size[r]:.0f}M total")

    print()
    print(f"=== TOTAL: {len(inventory)} videos across {len(SESSIONS)} sessions ===")
    print(f"  duration: {sum(e['duration_sec'] for e in inventory):.0f}s "
          f"({sum(e['duration_sec'] for e in inventory)/60:.1f} min total)")
    print(f"  size:     {sum(e['size_mb'] for e in inventory):.0f} MB "
          f"({sum(e['size_mb'] for e in inventory)/1024:.1f} GB total)")
    print(f"  saved:    {inv_path}")
    if failures:
        print()
        print("  WARNINGS:")
        for w in failures: print(f"    - {w}")


if __name__ == "__main__":
    main()
