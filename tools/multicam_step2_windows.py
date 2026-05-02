#!/usr/bin/env python3
"""multicam_step2_windows.py — ШАГ 2: сохранить 14 фиксированных
sampling windows + сгенерить 3 candidate preview frames для
session_174240_cam-24 (где user выбирает сам).

Лукапит реальный video file через inventory.json по (session, cam_num).
Не подменяет камеру — если файл нет, поднимает FATAL.
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

WORK = Path("/home/ipodrom/race_vision_bench/dinov2_prod/multicam_collection")
INV  = WORK / "inventory.json"
PREVIEWS_DIR = WORK / "candidate_previews_174240_cam24"
PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = WORK / "sampling_windows.json"

# Все 14 фиксированных окон от user (cam_id из spec → cam_num zero-padded)
FIXED_WINDOWS = [
    ("yaris_20260421_135242", "24", "0:55", "1:49"),
    ("yaris_20260421_140345", "24", "0:58", "1:10"),
    ("yaris_20260421_140345", "01", "2:00", "2:16"),
    ("yaris_20260421_175124", "24", "0:00", "0:36"),
    ("yaris_20260421_175124", "01", "1:20", "1:39"),
    ("yaris_20260421_175124", "02", "2:08", "2:30"),
    ("yaris_20260421_175124", "03", "2:58", "3:20"),
    ("yaris_20260421_175124", "04", "3:48", "4:10"),
    ("yaris_20260421_180010", "24", "1:17", "1:38"),
    ("yaris_20260421_180010", "01", "2:17", "2:35"),
    ("yaris_20260421_180010", "02", "3:00", "3:20"),
    ("yaris_20260421_180010", "03", "3:42", "4:00"),
    ("yaris_20260421_180010", "04", "4:17", "4:27"),
    ("yaris_20260421_180010", "05", "4:40", "4:50"),
]

# Candidates для 174240_cam-24 (длительность ~234s)
CANDIDATES_174240 = [
    ("A",  60,  90),   # ранняя зона
    ("B", 120, 150),   # середина
    ("C", 180, 210),   # поздняя зона
]


def mmss_to_sec(s: str) -> int:
    parts = s.split(":")
    return int(parts[0])*60 + int(parts[1])


def main():
    inv = json.loads(INV.read_text())
    by_key = {(e["session_short"], e["camera_num"]): e
              for e in inv if not e["is_recovery"]}

    windows = []
    print("=== fixed windows (14) ===")
    print(f"  {'session':<26s} {'cam':<6s} {'start':>6s} {'end':>6s} "
          f"{'len':>4s}  {'res':<10s} {'file':<50s}")
    print("  " + "-"*112)
    for sess, cam, ms_s, ms_e in FIXED_WINDOWS:
        sess_short = sess.split("_")[-1]
        ent = by_key.get((sess_short, cam))
        if not ent:
            print(f"  FATAL: no file for ({sess_short}, cam-{cam})", file=sys.stderr)
            sys.exit(1)
        s_sec = mmss_to_sec(ms_s)
        e_sec = mmss_to_sec(ms_e)
        if e_sec > ent["duration_sec"]:
            print(f"  FATAL: window {ms_s}-{ms_e} exceeds video "
                  f"duration {ent['duration_sec']:.1f}s for {ent['video_name']}",
                  file=sys.stderr)
            sys.exit(1)
        windows.append({
            "session":      sess,
            "session_short": sess_short,
            "camera":       f"cam-{cam}",
            "camera_num":   cam,
            "video_path":   ent["video_path"],
            "video_name":   ent["video_name"],
            "resolution_short": ent["resolution_short"],
            "resolution":   ent["resolution"],
            "fps":          ent["fps"],
            "windows": [{
                "start_sec":  s_sec,
                "end_sec":    e_sec,
                "len_sec":    e_sec - s_sec,
                "label":      f"{ms_s}-{ms_e}",
            }],
        })
        print(f"  {sess:<26s} cam-{cam:<3s} {s_sec:>5d}s {e_sec:>5d}s "
              f"{e_sec-s_sec:>3d}s  {ent['resolution_short']:<10s} {ent['video_name']}")
    print()

    # Candidates for 174240_cam-24
    cam24_174 = by_key[("174240", "24")]
    print("=== generating 3 candidate preview frames for 174240_cam-24 ===")
    print(f"  source: {cam24_174['video_name']}  (duration {cam24_174['duration_sec']:.1f}s)")
    print()
    candidate_meta = []
    for tag, s, e in CANDIDATES_174240:
        # extract 3 frames per candidate: start, middle, end
        for ts_label, ts in (("start", s), ("mid", (s+e)//2), ("end", e)):
            out = PREVIEWS_DIR / f"cand_{tag}_{ts_label}_t{ts:03d}s.jpg"
            cmd = ["ffmpeg", "-y", "-loglevel", "error",
                   "-ss", str(ts), "-i", cam24_174["video_path"],
                   "-frames:v", "1", "-q:v", "2", str(out)]
            subprocess.run(cmd, check=True)
        candidate_meta.append({
            "tag": tag, "start_sec": s, "end_sec": e,
            "len_sec": e - s,
            "preview_files": [f"cand_{tag}_start_t{s:03d}s.jpg",
                              f"cand_{tag}_mid_t{(s+e)//2:03d}s.jpg",
                              f"cand_{tag}_end_t{e:03d}s.jpg"],
        })
        print(f"  Candidate {tag}: {s}s-{e}s  (3 preview frames saved)")

    # Mini-gallery for visual pick
    parts = [
        '<!doctype html><html><head><meta charset="utf-8">',
        '<title>174240_cam-24 — pick 30s window</title>',
        '<style>',
        'body{font-family:system-ui;background:#1a1a1a;color:#ddd;margin:0;padding:14px;}',
        'h2{color:#fff;font-size:15px;margin:18px 0 6px 0;padding:6px 10px;background:#333;border-radius:4px;}',
        '.row{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:18px;}',
        '.cell{background:#2a2a2a;border-radius:4px;padding:6px;}',
        '.cell img{max-width:100%;display:block;margin:0 auto;}',
        '.meta{font-size:11px;color:#88ddff;padding:6px 0 0 0;text-align:center;}',
        '</style></head><body>',
        f'<h1 style="color:#fff">174240 cam-24 — выбери одно 30-сек окно (A/B/C)</h1>',
        f'<p style="color:#aaa;font-size:12px">Source: {cam24_174["video_name"]} '
        f'· duration {cam24_174["duration_sec"]:.0f}s · 3 кадра на окно (start/mid/end)</p>',
    ]
    for c in candidate_meta:
        parts.append(f'<h2>Candidate {c["tag"]} — {c["start_sec"]}s..{c["end_sec"]}s</h2>')
        parts.append('<div class="row">')
        for ts_label, fname in zip(["start","mid","end"], c["preview_files"]):
            ts_match = fname.split("_t")[1].replace("s.jpg","")
            parts.append(
                f'<div class="cell">'
                f'<img src="{fname}" loading="lazy">'
                f'<div class="meta">{ts_label} · t={ts_match}s</div>'
                f'</div>'
            )
        parts.append('</div>')
    parts.append('</body></html>')
    (PREVIEWS_DIR / "candidates.html").write_text("\n".join(parts))

    # Save without 174240 (waiting for user pick)
    out_doc = {
        "fixed_windows": windows,
        "pending_pick": {
            "session": "yaris_20260421_174240",
            "camera": "cam-24",
            "candidates": candidate_meta,
            "note": "User picks tag (A/B/C); 15-th window appended after.",
        },
    }
    OUT_JSON.write_text(json.dumps(out_doc, indent=2))

    print()
    print(f"  saved windows: {OUT_JSON}")
    print(f"  candidate gallery: {PREVIEWS_DIR / 'candidates.html'}")


if __name__ == "__main__":
    main()
