#!/usr/bin/env python3
"""multicam_step3b_check_1520p.py — quick activity check для 1520p
кандидатов: extract 3 preview frames + YOLO person count, чтобы
понять есть ли жокеи в указанном окне.
"""
from __future__ import annotations
import json, subprocess
from pathlib import Path
import cv2

WORK = Path("/home/ipodrom/race_vision_bench/dinov2_prod/multicam_collection")
INV  = json.loads((WORK / "inventory.json").read_text())
PREV_DIR = WORK / "candidate_previews_1520p"
PREV_DIR.mkdir(parents=True, exist_ok=True)

# (session_short, cam_num, start_sec, end_sec)
CANDIDATES = [
    ("135242", "15", 30, 60),  # primary user pick
    ("135242", "02", 30, 60),
    ("135242", "04", 30, 60),
    ("135242", "07", 30, 60),
    ("135242", "22", 30, 60),
    ("140345", "15", 30, 60),  # longer duration, potentially more activity
]

REPO = Path(__file__).resolve().parent.parent
YOLO_ONNX = REPO / "models" / "yolo11s_person_960.onnx"


def main():
    by_key = {(e["session_short"], e["camera_num"]): e
              for e in INV if not e["is_recovery"]}

    from ultralytics import YOLO
    yolo = YOLO(str(YOLO_ONNX), task="detect")

    print(f"  {'sess':<8s} {'cam':<6s} {'dur':>7s} {'30s_persons':>12s} "
          f"{'45s_persons':>12s} {'60s_persons':>12s} {'AVG':>5s}")
    print("  " + "-" * 70)
    rows = []
    for sess, cam, s, e in CANDIDATES:
        ent = by_key.get((sess, cam))
        if not ent:
            print(f"  {sess:<8s} cam-{cam}: missing"); continue
        if ent["duration_sec"] < e:
            print(f"  {sess:<8s} cam-{cam}: too short ({ent['duration_sec']:.1f}s < {e}s)")
            continue
        path = ent["video_path"]
        timestamps = [s, (s+e)//2, e-1]
        counts = []
        for ts in timestamps:
            out = PREV_DIR / f"{sess}_kamera{cam}_t{ts:03d}s.jpg"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-ss", str(ts), "-i", path,
                 "-frames:v", "1", "-q:v", "2", str(out)],
                check=True
            )
            img = cv2.imread(str(out))
            r = yolo.predict(source=img, imgsz=960, conf=0.25, iou=0.5,
                             classes=[0], device=0, half=False, verbose=False)
            n = 0
            if r and r[0].boxes is not None:
                # Count detections with bbox_h >= 25 (real jockey size filter)
                for box in r[0].boxes.xyxy.cpu().numpy():
                    if (box[3]-box[1]) >= 25: n += 1
            counts.append(n)
        avg = sum(counts) / len(counts)
        rows.append((sess, cam, ent["duration_sec"], counts, avg))
        print(f"  {sess:<8s} cam-{cam:<3s} {ent['duration_sec']:>6.1f}s "
              f"{counts[0]:>12d} {counts[1]:>12d} {counts[2]:>12d} {avg:>5.1f}")

    # HTML mini-gallery so user can visual-confirm
    parts = ['<!doctype html><html><head><meta charset="utf-8">',
             '<title>1520p candidates — preview</title>',
             '<style>body{font-family:system-ui;background:#1a1a1a;color:#ddd;padding:14px;}',
             'h2{color:#fff;font-size:14px;margin:14px 0 6px 0;padding:6px;background:#333;}',
             '.row{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;}',
             '.cell{background:#2a2a2a;padding:6px;border-radius:4px;}',
             '.cell img{max-width:100%;display:block;}',
             '.meta{font-size:11px;color:#88ddff;text-align:center;padding:4px;}',
             '</style></head><body>',
             '<h1 style="color:#fff">1520p candidates — preview frames + YOLO person count</h1>']
    for sess, cam, dur, counts, avg in rows:
        parts.append(f'<h2>{sess} cam-{cam} (1520p) — duration={dur:.0f}s · '
                     f'persons {counts} avg={avg:.1f}</h2>')
        parts.append('<div class="row">')
        for ts, n in zip([30, 45, 59], counts):
            parts.append(f'<div class="cell"><img src="{sess}_kamera{cam}_t{ts:03d}s.jpg">'
                         f'<div class="meta">t={ts}s · {n} persons detected</div></div>')
        parts.append('</div>')
    parts.append('</body></html>')
    (PREV_DIR / "candidates.html").write_text("\n".join(parts))
    print()
    print(f"  preview gallery: {PREV_DIR / 'candidates.html'}")


if __name__ == "__main__":
    main()
