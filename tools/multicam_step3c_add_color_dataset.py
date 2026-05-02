#!/usr/bin/env python3
"""multicam_step3c_add_color_dataset.py — добавить data/color_dataset/
(692 jpg, 4 цвета без purple) в multicam_collection как auto-labeled
поставку для prototype build.

Source structure:
  data/color_dataset/{blue,green,yellow,red,purple}/  (purple skip)
  filenames:
    exp10_cam2_f00096_d3_145x256.jpg                    → exp10/11 prefix
    kamera_13_163303_END163811_f00536_d2_HxW.jpg        → old session prefix

Per user: все эти crops имеют 720p source quality.

Действия:
  1. Walk 4 цвета (skip purple)
  2. Parse filename → source, frame_id, det_id, W×H
  3. ID: colordata_{source_norm}_f{frame:05d}_d{det}
  4. Copy file → multicam_collection/crops_raw/{ID}.jpg
  5. Append entry в crops_index.json (metadata: source=color_dataset)
  6. AUTO-add ID в labels.json под соответствующим цветом
  7. Print breakdown по (color × source_family)

Idempotent: если ID уже в crops_index — skip copy + не дублировать.
"""
from __future__ import annotations
import json, re, shutil
from collections import defaultdict
from pathlib import Path

REPO   = Path(__file__).resolve().parent.parent
SRC    = REPO / "data" / "color_dataset"
WORK   = Path("/home/ipodrom/race_vision_bench/dinov2_prod/multicam_collection")
DST    = WORK / "crops_raw"
INDEX  = WORK / "crops_index.json"
LABELS = WORK / "labels.json"

COLORS = ["blue", "green", "yellow", "red"]   # purple skip

# exp10_cam2_f00096_d3_145x256
RX_EXP    = re.compile(r"^(exp\d+)_cam(\d+)_f(\d+)_d(\d+)_(\d+)x(\d+)$")
# kamera_13_163303_END163811_f00536_d2_52x114
# kamera_05_162030_f00xxx_d2_HxW  (no _END suffix sometimes)
RX_KAMERA = re.compile(r"^kamera_(\d+)_(\d+)(?:_END\d+)?_f(\d+)_d(\d+)_(\d+)x(\d+)$")


def parse_filename(stem: str):
    m = RX_EXP.match(stem)
    if m:
        exp, cam_num, frame, det, w, h = m.groups()
        return {
            "source_family": exp,                              # 'exp10' / 'exp11'
            "source_id":     f"{exp}_cam{cam_num.zfill(2)}",
            "session":       exp,
            "camera":        f"cam-{cam_num.zfill(2)}",
            "camera_num":    cam_num.zfill(2),
            "frame_idx":     int(frame),
            "det_idx":       int(det),
            "bbox_w":        int(w),
            "bbox_h":        int(h),
        }
    m = RX_KAMERA.match(stem)
    if m:
        cam_num, time_id, frame, det, w, h = m.groups()
        return {
            "source_family": "old_session",
            "source_id":     f"kamera{cam_num.zfill(2)}_{time_id}",
            "session":       f"old_{time_id}",
            "camera":        f"cam-{cam_num.zfill(2)}",
            "camera_num":    cam_num.zfill(2),
            "frame_idx":     int(frame),
            "det_idx":       int(det),
            "bbox_w":        int(w),
            "bbox_h":        int(h),
        }
    return None


def make_id(parsed: dict) -> str:
    # Нормализованный ID для нового пула
    src = parsed["source_id"].replace("-", "")
    return f"colordata_{src}_f{parsed['frame_idx']:05d}_d{parsed['det_idx']}"


def main():
    if not INDEX.is_file():
        print(f"FATAL: {INDEX} missing. Run multicam_step3_extract.py first.")
        return

    crops_index = json.loads(INDEX.read_text())
    existing_ids = {e["id"] for e in crops_index}

    labels = json.loads(LABELS.read_text()) if LABELS.is_file() else {c: [] for c in COLORS}
    for c in COLORS:
        labels.setdefault(c, [])

    n_added = 0; n_skipped = 0; n_unparseable = 0
    breakdown = defaultdict(lambda: defaultdict(int))   # color → source_family → count
    new_entries = []
    label_added = defaultdict(int)

    for color in COLORS:
        cdir = SRC / color
        if not cdir.is_dir():
            print(f"  WARN: {cdir} missing"); continue
        files = sorted(cdir.glob("*.jpg"))
        for f in files:
            p = parse_filename(f.stem)
            if not p:
                n_unparseable += 1
                print(f"  WARN: cannot parse {f.name}")
                continue
            new_id = make_id(p)
            if new_id in existing_ids:
                n_skipped += 1
                continue
            # Copy file
            dst_path = DST / f"{new_id}.jpg"
            shutil.copyfile(f, dst_path)
            # Append entry
            entry = {
                "id":               new_id,
                "session":          p["session"],
                "session_short":    p["source_family"],
                "camera":           p["camera"],
                "camera_num":       p["camera_num"],
                "resolution":       "1280x720",       # user statement
                "resolution_short": "720p",
                "timestamp_ms":     None,             # not derivable from these
                "video_path":       None,
                "frame_path":       None,
                "crop_path":        str(dst_path),
                "bbox":             None,
                "bbox_with_margin": None,
                "det_conf":         None,
                "h":                p["bbox_h"],
                "w":                p["bbox_w"],
                "origin":           "color_dataset",
                "origin_source_id": p["source_id"],
                "origin_filename":  f.name,
            }
            new_entries.append(entry)
            existing_ids.add(new_id)
            # Auto-label
            if new_id not in labels[color]:
                labels[color].append(new_id)
                label_added[color] += 1
            n_added += 1
            breakdown[color][p["source_family"]] += 1

    crops_index.extend(new_entries)
    INDEX.write_text(json.dumps(crops_index, indent=2))
    LABELS.write_text(json.dumps(labels, indent=2))

    print()
    print("=" * 70)
    print(f"=== color_dataset integration done ===")
    print("=" * 70)
    print(f"  added:        {n_added} crops (copied + indexed + auto-labeled)")
    print(f"  skipped:      {n_skipped} (already in index)")
    print(f"  unparseable:  {n_unparseable}")
    print()
    print(f"  breakdown by (color × source_family):")
    print(f"    {'color':<8s} {'family':<14s} {'count':>6s}")
    print("    " + "-" * 32)
    for color in COLORS:
        fams = breakdown[color]
        if not fams:
            print(f"    {color:<8s} (none)")
            continue
        for fam, n in sorted(fams.items()):
            print(f"    {color:<8s} {fam:<14s} {n:>6d}")
    print()
    print(f"  new totals in labels.json:")
    for color in COLORS:
        print(f"    {color:<8s}: {len(labels[color])}  (+{label_added[color]} from color_dataset)")
    print()
    print(f"  NB: purple folder skipped (4-class only)")


if __name__ == "__main__":
    main()
