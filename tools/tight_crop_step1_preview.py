#!/usr/bin/env python3
"""tight_crop_step1_preview.py — генерация preview HTML с side-by-side
сравнением full bbox vs tight crop на 5 random multicam crops.

Cели по spec'у:
- 5 random crops с РАЗНЫХ камер (для разнообразия)
- Side-by-side: original | tight | composite (с overlay rectangles)
- Save в v2_tight_build/preview_tight/
"""
from __future__ import annotations
import json, sys
from html import escape
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from tight_crop_utils import (
    tight_crop_xyxy, TIGHT_X_LO, TIGHT_X_HI, TIGHT_Y_LO, TIGHT_Y_HI
)

WORK = Path("/home/ipodrom/race_vision_bench/dinov2_prod/multicam_collection")
OUT  = Path("/home/ipodrom/race_vision_bench/dinov2_prod/v2_tight_build/preview_tight")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    inv = json.loads((WORK / "crops_index.json").read_text())
    labels = json.loads((WORK / "labels.json").read_text())
    label_map = {cid: cls for cls, ids in labels.items() for cid in ids}

    # Filter: multicam only (origin != color_dataset, has frame_path + bbox)
    mc = [e for e in inv
          if e.get("origin") != "color_dataset"
          and e.get("frame_path") and e.get("bbox")
          and e["id"] in label_map]
    print(f"  multicam labeled crops: {len(mc)}")

    # Pick 5 from different cameras (deterministic seed)
    rng = np.random.default_rng(42)
    by_cam = {}
    for e in mc:
        by_cam.setdefault(e["camera"], []).append(e)
    cams = sorted(by_cam.keys())
    rng.shuffle(cams)
    chosen = []
    for cam in cams:
        if len(chosen) >= 5: break
        e = rng.choice(by_cam[cam])
        chosen.append(e)
    print(f"  selected {len(chosen)} crops from cams: {[e['camera'] for e in chosen]}")

    rows = []
    for e in chosen:
        cid = e["id"]
        true_label = label_map[cid]
        frame = cv2.imread(e["frame_path"])
        if frame is None:
            print(f"  WARN: cannot read {e['frame_path']}"); continue
        x1,y1,x2,y2 = e["bbox"]
        w_bbox = x2-x1; h_bbox = y2-y1

        # Full bbox crop
        full_crop = frame[max(0,y1):y2, max(0,x1):x2]

        # Tight crop
        tight_crop = tight_crop_xyxy(frame, [x1,y1,x2,y2])

        # Composite: full crop with tight rectangle drawn
        comp = full_crop.copy()
        tx_lo = int(w_bbox * TIGHT_X_LO); tx_hi = int(w_bbox * TIGHT_X_HI)
        ty_lo = int(h_bbox * TIGHT_Y_LO); ty_hi = int(h_bbox * TIGHT_Y_HI)
        cv2.rectangle(comp, (tx_lo, ty_lo), (tx_hi, ty_hi), (0, 255, 0), 2)

        # Save 3 images
        cv2.imwrite(str(OUT / f"{cid}_1full.jpg"),  full_crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(OUT / f"{cid}_2tight.jpg"), tight_crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(OUT / f"{cid}_3comp.jpg"),  comp,      [cv2.IMWRITE_JPEG_QUALITY, 92])

        rows.append({
            "id":         cid,
            "camera":     e["camera"],
            "session":    e["session_short"],
            "resolution": e["resolution_short"],
            "label":      true_label,
            "bbox":       [x1,y1,x2,y2],
            "full_hxw":   f"{full_crop.shape[0]}×{full_crop.shape[1]}",
            "tight_hxw":  f"{tight_crop.shape[0]}×{tight_crop.shape[1]}",
        })

    # HTML
    parts = [
        '<!doctype html><html><head><meta charset="utf-8">',
        '<title>Tight crop preview — 5 examples</title>',
        '<style>',
        'body{font-family:system-ui;background:#1a1a1a;color:#ddd;padding:14px;}',
        'h1{color:#fff;margin:0 0 14px 0;font-size:18px;}',
        'h2{color:#88ddff;font-size:14px;margin:18px 0 6px 0;padding:6px 10px;background:#262626;border-radius:3px;}',
        '.row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:18px;align-items:end;}',
        '.cell{background:#2a2a2a;padding:10px;border-radius:4px;text-align:center;}',
        '.cell img{display:block;margin:0 auto 8px auto;max-width:100%;max-height:380px;background:#000;}',
        '.cell .lbl{font-size:11px;color:#88ddff;font-weight:500;}',
        '.cell .meta{font-size:10px;color:#aaa;margin-top:4px;}',
        '.note{background:#333;padding:10px;border-radius:4px;font-size:13px;color:#ddd;margin:10px 0 18px;}',
        '</style></head><body>',
        '<h1>Tight crop preview (Y 10%-50% top, X 20%-80% center)</h1>',
        f'<div class="note">5 random multicam crops с разных камер. Слева = full bbox '
        f'(что сейчас идёт в SGIE). Центр = tight crop (что предлагается). '
        f'Справа = full с зелёным rectangle (где именно tight cuts). '
        f'Цель: силк остался виден, ноги/лошадь/соседи отрезаны.</div>',
    ]
    for r in rows:
        parts.append(f'<h2>{escape(r["id"])} — {r["camera"]} · {r["resolution"]} · '
                     f'session {r["session"]} · label: <b>{r["label"]}</b></h2>')
        parts.append('<div class="row">')
        parts.append(f'<div class="cell"><img src="{escape(r["id"])}_1full.jpg">'
                     f'<div class="lbl">FULL bbox (current)</div>'
                     f'<div class="meta">{r["full_hxw"]} px</div></div>')
        parts.append(f'<div class="cell"><img src="{escape(r["id"])}_2tight.jpg">'
                     f'<div class="lbl">TIGHT crop (proposed)</div>'
                     f'<div class="meta">{r["tight_hxw"]} px</div></div>')
        parts.append(f'<div class="cell"><img src="{escape(r["id"])}_3comp.jpg">'
                     f'<div class="lbl">FULL + tight rectangle</div>'
                     f'<div class="meta">green = tight zone</div></div>')
        parts.append('</div>')
    parts.append('</body></html>')

    html_path = OUT / "preview.html"
    html_path.write_text("\n".join(parts))
    print()
    print(f"  saved preview: {html_path}")
    print(f"  files: {len(rows) * 3} jpg + 1 html")


if __name__ == "__main__":
    main()
