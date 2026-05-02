#!/usr/bin/env python3
"""multicam_step3_extract.py — ШАГ 3: extraction по 15 sampling
windows + YOLO person detect + crops + gallery_all.html.

Naming format (per user spec):
  raw frames: {session_short}_kamera{NN}_{res}_t{ms:06d}.jpg
  crops:      {session_short}_kamera{NN}_{res}_t{ms:06d}_det{N}.jpg

Stride: 25 frames @ 25fps = 1 fps sampling.
YOLO: yolo11s_person_960.onnx, conf>=0.25, iou=0.5, person only.
Crop margin: 5% around bbox.
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from html import escape
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
WORK = Path("/home/ipodrom/race_vision_bench/dinov2_prod/multicam_collection")
WIN_JSON = WORK / "sampling_windows.json"
RAW_DIR  = WORK / "raw_frames"
CROPS_DIR = WORK / "crops_raw"
CROPS_INDEX_JSON = WORK / "crops_index.json"
GALLERY_HTML = WORK / "gallery_all.html"

YOLO_ONNX = REPO / "models" / "yolo11s_person_960.onnx"

SAMPLE_STRIDE = 25       # 1 fps на 25fps
DET_CONF      = 0.25
DET_IOU       = 0.5
DET_IMGSZ     = 960
CROP_MARGIN   = 0.05
MIN_BBOX_H    = 25


def crop_id(session_short, cam_num, res, ms, det_i):
    return f"{session_short}_kamera{cam_num}_{res}_t{ms:06d}_det{det_i}"


def frame_id(session_short, cam_num, res, ms):
    return f"{session_short}_kamera{cam_num}_{res}_t{ms:06d}"


def append_174240_pick(pick_tag: str):
    """Add the 15-th window (174240_cam-24, picked tag) to sampling_windows.json."""
    doc = json.loads(WIN_JSON.read_text())
    pending = doc["pending_pick"]
    cand = next(c for c in pending["candidates"] if c["tag"] == pick_tag)
    # Find inventory entry to fill resolution/path
    inv = json.loads((WORK / "inventory.json").read_text())
    by_key = {(e["session_short"], e["camera_num"]): e
              for e in inv if not e["is_recovery"]}
    ent = by_key[("174240", "24")]
    new_window = {
        "session":      "yaris_20260421_174240",
        "session_short": "174240",
        "camera":       "cam-24",
        "camera_num":   "24",
        "video_path":   ent["video_path"],
        "video_name":   ent["video_name"],
        "resolution_short": ent["resolution_short"],
        "resolution":   ent["resolution"],
        "fps":          ent["fps"],
        "windows": [{
            "start_sec":  cand["start_sec"],
            "end_sec":    cand["end_sec"],
            "len_sec":    cand["len_sec"],
            "label":      f"{cand['start_sec']}s-{cand['end_sec']}s (cand {pick_tag})",
        }],
    }
    if not any(w["session_short"] == "174240" and w["camera_num"] == "24"
               for w in doc["fixed_windows"]):
        doc["fixed_windows"].append(new_window)
    doc["pending_pick"]["resolved_to"] = pick_tag
    WIN_JSON.write_text(json.dumps(doc, indent=2))
    print(f"  added 15-th window: 174240 cam-24 (cand {pick_tag} = "
          f"{cand['start_sec']}s..{cand['end_sec']}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick-174240", choices=["A","B","C"], default="A",
                    help="Which 174240_cam-24 candidate to use")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Append 174240 pick to sampling_windows.json
    append_174240_pick(args.pick_174240)

    # 2. Load windows
    doc = json.loads(WIN_JSON.read_text())
    windows = doc["fixed_windows"]
    print(f"  total source windows: {len(windows)}")
    total_dur = sum(w["windows"][0]["len_sec"] for w in windows)
    print(f"  total sample-time: {total_dur}s (~{total_dur} frames @ 1fps)")
    print()

    # 3. Load YOLO once
    print(f"=== loading YOLO: {YOLO_ONNX.name} ===")
    from ultralytics import YOLO
    yolo = YOLO(str(YOLO_ONNX), task="detect")
    print()

    # 4. Per-window extraction
    crops_index = []
    n_frames_total = 0
    n_crops_total = 0
    breakdown = defaultdict(lambda: {"frames": 0, "crops": 0})  # (cam, res) → counts

    for w in windows:
        sess_short = w["session_short"]
        cam_num    = w["camera_num"]
        res        = w["resolution_short"]
        path       = w["video_path"]
        wnd        = w["windows"][0]
        start_sec  = wnd["start_sec"]
        end_sec    = wnd["end_sec"]
        fps        = w["fps"]
        start_frame = int(start_sec * fps)
        end_frame   = int(end_sec * fps)

        print(f"  {sess_short} kamera{cam_num} ({res}) {start_sec}s-{end_sec}s ...")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"    FATAL: cannot open {path}", file=sys.stderr); sys.exit(2)

        # Seek to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        cur_frame = start_frame
        n_frames_window = 0
        n_crops_window  = 0

        while cur_frame < end_frame:
            # Sample every SAMPLE_STRIDE frames
            ok = cap.grab()
            if not ok: break
            offset_in_window = cur_frame - start_frame
            if offset_in_window % SAMPLE_STRIDE == 0:
                ok2, frame = cap.retrieve()
                if not ok2 or frame is None:
                    cur_frame += 1; continue
                ms = int((cur_frame / fps) * 1000)
                fid = frame_id(sess_short, cam_num, res, ms)
                fp_raw = RAW_DIR / f"{fid}.jpg"
                cv2.imwrite(str(fp_raw), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                n_frames_window += 1

                # YOLO detect
                h_img, w_img = frame.shape[:2]
                results = yolo.predict(source=frame, imgsz=DET_IMGSZ,
                                       conf=DET_CONF, iou=DET_IOU,
                                       classes=[0], device=0, half=False, verbose=False)
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    for det_i, (box, det_conf) in enumerate(zip(boxes, confs)):
                        x1, y1, x2, y2 = box.tolist()
                        bbox_h = y2 - y1
                        if bbox_h < MIN_BBOX_H: continue
                        mw = (x2 - x1) * CROP_MARGIN
                        mh = (y2 - y1) * CROP_MARGIN
                        cx1 = max(0, int(x1 - mw))
                        cy1 = max(0, int(y1 - mh))
                        cx2 = min(w_img, int(x2 + mw))
                        cy2 = min(h_img, int(y2 + mh))
                        if cx2 <= cx1 or cy2 <= cy1: continue
                        crop = frame[cy1:cy2, cx1:cx2]
                        if crop.size == 0: continue
                        cid = crop_id(sess_short, cam_num, res, ms, det_i)
                        fp_crop = CROPS_DIR / f"{cid}.jpg"
                        cv2.imwrite(str(fp_crop), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                        crops_index.append({
                            "id":           cid,
                            "session":      w["session"],
                            "session_short": sess_short,
                            "camera":       w["camera"],
                            "camera_num":   cam_num,
                            "resolution":   w["resolution"],
                            "resolution_short": res,
                            "timestamp_ms": ms,
                            "video_path":   path,
                            "frame_path":   str(fp_raw),
                            "crop_path":    str(fp_crop),
                            "bbox":         [int(x1), int(y1), int(x2), int(y2)],
                            "bbox_with_margin": [cx1, cy1, cx2, cy2],
                            "det_conf":     float(det_conf),
                            "h":            int(bbox_h),
                            "w":            int(x2 - x1),
                        })
                        n_crops_window += 1
            cur_frame += 1
        cap.release()
        breakdown[(w["camera"], res)]["frames"] += n_frames_window
        breakdown[(w["camera"], res)]["crops"]  += n_crops_window
        n_frames_total += n_frames_window
        n_crops_total  += n_crops_window
        print(f"    -> {n_frames_window} frames, {n_crops_window} crops")

    # 5. Save crops_index.json
    CROPS_INDEX_JSON.write_text(json.dumps(crops_index, indent=2))

    # 6. Gallery — grouped by (camera + resolution)
    grouped = defaultdict(list)
    for r in crops_index:
        grouped[(r["camera"], r["resolution_short"])].append(r)

    LABELS = ["blue", "green", "yellow", "red", "skip"]
    parts = [
        '<!doctype html><html lang="ru"><head><meta charset="utf-8">',
        '<title>multicam_collection — label crops (4-class)</title>',
        '<style>',
        'body{font-family:system-ui;background:#1a1a1a;color:#ddd;margin:0;padding:12px;}',
        '.toolbar{position:sticky;top:0;background:#1a1a1a;padding:10px 0;z-index:50;'
        'border-bottom:1px solid #444;margin-bottom:10px;}',
        '.toolbar button{padding:6px 12px;margin-right:6px;cursor:pointer;background:#333;'
        'color:#fff;border:1px solid #555;border-radius:4px;}',
        '.toolbar button:hover{background:#444;}',
        '.stats{display:inline-block;margin-left:14px;color:#88ddff;font-size:12px;}',
        'h2{color:#fff;font-size:14px;margin:18px 0 6px 0;padding:6px 10px;background:#333;border-radius:4px;}',
        '.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:18px;}',
        '.card{background:#2a2a2a;border-radius:4px;padding:4px;border:2px solid transparent;}',
        '.card.labeled{border-color:#4a4;}',
        '.card img{max-width:100%;max-height:280px;display:block;margin:0 auto;}',
        '.meta{font-size:11px;color:#aaa;padding:4px;word-break:break-all;}',
        '.name{color:#88ddff;}',
        '.labels{display:flex;flex-wrap:wrap;gap:3px;padding:4px;}',
        '.labels label{font-size:11px;padding:3px 6px;background:#333;border-radius:3px;cursor:pointer;}',
        '.labels label:hover{background:#555;}',
        '.labels input{display:none;}',
        '.labels input:checked+span{color:#fff;}',
        '.labels input[value="blue"]:checked+span{background:#4af;padding:2px 5px;border-radius:3px;}',
        '.labels input[value="green"]:checked+span{background:#3a3;padding:2px 5px;border-radius:3px;}',
        '.labels input[value="yellow"]:checked+span{background:#aa3;padding:2px 5px;border-radius:3px;color:#000;}',
        '.labels input[value="red"]:checked+span{background:#a33;padding:2px 5px;border-radius:3px;}',
        '.labels input[value="skip"]:checked+span{background:#666;padding:2px 5px;border-radius:3px;}',
        'textarea#export{width:100%;min-height:240px;background:#000;color:#0f0;font-family:monospace;padding:8px;}',
        '.modal{display:none;position:fixed;top:5%;left:5%;right:5%;bottom:5%;'
        'background:#222;padding:16px;border:1px solid #555;z-index:100;overflow:auto;}',
        '.modal.active{display:block;}',
        '.modal-close{position:absolute;top:8px;right:12px;cursor:pointer;color:#aaa;font-size:20px;}',
        '</style></head><body>',
        '<div class="toolbar">',
        '  <button onclick="exportLabels()">📋 Export labels</button>',
        '  <button onclick="clearAll()">🗑 Clear all</button>',
        '  <span class="stats" id="stats"></span>',
        '</div>',
        f'<h1 style="color:#fff">multicam_collection — {len(crops_index)} crops · '
        '4-class labelling (blue/green/yellow/red/skip · NO not_jockey)</h1>',
        '<p style="color:#aaa;font-size:12px">Группировка: (camera + resolution). '
        'Метки сохраняются в localStorage. Export → готовый блок для копи-паста.</p>',
    ]

    # Sort: cam-XX numerical, then resolution
    def sort_key(k): return (int(k[0].split("-")[1]), k[1])
    for (cam, res) in sorted(grouped.keys(), key=sort_key):
        items = grouped[(cam, res)]
        parts.append(f'<h2>{cam} · {res} — {len(items)} crops</h2>')
        parts.append('<div class="grid">')
        for r in items:
            cid = r["id"]
            src = f"crops_raw/{r['id']}.jpg"
            meta = (f"<span class='name'>{escape(cid)}</span><br>"
                    f"sess={r['session_short']} · {r['camera']} · {r['resolution']}<br>"
                    f"t={r['timestamp_ms']/1000:.2f}s · "
                    f"conf={r['det_conf']:.2f} · {r['h']}×{r['w']}px")
            labels_html = ''.join(
                f'<label><input type="radio" name="lbl_{cid}" value="{lbl}" '
                f'onchange="setLabel(\'{cid}\', \'{lbl}\')"><span>{lbl}</span></label>'
                for lbl in LABELS
            )
            parts.append(
                f'<div class="card" id="card_{cid}">'
                f'<img src="{escape(src)}" loading="lazy">'
                f'<div class="meta">{meta}</div>'
                f'<div class="labels">{labels_html}</div>'
                f'</div>'
            )
        parts.append('</div>')

    parts.append(
        '<div class="modal" id="exportModal">'
        '<span class="modal-close" onclick="document.getElementById(\'exportModal\').classList.remove(\'active\')">✕</span>'
        '<h2 style="color:#fff">Labels (copy this block)</h2>'
        '<textarea id="export" readonly></textarea>'
        '</div>'
    )
    parts.append('''<script>
const KEY = 'multicam_collection_labels';
function load(){try{return JSON.parse(localStorage.getItem(KEY)||'{}');}catch(e){return{};}}
function save(o){localStorage.setItem(KEY, JSON.stringify(o));}
function setLabel(id,lbl){const o=load();o[id]=lbl;save(o);refreshCard(id,lbl);refreshStats();}
function refreshCard(id,lbl){const c=document.getElementById('card_'+id);if(!c)return;c.classList.remove('labeled');if(lbl&&lbl!=='skip')c.classList.add('labeled');}
function refreshStats(){const o=load();const cnt={};for(const v of Object.values(o))cnt[v]=(cnt[v]||0)+1;
  const tot=Object.keys(o).length;const all=document.querySelectorAll('.card').length;
  const p=[`labeled ${tot}/${all}`];for(const[k,v]of Object.entries(cnt).sort())p.push(`${k}=${v}`);
  document.getElementById('stats').textContent=p.join(' · ');}
function exportLabels(){const o=load();const g={};
  for(const[cid,lbl]of Object.entries(o)){if(lbl==='skip')continue;if(!g[lbl])g[lbl]=[];g[lbl].push(cid);}
  let t='';for(const[lbl,ids]of Object.entries(g).sort()){ids.sort();t+=`${lbl}: [\\n  ${ids.join(',\\n  ')}\\n]\\n\\n`;}
  if(!t)t='(no labels)';document.getElementById('export').value=t;
  document.getElementById('exportModal').classList.add('active');}
function clearAll(){if(confirm('Clear all labels?')){localStorage.removeItem(KEY);location.reload();}}
window.addEventListener('load',()=>{const o=load();for(const[id,lbl]of Object.entries(o)){
  const r=document.querySelector(`input[name="lbl_${id}"][value="${lbl}"]`);if(r){r.checked=true;refreshCard(id,lbl);}}refreshStats();});
</script></body></html>''')
    GALLERY_HTML.write_text("\n".join(parts))

    # Print breakdown
    print()
    print("=" * 76)
    print("=== ШАГ 3 EXTRACTION DONE ===")
    print("=" * 76)
    print(f"  total raw frames:  {n_frames_total}")
    print(f"  total crops:       {n_crops_total}")
    print()
    print(f"  breakdown by (camera × resolution):")
    print(f"  {'camera':<10s} {'res':<8s} {'frames':>7s} {'crops':>7s} "
          f"{'crops/frame':>12s}")
    print("  " + "-"*46)
    for (cam, res) in sorted(breakdown.keys(), key=sort_key):
        b = breakdown[(cam, res)]
        cf = b["crops"] / max(b["frames"], 1)
        print(f"  {cam:<10s} {res:<8s} {b['frames']:>7d} {b['crops']:>7d} "
              f"{cf:>12.2f}")
    print()
    print(f"  crops_index: {CROPS_INDEX_JSON}")
    print(f"  raw_frames:  {RAW_DIR}/")
    print(f"  crops_raw:   {CROPS_DIR}/")
    print(f"  gallery:     {GALLERY_HTML}")


if __name__ == "__main__":
    main()
