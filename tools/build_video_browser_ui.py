#!/usr/bin/env python3
"""build_video_browser_ui.py — HTML browser для всех mp4 в data/videos/.

Сканирует, генерирует thumbnails (1 кадр середины), строит index.html
с иерархией top-level / session / camera. HTML5 <video> для стриминга
прямо из браузера через тот же http.server.
"""
from __future__ import annotations
import json, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import escape
from pathlib import Path

REPO = Path("/home/ipodrom/Рабочий стол/Ipodrom-Project/user/race_vision")
ROOT = REPO / "data" / "videos"
THUMBS = ROOT / ".thumbnails"
INDEX = ROOT / "index.html"
META  = ROOT / ".metadata.json"

CAM_RX = re.compile(r"^kamera_(\d+)(?:_(r\d+))?_")


def ffprobe(path: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,r_frame_rate,duration,nb_frames",
           "-show_entries", "format=duration,size",
           "-of", "json", str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0: return {}
    j = json.loads(r.stdout)
    s = j["streams"][0] if j.get("streams") else {}
    fmt = j.get("format", {})
    rfr = s.get("r_frame_rate", "0/1").split("/")
    fps = float(rfr[0]) / float(rfr[1]) if len(rfr) == 2 and float(rfr[1]) else 0.0
    dur = float(s.get("duration") or fmt.get("duration") or 0)
    return {
        "width":  int(s.get("width", 0)),
        "height": int(s.get("height", 0)),
        "fps":    round(fps, 2),
        "duration_sec": round(dur, 1),
        "size_mb": round(int(fmt.get("size", path.stat().st_size)) / 1024 / 1024, 1),
    }


def gen_thumb(video: Path, thumb: Path, t_sec: float):
    if thumb.exists(): return True
    thumb.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t_sec),
           "-i", str(video), "-frames:v", "1",
           "-vf", "scale=320:-1", "-q:v", "5", str(thumb)]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0


def process_video(video: Path) -> dict:
    rel = video.relative_to(ROOT)
    rel_thumb = Path(".thumbnails") / rel.with_suffix(".jpg")
    abs_thumb = ROOT / rel_thumb
    meta = ffprobe(video)
    if not meta.get("duration_sec"):
        return {"path": str(rel), "error": "ffprobe failed"}
    t_mid = meta["duration_sec"] / 2
    gen_thumb(video, abs_thumb, t_mid)
    parts = rel.parts
    top = parts[0] if len(parts) > 0 else "(root)"
    session = parts[1] if len(parts) > 1 else "(direct)"
    name = video.name
    m = CAM_RX.match(name)
    cam_num = m.group(1).zfill(2) if m else None
    return {
        "path":      str(rel),
        "thumb":     str(rel_thumb),
        "name":      name,
        "top":       top,
        "session":   session,
        "camera":    f"cam-{cam_num}" if cam_num else None,
        **meta,
    }


def main():
    print(f"  scanning {ROOT}/")
    videos = sorted(ROOT.rglob("*.mp4"))
    print(f"  found {len(videos)} mp4 files")
    THUMBS.mkdir(parents=True, exist_ok=True)

    print(f"  generating thumbnails (parallel) ...")
    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(process_video, v): v for v in videos}
        for i, fut in enumerate(as_completed(futures)):
            rows.append(fut.result())
            if (i+1) % 25 == 0:
                print(f"    {i+1}/{len(videos)}")
    print(f"    done: {len(rows)} entries")

    META.write_text(json.dumps(rows, indent=2))

    # Group: top → session → entries
    grouped = {}
    for r in rows:
        if r.get("error"): continue
        grouped.setdefault(r["top"], {}).setdefault(r["session"], []).append(r)

    # Sort entries by camera number (where applicable)
    for top in grouped:
        for sess in grouped[top]:
            grouped[top][sess].sort(key=lambda r: (
                int(r["camera"].split("-")[1]) if r.get("camera") else 999,
                r["name"]))

    # Build HTML
    parts = ['<!doctype html><html lang="ru"><head><meta charset="utf-8">',
             '<title>Race Vision — videos browser</title>',
             '<style>',
             'body{font-family:system-ui,Segoe UI,Roboto,sans-serif;background:#1a1a1a;color:#ddd;margin:0;padding:14px;}',
             'h1{color:#fff;font-size:18px;margin:6px 0 14px 0;}',
             'h2{color:#fff;font-size:15px;margin:18px 0 8px 0;padding:8px 12px;background:#333;border-radius:4px;cursor:pointer;}',
             'h2:hover{background:#3a3a3a;}',
             'h3{color:#88ddff;font-size:13px;margin:10px 0 6px 0;padding:4px 10px;background:#262626;border-radius:3px;cursor:pointer;}',
             'h3:hover{background:#2e2e2e;}',
             '.section{margin-bottom:14px;}',
             '.collapsed h3+.grid{display:none;}',
             '.collapsed h2+.subsections{display:none;}',
             '.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px;margin-bottom:14px;}',
             '.card{background:#2a2a2a;border-radius:4px;padding:6px;cursor:pointer;transition:background 0.1s;}',
             '.card:hover{background:#333;}',
             '.card img{width:100%;display:block;border-radius:3px;background:#000;aspect-ratio:16/9;object-fit:cover;}',
             '.card .meta{font-size:10px;color:#aaa;padding:4px 2px;line-height:1.4;}',
             '.card .name{color:#88ddff;word-break:break-all;font-weight:500;}',
             '.tag{display:inline-block;padding:1px 5px;background:#444;border-radius:2px;margin-right:3px;font-size:10px;}',
             '.tag.res-720p{background:#444;}',
             '.tag.res-1080p{background:#3a5a3a;}',
             '.tag.res-1520p{background:#5a3a5a;}',
             '.tag.res-other{background:#5a3a3a;}',
             '.modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.92);z-index:200;align-items:center;justify-content:center;}',
             '.modal.active{display:flex;}',
             '.modal video{max-width:96vw;max-height:90vh;}',
             '.modal-close{position:fixed;top:16px;right:24px;color:#fff;font-size:30px;cursor:pointer;background:#000;padding:4px 14px;border-radius:4px;}',
             '.controls{position:fixed;bottom:14px;left:50%;transform:translateX(-50%);background:rgba(0,0,0,0.85);padding:8px 14px;border-radius:6px;font-size:12px;color:#ddd;}',
             '.search{padding:8px 12px;width:100%;max-width:380px;background:#2a2a2a;color:#fff;border:1px solid #444;border-radius:4px;font-size:13px;}',
             '.toolbar{position:sticky;top:0;background:#1a1a1a;padding:10px 0;z-index:50;border-bottom:1px solid #444;margin-bottom:10px;}',
             '.summary{color:#88ddff;font-size:11px;margin-left:10px;}',
             '</style></head><body>',
             '<div class="toolbar">',
             '<input class="search" id="search" placeholder="filter: kamera, session, 1080p, 720p..." oninput="filter()">',
             f'<span class="summary">{len(rows)} videos · ',
             f'{sum(r.get("size_mb",0) for r in rows)/1024:.1f} GB total · ',
             f'click thumbnail to play</span>',
             '<button onclick="expandAll()" style="margin-left:10px;padding:4px 10px;background:#333;color:#fff;border:1px solid #555;border-radius:3px;cursor:pointer;">Expand all</button>',
             '<button onclick="collapseAll()" style="margin-left:4px;padding:4px 10px;background:#333;color:#fff;border:1px solid #555;border-radius:3px;cursor:pointer;">Collapse all</button>',
             '</div>',
             '<h1>Race Vision data/videos browser</h1>',
             ]

    # Build sections
    for top in sorted(grouped.keys()):
        sessions = grouped[top]
        n_top = sum(len(v) for v in sessions.values())
        sz_top = sum(r.get("size_mb",0) for ses in sessions.values() for r in ses)/1024
        parts.append(f'<div class="section">')
        parts.append(f'<h2 onclick="this.parentElement.classList.toggle(\'collapsed\')">'
                     f'📁 {escape(top)} '
                     f'<span style="color:#aaa;font-weight:normal;font-size:12px">'
                     f'· {n_top} videos · {sz_top:.1f} GB</span></h2>')
        parts.append(f'<div class="subsections">')
        for sess in sorted(sessions.keys()):
            ents = sessions[sess]
            n_sess = len(ents)
            sz_sess = sum(r.get("size_mb",0) for r in ents)/1024
            parts.append(f'<h3 onclick="this.parentElement.classList.toggle(\'collapsed\')">'
                         f'📂 {escape(sess)} '
                         f'<span style="color:#888;font-weight:normal">· {n_sess} videos · '
                         f'{sz_sess:.1f} GB</span></h3>')
            parts.append('<div class="grid">')
            for r in ents:
                res_tag = (f"{r['height']}p" if r.get("height") in (720, 1080)
                           else f"{r['height']}p" if r.get("height") == 1520
                           else "other")
                res_class = (f"res-{res_tag}" if res_tag in ("720p","1080p","1520p")
                             else "res-other")
                meta_line = (
                    f"<span class='tag {res_class}'>{r.get('width',0)}×{r.get('height',0)}</span> "
                    f"<span class='tag'>{r.get('duration_sec',0):.0f}s</span> "
                    f"<span class='tag'>{r.get('size_mb',0):.0f}M</span>"
                )
                cam_str = f"{r.get('camera','?')} · " if r.get("camera") else ""
                parts.append(
                    f'<div class="card" data-search="{escape(r["name"].lower())} {escape(top.lower())} {escape(sess.lower())} {res_tag}" '
                    f'onclick="play(\'{escape(r["path"])}\', \'{escape(r["name"])}\')">'
                    f'<img src="{escape(r["thumb"])}" loading="lazy" '
                    f'onerror="this.style.background=\'#444\';">'
                    f'<div class="meta">'
                    f'<span class="name">{escape(r["name"])}</span><br>'
                    f'{cam_str}{meta_line}'
                    f'</div></div>'
                )
            parts.append('</div>')
        parts.append('</div></div>')

    parts.append(
        '<div class="modal" id="modal" onclick="if(event.target===this)closeModal()">'
        '<span class="modal-close" onclick="closeModal()">✕</span>'
        '<video id="player" controls></video>'
        '<div class="controls" id="controls">filename</div>'
        '</div>'
    )
    parts.append('''<script>
function play(path, name){
  const m=document.getElementById('modal'), v=document.getElementById('player');
  v.src=path; v.play().catch(()=>{}); m.classList.add('active');
  document.getElementById('controls').textContent=name+' (Esc to close)';
}
function closeModal(){
  const m=document.getElementById('modal'), v=document.getElementById('player');
  v.pause(); v.src=''; m.classList.remove('active');
}
window.addEventListener('keydown',e=>{if(e.key==='Escape')closeModal();});
function filter(){
  const q=document.getElementById('search').value.toLowerCase().trim();
  document.querySelectorAll('.card').forEach(c=>{
    c.style.display=(!q||c.dataset.search.includes(q))?'':'none';
  });
}
function expandAll(){document.querySelectorAll('.section,.subsections,h3').forEach(s=>s.classList?.remove('collapsed'));
  document.querySelectorAll('.section').forEach(s=>s.classList.remove('collapsed'));
  document.querySelectorAll('h3').forEach(h=>h.parentElement&&h.parentElement.classList&&h.parentElement.classList.remove('collapsed'));}
function collapseAll(){
  document.querySelectorAll('.section').forEach(s=>s.classList.add('collapsed'));
}
</script></body></html>''')
    INDEX.write_text("\n".join(parts))
    print(f"  index: {INDEX}")
    print(f"  metadata: {META}")
    by_res = {}
    for r in rows:
        h = r.get("height", 0)
        by_res[h] = by_res.get(h, 0) + 1
    print(f"  by resolution: {by_res}")


if __name__ == "__main__":
    main()
