#!/usr/bin/env python3
"""
review_reid_gallery.py — Веб-просмотр и очистка галереи ReID.

Usage:
    python3 tools/review_reid_gallery.py
    python3 tools/review_reid_gallery.py --gallery data/reid --port 8888
"""

import argparse
import json
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

COLORS = ["blue", "green", "purple", "red", "yellow"]


def scan_gallery(gallery_dir: Path) -> dict:
    data = {}
    for color in COLORS:
        d = gallery_dir / color
        if not d.exists():
            continue
        files = sorted(
            [f for f in d.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')],
            key=lambda f: f.stat().st_size, reverse=True
        )
        data[color] = [
            {"path": str(f), "name": f.name, "size_kb": round(f.stat().st_size / 1024, 1)}
            for f in files
        ]
    return data


HTML_PAGE = r"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>ReID Gallery Review</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:monospace;background:#1a1a2e;color:#eee;padding:20px}
h1{text-align:center;margin-bottom:10px;color:#e94560}
#stats{text-align:center;margin-bottom:20px;color:#aaa;font-size:14px}
#tabs{display:flex;justify-content:center;gap:8px;margin-bottom:20px}
.tab{padding:8px 20px;border-radius:6px;cursor:pointer;font-weight:bold;
     border:2px solid #333;background:#16213e;user-select:none}
.tab.active{border-color:#e94560;background:#0f3460}
.tab .cnt{font-size:11px;color:#aaa}
#bulk{text-align:center;margin-bottom:15px}
#bulk button{padding:8px 16px;margin:0 5px;border:none;border-radius:4px;
             cursor:pointer;font-weight:bold;font-size:13px}
#grid{display:flex;flex-wrap:wrap;gap:10px;justify-content:center}
.card{border:3px solid #333;border-radius:8px;overflow:hidden;
      transition:all 0.2s;background:#16213e;width:210px}
.card:hover{border-color:#e94560;transform:scale(1.02)}
.card.del{opacity:0.15;pointer-events:none}
.card.kept{border-color:#0a8754}
.card img{display:block;width:200px;height:200px;object-fit:contain;
          background:#000;image-rendering:pixelated;margin:5px auto 0}
.card .info{padding:4px 8px;font-size:11px;color:#aaa;text-align:center}
.card .acts{display:flex}
.card .acts button{flex:1;padding:6px;border:none;cursor:pointer;
                   font-weight:bold;font-size:13px}
.bk{background:#0a8754;color:#fff}.bk:hover{background:#0db96d}
.bd{background:#c62828;color:#fff}.bd:hover{background:#f44336}
</style>
</head><body>
<h1>ReID Gallery Review</h1>
<div id="stats"></div>
<div id="tabs"></div>
<div id="bulk">
  <button id="btnKeepAll" style="background:#0a8754;color:#fff">Keep All</button>
  <button id="btnDelSmall" style="background:#c62828;color:#fff">Delete &lt;3KB</button>
</div>
<div id="grid"></div>
<script>
var DATA = __DATA_PLACEHOLDER__;
var deletedSet = {};
var curColor = Object.keys(DATA)[0] || "blue";

function render() {
  // tabs
  var h = "";
  var colors = Object.keys(DATA);
  for (var ci = 0; ci < colors.length; ci++) {
    var c = colors[ci];
    var items = DATA[c];
    var alive = 0;
    for (var j = 0; j < items.length; j++) {
      if (!deletedSet[items[j].path]) alive++;
    }
    var cls = c === curColor ? "tab active" : "tab";
    h += "<div class='" + cls + "' data-color='" + c + "'>"
       + c + " <span class='cnt'>" + alive + "/" + items.length + "</span></div>";
  }
  document.getElementById("tabs").innerHTML = h;

  // grid
  var items = DATA[curColor] || [];
  var g = "";
  for (var i = 0; i < items.length; i++) {
    var item = items[i];
    var isDel = !!deletedSet[item.path];
    var cls = "card" + (isDel ? " del" : "");
    g += "<div class='" + cls + "' data-idx='" + i + "'>"
       + "<img src='/img?path=" + encodeURIComponent(item.path) + "'>"
       + "<div class='info'>" + item.name + " (" + item.size_kb + " KB)</div>"
       + "<div class='acts'>"
       + "<button class='bk' data-act='keep' data-idx='" + i + "'>Keep</button>"
       + "<button class='bd' data-act='del' data-idx='" + i + "'>Del</button>"
       + "</div></div>";
  }
  document.getElementById("grid").innerHTML = g;
  updStats();
}

function updStats() {
  var total = 0, alive = 0, del = 0;
  var colors = Object.keys(DATA);
  for (var ci = 0; ci < colors.length; ci++) {
    var items = DATA[colors[ci]];
    total += items.length;
    for (var j = 0; j < items.length; j++) {
      if (deletedSet[items[j].path]) del++; else alive++;
    }
  }
  document.getElementById("stats").textContent =
    "Remaining: " + alive + "/" + total + " | Deleted: " + del;
}

function doDelete(path, idx) {
  var xhr = new XMLHttpRequest();
  xhr.open("POST", "/delete", true);
  xhr.setRequestHeader("Content-Type", "application/json");
  xhr.onload = function() {
    deletedSet[path] = true;
    var card = document.querySelector("[data-idx='" + idx + "'].card");
    if (card) card.className = "card del";
    updStats();
    // update tab count
    var tabs = document.getElementById("tabs");
    render();
  };
  xhr.send(JSON.stringify({path: path}));
}

// Event delegation - tabs
document.getElementById("tabs").addEventListener("click", function(e) {
  var tab = e.target.closest(".tab");
  if (tab && tab.dataset.color) {
    curColor = tab.dataset.color;
    render();
  }
});

// Event delegation - grid buttons
document.getElementById("grid").addEventListener("click", function(e) {
  var btn = e.target.closest("button");
  if (!btn) return;
  var idx = parseInt(btn.dataset.idx);
  var items = DATA[curColor] || [];
  if (idx < 0 || idx >= items.length) return;

  if (btn.dataset.act === "del") {
    doDelete(items[idx].path, idx);
  } else if (btn.dataset.act === "keep") {
    var card = btn.closest(".card");
    if (card) card.className = "card kept";
  }
});

// Bulk buttons
document.getElementById("btnKeepAll").addEventListener("click", function() {
  var cards = document.querySelectorAll(".card:not(.del)");
  for (var i = 0; i < cards.length; i++) cards[i].className = "card kept";
});

document.getElementById("btnDelSmall").addEventListener("click", function() {
  var items = DATA[curColor] || [];
  var count = 0;
  for (var i = 0; i < items.length; i++) {
    if (items[i].size_kb < 3 && !deletedSet[items[i].path]) count++;
  }
  if (!count) { alert("No files < 3KB"); return; }
  if (!confirm("Delete " + count + " files < 3KB?")) return;
  for (var i = 0; i < items.length; i++) {
    if (items[i].size_kb < 3 && !deletedSet[items[i].path]) {
      doDelete(items[i].path, i);
    }
  }
});

render();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    gallery_data = None

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", ""):
            data_json = json.dumps(self.gallery_data)
            html = HTML_PAGE.replace("__DATA_PLACEHOLDER__", data_json)
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/img":
            qs = parse_qs(parsed.query)
            fpath = Path(qs.get("path", [""])[0])
            if fpath.exists() and fpath.suffix.lower() in (".jpg", ".jpeg", ".png"):
                data = fpath.read_bytes()
                ct = "image/png" if fpath.suffix.lower() == ".png" else "image/jpeg"
                self.send_response(200)
                self.send_header("Content-Type", ct)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/delete":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            fpath = Path(body["path"])
            if fpath.exists():
                fpath.unlink()
                resp = json.dumps({"ok": True}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
                print(f"  DELETED: {fpath.name}")
            else:
                self.send_error(404, "File not found")
        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery", default="data/reid")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()

    gallery_dir = Path(args.gallery)
    if not gallery_dir.exists():
        print(f"Gallery not found: {gallery_dir}")
        sys.exit(1)

    data = scan_gallery(gallery_dir)
    total = sum(len(v) for v in data.values())
    print(f"Gallery: {total} photos in {len(data)} colors")
    for color, items in data.items():
        print(f"  {color}: {len(items)}")

    Handler.gallery_data = data

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"\nOpen: http://localhost:{args.port}")
    print("Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        remaining = scan_gallery(gallery_dir)
        for color, items in remaining.items():
            print(f"  {color}: {len(items)}")


if __name__ == "__main__":
    main()
