#!/usr/bin/env python3
"""Simple server for ROI editor tool."""

import json
import os
import sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

import cv2

FRAMES_DIR = Path("/tmp/roi_frames")
ROI_PATH = Path("configs/camera_roi.json")
CAMERAS_PATH = Path(os.environ.get("RV_CAMERAS", "configs/cameras_live.json"))
PORT = 8899

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/cameras":
            with open(CAMERAS_PATH) as f:
                cfg = json.load(f)
            cams = [c["id"] for c in cfg.get("analytics", [])]
            self._json(cams)
        elif self.path == "/api/roi":
            if ROI_PATH.exists():
                with open(ROI_PATH) as f:
                    self._json(json.load(f))
            else:
                self._json({})
        elif self.path.startswith("/frames/"):
            fname = self.path.split("/frames/")[1].split("?")[0]
            fpath = FRAMES_DIR / fname
            if fpath.exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                self.wfile.write(fpath.read_bytes())
            else:
                self.send_error(404)
        elif self.path == "/" or self.path == "/index.html":
            html = Path("tools/roi_editor.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/roi":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            ROI_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(ROI_PATH, "w") as f:
                json.dump(body, f, indent=2)
            self._json({"ok": True, "path": str(ROI_PATH)})
        else:
            self.send_error(404)

    def _json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # quiet

def grab_frame(url: str, out_path: Path,
               warmup: int = 25, samples: int = 8) -> bool:
    """Grab a clean RTSP frame via OpenCV.

    Strategy: open capture, burn `warmup` frames to let the decoder
    align on a GOP boundary, then read `samples` frames and keep the
    sharpest one (Laplacian variance as a proxy for image quality).
    Much more robust than taking the first frame — the first few frames
    after RTSP connect are often partially decoded / half-painted.
    """
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                          "rtsp_transport;tcp|buffer_size;1048576")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return False
    try:
        # Let decoder settle on a keyframe
        for _ in range(warmup):
            if not cap.grab():
                break

        best_frame = None
        best_score = -1.0
        for _ in range(samples):
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if score > best_score:
                best_score = score
                best_frame = frame
    finally:
        cap.release()

    if best_frame is None:
        return False
    h, w = best_frame.shape[:2]
    cv2.imwrite(str(out_path), best_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"    kept frame {w}x{h} sharpness={best_score:.0f}")
    return True


if __name__ == "__main__":
    print(f"Camera config: {CAMERAS_PATH}")
    if not CAMERAS_PATH.is_file():
        print(f"ERROR: config not found. Override with RV_CAMERAS=path/to/cameras.json")
        sys.exit(1)

    use_sub = os.environ.get("RV_ROI_SUBSTREAM", "0") == "1"
    only = set(filter(None, os.environ.get("RV_ROI_ONLY", "").split(",")))
    print(f"Substream (Channels/102): {use_sub}")
    if only:
        print(f"Re-grabbing only: {sorted(only)}")
    print("Grabbing one frame per camera via OpenCV...")
    FRAMES_DIR.mkdir(exist_ok=True)
    with open(CAMERAS_PATH) as f:
        cfg = json.load(f)
    ok_count = 0
    for cam in cfg.get("analytics", []):
        if only and cam["id"] not in only:
            continue
        out = FRAMES_DIR / f"{cam['id']}.jpg"
        url = cam["url"]
        if use_sub:
            # Hikvision main→sub: /Streaming/Channels/101 → 102
            url = url.replace("/Streaming/Channels/101", "/Streaming/Channels/102")
        print(f"  {cam['id']:8s} grabbing {url[:70]}")
        ok = grab_frame(url, out)
        status = "OK" if ok else "FAIL"
        print(f"  {cam['id']:8s} {status}")
        if ok:
            ok_count += 1

    print(f"\nGrabbed {ok_count}/{len(cfg.get('analytics', []))} frames into {FRAMES_DIR}")
    print(f"ROI Editor: http://localhost:{PORT}")
    print("Shortcuts: arrows=nav, Enter=close polygon, z=undo, Esc=cancel, Ctrl+S=save")
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
