#!/usr/bin/env python3
"""
Force Hikvision RTSP main streams to a target resolution and keep them
there. Reads camera IP + credentials from a JSON config (cameras_live.json
format), talks to /ISAPI/Streaming/channels/101, rewrites the resolution
fields, PUTs the XML back. Optionally polls on a schedule to re-apply if
the setting drifts (cam reboot, manual change, etc).

Usage:
    # One-shot: set all cameras to 2560x1440 and exit
    python3 tools/set_camera_resolution.py

    # Subset (comma-separated cam IDs)
    RV_CAMS=cam-01,cam-13 python3 tools/set_camera_resolution.py

    # Different resolution / target channel
    python3 tools/set_camera_resolution.py --width 1920 --height 1080

    # Watcher: re-apply every N seconds
    python3 tools/set_camera_resolution.py --watch 60

    # Substream (Channels/102) instead of main (101)
    python3 tools/set_camera_resolution.py --channel 102 --width 704 --height 480

Deps: requests (standard library urllib would work but digest auth is
annoying without requests).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

try:
    import requests
    from requests.auth import HTTPDigestAuth
except ImportError:
    sys.stderr.write("requires `pip install requests`\n")
    sys.exit(1)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CAMERAS = REPO_ROOT / "configs" / "cameras_live.json"


def parse_rtsp_url(url: str) -> dict:
    """Extract host/user/password from rtsp://user:pass@ip:port/path."""
    u = urlparse(url)
    return {
        "host":     u.hostname,
        "port":     554,
        "user":     u.username or "admin",
        "password": u.password or "",
    }


def get_stream_xml(auth: HTTPDigestAuth, host: str, channel: int) -> str:
    """GET the current StreamingChannel XML from ISAPI."""
    url = f"http://{host}/ISAPI/Streaming/channels/{channel}"
    r = requests.get(url, auth=auth, timeout=6)
    r.raise_for_status()
    return r.text


def put_stream_xml(auth: HTTPDigestAuth, host: str, channel: int, xml_body: str) -> str:
    """PUT the (modified) StreamingChannel XML back."""
    url = f"http://{host}/ISAPI/Streaming/channels/{channel}"
    r = requests.put(url, data=xml_body.encode("utf-8"), auth=auth,
                     headers={"Content-Type": "application/xml"}, timeout=6)
    r.raise_for_status()
    return r.text


def apply_resolution(xml_body: str, width: int, height: int) -> tuple[str, tuple[int, int] | None]:
    """Rewrite <videoResolutionWidth>/<videoResolutionHeight> in the XML.
    Returns (new_xml, old_resolution_or_None)."""
    m_w = re.search(r"<videoResolutionWidth>(\d+)</videoResolutionWidth>", xml_body)
    m_h = re.search(r"<videoResolutionHeight>(\d+)</videoResolutionHeight>", xml_body)
    old = (int(m_w.group(1)), int(m_h.group(1))) if (m_w and m_h) else None

    new = re.sub(
        r"<videoResolutionWidth>\d+</videoResolutionWidth>",
        f"<videoResolutionWidth>{width}</videoResolutionWidth>",
        xml_body, count=1,
    )
    new = re.sub(
        r"<videoResolutionHeight>\d+</videoResolutionHeight>",
        f"<videoResolutionHeight>{height}</videoResolutionHeight>",
        new, count=1,
    )
    return new, old


def set_one_camera(cam: dict, channel: int, width: int, height: int,
                   dry_run: bool, dump_xml: Path | None = None) -> str:
    """Handle one camera. Returns a status string for logging."""
    creds = parse_rtsp_url(cam["url"])
    if not creds["host"]:
        return "SKIP (no host)"
    auth = HTTPDigestAuth(creds["user"], creds["password"])
    try:
        xml = get_stream_xml(auth, creds["host"], channel)
    except requests.HTTPError as e:
        return f"GET FAIL ({e.response.status_code})"
    except Exception as e:
        return f"GET FAIL ({type(e).__name__}: {e})"

    if dump_xml is not None:
        dump_xml.write_text(xml)
        print(f"    dumped GET XML ({len(xml)} bytes) → {dump_xml}")

    new_xml, old_res = apply_resolution(xml, width, height)
    if old_res == (width, height):
        return f"already {width}x{height}"

    if dry_run:
        return f"DRY would change {old_res} → {width}x{height}"

    try:
        resp_text = put_stream_xml(auth, creds["host"], channel, new_xml)
    except requests.HTTPError as e:
        msg = f"PUT FAIL ({e.response.status_code})"
        # Save response body on first 500 so we can read the real error
        if dump_xml is not None:
            (dump_xml.parent / (dump_xml.stem + "_put_response.xml")).write_text(
                e.response.text or "<empty>")
            try:
                (dump_xml.parent / (dump_xml.stem + "_put_sent.xml")).write_text(new_xml)
            except Exception:
                pass
        return msg
    except Exception as e:
        return f"PUT FAIL ({type(e).__name__}: {e})"
    return f"OK {old_res} → {width}x{height}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--cameras", default=str(DEFAULT_CAMERAS),
                    help="cameras_live.json-style config")
    ap.add_argument("--channel", type=int, default=101,
                    help="ISAPI streaming channel (101=main, 102=sub)")
    # 2688x1520 = native 4MP Hikvision; supported by all our cameras.
    # 2560x1440 sounds right but is NOT in their capability list.
    ap.add_argument("--width",   type=int, default=2688)
    ap.add_argument("--height",  type=int, default=1520)
    ap.add_argument("--watch",   type=int, default=0,
                    help="if > 0, re-apply every N seconds instead of exiting")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dump", default=None,
                    help="on failure/first pass, dump GET XML and PUT response "
                         "for the FIRST camera to this file (for debugging).")
    ap.add_argument("--show-caps", action="store_true",
                    help="Query /ISAPI/Streaming/channels/{ch}/capabilities for "
                         "each camera and print supported resolutions. "
                         "Nothing is modified.")
    ap.add_argument("--auto", action="store_true",
                    help="Pick the HIGHEST resolution supported by ALL cameras "
                         "(queries /capabilities first). Overrides --width/--height.")
    args = ap.parse_args()

    cfg = json.loads(Path(args.cameras).read_text())
    cams = cfg.get("analytics", [])
    only = set(filter(None, os.environ.get("RV_CAMS", "").split(",")))
    if only:
        cams = [c for c in cams if c["id"] in only]

    if args.show_caps:
        print(f"Querying capabilities on channel {args.channel} ...")
        for cam in cams:
            creds = parse_rtsp_url(cam["url"])
            if not creds["host"]:
                continue
            auth = HTTPDigestAuth(creds["user"], creds["password"])
            url = f"http://{creds['host']}/ISAPI/Streaming/channels/{args.channel}/capabilities"
            try:
                r = requests.get(url, auth=auth, timeout=5)
                r.raise_for_status()
                body = r.text
            except Exception as e:
                print(f"  {cam['id']:8s} {creds['host']:15s}  FAIL ({type(e).__name__})")
                continue
            # Parse <videoResolutionWidth opt="W1,W2,..."> and height equivalent
            w_opt = re.search(r'videoResolutionWidth[^>]*opt="([^"]+)"', body)
            h_opt = re.search(r'videoResolutionHeight[^>]*opt="([^"]+)"', body)
            codec  = re.search(r'videoCodecType[^>]*opt="([^"]+)"', body)
            if w_opt and h_opt:
                widths  = w_opt.group(1).split(",")
                heights = h_opt.group(1).split(",")
                combos  = set()
                for w, h in zip(widths, heights):
                    combos.add((w.strip(), h.strip()))
                pairs = sorted({(int(w), int(h)) for w, h in combos}, reverse=True)
                pair_str = ", ".join(f"{w}x{h}" for w, h in pairs)
                codec_s = codec.group(1) if codec else "?"
                print(f"  {cam['id']:8s} {creds['host']:15s}  codec={codec_s}  resolutions: {pair_str}")
            else:
                print(f"  {cam['id']:8s} {creds['host']:15s}  opt attr missing — dumping:")
                print(body[:500])
        return

    if args.auto:
        # Intersect supported resolutions across all reachable cameras
        common = None
        for cam in cams:
            creds = parse_rtsp_url(cam["url"])
            if not creds["host"]:
                continue
            auth = HTTPDigestAuth(creds["user"], creds["password"])
            url = f"http://{creds['host']}/ISAPI/Streaming/channels/{args.channel}/capabilities"
            try:
                r = requests.get(url, auth=auth, timeout=5)
                r.raise_for_status()
            except Exception:
                continue
            w_opt = re.search(r'videoResolutionWidth[^>]*opt="([^"]+)"', r.text)
            h_opt = re.search(r'videoResolutionHeight[^>]*opt="([^"]+)"', r.text)
            if not (w_opt and h_opt):
                continue
            combos = {(int(w.strip()), int(h.strip()))
                      for w, h in zip(w_opt.group(1).split(","),
                                      h_opt.group(1).split(","))}
            common = combos if common is None else (common & combos)
        if not common:
            print("AUTO: couldn't build a common resolution set; falling back to "
                  f"{args.width}x{args.height}")
        else:
            # Pick largest by total pixels; tie-break by width
            best = max(common, key=lambda p: (p[0] * p[1], p[0]))
            args.width, args.height = best
            print(f"AUTO picked {args.width}x{args.height} (common to {len(cams)} cams)")

    print(f"Target: channel={args.channel}  resolution={args.width}x{args.height}")
    print(f"Cameras: {len(cams)}  (dry_run={args.dry_run}, watch={args.watch}s)")

    def pass_once():
        for i, cam in enumerate(cams):
            dump = Path(args.dump) if (args.dump and i == 0) else None
            status = set_one_camera(cam, args.channel, args.width, args.height,
                                    args.dry_run, dump_xml=dump)
            creds = parse_rtsp_url(cam["url"])
            print(f"  {cam['id']:8s} {creds['host']:15s}  {status}")

    if args.watch <= 0:
        pass_once()
        return

    while True:
        t0 = time.time()
        print(f"\n=== {time.strftime('%H:%M:%S')} pass ===")
        pass_once()
        sleep_s = max(5, args.watch - (time.time() - t0))
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
