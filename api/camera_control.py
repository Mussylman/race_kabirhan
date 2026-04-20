"""
Hikvision ISAPI — force & hold camera stream resolution.

Used by api/server.py to maintain main-stream resolution across camera
reboots / manual overrides. Standalone CLI variant lives at
tools/set_camera_resolution.py.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import requests
from requests.auth import HTTPDigestAuth

log = logging.getLogger("rv.camera_control")


@dataclass
class CameraEntry:
    cam_id:   str
    url:      str
    host:     str
    user:     str
    password: str


def parse_rtsp_url(url: str) -> tuple[str, str, str] | None:
    u = urlparse(url)
    if not u.hostname:
        return None
    return u.hostname, (u.username or "admin"), (u.password or "")


def _rewrite_resolution(xml_body: str, width: int, height: int,
                        bitrate_kbps: int | None = None
                       ) -> tuple[str, tuple[int, int] | None, int | None]:
    m_w  = re.search(r"<videoResolutionWidth>(\d+)</videoResolutionWidth>", xml_body)
    m_h  = re.search(r"<videoResolutionHeight>(\d+)</videoResolutionHeight>", xml_body)
    m_vu = re.search(r"<vbrUpperCap>(\d+)</vbrUpperCap>", xml_body)
    old_res = (int(m_w.group(1)), int(m_h.group(1))) if (m_w and m_h) else None
    old_vu  = int(m_vu.group(1)) if m_vu else None
    new = re.sub(r"<videoResolutionWidth>\d+</videoResolutionWidth>",
                 f"<videoResolutionWidth>{width}</videoResolutionWidth>",
                 xml_body, count=1)
    new = re.sub(r"<videoResolutionHeight>\d+</videoResolutionHeight>",
                 f"<videoResolutionHeight>{height}</videoResolutionHeight>",
                 new, count=1)
    if bitrate_kbps is not None:
        new = re.sub(r"<constantBitRate>\d+</constantBitRate>",
                     f"<constantBitRate>{bitrate_kbps}</constantBitRate>",
                     new, count=1)
        new = re.sub(r"<vbrUpperCap>\d+</vbrUpperCap>",
                     f"<vbrUpperCap>{bitrate_kbps}</vbrUpperCap>",
                     new, count=1)
    return new, old_res, old_vu


def set_camera_resolution(cam: CameraEntry, channel: int,
                          width: int, height: int,
                          bitrate_kbps: int | None = None,
                          timeout_s: float = 6.0) -> str:
    """Return a short status string."""
    auth = HTTPDigestAuth(cam.user, cam.password)
    base = f"http://{cam.host}/ISAPI/Streaming/channels/{channel}"
    try:
        r = requests.get(base, auth=auth, timeout=timeout_s)
        r.raise_for_status()
    except requests.HTTPError as e:
        return f"GET FAIL ({e.response.status_code})"
    except Exception as e:
        return f"GET FAIL ({type(e).__name__})"

    new_xml, old_res, old_vu = _rewrite_resolution(r.text, width, height, bitrate_kbps)
    res_same = (old_res == (width, height))
    br_same  = (bitrate_kbps is None) or (old_vu == bitrate_kbps)
    if res_same and br_same:
        return f"already {width}x{height}" + (f" @ {bitrate_kbps}kbps" if bitrate_kbps else "")

    try:
        resp = requests.put(base, data=new_xml.encode("utf-8"), auth=auth,
                            headers={"Content-Type": "application/xml"},
                            timeout=timeout_s)
        resp.raise_for_status()
    except requests.HTTPError as e:
        return f"PUT FAIL ({e.response.status_code})"
    except Exception as e:
        return f"PUT FAIL ({type(e).__name__})"
    parts = []
    if not res_same: parts.append(f"{old_res} → {width}x{height}")
    if not br_same:  parts.append(f"bitrate {old_vu} → {bitrate_kbps}kbps")
    return f"OK: {' + '.join(parts)}"


def load_cameras_from_analytics(analytics: list[dict]) -> list[CameraEntry]:
    out: list[CameraEntry] = []
    for entry in analytics:
        parsed = parse_rtsp_url(entry["url"])
        if parsed is None:
            continue  # not RTSP (file://), skip silently
        host, user, passwd = parsed
        out.append(CameraEntry(entry["id"], entry["url"], host, user, passwd))
    return out


class ResolutionEnforcer:
    """Background thread that periodically re-asserts main-stream resolution
    (and optionally bitrate) on a set of Hikvision cameras."""

    def __init__(self, cameras: list[CameraEntry], channel: int,
                 width: int, height: int, interval_s: int,
                 bitrate_kbps: int | None = None):
        self.cameras  = cameras
        self.channel  = channel
        self.width    = width
        self.height   = height
        self.bitrate  = bitrate_kbps
        self.interval = max(10, int(interval_s))
        self._stop    = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_status: dict[str, str] = {}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="ResolutionEnforcer",
                                        daemon=True)
        self._thread.start()
        br = f" @ {self.bitrate}kbps" if self.bitrate else ""
        log.info("ResolutionEnforcer started — channel=%d %dx%d%s every %ds on %d cameras",
                 self.channel, self.width, self.height, br,
                 self.interval, len(self.cameras))

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        # One pass immediately on startup, then every `interval` seconds.
        while not self._stop.is_set():
            t0 = time.monotonic()
            changed = []
            for cam in self.cameras:
                if self._stop.is_set():
                    break
                status = set_camera_resolution(cam, self.channel,
                                               self.width, self.height, self.bitrate)
                self.last_status[cam.cam_id] = status
                if status.startswith("OK"):
                    changed.append(cam.cam_id)
                elif "FAIL" in status:
                    log.warning("resolution enforce %s (%s): %s",
                                cam.cam_id, cam.host, status)
            if changed:
                log.info("resolution re-applied on %d cam(s): %s",
                         len(changed), ", ".join(changed))
            # Sleep remainder of the interval
            wait = max(1.0, self.interval - (time.monotonic() - t0))
            if self._stop.wait(timeout=wait):
                break
