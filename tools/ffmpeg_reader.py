"""
ffmpeg_reader.py — Universal RTSP / video file reader via ffmpeg subprocess.

Standalone library: copy this single file into any project.
Dependencies: numpy + ffmpeg binary. OpenCV optional (CLI preview only).

Architecture:
    find_ffmpeg()       → locate ffmpeg binary
    probe_stream()      → codec, resolution, fps in one call
    FFmpegPipe          → low-level: one subprocess, raw frame reads
    CameraStream        → mid-level: one camera + auto-reconnect in a thread
    MultiCameraReader   → high-level: N cameras, unified API

Usage:
    from ffmpeg_reader import MultiCameraReader

    reader = MultiCameraReader(gpu=True)
    reader.add("cam1", "rtsp://admin:pass@192.168.1.10:554/stream")
    reader.add("cam2", "rtsp://admin:pass@192.168.1.11:554/stream")
    reader.start_all()

    frame = reader.grab("cam1")   # numpy array or None
    frames = reader.grab_all()    # {"cam1": array, "cam2": array}
    reader.stop_all()

CLI:
    python tools/ffmpeg_reader.py probe  "rtsp://..."
    python tools/ffmpeg_reader.py preview "rtsp://..." --gpu
    python tools/ffmpeg_reader.py record "rtsp://..." --duration 10 --output clip.mp4
    python tools/ffmpeg_reader.py test   "rtsp://..." --gpu
"""

import os
import re
import sys
import time
import logging
import platform
import subprocess
import shutil
import threading
from pathlib import Path
from typing import Optional, Callable

import numpy as np

__all__ = [
    "find_ffmpeg",
    "probe_stream",
    "FFmpegPipe",
    "CameraStream",
    "MultiCameraReader",
]

log = logging.getLogger("ffmpeg_reader")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Windows: hide console windows spawned by subprocess
_SUBPROCESS_FLAGS = 0
if platform.system() == "Windows":
    _SUBPROCESS_FLAGS = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]


def _mask_url(url: str) -> str:
    """Hide credentials in RTSP URL for logging."""
    if "@" in url and "://" in url:
        scheme_end = url.index("://") + 3
        at_pos = url.index("@")
        return url[:scheme_end] + "***:***@" + url[at_pos + 1:]
    return url


# ---------------------------------------------------------------------------
# find_ffmpeg
# ---------------------------------------------------------------------------

def find_ffmpeg(path: Optional[str] = None) -> str:
    """Locate ffmpeg binary.

    Search order:
        1. Explicit *path* argument
        2. FFMPEG_PATH environment variable
        3. imageio_ffmpeg package (bundled binary)
        4. shutil.which("ffmpeg")

    Returns:
        Absolute path to ffmpeg executable.

    Raises:
        FileNotFoundError: if ffmpeg cannot be found anywhere.
    """
    # 1. Explicit argument
    if path and Path(path).is_file():
        return str(Path(path).resolve())

    # 2. Environment variable
    env_path = os.environ.get("FFMPEG_PATH")
    if env_path and Path(env_path).is_file():
        return str(Path(env_path).resolve())

    # 3. imageio_ffmpeg
    try:
        import imageio_ffmpeg
        iio_path = imageio_ffmpeg.get_ffmpeg_exe()
        if iio_path and Path(iio_path).is_file():
            return str(Path(iio_path).resolve())
    except Exception:
        pass

    # 4. System PATH
    which = shutil.which("ffmpeg")
    if which:
        return str(Path(which).resolve())

    raise FileNotFoundError(
        "ffmpeg not found. Install it or set FFMPEG_PATH environment variable."
    )


# ---------------------------------------------------------------------------
# probe_stream
# ---------------------------------------------------------------------------

def probe_stream(
    source: str,
    ffmpeg: Optional[str] = None,
    timeout: float = 15.0,
) -> dict:
    """Probe a video source (RTSP URL or file) and return stream info.

    Returns:
        {
            "width": int,
            "height": int,
            "codec": str,      # e.g. "h264", "hevc"
            "fps": float,
            "raw": str,        # full ffmpeg stderr for debugging
        }
        On failure, width/height are 0 and codec is "h264" (safe fallback).
    """
    ffmpeg = ffmpeg or find_ffmpeg()

    cmd = [ffmpeg]

    # RTSP-specific flags
    if source.startswith("rtsp://"):
        cmd += ["-rtsp_transport", "tcp"]

    cmd += ["-i", source, "-frames:v", "1", "-f", "null", "-"]

    info = {"width": 0, "height": 0, "codec": "h264", "fps": 0.0, "raw": ""}

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=_SUBPROCESS_FLAGS,
        )
        stderr = result.stderr
        info["raw"] = stderr

        for line in stderr.split("\n"):
            if "Video:" not in line:
                continue

            lower = line.lower()

            # Codec
            if "hevc" in lower or "h265" in lower or "h.265" in lower:
                info["codec"] = "hevc"
            elif "h264" in lower or "h.264" in lower or "avc" in lower:
                info["codec"] = "h264"

            # Resolution (WxH)
            m = re.search(r"(\d{3,5})x(\d{3,5})", line)
            if m:
                info["width"] = int(m.group(1))
                info["height"] = int(m.group(2))

            # FPS
            fps_match = re.search(r"(\d+(?:\.\d+)?)\s*fps", lower)
            if fps_match:
                info["fps"] = float(fps_match.group(1))
            else:
                # Try tbr
                tbr_match = re.search(r"(\d+(?:\.\d+)?)\s*tbr", lower)
                if tbr_match:
                    info["fps"] = float(tbr_match.group(1))

            break  # first video stream only

    except subprocess.TimeoutExpired:
        log.warning("probe_stream: timeout (%.0fs) for %s", timeout, _mask_url(source))
    except FileNotFoundError:
        log.error("probe_stream: ffmpeg not found at %s", ffmpeg)
    except Exception as e:
        log.error("probe_stream: %s", e)

    return info


# ---------------------------------------------------------------------------
# FFmpegPipe — low-level: one subprocess → raw frames
# ---------------------------------------------------------------------------

class FFmpegPipe:
    """Low-level ffmpeg subprocess that decodes video to raw BGR24 frames.

    Works with both RTSP streams and local video files.

    Usage:
        pipe = FFmpegPipe(source, width=1920, height=1080, ffmpeg=find_ffmpeg())
        pipe.start()
        ok, frame = pipe.read()   # numpy (H, W, 3) uint8 or (False, None)
        pipe.stop()
    """

    def __init__(
        self,
        source: str,
        width: int,
        height: int,
        *,
        ffmpeg: Optional[str] = None,
        gpu: bool = False,
        codec: str = "h264",
    ):
        self.source = source
        self.width = width
        self.height = height
        self.ffmpeg = ffmpeg or find_ffmpeg()
        self.gpu = gpu
        self.codec = codec
        self.frame_size = width * height * 3
        self._process: Optional[subprocess.Popen] = None

    def start(self):
        """Launch ffmpeg subprocess."""
        cmd = [self.ffmpeg]

        # RTSP-specific flags
        if self.source.startswith("rtsp://"):
            cmd += [
                "-rtsp_transport", "tcp",
                "-fflags", "nobuffer",
                "-flags", "low_delay",
            ]

        # GPU decode
        if self.gpu:
            cuvid = "hevc_cuvid" if self.codec == "hevc" else "h264_cuvid"
            cmd += ["-hwaccel", "cuda", "-c:v", cuvid]

        cmd += [
            "-i", self.source,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an",
            "-",
        ]

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self.frame_size * 2,
            creationflags=_SUBPROCESS_FLAGS,
        )

    def read(self) -> tuple:
        """Read one frame.

        Returns:
            (True, numpy_array) on success, (False, None) on failure/EOF.
        """
        if self._process is None or self._process.stdout is None:
            return False, None

        raw = self._process.stdout.read(self.frame_size)
        if len(raw) != self.frame_size:
            return False, None

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            (self.height, self.width, 3)
        ).copy()
        return True, frame

    def stop(self):
        """Terminate ffmpeg process."""
        proc = self._process
        if proc is None:
            return
        self._process = None

        try:
            proc.terminate()
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
        except Exception:
            pass

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def __del__(self):
        self.stop()


# ---------------------------------------------------------------------------
# CameraStream — one camera + auto-reconnect in a daemon thread
# ---------------------------------------------------------------------------

class CameraStream:
    """Reads frames from a single RTSP/video source in a background thread.

    Features:
        - Auto-reconnect with configurable delay
        - Thread-safe grab() for latest frame
        - Callbacks: on_frame, on_disconnect, on_reconnect
        - Fast-fail protection: if pipe dies <2s three times in a row → stop

    Usage:
        cam = CameraStream("cam1", "rtsp://...", gpu=True)
        cam.start()
        frame = cam.grab()   # numpy array or None
        cam.stop()
    """

    FAST_FAIL_THRESHOLD = 2.0   # seconds — if pipe dies faster → counts as fast fail
    FAST_FAIL_MAX = 3           # consecutive fast fails → give up

    def __init__(
        self,
        cam_id: str,
        source: str,
        *,
        ffmpeg: Optional[str] = None,
        gpu: bool = False,
        reconnect_delay: float = 5.0,
        on_frame: Optional[Callable] = None,
        on_disconnect: Optional[Callable] = None,
        on_reconnect: Optional[Callable] = None,
    ):
        self.cam_id = cam_id
        self.source = source
        self.ffmpeg = ffmpeg or find_ffmpeg()
        self.gpu = gpu
        self.reconnect_delay = reconnect_delay

        # Callbacks: (cam_id, ...) signatures
        self.on_frame = on_frame                # (cam_id, frame)
        self.on_disconnect = on_disconnect      # (cam_id, error_msg)
        self.on_reconnect = on_reconnect        # (cam_id, probe_info)

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_time: float = 0.0
        self._frame_count: int = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._pipe: Optional[FFmpegPipe] = None

        # Stream info (populated after first successful probe)
        self.width: int = 0
        self.height: int = 0
        self.codec: str = "h264"
        self.fps: float = 0.0
        self.connected: bool = False

    def start(self):
        """Start background reader thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name=f"cam-{self.cam_id}", daemon=True
        )
        self._thread.start()

    def stop(self):
        """Stop background thread and release resources."""
        self._running = False
        if self._pipe:
            self._pipe.stop()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def grab(self) -> Optional[np.ndarray]:
        """Get the latest frame (thread-safe). Returns None if no frame yet."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def grab_with_info(self) -> tuple:
        """Returns (frame_or_None, frame_time, frame_count)."""
        with self._lock:
            f = self._frame.copy() if self._frame is not None else None
            return f, self._frame_time, self._frame_count

    @property
    def frame_count(self) -> int:
        return self._frame_count

    # -- internal --

    def _run(self):
        """Main loop: probe → connect → read → reconnect."""
        fast_fails = 0

        while self._running:
            # Probe
            log.info("[%s] Probing %s ...", self.cam_id, _mask_url(self.source))
            info = probe_stream(self.source, ffmpeg=self.ffmpeg)

            if info["width"] == 0 or info["height"] == 0:
                log.warning(
                    "[%s] Probe failed (no resolution), retrying in %.0fs",
                    self.cam_id, self.reconnect_delay,
                )
                if self.on_disconnect:
                    self.on_disconnect(self.cam_id, "probe failed")
                self._sleep(self.reconnect_delay)
                continue

            self.width = info["width"]
            self.height = info["height"]
            self.codec = info["codec"]
            self.fps = info["fps"]

            log.info(
                "[%s] %dx%d %s %.1ffps, GPU=%s",
                self.cam_id, self.width, self.height,
                self.codec.upper(), self.fps, self.gpu,
            )

            if self.on_reconnect:
                self.on_reconnect(self.cam_id, info)

            # Connect
            pipe = FFmpegPipe(
                self.source,
                self.width, self.height,
                ffmpeg=self.ffmpeg,
                gpu=self.gpu,
                codec=self.codec,
            )
            pipe.start()
            self._pipe = pipe
            self.connected = True
            connect_time = time.monotonic()

            # Read loop
            while self._running:
                ok, frame = pipe.read()
                if not ok:
                    break
                with self._lock:
                    self._frame = frame
                    self._frame_time = time.time()
                    self._frame_count += 1

                if self.on_frame:
                    self.on_frame(self.cam_id, frame)

            # Disconnected
            pipe.stop()
            self._pipe = None
            self.connected = False

            elapsed = time.monotonic() - connect_time
            if elapsed < self.FAST_FAIL_THRESHOLD:
                fast_fails += 1
                log.warning(
                    "[%s] Pipe died after %.1fs (fast fail %d/%d)",
                    self.cam_id, elapsed, fast_fails, self.FAST_FAIL_MAX,
                )
                if fast_fails >= self.FAST_FAIL_MAX:
                    log.error(
                        "[%s] Too many fast failures — possible GPU/driver issue. Stopping.",
                        self.cam_id,
                    )
                    if self.on_disconnect:
                        self.on_disconnect(self.cam_id, "fast fail limit reached")
                    self._running = False
                    return
            else:
                fast_fails = 0

            if self.on_disconnect:
                self.on_disconnect(self.cam_id, f"pipe ended after {elapsed:.1f}s")

            if self._running:
                log.info(
                    "[%s] Reconnecting in %.0fs ...",
                    self.cam_id, self.reconnect_delay,
                )
                self._sleep(self.reconnect_delay)

    def _sleep(self, seconds: float):
        """Interruptible sleep."""
        deadline = time.monotonic() + seconds
        while self._running and time.monotonic() < deadline:
            time.sleep(min(0.5, max(0, deadline - time.monotonic())))


# ---------------------------------------------------------------------------
# MultiCameraReader — N cameras, unified API
# ---------------------------------------------------------------------------

class MultiCameraReader:
    """Manage multiple CameraStream instances.

    Usage:
        with MultiCameraReader(gpu=True) as reader:
            reader.add("cam1", "rtsp://...")
            reader.add("cam2", "rtsp://...")
            reader.start_all()

            frame = reader.grab("cam1")
            all_frames = reader.grab_all()

    Or without context manager:
        reader = MultiCameraReader()
        reader.add(...)
        reader.start_all()
        ...
        reader.stop_all()
    """

    def __init__(
        self,
        *,
        ffmpeg: Optional[str] = None,
        gpu: bool = False,
        reconnect_delay: float = 5.0,
        on_frame: Optional[Callable] = None,
        on_disconnect: Optional[Callable] = None,
        on_reconnect: Optional[Callable] = None,
    ):
        self.ffmpeg = ffmpeg or find_ffmpeg()
        self.gpu = gpu
        self.reconnect_delay = reconnect_delay
        self.on_frame = on_frame
        self.on_disconnect = on_disconnect
        self.on_reconnect = on_reconnect

        self._cameras: dict[str, CameraStream] = {}

    def add(self, cam_id: str, source: str, **kwargs):
        """Add a camera. Extra kwargs override defaults (gpu, reconnect_delay, etc.)."""
        if cam_id in self._cameras:
            raise ValueError(f"Camera '{cam_id}' already exists")

        cam = CameraStream(
            cam_id,
            source,
            ffmpeg=kwargs.get("ffmpeg", self.ffmpeg),
            gpu=kwargs.get("gpu", self.gpu),
            reconnect_delay=kwargs.get("reconnect_delay", self.reconnect_delay),
            on_frame=kwargs.get("on_frame", self.on_frame),
            on_disconnect=kwargs.get("on_disconnect", self.on_disconnect),
            on_reconnect=kwargs.get("on_reconnect", self.on_reconnect),
        )
        self._cameras[cam_id] = cam
        return cam

    def remove(self, cam_id: str):
        """Stop and remove a camera."""
        cam = self._cameras.pop(cam_id, None)
        if cam:
            cam.stop()

    def start_all(self):
        """Start all cameras."""
        for cam in self._cameras.values():
            cam.start()

    def stop_all(self):
        """Stop all cameras."""
        for cam in self._cameras.values():
            cam.stop()

    def grab(self, cam_id: str) -> Optional[np.ndarray]:
        """Get latest frame from a specific camera."""
        cam = self._cameras.get(cam_id)
        if cam is None:
            return None
        return cam.grab()

    def grab_all(self) -> dict:
        """Get latest frames from all cameras. {cam_id: frame_or_None}."""
        return {cid: cam.grab() for cid, cam in self._cameras.items()}

    def get_camera(self, cam_id: str) -> Optional[CameraStream]:
        """Get CameraStream instance by ID."""
        return self._cameras.get(cam_id)

    @property
    def cameras(self) -> dict:
        """Read-only dict of {cam_id: CameraStream}."""
        return dict(self._cameras)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop_all()

    def __del__(self):
        self.stop_all()


# ===========================================================================
# CLI
# ===========================================================================

def _cli_probe(args):
    """CLI: probe a stream."""
    ffmpeg = find_ffmpeg(args.ffmpeg)
    print(f"ffmpeg: {ffmpeg}")
    print(f"source: {_mask_url(args.source)}")
    print()

    info = probe_stream(args.source, ffmpeg=ffmpeg)

    if info["width"] == 0:
        print("FAILED — could not detect stream info")
        # Show raw output for debugging
        if info["raw"]:
            print("\nffmpeg output (last 10 lines):")
            for line in info["raw"].strip().split("\n")[-10:]:
                print(f"  {line.rstrip()}")
        return 1

    print(f"  Resolution : {info['width']}x{info['height']}")
    print(f"  Codec      : {info['codec'].upper()}")
    print(f"  FPS        : {info['fps']:.1f}")
    return 0


def _cli_test(args):
    """CLI: read N frames, measure throughput."""
    ffmpeg = find_ffmpeg(args.ffmpeg)
    print(f"ffmpeg: {ffmpeg}")
    print(f"source: {_mask_url(args.source)}")
    print(f"GPU   : {args.gpu}")
    print()

    # Probe
    print("Probing...", end=" ", flush=True)
    info = probe_stream(args.source, ffmpeg=ffmpeg)
    if info["width"] == 0:
        print("FAILED")
        return 1
    print(f"{info['width']}x{info['height']} {info['codec'].upper()} {info['fps']:.0f}fps")

    # Connect
    print("Connecting...", end=" ", flush=True)
    pipe = FFmpegPipe(
        args.source, info["width"], info["height"],
        ffmpeg=ffmpeg, gpu=args.gpu, codec=info["codec"],
    )
    pipe.start()

    t0 = time.time()
    ok, frame = pipe.read()
    if not ok:
        print(f"FAILED ({time.time() - t0:.1f}s)")
        pipe.stop()
        return 1
    print(f"OK ({time.time() - t0:.1f}s)")

    # Read frames
    n_frames = args.frames
    print(f"\nReading {n_frames} frames...", end=" ", flush=True)
    t0 = time.time()
    good = 1  # already read one
    for _ in range(n_frames - 1):
        ok, f = pipe.read()
        if ok:
            good += 1
            frame = f
        else:
            break
    elapsed = time.time() - t0
    fps = good / elapsed if elapsed > 0 else 0

    pipe.stop()

    print(f"{good}/{n_frames} OK")
    print(f"  Throughput : {fps:.1f} fps")
    print(f"  Per frame  : {elapsed / max(good, 1) * 1000:.1f} ms")
    print(f"  Frame shape: {frame.shape}")

    return 0


def _cli_preview(args):
    """CLI: live preview window (requires OpenCV)."""
    try:
        import cv2
    except ImportError:
        print("ERROR: OpenCV (cv2) is required for preview mode")
        return 1

    ffmpeg = find_ffmpeg(args.ffmpeg)
    print(f"source: {_mask_url(args.source)}")
    print(f"GPU   : {args.gpu}")
    print()

    # Probe
    info = probe_stream(args.source, ffmpeg=ffmpeg)
    if info["width"] == 0:
        print("FAILED — could not probe stream")
        return 1

    print(f"{info['width']}x{info['height']} {info['codec'].upper()} {info['fps']:.0f}fps")

    # Connect
    pipe = FFmpegPipe(
        args.source, info["width"], info["height"],
        ffmpeg=ffmpeg, gpu=args.gpu, codec=info["codec"],
    )
    pipe.start()

    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
    w, h = info["width"], info["height"]
    if w > 1920:
        cv2.resizeWindow("Preview", w // 2, h // 2)
    else:
        cv2.resizeWindow("Preview", w, h)

    print("Press 'q' or ESC to quit")

    frame_count = 0
    t0 = time.time()

    while True:
        ok, frame = pipe.read()
        if not ok:
            print("Stream ended / connection lost")
            break

        frame_count += 1
        elapsed = time.time() - t0
        fps = frame_count / elapsed if elapsed > 0 else 0

        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | Frame: {frame_count}",
            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2,
        )
        cv2.imshow("Preview", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

        if args.duration > 0 and elapsed >= args.duration:
            break

    cv2.destroyAllWindows()
    pipe.stop()

    print(f"Frames: {frame_count}, avg FPS: {fps:.1f}")
    return 0


def _cli_record(args):
    """CLI: record a clip."""
    ffmpeg = find_ffmpeg(args.ffmpeg)
    source = args.source
    duration = args.duration
    output = args.output

    print(f"source  : {_mask_url(source)}")
    print(f"duration: {duration}s")
    print(f"output  : {output}")
    print(f"GPU     : {args.gpu}")
    print()

    # Probe to detect codec
    info = probe_stream(source, ffmpeg=ffmpeg)
    codec = info["codec"]
    print(f"Codec: {codec.upper()}")

    # Build ffmpeg command
    cmd = [ffmpeg, "-y", "-rtsp_transport", "tcp"]

    if args.gpu:
        cuvid = "hevc_cuvid" if codec == "hevc" else "h264_cuvid"
        cmd += ["-hwaccel", "cuda", "-c:v", cuvid]
        cmd += ["-i", source, "-t", str(duration)]
        cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]
        print(f"Decode: {cuvid}, Encode: h264_nvenc")
    else:
        cmd += ["-i", source, "-t", str(duration), "-c:v", "copy"]

    cmd += ["-c:a", "copy", "-movflags", "+faststart", output]

    print("Recording...", end=" ", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=duration + 30,
            creationflags=_SUBPROCESS_FLAGS,
        )
        elapsed = time.time() - t0

        if result.returncode == 0 and Path(output).exists():
            size_mb = Path(output).stat().st_size / 1024 / 1024
            print(f"OK ({elapsed:.1f}s, {size_mb:.1f} MB)")
            print(f"  File: {output}")
            return 0
        else:
            print(f"FAILED (code {result.returncode})")
            for line in result.stderr.strip().split("\n")[-3:]:
                print(f"  {line.rstrip()}")
            return 1
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT ({duration + 30}s)")
        return 1


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ffmpeg_reader — Universal RTSP/video reader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ffmpeg", default=None,
        help="Path to ffmpeg binary (auto-detected if omitted)",
    )
    sub = parser.add_subparsers(dest="command")

    # probe
    p_probe = sub.add_parser("probe", help="Probe stream info (codec, resolution, fps)")
    p_probe.add_argument("source", help="RTSP URL or video file path")

    # test
    p_test = sub.add_parser("test", help="Read N frames, measure throughput")
    p_test.add_argument("source", help="RTSP URL or video file path")
    p_test.add_argument("--gpu", action="store_true", help="Use GPU decode")
    p_test.add_argument("--frames", type=int, default=60, help="Number of frames to read")

    # preview
    p_preview = sub.add_parser("preview", help="Live preview window (requires OpenCV)")
    p_preview.add_argument("source", help="RTSP URL or video file path")
    p_preview.add_argument("--gpu", action="store_true", help="Use GPU decode")
    p_preview.add_argument("--duration", type=float, default=0, help="Auto-stop after N seconds (0=infinite)")

    # record
    p_record = sub.add_parser("record", help="Record a clip")
    p_record.add_argument("source", help="RTSP URL or video file path")
    p_record.add_argument("--gpu", action="store_true", help="Use GPU encode (h264_nvenc)")
    p_record.add_argument("--duration", type=int, default=10, help="Duration in seconds")
    p_record.add_argument("--output", default="clip.mp4", help="Output file path")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "probe": _cli_probe,
        "test": _cli_test,
        "preview": _cli_preview,
        "record": _cli_record,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
