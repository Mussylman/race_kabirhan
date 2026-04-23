"""
shm_reader.py — POSIX shared memory reader for DeepStream detection results.

Attaches to /rv_detections shared memory created by the DeepStream C++ pipeline.
Parses binary Detection structs into CameraDetections objects compatible with
the existing VoteEngine + FusionEngine pipeline.

Protocol:
    - C++ writes detection results into a fixed-layout shared memory segment
    - C++ increments write_seq atomically and calls sem_post
    - Python calls sem_timedwait, reads if seq changed
    - ~30KB ring buffer, no serialization overhead

Usage:
    reader = SharedMemoryReader()
    reader.attach()

    while running:
        results = reader.read()  # blocks up to 200ms
        if results:
            for cam_det in results:
                # cam_det is a CameraDetections object
                fusion.update([cam_det])

    reader.detach()
"""

import time
import struct
import ctypes
import ctypes.util
import logging
import mmap
from typing import Optional

from .detections import CameraDetections
from .log_utils import slog, throttle, agg, LOG_GEOMETRY

log = logging.getLogger("pipeline.shm_reader")

# ── Constants (must match deepstream/src/config.h) ──────────────────

SHM_NAME = "/rv_detections"
SEM_NAME = "/rv_detections_sem"

MAX_CAMERAS    = 25
MAX_DETECTIONS = 20
NUM_COLORS     = 5
CAM_ID_LEN     = 16

COLOR_NAMES = ["blue", "green", "purple", "red", "yellow"]

# Struct sizes (must match C++ packed structs)
DETECTION_SIZE    = 56   # 6 floats + uint32 + float + 5 floats + uint32 = 56 bytes
CAMERA_SLOT_SIZE  = CAM_ID_LEN + 8 + 4 + 4 + 4 + 4 + (DETECTION_SIZE * MAX_DETECTIONS)
SHM_HEADER_SIZE   = 8 + 4 + 4  # write_seq(8) + num_cameras(4) + reserved(4)
SHM_TOTAL_SIZE    = SHM_HEADER_SIZE + (CAMERA_SLOT_SIZE * MAX_CAMERAS)

# struct format strings
# Detection: x1,y1,x2,y2,center_x,det_conf (6f), color_id (I), color_conf (f),
#            color_probs[5] (5f), track_id (I)
DETECTION_FMT = "<6fIf5fI"
assert struct.calcsize(DETECTION_FMT) == DETECTION_SIZE

# CameraSlot header: cam_id[16] (16s), timestamp_us (Q), frame_w (I), frame_h (I),
#                     num_detections (I), _pad (I)
CAMERA_SLOT_HEADER_FMT = "<16sQIIII"
CAMERA_SLOT_HEADER_SIZE = struct.calcsize(CAMERA_SLOT_HEADER_FMT)

# ShmHeader: write_seq (Q), num_cameras (I), _reserved (I)
SHM_HEADER_FMT = "<QII"

# ── POSIX SHM/SEM via ctypes ───────────────────────────────────────

_librt = None

def _load_librt():
    """Load librt for shm_open / sem_open / sem_timedwait."""
    global _librt
    if _librt is not None:
        return _librt

    librt_path = ctypes.util.find_library("rt")
    if librt_path:
        _librt = ctypes.CDLL(librt_path, use_errno=True)
    else:
        _librt = ctypes.CDLL("librt.so.1", use_errno=True)
    return _librt


class _Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


# ── SharedMemoryReader ─────────────────────────────────────────────

class SharedMemoryReader:
    """Reads detection results from POSIX shared memory written by DeepStream C++."""

    def __init__(self, timeout_ms: int = 200):
        """
        Args:
            timeout_ms: sem_timedwait timeout in milliseconds.
        """
        self.timeout_ms = timeout_ms
        self._shm_fd: int = -1
        self._shm_buf: Optional[mmap.mmap] = None
        self._sem = None
        self._last_seq: int = 0
        self._attached = False
        self._librt = None
        self._libc = None
        self._libpthread = None

    def attach(self) -> bool:
        """Attach to existing shared memory and semaphore.

        Returns True on success. The C++ DeepStream process must be running first.
        """
        import os
        import errno

        self._librt = _load_librt()

        # Open shared memory
        O_RDONLY = 0
        shm_name = SHM_NAME.encode()

        self._librt.shm_open.restype = ctypes.c_int
        self._librt.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]

        self._shm_fd = self._librt.shm_open(shm_name, O_RDONLY, 0o666)
        if self._shm_fd < 0:
            err = ctypes.get_errno()
            log.error("shm_open(%s) failed: errno=%d (%s)", SHM_NAME, err, os.strerror(err))
            return False

        # mmap the shared memory
        try:
            self._shm_buf = mmap.mmap(self._shm_fd, SHM_TOTAL_SIZE,
                                      mmap.MAP_SHARED, mmap.PROT_READ)
        except Exception as e:
            log.error("mmap failed: %s", e)
            os.close(self._shm_fd)
            self._shm_fd = -1
            return False

        # Open semaphore
        self._librt.sem_open.restype = ctypes.c_void_p
        self._librt.sem_open.argtypes = [ctypes.c_char_p, ctypes.c_int]

        sem_name = SEM_NAME.encode()
        self._sem = self._librt.sem_open(sem_name, 0)
        SEM_FAILED = ctypes.c_void_p(-1).value
        if self._sem == SEM_FAILED or self._sem is None:
            err = ctypes.get_errno()
            log.error("sem_open(%s) failed: errno=%d (%s)", SEM_NAME, err, os.strerror(err))
            self._shm_buf.close()
            os.close(self._shm_fd)
            self._shm_fd = -1
            self._shm_buf = None
            return False

        self._attached = True
        log.info("Attached to SHM '%s' (%d bytes), SEM '%s'",
                 SHM_NAME, SHM_TOTAL_SIZE, SEM_NAME)
        return True

    def detach(self):
        """Detach from shared memory and semaphore."""
        import os

        if self._sem is not None and self._librt is not None:
            self._librt.sem_close(ctypes.c_void_p(self._sem))
            self._sem = None

        if self._shm_buf is not None:
            self._shm_buf.close()
            self._shm_buf = None

        if self._shm_fd >= 0:
            os.close(self._shm_fd)
            self._shm_fd = -1

        self._attached = False
        log.info("Detached from SHM")

    def read(self) -> Optional[list[CameraDetections]]:
        """Wait for new data and parse detection results.

        Blocks up to timeout_ms. Returns None if no new data.
        Returns list of CameraDetections if new data is available.
        """
        if not self._attached:
            return None

        # sem_timedwait
        if not self._wait_semaphore():
            return None

        # Check if sequence changed
        self._shm_buf.seek(0)
        header_data = self._shm_buf.read(SHM_HEADER_SIZE)
        write_seq, num_cameras, _ = struct.unpack(SHM_HEADER_FMT, header_data)

        if write_seq == self._last_seq:
            return None  # No new data

        self._last_seq = write_seq

        # Parse camera slots
        results = []
        num_cameras = min(num_cameras, MAX_CAMERAS)

        for i in range(num_cameras):
            offset = SHM_HEADER_SIZE + i * CAMERA_SLOT_SIZE
            self._shm_buf.seek(offset)

            # Read camera slot header
            slot_header = self._shm_buf.read(CAMERA_SLOT_HEADER_SIZE)
            (cam_id_raw, timestamp_us, frame_w, frame_h,
             num_dets, _pad) = struct.unpack(CAMERA_SLOT_HEADER_FMT, slot_header)

            cam_id = cam_id_raw.rstrip(b'\x00').decode('ascii', errors='replace')
            num_dets = min(num_dets, MAX_DETECTIONS)

            ts_capture = timestamp_us / 1e6

            if num_dets == 0:
                # No detections — still create empty CameraDetections for stale tracking
                cam_result = CameraDetections(cam_id, frame_w, frame_h)
                cam_result.timestamp = ts_capture
                cam_result.frame_seq = write_seq
                results.append(cam_result)
                continue

            cam_result = CameraDetections(cam_id, frame_w, frame_h)
            cam_result.timestamp = ts_capture
            cam_result.frame_seq = write_seq

            # Read detections
            for j in range(num_dets):
                det_data = self._shm_buf.read(DETECTION_SIZE)
                (x1, y1, x2, y2, center_x, det_conf,
                 color_id, color_conf,
                 p0, p1, p2, p3, p4,
                 track_id) = struct.unpack(DETECTION_FMT, det_data)

                # Map color_id to color name
                if color_id < NUM_COLORS:
                    color = COLOR_NAMES[color_id]
                else:
                    color = "unknown"

                # Build prob_dict
                prob_dict = {
                    COLOR_NAMES[k]: round(p, 4)
                    for k, p in enumerate([p0, p1, p2, p3, p4])
                }

                det = {
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'center_x': float(center_x),
                    'det_conf': float(det_conf),
                    'color': color,
                    'conf': float(color_conf),
                    'prob_dict': prob_dict,
                    'cam_id': cam_id,
                    'track_id': int(track_id),
                }
                cam_result.add(det)

            # SHM_READ log — throttled to 1/2s per camera (every frame when LOG_TIMING)
            if num_dets > 0:
                now = time.time()
                age_ms = (now - ts_capture) * 1000
                agg.record_shm(cam_id, age_ms)
                key = f"SHM_READ:{cam_id}"
                extra: dict = {"dets": num_dets, "ts_capture": ts_capture, "age_ms": age_ms}
                if LOG_GEOMETRY and cam_result.detections:
                    first = cam_result.detections[0]
                    extra["bbox0"] = str(first.get("bbox", "?"))
                if throttle.allow(key, interval=2.0):
                    slog("SHM_READ", cam_id, write_seq, now, **extra)

            results.append(cam_result)

        agg.flush_if_due()
        return results if results else None

    def _wait_semaphore(self) -> bool:
        """Wait on semaphore with timeout. Returns True if signaled."""
        import os
        import errno

        # Compute absolute timeout
        now = time.time()
        deadline = now + self.timeout_ms / 1000.0
        ts = _Timespec()
        ts.tv_sec = int(deadline)
        ts.tv_nsec = int((deadline - int(deadline)) * 1e9)

        self._librt.sem_timedwait.restype = ctypes.c_int
        self._librt.sem_timedwait.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Timespec)]

        ret = self._librt.sem_timedwait(ctypes.c_void_p(self._sem), ctypes.byref(ts))
        if ret != 0:
            err = ctypes.get_errno()
            if err == errno.ETIMEDOUT:
                return False  # Normal timeout
            log.warning("sem_timedwait error: errno=%d (%s)", err, os.strerror(err))
            return False

        return True

    @property
    def is_attached(self) -> bool:
        return self._attached

    @property
    def last_seq(self) -> int:
        return self._last_seq
