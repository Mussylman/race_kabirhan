"""
deepstream/rv_plugin.py — Python ctypes binding for libnvdsinfer_racevision.so

Mirrors the structs in deepstream/src/config.h and exposes thin wrappers
around the C entry points in deepstream/src/plugin.cpp.

Usage:
    from deepstream.rv_plugin import RVPlugin, CameraSlot, Detection, ColorId

    plugin = RVPlugin.load()
    shm = plugin.create_shm(["cam-01", "cam-02"])
    ...
    slot = plugin.make_camera_slot("cam-01", frame_w, frame_h, detections)
    plugin.write_camera(shm, 0, slot)
    plugin.commit(shm)
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# ── Constants (must match deepstream/src/config.h) ───────────────────
MAX_CAMERAS    = 25
MAX_DETECTIONS = 20
NUM_COLORS     = 5
CAM_ID_LEN     = 16

COLOR_BLUE    = 0
COLOR_GREEN   = 1
COLOR_PURPLE  = 2
COLOR_RED     = 3
COLOR_YELLOW  = 4
COLOR_UNKNOWN = 255

DEFAULT_LIB_PATH = (
    Path(__file__).resolve().parent / "build" / "libnvdsinfer_racevision.so"
)


# ── ctypes structs (must match config.h byte layout) ─────────────────

class Detection(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("x1",          ctypes.c_float),
        ("y1",          ctypes.c_float),
        ("x2",          ctypes.c_float),
        ("y2",          ctypes.c_float),
        ("center_x",    ctypes.c_float),
        ("det_conf",    ctypes.c_float),
        ("color_id",    ctypes.c_uint32),
        ("color_conf",  ctypes.c_float),
        ("color_probs", ctypes.c_float * NUM_COLORS),
        ("track_id",    ctypes.c_uint32),
    ]


assert ctypes.sizeof(Detection) == 56, f"Detection size {ctypes.sizeof(Detection)}"


class CameraSlot(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("cam_id",           ctypes.c_char * CAM_ID_LEN),
        ("timestamp_us",     ctypes.c_uint64),
        ("frame_width",      ctypes.c_uint32),
        ("frame_height",     ctypes.c_uint32),
        ("num_detections",   ctypes.c_uint32),
        ("source_frame_num", ctypes.c_uint32),
        ("detections",       Detection * MAX_DETECTIONS),
    ]


_EXPECTED_SLOT_SIZE = CAM_ID_LEN + 8 + 4 + 4 + 4 + 4 + 56 * MAX_DETECTIONS
assert ctypes.sizeof(CameraSlot) == _EXPECTED_SLOT_SIZE, \
    f"CameraSlot size {ctypes.sizeof(CameraSlot)} != {_EXPECTED_SLOT_SIZE}"


# ── High-level wrapper ───────────────────────────────────────────────

@dataclass
class _Symbols:
    rv_color_create:        ctypes._FuncPointer
    rv_color_destroy:       ctypes._FuncPointer
    rv_color_classify:      ctypes._FuncPointer
    rv_shm_create:          ctypes._FuncPointer
    rv_shm_destroy:         ctypes._FuncPointer
    rv_shm_write_camera:    ctypes._FuncPointer
    rv_shm_commit:          ctypes._FuncPointer


class RVPlugin:
    def __init__(self, lib: ctypes.CDLL):
        self.lib = lib

        # Color
        lib.rv_color_create.argtypes = [ctypes.c_char_p]
        lib.rv_color_create.restype  = ctypes.c_void_p

        lib.rv_color_destroy.argtypes = [ctypes.c_void_p]
        lib.rv_color_destroy.restype  = None

        lib.rv_color_classify.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.rv_color_classify.restype = ctypes.c_int

        # SHM
        lib.rv_shm_create.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_char_p)]
        lib.rv_shm_create.restype  = ctypes.c_void_p

        lib.rv_shm_destroy.argtypes = [ctypes.c_void_p]
        lib.rv_shm_destroy.restype  = None

        lib.rv_shm_write_camera.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
        lib.rv_shm_write_camera.restype  = None

        lib.rv_shm_commit.argtypes = [ctypes.c_void_p]
        lib.rv_shm_commit.restype  = None

        # Constants probes (sanity check)
        for fn in ("rv_get_max_cameras", "rv_get_max_detections",
                   "rv_get_num_colors", "rv_get_camera_slot_size",
                   "rv_get_detection_size"):
            getattr(lib, fn).restype = ctypes.c_uint32

        self._verify_layout()

    @classmethod
    def load(cls, lib_path: os.PathLike | None = None) -> "RVPlugin":
        path = Path(lib_path) if lib_path else DEFAULT_LIB_PATH
        if not path.is_file():
            raise FileNotFoundError(f"plugin .so not found: {path}")
        return cls(ctypes.CDLL(str(path)))

    def _verify_layout(self):
        if self.lib.rv_get_max_cameras()    != MAX_CAMERAS:
            raise RuntimeError("MAX_CAMERAS mismatch")
        if self.lib.rv_get_max_detections() != MAX_DETECTIONS:
            raise RuntimeError("MAX_DETECTIONS mismatch")
        if self.lib.rv_get_num_colors()     != NUM_COLORS:
            raise RuntimeError("NUM_COLORS mismatch")
        if self.lib.rv_get_detection_size() != ctypes.sizeof(Detection):
            raise RuntimeError("Detection size mismatch with C struct")
        if self.lib.rv_get_camera_slot_size() != ctypes.sizeof(CameraSlot):
            raise RuntimeError("CameraSlot size mismatch with C struct")

    # ── SHM helpers ────────────────────────────────────────────

    def create_shm(self, cam_ids: Sequence[str]) -> ctypes.c_void_p:
        n = len(cam_ids)
        if n == 0 or n > MAX_CAMERAS:
            raise ValueError(f"cam_ids must be 1..{MAX_CAMERAS}")
        arr_t = ctypes.c_char_p * n
        arr   = arr_t(*[c.encode("ascii") for c in cam_ids])
        handle = self.lib.rv_shm_create(n, arr)
        if not handle:
            raise RuntimeError("rv_shm_create failed (already mapped? perms?)")
        # Keep a reference to arr so it isn't GC'd while C reads it (rv_shm_create
        # only reads during the call, so this is just defensive).
        self._cam_ids_keepalive = arr
        return ctypes.c_void_p(handle)

    def destroy_shm(self, handle: ctypes.c_void_p):
        if handle:
            self.lib.rv_shm_destroy(handle)

    def write_camera(self, handle: ctypes.c_void_p, cam_index: int,
                     slot: CameraSlot) -> None:
        self.lib.rv_shm_write_camera(handle, cam_index, ctypes.byref(slot))

    def commit(self, handle: ctypes.c_void_p) -> None:
        self.lib.rv_shm_commit(handle)


# ── Helper: build a Detection from a bbox ────────────────────────────

def make_detection(x1: float, y1: float, x2: float, y2: float,
                   det_conf: float,
                   color_id: int = COLOR_UNKNOWN,
                   color_conf: float = 0.0,
                   color_probs: Sequence[float] | None = None,
                   track_id: int = 0) -> Detection:
    d = Detection()
    d.x1 = x1
    d.y1 = y1
    d.x2 = x2
    d.y2 = y2
    d.center_x = (x1 + x2) * 0.5
    d.det_conf = det_conf
    d.color_id = color_id
    d.color_conf = color_conf
    if color_probs is not None:
        for i in range(min(NUM_COLORS, len(color_probs))):
            d.color_probs[i] = float(color_probs[i])
    d.track_id = track_id
    return d


def make_camera_slot(cam_id: str, frame_w: int, frame_h: int,
                     timestamp_us: int,
                     detections: Sequence[Detection],
                     source_frame_num: int = 0) -> CameraSlot:
    slot = CameraSlot()
    enc = cam_id.encode("ascii")[:CAM_ID_LEN - 1]
    slot.cam_id = enc.ljust(CAM_ID_LEN, b"\x00")
    slot.timestamp_us = timestamp_us
    slot.frame_width  = frame_w
    slot.frame_height = frame_h
    n = min(len(detections), MAX_DETECTIONS)
    slot.num_detections = n
    slot.source_frame_num = int(source_frame_num) & 0xFFFFFFFF
    for i in range(n):
        slot.detections[i] = detections[i]
    return slot
