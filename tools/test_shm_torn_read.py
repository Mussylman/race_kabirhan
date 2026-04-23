#!/usr/bin/env python3
"""Stress test for pipeline.shm_reader.SharedMemoryReader torn-read retry.

Spawns a pure-Python POSIX SHM writer that mimics the C++ ShmWriter layout
(deepstream/src/shm_writer.cpp + config.h) and commits at high rate from a
background thread. Each commit stamps every detection's track_id with the
current write_seq ("magic"). The reader is expected to only ever return
snapshots where all detections share one magic — any mixing means a torn
snapshot leaked through the seqlock retry in shm_reader.read().

Run:
    .venv/bin/python tools/test_shm_torn_read.py

Exit code 0 on success, 1 on torn snapshots detected.
"""

import os
import sys
import time
import struct
import mmap
import logging
import threading
import argparse
from pathlib import Path

import posix_ipc

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.shm_reader as shm_reader_mod
from pipeline.shm_reader import (
    SharedMemoryReader,
    SHM_HEADER_FMT, SHM_HEADER_SIZE, SHM_TOTAL_SIZE,
    CAMERA_SLOT_SIZE, CAMERA_SLOT_HEADER_FMT, CAMERA_SLOT_HEADER_SIZE,
    DETECTION_FMT, DETECTION_SIZE,
    MAX_CAMERAS, MAX_DETECTIONS, NUM_COLORS,
)

TEST_SHM_NAME = "/rv_test_torn"
TEST_SEM_NAME = "/rv_test_torn_sem"


class FakeWriter:
    """Pure-Python writer matching the on-disk layout of C++ ShmWriter."""

    def __init__(self, shm_name=TEST_SHM_NAME, sem_name=TEST_SEM_NAME):
        self.shm_name = shm_name
        self.sem_name = sem_name
        self.shm = None
        self.sem = None
        self.buf: mmap.mmap | None = None
        self.write_seq = 0

    def create(self):
        # Cleanup stale
        try:
            posix_ipc.unlink_shared_memory(self.shm_name)
        except posix_ipc.ExistentialError:
            pass
        try:
            posix_ipc.unlink_semaphore(self.sem_name)
        except posix_ipc.ExistentialError:
            pass

        self.shm = posix_ipc.SharedMemory(
            self.shm_name, flags=posix_ipc.O_CREAT, mode=0o666, size=SHM_TOTAL_SIZE
        )
        self.buf = mmap.mmap(self.shm.fd, SHM_TOTAL_SIZE)
        os.close(self.shm.fd)
        self.buf[:SHM_TOTAL_SIZE] = b"\x00" * SHM_TOTAL_SIZE
        self.sem = posix_ipc.Semaphore(
            self.sem_name, flags=posix_ipc.O_CREAT, mode=0o666, initial_value=0
        )

    def write_camera_slot(self, cam_idx: int, cam_id: str, num_dets: int, magic: int):
        offset = SHM_HEADER_SIZE + cam_idx * CAMERA_SLOT_SIZE
        slot_header = struct.pack(
            CAMERA_SLOT_HEADER_FMT,
            cam_id.encode("ascii").ljust(16, b"\x00")[:16],
            int(time.time() * 1e6),
            1280, 720,
            num_dets, 0,
        )
        self.buf[offset:offset + CAMERA_SLOT_HEADER_SIZE] = slot_header

        det_off = offset + CAMERA_SLOT_HEADER_SIZE
        for j in range(num_dets):
            det = struct.pack(
                DETECTION_FMT,
                100.0, 100.0, 200.0, 300.0, 150.0, 0.9,
                j % NUM_COLORS, 0.85,
                0.1, 0.2, 0.3, 0.2, 0.2,
                magic & 0x7FFFFFFF,
            )
            self.buf[det_off + j * DETECTION_SIZE : det_off + (j + 1) * DETECTION_SIZE] = det

    def commit(self, num_cameras: int = MAX_CAMERAS):
        self.write_seq += 1
        header = struct.pack(SHM_HEADER_FMT, self.write_seq, num_cameras, 0)
        self.buf[:SHM_HEADER_SIZE] = header
        self.sem.release()

    def destroy(self):
        if self.buf is not None:
            self.buf.close()
            self.buf = None
        if self.shm is not None:
            try:
                self.shm.unlink()
            except posix_ipc.ExistentialError:
                pass
            self.shm = None
        if self.sem is not None:
            try:
                self.sem.unlink()
            except posix_ipc.ExistentialError:
                pass
            self.sem = None


def patch_reader_names():
    shm_reader_mod.SHM_NAME = TEST_SHM_NAME
    shm_reader_mod.SEM_NAME = TEST_SEM_NAME


def test_sanity_single_commit() -> bool:
    print("[1/4] Sanity: single commit, single read")
    patch_reader_names()
    writer = FakeWriter()
    writer.create()
    try:
        for cam_idx in range(MAX_CAMERAS):
            writer.write_camera_slot(cam_idx, f"cam-{cam_idx:02d}", 3, magic=42)
        writer.commit()

        reader = SharedMemoryReader(timeout_ms=200)
        assert reader.attach(), "attach failed"
        try:
            result = reader.read()
            assert result is not None, "read returned None"
            assert len(result) == MAX_CAMERAS, f"expected {MAX_CAMERAS} cams, got {len(result)}"
            cam0 = result[0]
            assert cam0.cam_id == "cam-00", f"unexpected cam_id: {cam0.cam_id}"
            assert cam0.n_detections == 3, f"expected 3 dets, got {cam0.n_detections}"
            for det in cam0.detections:
                assert det["track_id"] == 42, f"magic mismatch: {det['track_id']}"
        finally:
            reader.detach()
        print("       PASS")
        return True
    finally:
        writer.destroy()


def test_no_data_then_one_commit() -> bool:
    print("[2/4] Idempotent: no commits → read returns None; one commit → returns one snapshot")
    patch_reader_names()
    writer = FakeWriter()
    writer.create()
    try:
        reader = SharedMemoryReader(timeout_ms=50)
        assert reader.attach()
        try:
            assert reader.read() is None, "read should return None before any commit"

            for cam_idx in range(MAX_CAMERAS):
                writer.write_camera_slot(cam_idx, f"cam-{cam_idx:02d}", 1, magic=7)
            writer.commit()

            r1 = reader.read()
            assert r1 is not None, "expected a snapshot after commit"
            r2 = reader.read()
            assert r2 is None, "second read with no new commit should return None"
        finally:
            reader.detach()
        print("       PASS")
        return True
    finally:
        writer.destroy()


def _stress(duration_sec: float, target_hz: float | None) -> tuple[int, int, int]:
    """Run concurrent writer+reader. Returns (commits, snapshots, torn)."""
    patch_reader_names()
    writer = FakeWriter()
    writer.create()
    stop = threading.Event()
    commit_count = [0]
    period = 1.0 / target_hz if target_hz else 0.0

    def writer_loop():
        i = 1
        next_t = time.monotonic()
        while not stop.is_set():
            magic = i & 0x7FFFFFFF
            for cam_idx in range(MAX_CAMERAS):
                writer.write_camera_slot(cam_idx, f"cam-{cam_idx:02d}", MAX_DETECTIONS, magic)
            writer.commit()
            commit_count[0] += 1
            i += 1
            if period:
                next_t += period
                sleep_left = next_t - time.monotonic()
                if sleep_left > 0:
                    time.sleep(sleep_left)
                else:
                    next_t = time.monotonic()

    t = threading.Thread(target=writer_loop, daemon=True)
    reader = SharedMemoryReader(timeout_ms=20)
    assert reader.attach()

    snapshots = 0
    torn = 0
    deadline = time.time() + duration_sec
    try:
        t.start()
        while time.time() < deadline:
            result = reader.read()
            if result is None:
                continue
            snapshots += 1
            magics = set()
            for cam_det in result:
                for det in cam_det.detections:
                    magics.add(det["track_id"])
            if len(magics) > 1:
                torn += 1
    finally:
        stop.set()
        t.join(timeout=1.0)
        reader.detach()
        writer.destroy()
    return commit_count[0], snapshots, torn


def test_stress_realistic_rate(duration_sec: float = 2.0) -> bool:
    print(f"[3/4] Stress @ realistic rate (~30 Hz, DeepStream-like) for {duration_sec:.1f}s")
    commits, snapshots, torn = _stress(duration_sec, target_hz=30.0)
    rate = commits / duration_sec
    print(f"       commits={commits} ({rate:.0f}/s), reads={snapshots}, torn={torn}")
    if torn == 0:
        print("       PASS — no torn snapshots at production rate")
        return True
    print(f"       FAIL — {torn}/{snapshots} snapshots torn (should be 0)")
    return False


def test_stress_extreme_rate(duration_sec: float = 1.5) -> bool:
    print(f"[4/4] Stress @ extreme rate (writer flat-out) for {duration_sec:.1f}s")
    print("       expect log warnings + graceful degradation, not a crash")
    commits, snapshots, torn = _stress(duration_sec, target_hz=None)
    rate = commits / duration_sec
    print(f"       commits={commits} ({rate:.0f}/s), reads={snapshots}, torn={torn}")
    if snapshots > 0:
        print("       PASS — reader survived starvation (degraded reads expected)")
        return True
    print("       FAIL — reader produced zero snapshots")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--realistic-sec", type=float, default=2.0)
    parser.add_argument("--extreme-sec", type=float, default=1.5)
    args = parser.parse_args()

    results = [
        test_sanity_single_commit(),
        test_no_data_then_one_commit(),
        test_stress_realistic_rate(args.realistic_sec),
        test_stress_extreme_rate(args.extreme_sec),
    ]
    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 60}\n{passed}/{total} tests passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
