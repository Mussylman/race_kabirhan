"""
deepstream_logic.py — Мост между C++ DeepStream и Python бизнес-логикой.

Архитектура (из DOCS.md):

    C++ DeepStream (GPU)                    Python (CPU)
    ┌───────────────────┐                   ┌────────────────────────────┐
    │ TriggerPipeline   │──/rv_trigger──→   │ TriggerShmReader           │
    │ YOLOv8n @ 640     │  (active_mask)    │ → CameraManager.activate() │
    │ 25 камер, 3 fps   │                   │                            │
    ├───────────────────┤                   ├────────────────────────────┤
    │ AnalysisPipeline  │──/rv_detections─→ │ DetectionProcessor         │
    │ YOLOv8s @ 800     │  (bbox+color)     │ → VoteEngine (per cam)     │
    │ valve-gated       │                   │ → FusionEngine (global)    │
    └───────────────────┘                   │ → ranking_update → WS      │
                                            └────────────────────────────┘

Жизненный цикл камеры:
    IDLE → (trigger: horses detected) → ACTIVE → (vote ready OR timeout) → COMPLETED

Использование:
    logic = DeepStreamLogic(cameras_json="cameras.json")
    logic.start()            # запускает 2 потока: trigger_reader + detection_reader
    ...
    ranking = logic.get_ranking()   # для WebSocket broadcast
    status  = logic.get_status()    # для оператора
    ...
    logic.reset()            # новая гонка
    logic.stop()
"""

import time
import struct
import ctypes
import ctypes.util
import logging
import mmap
import threading
from typing import Optional, Callable

from .camera_manager import CameraManager
from .track_topology import TrackTopology
from .detections import CameraDetections
from .fusion import FusionEngine
from .vote_engine import VoteEngine
from .shm_reader import (
    SharedMemoryReader,
    SHM_NAME, SEM_NAME,
    COLOR_NAMES, NUM_COLORS,
)

log = logging.getLogger("pipeline.deepstream_logic")

# ── Trigger SHM constants (match deepstream/src/config.h) ──────────

TRIGGER_SHM_NAME = "/rv_trigger"
TRIGGER_SEM_NAME = "/rv_trigger_sem"
TRIGGER_MAX_CAMERAS = 25

# struct TriggerShmHeader:
#   write_seq (Q), active_mask (I), num_cameras (I),
#   detection_counts[25] (25I), timestamp_us (Q)
TRIGGER_SHM_FMT = f"<QII{TRIGGER_MAX_CAMERAS}IQ"
TRIGGER_SHM_SIZE = struct.calcsize(TRIGGER_SHM_FMT)


# =====================================================================
# TriggerShmReader — читает /rv_trigger, обновляет CameraManager
# =====================================================================

class TriggerShmReader:
    """Читает активные камеры из /rv_trigger SHM.

    C++ TriggerPipeline пишет:
      - active_mask  (bitmask: бит i = камера i активна)
      - detection_counts[25] (сколько людей на каждой камере)

    Python обновляет CameraManager.activate/deactivate.
    """

    def __init__(self, timeout_ms: int = 200):
        self.timeout_ms = timeout_ms
        self._shm_fd: int = -1
        self._shm_buf: Optional[mmap.mmap] = None
        self._sem = None
        self._last_seq: int = 0
        self._attached = False
        self._librt = None

    def attach(self) -> bool:
        import os
        self._librt = self._load_librt()

        # shm_open
        self._librt.shm_open.restype = ctypes.c_int
        self._librt.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]
        self._shm_fd = self._librt.shm_open(TRIGGER_SHM_NAME.encode(), 0, 0o666)
        if self._shm_fd < 0:
            return False

        try:
            self._shm_buf = mmap.mmap(self._shm_fd, TRIGGER_SHM_SIZE,
                                      mmap.MAP_SHARED, mmap.PROT_READ)
        except Exception:
            os.close(self._shm_fd)
            self._shm_fd = -1
            return False

        # sem_open
        self._librt.sem_open.restype = ctypes.c_void_p
        self._librt.sem_open.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self._sem = self._librt.sem_open(TRIGGER_SEM_NAME.encode(), 0)
        SEM_FAILED = ctypes.c_void_p(-1).value
        if self._sem == SEM_FAILED or self._sem is None:
            self._shm_buf.close()
            os.close(self._shm_fd)
            self._shm_fd = -1
            self._shm_buf = None
            return False

        self._attached = True
        log.info("Attached to trigger SHM '%s'", TRIGGER_SHM_NAME)
        return True

    def read(self) -> Optional[dict]:
        """Блокирует до timeout_ms. Возвращает {cam_index: det_count} + active_mask."""
        if not self._attached:
            return None

        if not self._wait_sem():
            return None

        self._shm_buf.seek(0)
        data = self._shm_buf.read(TRIGGER_SHM_SIZE)
        parsed = struct.unpack(TRIGGER_SHM_FMT, data)

        write_seq = parsed[0]
        if write_seq == self._last_seq:
            return None
        self._last_seq = write_seq

        active_mask = parsed[1]
        num_cameras = parsed[2]
        detection_counts = parsed[3:3 + TRIGGER_MAX_CAMERAS]
        timestamp_us = parsed[3 + TRIGGER_MAX_CAMERAS]

        return {
            "active_mask": active_mask,
            "num_cameras": num_cameras,
            "detection_counts": list(detection_counts[:num_cameras]),
            "timestamp": timestamp_us / 1e6,
        }

    def detach(self):
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

    @property
    def is_attached(self) -> bool:
        return self._attached

    def _wait_sem(self) -> bool:
        import os
        import errno

        class _Timespec(ctypes.Structure):
            _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]

        deadline = time.time() + self.timeout_ms / 1000.0
        ts = _Timespec()
        ts.tv_sec = int(deadline)
        ts.tv_nsec = int((deadline - int(deadline)) * 1e9)

        self._librt.sem_timedwait.restype = ctypes.c_int
        self._librt.sem_timedwait.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Timespec)]

        ret = self._librt.sem_timedwait(ctypes.c_void_p(self._sem), ctypes.byref(ts))
        return ret == 0

    @staticmethod
    def _load_librt():
        librt_path = ctypes.util.find_library("rt")
        if librt_path:
            return ctypes.CDLL(librt_path, use_errno=True)
        return ctypes.CDLL("librt.so.1", use_errno=True)


# =====================================================================
# DeepStreamLogic — полная логика гонки поверх DeepStream
# =====================================================================

class DeepStreamLogic:
    """Два SHM-ридера + VoteEngine + FusionEngine = полная логика гонки.

    Поток данных:
        /rv_trigger  → TriggerShmReader → CameraManager (IDLE↔ACTIVE)
        /rv_detections → SharedMemoryReader → VoteEngine → CameraManager (COMPLETED)
                                            → FusionEngine → ranking
    """

    def __init__(
        self,
        camera_manager: CameraManager,
        topology: TrackTopology,
        colors: list[str] = None,
        *,
        frame_skip: int = 6,           # ~25fps SHM → ~4fps эффективных
        max_analysis_sec: float = 8.0,  # таймаут анализа камеры
        grace_period_sec: float = 2.0,  # после появления всех 5 цветов
    ):
        self.camera_manager = camera_manager
        self.topology = topology
        self.colors = colors or ["green", "red", "yellow"]

        self.fusion = FusionEngine(topology, colors=self.colors)

        # SHM readers
        self._trigger_reader = TriggerShmReader(timeout_ms=200)
        self._detection_reader = SharedMemoryReader(timeout_ms=200)

        # Per-camera vote engines
        self._vote_engines: dict[str, VoteEngine] = {}

        # Camera lifecycle tracking
        self._cam_first_analysis: dict[str, float] = {}
        self._cam_all_visible_time: dict[str, float] = {}
        self._cam_completed: set[str] = set()
        self._cam_frame_count: dict[str, int] = {}

        # Parameters
        self.frame_skip = frame_skip
        self.max_analysis_sec = max_analysis_sec
        self.grace_period_sec = grace_period_sec

        # Callbacks
        self.on_ranking_update: Optional[Callable[[list[dict]], None]] = None
        self.on_camera_complete: Optional[Callable[[str, list[str]], None]] = None

        # Results queue for WebSocket broadcast
        self.pending_camera_results: list[dict] = []

        # Live detection status (for operator panel)
        self.live_detections: dict[str, list] = {}

        # Threads
        self._running = False
        self._trigger_thread: Optional[threading.Thread] = None
        self._detection_thread: Optional[threading.Thread] = None

        # Stats
        self.stats = {
            "trigger_cycles": 0,
            "detection_cycles": 0,
            "frames_voted": 0,
            "cameras_completed": 0,
        }

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self):
        """Запуск двух потоков: trigger reader + detection reader."""
        self._running = True

        self._trigger_thread = threading.Thread(
            target=self._trigger_loop, daemon=True, name="DS-TriggerReader")
        self._detection_thread = threading.Thread(
            target=self._detection_loop, daemon=True, name="DS-DetectionReader")

        self._trigger_thread.start()
        self._detection_thread.start()
        log.info("DeepStreamLogic started (2 reader threads)")

    def stop(self):
        self._running = False
        self._trigger_reader.detach()
        self._detection_reader.detach()
        log.info("DeepStreamLogic stopped")

    def reset(self):
        """Сброс для новой гонки."""
        self.fusion.reset()
        for engine in self._vote_engines.values():
            engine.reset()
        self._cam_first_analysis.clear()
        self._cam_all_visible_time.clear()
        self._cam_completed.clear()
        self._cam_frame_count.clear()
        self.pending_camera_results.clear()
        self.camera_manager.reset_completed()
        self.stats["cameras_completed"] = 0
        log.info("DeepStreamLogic reset (new race)")

    # ── Thread 1: Trigger reader ─────────────────────────────────────

    def _trigger_loop(self):
        """Читает /rv_trigger → обновляет CameraManager.activate/deactivate.

        Жизненный цикл:
            IDLE ──(trigger: count > 0)──→ ACTIVE
            ACTIVE ──(no detections + cooldown)──→ IDLE
            COMPLETED камеры игнорируются триггером.
        """
        self._attach_with_retry(self._trigger_reader, "trigger")

        cameras = self.camera_manager.get_analytics_cameras()
        cam_ids = [c.cam_id for c in cameras]

        while self._running:
            result = self._trigger_reader.read()
            if result is None:
                self._reattach_if_needed(self._trigger_reader, "trigger")
                continue

            self.stats["trigger_cycles"] += 1

            # Обновить CameraManager из bitmask + counts
            trigger_results = {}
            for i, cam_id in enumerate(cam_ids):
                if i < result["num_cameras"]:
                    trigger_results[cam_id] = result["detection_counts"][i]
                else:
                    trigger_results[cam_id] = 0

            self.camera_manager.update_trigger_results(trigger_results)

    # ── Thread 2: Detection reader ───────────────────────────────────

    def _detection_loop(self):
        """Читает /rv_detections → VoteEngine → FusionEngine → ranking.

        Для каждой камеры:
            1. Frame skip (25fps → 4fps)
            2. VoteEngine.submit_frame (сортировка по X, голосование)
            3. Проверка завершения:
               - is_result_ready() + grace period → COMPLETED
               - vote_frames >= 8 + confident → COMPLETED
               - timeout 8s + есть голоса → COMPLETED
            4. При COMPLETED:
               - compute_result() → order (e.g. ["red", "green", "yellow"])
               - FusionEngine.update() → global ranking
               - camera_manager.mark_completed()
        """
        self._attach_with_retry(self._detection_reader, "detection")

        while self._running:
            cam_results = self._detection_reader.read()
            if cam_results is None:
                self._reattach_if_needed(self._detection_reader, "detection")
                continue

            self.stats["detection_cycles"] += 1

            # Обновить live detections (для оператора — видно всегда)
            self._update_live_detections(cam_results)

            # Обработка через VoteEngine (только во время гонки)
            for cam_det in cam_results:
                self._process_camera(cam_det)

    def _process_camera(self, cam_det: CameraDetections):
        """Обработка одной камеры: vote → check completion → fusion."""
        cam_id = cam_det.cam_id

        # Пропуск завершённых камер
        if cam_id in self._cam_completed:
            return

        if cam_det.n_detections == 0:
            return

        # Frame skip: ~25fps → ~4fps
        self._cam_frame_count[cam_id] = self._cam_frame_count.get(cam_id, 0) + 1
        if self._cam_frame_count[cam_id] % self.frame_skip != 0:
            return

        # VoteEngine: сортировка по X (правее = впереди), голосование
        engine = self._get_vote_engine(cam_id)
        assigned, weight = engine.submit_frame(cam_det.detections)
        self.stats["frames_voted"] += 1

        now = time.monotonic()

        # Трекинг первого анализа камеры
        if cam_id not in self._cam_first_analysis:
            self._cam_first_analysis[cam_id] = now
            log.info("ANALYSIS START  %s  (%d detections)", cam_id, cam_det.n_detections)

        # Трекинг появления всех цветов
        if weight > 0:
            visible = set(d['color'] for d in assigned)
            if len(visible) >= len(self.colors) and cam_id not in self._cam_all_visible_time:
                self._cam_all_visible_time[cam_id] = now
                log.info("ALL COLORS VISIBLE  %s  [%s]",
                         cam_id, ", ".join(sorted(visible)))

        # ── Проверка завершения ──────────────────────────────────
        should_complete, reason = self._check_completion(cam_id, engine, now)

        if should_complete:
            self._complete_camera(cam_id, cam_det, engine, assigned, reason)

    def _check_completion(self, cam_id: str, engine: VoteEngine, now: float) -> tuple[bool, str]:
        """4 условия завершения камеры (из DOCS.md).

        1. Confident + grace: все позиции заполнены + N сек после всех цветов
        2. Confident + frames: is_result_ready() + >= 8 кадров
        3. Timeout + votes: >= 8 сек + есть голоса
        4. Partial timeout: >= 8 сек без голосов (отправить что есть)
        """
        # Условие 1: уверенный результат + grace period
        if engine.is_result_ready():
            if cam_id in self._cam_all_visible_time:
                elapsed = now - self._cam_all_visible_time[cam_id]
                if elapsed >= self.grace_period_sec:
                    return True, f"confident + grace {elapsed:.1f}s"
            # Условие 2: уверенный результат + много кадров
            if engine.vote_frames >= 8:
                return True, f"confident + {engine.vote_frames} frames"

        # Условие 3: timeout с голосами
        if cam_id in self._cam_first_analysis:
            elapsed = now - self._cam_first_analysis[cam_id]
            if elapsed >= self.max_analysis_sec:
                if engine.vote_frames >= 2:
                    return True, f"timeout ({elapsed:.1f}s, {engine.vote_frames} votes)"
                # Условие 4: partial timeout
                return True, f"partial timeout ({elapsed:.1f}s)"

        return False, ""

    def _complete_camera(
        self,
        cam_id: str,
        cam_det: CameraDetections,
        engine: VoteEngine,
        assigned: list[dict],
        reason: str,
    ):
        """Завершение камеры: compute result → fusion → mark completed."""
        self._cam_completed.add(cam_id)
        vote_result = engine.compute_result()

        log.info("CAMERA COMPLETE  %s  order=[%s]  (%s, %d vote frames)",
                 cam_id, " > ".join(vote_result), reason, engine.vote_frames)

        # Обновить FusionEngine (один раз!)
        voted = CameraDetections(cam_id, cam_det.frame_width, cam_det.frame_height)
        voted.timestamp = cam_det.timestamp
        for d in assigned:
            voted.add(d)
        self.fusion.update([voted])

        # Пометить камеру как завершённую
        self.camera_manager.mark_completed(cam_id)
        self.stats["cameras_completed"] += 1

        # Результат для WebSocket broadcast
        self.pending_camera_results.append({
            "type": "camera_result",
            "camera_id": cam_id,
            "ranking": vote_result,
            "vote_frames": engine.vote_frames,
        })

        # Callback
        if self.on_camera_complete:
            self.on_camera_complete(cam_id, vote_result)

    # ── Queries ──────────────────────────────────────────────────────

    def get_ranking(self) -> list[dict]:
        """Текущий глобальный рейтинг (для WebSocket ranking_update)."""
        return self.fusion.get_ranking()

    def get_status(self) -> dict:
        """Полный статус для оператора."""
        return {
            "cameras": self.camera_manager.get_status(),
            "fusion": self.fusion.get_stats(),
            "completed": list(self._cam_completed),
            "live_detections": self.live_detections,
            **self.stats,
        }

    def get_vote_table(self, cam_id: str) -> list[dict]:
        """Таблица голосов для камеры (диагностика)."""
        engine = self._vote_engines.get(cam_id)
        return engine.get_vote_table() if engine else []

    # ── Internal helpers ─────────────────────────────────────────────

    def _get_vote_engine(self, cam_id: str) -> VoteEngine:
        if cam_id not in self._vote_engines:
            self._vote_engines[cam_id] = VoteEngine(self.colors)
        return self._vote_engines[cam_id]

    def _update_live_detections(self, cam_results: list[CameraDetections]):
        """Обновить карту live detections (для operator panel)."""
        live = {}
        for cam_det in cam_results:
            if cam_det.n_detections > 0:
                live[cam_det.cam_id] = [
                    {
                        "color": d.get("color", "unknown"),
                        "conf": round(d.get("conf", 0) * 100),
                        "track_id": d.get("track_id", 0),
                    }
                    for d in cam_det.detections
                ]
        self.live_detections = live

    def _attach_with_retry(self, reader, name: str):
        """Подключиться к SHM с retry (ждём пока C++ процесс запустится)."""
        while self._running:
            if hasattr(reader, 'attach') and reader.attach():
                log.info("Attached to %s SHM", name)
                return
            log.info("Waiting for %s SHM (C++ not ready)...", name)
            time.sleep(2.0)

    def _reattach_if_needed(self, reader, name: str):
        """Re-attach если SHM отвалился."""
        if not reader.is_attached:
            self._attach_with_retry(reader, name)
