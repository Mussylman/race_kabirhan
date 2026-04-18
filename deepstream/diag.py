"""
deepstream/diag.py — Diagnostics for the Python DS pipeline.

Replaces the C++ `diag_logger.cpp` (~150 lines) with structured Python
logging + optional CSV throughput stats. Use `setup_logging()` once at
process start, then `log = get_logger("rv.pipeline")` anywhere.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional


LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-5s %(name)s | %(message)s"
LOG_DATEFMT = "%H:%M:%S"


def setup_logging(level: str = "INFO", log_dir: Optional[Path] = None) -> Path | None:
    """Configure root logger.

    @param level    INFO, DEBUG, WARNING, ERROR
    @param log_dir  optional directory to also write rv_pipeline.log
    @return         path of the file log if one was opened, else None
    """
    root = logging.getLogger()
    if root.handlers:                 # already configured
        return None
    root.setLevel(level.upper())

    fmt = logging.Formatter(LOG_FORMAT, LOG_DATEFMT)

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    file_path: Path | None = None
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_path = log_dir / "rv_pipeline.log"
        fh = logging.FileHandler(file_path)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    return file_path


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class FpsTracker:
    """Cheap rolling-window FPS counter.

    `tick()` once per processed frame. `fps()` returns the average FPS
    across the most recent `window` ticks.
    """

    def __init__(self, window: int = 100):
        self.window = window
        self.timestamps: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self.timestamps.append(time.time())

    def fps(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        span = self.timestamps[-1] - self.timestamps[0]
        if span <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / span


class CsvMetricsWriter:
    """Append rows to a CSV inside `log_dir`. Header is written on first row."""

    def __init__(self, log_dir: Path | str, name: str = "metrics.csv"):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / name
        self._writer: csv.DictWriter | None = None
        self._fh = None

    def write(self, **row) -> None:
        if self._writer is None:
            self._fh = open(self.path, "w", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None
