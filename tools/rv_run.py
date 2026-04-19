"""
Central per-run folder helper.

Every run (debug script, main pipeline, anything else) gets a folder
    runs/NN_<label>/
with globally auto-incremented NN. Use inside a script:

    from tools.rv_run import new_run_dir, tee_stdout
    run_dir = new_run_dir("debug_cam13")
    tee_stdout(run_dir / "log.txt")     # stdout/stderr → terminal + file
    ...
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO_ROOT / "runs"


def _next_index() -> int:
    RUNS_ROOT.mkdir(exist_ok=True)
    existing = []
    for p in RUNS_ROOT.iterdir():
        if not p.is_dir():
            continue
        head = p.name.split("_", 1)[0]
        if head.isdigit():
            existing.append(int(head))
    return (max(existing) + 1) if existing else 1


def new_run_dir(label: str) -> Path:
    """Create runs/NN_<label>/ with globally auto-incremented NN."""
    idx = _next_index()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = label.replace("/", "_").replace(" ", "_")
    d = RUNS_ROOT / f"{idx:03d}_{safe_label}_{ts}"
    d.mkdir(parents=True, exist_ok=False)
    return d


class _Tee:
    def __init__(self, fp, orig):
        self.fp, self.orig = fp, orig

    def write(self, data):
        self.orig.write(data)
        self.fp.write(data)
        self.fp.flush()

    def flush(self):
        self.orig.flush()
        self.fp.flush()

    def isatty(self):
        return getattr(self.orig, "isatty", lambda: False)()


def tee_stdout(log_path: Path) -> None:
    """Duplicate stdout + stderr into log_path from now until process exit."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fp = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(fp, sys.stdout)
    sys.stderr = _Tee(fp, sys.stderr)


def write_meta(run_dir: Path, meta: dict) -> None:
    """Write key=value meta lines to run_dir/meta.txt."""
    (run_dir / "meta.txt").write_text(
        "\n".join(f"{k}={v}" for k, v in meta.items()) + "\n"
    )
