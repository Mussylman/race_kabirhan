# Race Vision — Hybrid C++/Python Architecture

Status: current as of Phase 8 of the DeepStream → Python migration.
Supersedes the pure-C++ design described in
[PLAN_HYBRID_MIGRATION.md](PLAN_HYBRID_MIGRATION.md) and
[AUDIT_DEEPSTREAM_DETECTION.md](AUDIT_DEEPSTREAM_DETECTION.md).

---

## 1. Overview

Race Vision is a real-time 25-camera jockey tracking system running on a
single RTX 5070 Ti server. After Phases 1–7 the stack is **hybrid**:
GStreamer pipeline orchestration moved from ~900 lines of C++
(`main.cpp + pipeline.cpp + analysis_pipeline.cpp + dual_pipeline.cpp +
trigger_pipeline.cpp`) into ~250 lines of Python built on NVIDIA's
**pyservicemaker** (DeepStream 9.0 Python API). GPU-critical code —
custom YOLO bbox parser, CUDA preprocessing kernels, POSIX SHM writer —
stays in C++ inside a single combined plugin `libnvdsinfer_racevision.so`
and is called from Python via `ctypes`. This removes boilerplate
(GMainLoop, bus handlers, manual pad linking) while keeping inference
and the hot write path native.

---

## 2. Process Layout

```
┌────────────────────────────────────────────────────────────────────┐
│  Python process #1 — deepstream/pipeline.py (pyservicemaker)       │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  GStreamer: uridecodebin×N → nvstreammux → PGIE → SGIE →      │ │
│  │             probe → fakesink                                  │ │
│  │                              │                                │ │
│  │                              ▼  ctypes → libnvdsinfer_...so   │ │
│  │                         rv_shm_write_camera + rv_shm_commit   │ │
│  └────────────────────────────────────┬──────────────────────────┘ │
└───────────────────────────────────────┼────────────────────────────┘
                                        │   POSIX SHM  /rv_detections
                                        ▼              /rv_detections_sem
┌────────────────────────────────────────────────────────────────────┐
│  Python process #2 — api/server.py (FastAPI + uvicorn, port 8000)  │
│     └─ pipeline/shm_reader.py (posix_ipc) → fusion → WebSocket     │
└────────────────────────────────────────┬───────────────────────────┘
                                         │  ws:// detections stream
                                         ▼
                        React frontend (Kabirhan-Frontend, Vite)
```

Single-writer / single-reader SHM: the DeepStream pipeline is the only
writer; the FastAPI server is the only reader. A named semaphore
(`/rv_detections_sem`) protects the monotonic `write_seq` counter.

---

## 3. C++ component — `libnvdsinfer_racevision.so`

Everything that needs direct GPU / kernel-level access lives here.
Built from `deepstream/src/` via `deepstream/CMakeLists.txt`.

| File | Role |
|---|---|
| [`yolo_parser.cpp`](../deepstream/src/yolo_parser.cpp) | Custom YOLOv8 bbox parser. Registered with nvinfer via `parse-bbox-func-name=NvDsInferParseYoloV8`; runs inside the PGIE element. |
| [`color_infer.cpp`](../deepstream/src/color_infer.cpp) | Standalone TRT engine + CUDA preprocessing kernels for torso color classification. **Currently unused** in the runtime pipeline (SGIE handles color), but kept exported as `rv_color_create / rv_color_destroy / rv_color_classify` for future direct-call use cases. |
| [`shm_writer.cpp`](../deepstream/src/shm_writer.cpp) | POSIX SHM writer (`shm_open` + `mmap` + named semaphore). Exported as `rv_shm_create / rv_shm_destroy / rv_shm_write_camera / rv_shm_commit`. Called from Python on every batch. |
| [`plugin.cpp`](../deepstream/src/plugin.cpp) | Thin `extern "C"` wrappers used by the ctypes binding; also exports `rv_get_*` size-probe functions that `rv_plugin.py` calls at load time to sanity-check struct layout. |
| [`config.h`](../deepstream/src/config.h) | Shared SHM struct layout (`Detection`, `CameraSlot`, `ShmHeader`). The single source of truth — [`rv_plugin.py`](../deepstream/rv_plugin.py) mirrors it byte-for-byte with `_pack_ = 1` ctypes.Structure and `static_assert` matches Python `assert`. |

Legacy files still in `deepstream/src/` (`main.cpp`, `pipeline.cpp`,
`analysis_pipeline.cpp`, `dual_pipeline.cpp`, `trigger_pipeline.cpp`,
`trigger_shm.cpp`, `diag_logger.cpp`) are **not** compiled into the new
`.so`. They remain for reference only and will be removed in a later
cleanup pass.

---

## 4. Python component

New files under `deepstream/`:

| File | Role |
|---|---|
| [`pipeline.py`](../deepstream/pipeline.py) | pyservicemaker pipeline builder, probe, CLI. Replaces ~900 lines of C++ orchestration. |
| [`rv_plugin.py`](../deepstream/rv_plugin.py) | ctypes binding: `Detection`, `CameraSlot` structs, `RVPlugin` wrapper with `create_shm / write_camera / commit / destroy_shm`. Calls `rv_get_*` probes to assert layout matches `config.h`. |
| `main.py` | Phase 6 entry point: loads config, wires logging, spawns the pipeline subprocess. |
| `config.py` | Camera / mux / logging config helpers used by `main.py`. |
| `diag.py` | Runtime diagnostics (per-camera FPS, detections/frame, SHM write rate). |

SHM **reading** side lives under `pipeline/`:

| File | Role |
|---|---|
| [`pipeline/shm_reader.py`](../pipeline/shm_reader.py) | `posix_ipc` reader that mmaps `/rv_detections`, waits on seq change, yields `CameraSlot` dicts to the FastAPI server. |

---

## 5. Pipeline Topology

Built in [`build_pipeline()`](../deepstream/pipeline.py) using
pyservicemaker `Pipeline.add()` / `Pipeline.link()`:

```
src_0 uridecodebin ─┐
src_1 uridecodebin ─┤
   ...              ├─► nvstreammux (batch=N, 800x800)
src_{N-1}           ┘        │
                             ▼
                  nvinfer "infer"  (PGIE, unique-id=1)
                    config: deepstream/configs/nvinfer_racevision.txt
                    parse-bbox-func-name=NvDsInferParseYoloV8
                    custom-lib-path=.../libnvdsinfer_racevision.so
                             │
                             ▼
                  nvinfer "sgie_color"  (SGIE, unique-id=2)
                    config: deepstream/configs/sgie_color.txt
                    network-type=1, process-mode=2
                    operate-on-gie-id=1, operate-on-class-ids=0
                             │
                             ▼
                  Probe "rv_probe"  (DetectionProbe.handle_metadata)
                    - filters: class_id==0, min_h=40, min_ar=0.2
                    - reads classifier_items → color_id / color_conf
                    - packs Detection + CameraSlot structs
                    - rv_shm_write_camera() per frame
                    - rv_shm_commit() at end of batch
                             │
                             ▼
                          fakesink
```

SGIE is optional; passing `--sgie ""` disables it and falls back to
`COLOR_UNKNOWN`. The probe attaches to the **last** inference element
so it sees classifier_meta.

---

## 6. Configuration Files

| Path | Purpose |
|---|---|
| [`configs/cameras_live.json`](../configs/cameras_live.json) | Production camera list — 25 RTSP URLs from go2rtc. |
| [`configs/cameras_test_files.json`](../configs/cameras_test_files.json) | Offline testing — local `.mp4` file URIs. Default in `pipeline.py`. |
| [`deepstream/configs/nvinfer_racevision.txt`](../deepstream/configs/nvinfer_racevision.txt) | PGIE (YOLOv8 person). 960×960, FP16, `cluster-mode=4` (NMS inside parser), `pre-cluster-threshold=0.35`. |
| [`deepstream/configs/sgie_color.txt`](../deepstream/configs/sgie_color.txt) | SGIE (color classifier v4). 128×128, FP16, ImageNet normalization, `classifier-threshold=0.30`, `classifier-async-mode=0` so the probe sees results synchronously. |
| [`deepstream/configs/labels_color.txt`](../deepstream/configs/labels_color.txt) | 5 labels in ColorId order: blue, green, purple, red, yellow. |

---

## 7. Build

```bash
cd deepstream && mkdir -p build && cd build
cmake .. \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=120
make nvdsinfer_racevision -j4
# → deepstream/build/libnvdsinfer_racevision.so
```

`sm_120` = Blackwell / RTX 5070 Ti. `rv_plugin.py` asserts struct sizes
match `config.h` at load time, so a stale build will fail fast.

---

## 8. Run

Activate the venv first: `source .venv/bin/activate`.

```bash
# 25 live RTSP cameras
python -m deepstream.pipeline \
    --cameras configs/cameras_live.json \
    --config  deepstream/configs/nvinfer_racevision.txt \
    --sgie    deepstream/configs/sgie_color.txt

# Single file, no color SGIE (smoke test)
python -m deepstream.pipeline \
    --cameras configs/cameras_test_files.json \
    --limit 1 \
    --sgie ""
```

In a second shell start the API/WebSocket server:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

---

## 9. Migration History

- **Phase 1** — critical detection bug fixes (bbox filters, class_id).
- **Phase 2** — color classification + speed calculation fixes.
- **Phase 3** — threshold tuning, speed units normalized.
- **Phase 4** — combined plugin `.so`: parser + color + shm + ctypes entry points.
- **Phase 5** — pyservicemaker pipeline in Python; ~900 → ~250 lines.
- **Phase 6** — `main.py` / `config.py` / `diag.py` entry point + diagnostics.
- **Phase 7** — CMake cleanup; only `nvdsinfer_racevision` target remains.
- **Phase 8** — docs update (this file + `requirements_deepstream.txt`).
