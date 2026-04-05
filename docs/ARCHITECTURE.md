# Race Vision — Архитектура системы

Система трекинга позиций лошадей на скачках в реальном времени.
25 аналитических камер покрывают 2500м трек. 5 лошадей определяются по цвету жокея.

---

## Общая схема

```
┌─────────────────────────────────────────────────────────────────────┐
│                        25 RTSP камер (аналитика)                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
     ┌─────────────────┐               ┌──────────────────┐
     │   go2rtc         │               │  DeepStream C++  │
     │   (RTSP relay)   │               │  (GPU inference) │
     │   port 1984      │               │                  │
     └────────┬─────────┘               └────────┬─────────┘
              │                                  │
              │ WebRTC/MSE                  POSIX SHM
              │                          /rv_detections
              │                                  │
              │                         ┌────────┴─────────┐
              │                         │  Python Backend   │
              │                         │  FastAPI :8000    │
              │                         │  SHM Reader       │
              │                         │  Fusion Engine    │
              │                         └────────┬─────────┘
              │                                  │
              │                            WebSocket
              │                         (live_detections)
              │                                  │
              ▼                                  ▼
     ┌──────────────────────────────────────────────────┐
     │              React Frontend :5173                 │
     │    go2rtc видео ◄─────── bbox overlay (canvas)    │
     │    WebRTC player          detectionBuffer          │
     └──────────────────────────────────────────────────┘
```

---

## Компоненты

### 1. DeepStream C++ Pipeline

Высокопроизводительный GPU-пайплайн на GStreamer + TensorRT.

**Бинарник:** `deepstream/build/race_vision_deepstream`

**Режимы работы:**
- **Single pipeline** (по умолчанию): все камеры через YOLOv11s → nvtracker → ColorCNN
- **Dual pipeline** (`--dual`): trigger (YOLOv8n@640) + analysis (YOLOv8s@800) с valve-gating

**Конвейер GStreamer (single mode):**
```
uridecodebin × N → nvstreammux (1520×1520) → nvinfer (YOLOv11s@800) → nvtracker (IOU) → probe → fakesink
                                                                                          │
                                                                               ColorCNN (TRT, CUDA crop)
                                                                                          │
                                                                                    SHM writer
                                                                                  /rv_detections
```

**Модели (рабочие):**
| Модель | Engine | Назначение |
|--------|--------|-----------|
| YOLOv11s | `jockey_yolov11s.onnx_b25_gpu0_fp16.engine` | Детекция жокеев (1 класс) |
| ColorCNN v3 | `color_classifier_v3.engine` | Классификация цвета (5 классов) |

**Конфиги:**
| Файл | Описание |
|------|----------|
| `deepstream/configs/nvinfer_jockey.txt` | YOLOv11s, batch=25, 800×800, FP16, threshold=0.15 |
| `deepstream/configs/tracker_iou.yml` | IOU tracker, probationAge=1 |

**Команда запуска:**
```bash
./deepstream/build/race_vision_deepstream \
  --config cameras.json \
  --yolo-engine deepstream/configs/nvinfer_jockey.txt \
  --color-engine models/color_classifier_v3.engine \
  --mux-width 1520 --mux-height 1520 \
  --file-mode   # для файлов, без -- для RTSP
```

**Исходники:**
| Файл | Назначение |
|------|-----------|
| `deepstream/src/main.cpp` | Точка входа, CLI парсинг |
| `deepstream/src/pipeline.cpp/h` | GStreamer pipeline, probe callback |
| `deepstream/src/yolo_parser.cpp` | Custom NvDsInferParseYoloV8 (авто-детект layout) |
| `deepstream/src/color_infer.cpp/h` | TRT ColorCNN + CUDA crop/resize/normalize |
| `deepstream/src/shm_writer.cpp/h` | POSIX SHM запись в `/rv_detections` |
| `deepstream/src/config.h` | Бинарный протокол SHM (Detection=56B, CameraSlot, ShmHeader) |
| `deepstream/src/diag_logger.cpp/h` | Диагностический лог |

---

### 2. POSIX Shared Memory Protocol

Бинарный протокол между C++ и Python через `/rv_detections`.

**Структура SHM (~30KB):**
```
ShmHeader (16 bytes)
├── write_seq     : uint64  — монотонный счётчик (атомарно инкрементируется после записи)
├── num_cameras   : uint32  — количество камер
├── _reserved     : uint32
└── cameras[25]   : CameraSlot[]

CameraSlot (1160 bytes)
├── cam_id        : char[16]  — "cam-01"
├── timestamp_us  : uint64    — микросекунды с epoch
├── frame_width   : uint32
├── frame_height  : uint32
├── num_detections: uint32    — 0..20
├── _pad          : uint32
└── detections[20]: Detection[]

Detection (56 bytes)
├── x1, y1, x2, y2  : float × 4  — bbox в mux-координатах
├── center_x         : float      — (x1+x2)/2
├── det_conf         : float      — YOLO confidence
├── color_id         : uint32     — 0=blue, 1=green, 2=purple, 3=red, 4=yellow, 255=unknown
├── color_conf       : float      — top-1 color confidence
├── color_probs[5]   : float × 5  — softmax [blue, green, purple, red, yellow]
└── track_id         : uint32     — nvtracker persistent ID
```

**Синхронизация:** семафор `/rv_detections_sem`, writer делает `sem_post()` после каждой записи.

---

### 3. Python Backend (FastAPI)

**Точка входа:** `api/server.py` (port 8000)

**Ключевые классы:**

| Класс/Модуль | Файл | Назначение |
|-------------|------|-----------|
| `DeepStreamSubprocess` | `api/deepstream_pipeline.py` | Запуск C++ binary как subprocess |
| `DeepStreamPipeline` | `api/deepstream_pipeline.py` | 3-thread: SHM reader → inference → broadcast |
| `SharedMemoryReader` | `pipeline/shm_reader.py` | Чтение SHM, парсинг в CameraDetections |
| `FusionEngine` | `pipeline/fusion.py` | Мульти-камерное слияние, глобальный ranking |
| `CameraDetections` | `pipeline/detections.py` | Data class для детекций одной камеры |

**Потоки данных:**
```
SHM Reader Thread ──► DetectionBuffer ──► Inference Thread ──► FusionEngine
                                                                    │
                                              WebSocket broadcast ◄─┘
                                              (live_detections msg)
```

**WebSocket сообщения (backend → frontend):**
```json
{
  "type": "live_detections",
  "data": {
    "cam-01": {
      "frame_w": 1520, "frame_h": 1520,
      "ts_capture": 1775025890.15,
      "frame_seq": 2041,
      "detections": [
        {"color": "red", "conf": 100, "track_id": 1, "bbox": [1116, 696, 1183, 808]}
      ]
    }
  },
  "ts_server_send": 1775025890.21
}
```

**Env-флаги логирования:**
| Переменная | Что включает |
|-----------|-------------|
| `LOG_LEVEL` | Уровень лога (DEBUG/INFO) |
| `LOG_TIMING` | Тайминг каждого кадра без throttle |
| `LOG_GEOMETRY` | Координаты bbox в логах |
| `LOG_FUSION` | Accepted fusion updates |
| `LOG_STATS` | Per-second STATS_1S агрегация |

---

### 4. React Frontend

**Стек:** React 19 + TypeScript + Zustand + Framer Motion + Vite

**Ключевые файлы:**

| Файл | Назначение |
|------|-----------|
| `services/backendConnection.ts` | WebSocket клиент, heartbeat, reconnect |
| `services/detectionBuffer.ts` | Per-camera temporal buffer для bbox overlay |
| `utils/frameLogger.ts` | Frontend structured logging |
| `components/operator/CameraGrid.tsx` | Сетка камер, canvas overlay, DRAW/CLEAR логика |
| `config/go2rtc.ts` | Конфиг go2rtc WebRTC плеера |

**Рендеринг bbox:**
1. WebSocket `live_detections` → `pushDetectionFrame()` в buffer
2. `requestAnimationFrame` loop → `selectFrame(camId, video.currentTime)`
3. Если frame свежий → рисует bbox на canvas overlay поверх видео
4. Если stale (>1000ms) → очищает canvas

**Env-флаги:**
| Переменная | Что включает |
|-----------|-------------|
| `VITE_LOG_WS` | WS_RECV логи |
| `VITE_LOG_BUFFER` | BUFFER_PUSH, FRAME_SELECT |
| `VITE_LOG_VIDEO` | Video element events |
| `VITE_LOG_TIMING` | Timing details |
| `VITE_LOG_STATS` | Per-second stats |

Runtime: `window.__LOG_FLAGS = { ws: true, buffer: true }`

---

### 5. go2rtc (Video Relay)

RTSP → WebRTC/MSE passthrough без транскодирования.

**Конфиг:** `configs/go2rtc.yaml` или `go2rtc_25cam.yaml`
**Port:** 1984

Для файлового режима — exec с `-stream_loop -1`:
```yaml
streams:
  cam-05:
    - exec:ffmpeg -stream_loop -1 -re -i /path/to/video.mp4 -c copy -f rtsp {output}
```

---

## Docker

**3 сервиса** в `docker-compose.yml`:

| Сервис | Образ | Port |
|--------|-------|------|
| go2rtc | go2rtc base | 1984 |
| backend | Multi-stage: DeepStream 7.1 builder → runtime + Python | 8000 |
| frontend | Node build → Nginx | 3000 |

**Dockerfile** (multi-stage):
1. Builder: компилирует C++ binary + custom parser lib
2. Runtime: DeepStream 7.1 + Python + pip deps из `requirements_deepstream.txt`
3. C++ binary: `/app/bin/race_vision_deepstream`, parser: `/app/lib/`

**КРИТИЧНО:** TRT engine собирать ТОЛЬКО внутри Docker (TensorRT 10.3.0), не на хосте.

---

## Модели

### Детекция: YOLOv11s (jockey)
- **Задача:** 1 класс — жокей на лошади
- **Input:** 800×800 RGB, FP16
- **Output:** `[5, 13125]` — 4 bbox + 1 class confidence
- **Обучение:** `tools/train_jockey.py`
- **Файлы:** `models/jockey_yolov11s.pt` → `.onnx` → `.engine`

### Классификация цвета: ColorCNN v3
- **Задача:** 5 классов — blue, green, purple, red, yellow
- **Input:** 64×64 RGB crop торса жокея, ImageNet normalization
- **Output:** softmax[5]
- **Обучение:** `tools/train_color_classifier.py`
- **Файлы:** `models/color_classifier_v3.pt` → `.onnx` → `.engine`

### Трекинг: IOU Tracker (nvtracker)
- **Алгоритм:** NvMultiObjectTracker с IOU matching
- **probationAge:** 1 (минимум кадров для подтверждения трека)
- **Конфиг:** `deepstream/configs/tracker_iou.yml`

---

## Pipeline Tuning

| Параметр | Значение | Где |
|---------|---------|-----|
| `MIN_BBOX_HEIGHT` | 35px | `pipeline.h` |
| `MIN_ASPECT_RATIO` | 0.25 | `pipeline.h` |
| `MAX_ASPECT_RATIO` | 2.5 | `pipeline.h` |
| `COLOR_EMA_ALPHA` | 0.35 | `pipeline.cpp` |
| `MIN_COLOR_CONF` | 0.60 | `pipeline.cpp` |
| `STALE_THRESHOLD_MS` | 1000 | `CameraGrid.tsx` |
| mux resolution | 1520×1520 | CLI `--mux-width/height` |

---

## Structured Logging (сквозное)

Уникальный ключ: `camera_id + frame_seq + ts_capture`

### Стадии

```
C++ DeepStream
   │
   ▼ (SHM write)
SHM_READ ──► LIVE_UPDATE ──► WS_SEND ──► WS_RECV ──► BUFFER_PUSH ──► FRAME_SELECT ──► DRAW/CLEAR
 Python        Python         Python      Browser      Browser         Browser          Browser
```

### Формат лога (backend)
```
SHM_READ       cam=cam-05  frame=14823  ts=1743417600.142  dets=3  age_ms=2.1
LIVE_UPDATE    cam=cam-05  frame=14823  ts=1743417600.145  colors=['red','green','yellow']
WS_SEND        cam=cam-05  frame=14823  ts=1743417600.150  clients=3  dets=3  age_ms=8.0
```

### Утилиты
- `pipeline/log_utils.py` — `slog()`, `ThrottleMap`, `PerCameraAggregator`
- `Kabirhan-Frontend/src/utils/frameLogger.ts` — `flog()`, `throttledLog()`, `frontendAgg`

---

## Инструменты

| Скрипт | Назначение |
|--------|-----------|
| `tools/live_frame_saver.py` | Realtime: сохраняет кадры с bbox из SHM параллельно с DeepStream |
| `tools/extract_analysis_frames.py` | Post-run: извлекает кадры из JSONL лога |
| `tools/train_jockey.py` | Обучение YOLO на жокеев |
| `tools/train_color_classifier.py` | Обучение ColorCNN |
| `tools/export_trt.py` | Экспорт .pt → .onnx → .engine |
| `tools/build_engine.py` | Сборка TRT engine |
| `tools/ffmpeg_reader.py` | RTSP/file reader с GPU decode |
| `scripts/run_all.sh` | Запуск всей системы |

---

## Камеры

### Аналитические (25 шт)
- Покрывают 2500м трек с перекрытием ~10м
- RTSP потоки, разрешение 1280×720 (2880×1620 нативное)
- Конфиг: `cameras_example.json` (JSON с track_start/track_end для каждой)

### Дисплейные (4 шт)
- PTZ камеры для трансляции
- Через go2rtc → WebRTC в браузер

### Файловый режим
- Записи в `/home/user/recordings/yaris_YYYYMMDD_HHMMSS/`
- `kamera_NN_*.mp4` — по файлу на камеру
- Запуск с `--file-mode` + `file://` URL в конфиге

---

## Запуск (полная система)

```bash
# 1. go2rtc
docker compose up go2rtc -d

# 2. DeepStream
./deepstream/build/race_vision_deepstream \
  --config cameras_example.json \
  --yolo-engine deepstream/configs/nvinfer_jockey.txt \
  --color-engine models/color_classifier_v3.engine \
  --mux-width 1520 --mux-height 1520

# 3. Python backend
.venv/bin/python -m api.server --config cameras_example.json --deepstream --auto-start

# 4. Frontend
cd Kabirhan-Frontend && npx vite --host 0.0.0.0
```

Или через скрипт: `bash scripts/run_all.sh --25cam`

---

## GPU: NVIDIA RTX 3060 12GB

| Нагрузка | VRAM |
|---------|------|
| YOLOv11s (batch=25, 800×800, FP16) | ~2 GB |
| ColorCNN v3 (batch≤128, 64×64) | ~0.2 GB |
| nvstreammux (25 sources, 1520×1520) | ~3 GB |
| CUDA decode (25 RTSP) | ~2 GB |
| **Итого** | **~7-8 GB из 12 GB** |
