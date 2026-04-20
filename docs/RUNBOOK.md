# Race Vision — RUNBOOK

Короткая инструкция: что запускать, какие env-vars, что НЕ трогать.

---

## 1. Что активно (whitelist)

### DeepStream pipeline (Python через pyservicemaker)

```
deepstream/
  main.py                 — entry point
  pipeline.py             — pipeline + DetectionProbe
  config.py               — paths, mux defaults
  rv_plugin.py            — ctypes binding к .so
  diag.py                 — logging
  configs/
    nvinfer_racevision.txt — PGIE (YOLO)
    sgie_color.txt         — SGIE (color classifier)
    labels_color.txt       — 5 цветов
    tracker_iou.yml        — tracker (отключён по умолчанию)
  src/ (C++ plugin)
    yolo_parser.cpp/.h     — PGIE bbox parser
    color_classifier_parser.cpp — SGIE softmax+argmax (Phase 9)
    color_infer.cpp/.h     — CUDA kernels
    shm_writer.cpp/.h      — SHM writer
    plugin.cpp             — C API
    config.h               — shared structs
  build/libnvdsinfer_racevision.so — собранный плагин
  CMakeLists.txt
```

### Backend API

```
api/server.py               — FastAPI + WebSocket /ws
api/deepstream_pipeline.py  — SHM reader + inference loop + live fusion
api/shared.py               — COLOR_TO_HORSE маппинг
api/camera_io.py
```

### Pipeline (Python общие)

```
pipeline/shm_reader.py      — POSIX SHM reader
pipeline/fusion.py          — EMA position fusion
pipeline/track_topology.py  — pixel_x → global_track_m
pipeline/vote_engine.py     — finish-line ordering
pipeline/detections.py, camera_manager.py, log_utils.py
```

### Frontend

```
Kabirhan-Frontend/          — React + Vite + Zustand
```

### Tools (активные)

```
tools/rv_run.py                    — runs/ folder helper (Phase 9)
tools/debug_detect_classify.py     — debug видео с PyTorch (Phase 9)
tools/export_color_with_norm.py    — экспорт ONNX с вшитой нормой (dormant)
tools/train_color_v3.py            — обучение классификатора
```

### Модели (используются)

```
models/color_classifier_v4.pt        — веса
models/color_classifier_v4.onnx      — SGIE (без вшитой нормы)
models/yolo11s_person_960.onnx       — PGIE
models/*.engine                       — auto-rebuild из ONNX
```

### Configs (камеры)

```
configs/cameras_live.json                — production RTSP
configs/cameras_test_files.json          — тест из файлов (порядок по ID)
configs/cameras_test_files_ordered.json  — тест, порядок по появлению жокеев
configs/go2rtc_live.yaml                 — go2rtc live streams
```

---

## 2. Что НЕ использовать (archived)

```
archive/cpp_legacy/         ← pre-Phase 4 C++ (main.cpp, pipeline.cpp и т.д.)
archive/root_models/        ← старые yolo weights из корня
archive/notes/              ← старые session-логи
models/archive/             ← v1/v2/v3 классификаторы, jockey_yolov11s
configs/archive/            ← старые камера-конфиги
deepstream/configs/archive/ ← старые nvinfer-конфиги
tools/archive/              ← старые тулы
```

**Правило:** если нужно — копируй из `archive/` обратно, не линкуй напрямую.

---

## 3. Команды запуска (production)

### Порядок: DeepStream → API → Frontend

```bash
cd ~/"Рабочий стол/Ipodrom-Project/user/race_vision"

# 1. DeepStream pipeline (пишет в /dev/shm/rv_detections)
python3 -m deepstream.main --cameras configs/cameras_test_files_ordered.json

# 2. Backend API (в другом терминале)
python3 -m api.server --config configs/cameras_test_files_ordered.json \
                     --deepstream --auto-start --file-mode

# 3. Frontend (в третьем терминале)
cd Kabirhan-Frontend && npm run dev
```

Доступ:
- WebSocket: `ws://localhost:8000/ws`
- Frontend: `http://localhost:5173`
- MJPEG preview: `http://localhost:8000/stream/cam1`
- REST stats: `http://localhost:8000/api/stats`

### 3a. Форсировать разрешение Hikvision-камер (в составе API)

API-сервер может сам через ISAPI держать главный поток на заданном разрешении
(проверка и re-apply каждые N секунд). Credentials берутся из RTSP-URL:

```bash
python3 -m api.server --config configs/cameras_live_ordered.json \
                     --deepstream --auto-start \
                     --enforce-resolution 2560x1440 \
                     --resolution-interval 60
```

| Флаг | По умолчанию | Что делает |
|---|---|---|
| `--enforce-resolution WxH` | отключён | целевое разрешение (например `2560x1440`) |
| `--resolution-channel` | `101` | `101`=main, `102`=sub |
| `--resolution-interval` | `60` | как часто перепроверять (сек, min 10) |

Работает только на RTSP-конфигах (`file://` игнорируется). Для одноразового
применения без API: `python3 tools/set_camera_resolution.py` (тот же протокол).

---

## 4. Env-vars (для конкретной гонки)

| Env var | По умолчанию | Что делает |
|---|---|---|
| `RV_ACTIVE_COLORS` | `green,red,yellow` | Цвета силков сегодня. Остальное → UNK (фильтр зрителей). |
| `RV_INVERT_TRACK` | `1` | Инвертировать ось трассы (leftmost = лидер). `0` если жокеи движутся слева→направо. |
| `RV_ROI_FILE` | `configs/camera_roi_normalized.json` | JSON с полигонами ROI (нормализованные 0-1). Детекция центром вне ROI → отбрасывается. Если файла нет — фильтр не применяется. |
| `RV_DEBUG_PROBE` | `0` | Verbose probe, отключает OSD (диагностика). |
| `RV_DUMP_CSV` | unset | Путь к CSV для всех детекций. Для оффлайн-анализа. |

## 4a. Редактирование ROI (каждая камера → полигон "где трасса")

Инструмент-редактор рисует полигон поверх реального стоп-кадра с каждой камеры.

```bash
cd ~/"Рабочий стол/Ipodrom-Project/user/race_vision"
python3 tools/roi_server.py
# Автоматически снимет по 1 кадру с каждой RTSP-камеры в /tmp/roi_frames/
# и запустит HTTP-редактор на http://localhost:8899
```

В браузере: ← / → переключение камер, клик = точка, Enter = закрыть полигон,
Ctrl+S = сохранить в `configs/camera_roi.json`. Для нормализации (чтобы
ROI работала на любом mux resolution):

```bash
python3 -c "
import json
from pathlib import Path
r = json.loads(Path('configs/camera_roi.json').read_text())
# frame size: 1280x720 for our recorded files, 2688x1520 for live Hik cams.
# roi_editor сохраняет в оригинальных пикселях кадра. Вот нормализация:
sizes = {}  # fill per cam: {'cam-01': (1280, 720), ...}
import subprocess
for cam_id in r:
    # Read frame size from saved JPEG
    try:
        out = subprocess.check_output(['identify', '-format', '%wx%h',
                                       f'/tmp/roi_frames/{cam_id}.jpg'])
        w, h = map(int, out.decode().strip().split('x'))
        sizes[cam_id] = (w, h)
    except Exception:
        sizes[cam_id] = (1280, 720)
norm = {}
for cam_id, polys in r.items():
    w, h = sizes[cam_id]
    norm[cam_id] = [[{'x': p['x']/w, 'y': p['y']/h} for p in poly] for poly in polys]
Path('configs/camera_roi_normalized.json').write_text(json.dumps(norm, indent=2))
print(f'wrote normalized ROI for {len(norm)} cameras')
"
```

ROI применяется автоматически при следующем запуске DS pipeline. Если надо
переопределить путь:

```bash
RV_ROI_FILE=/path/to/my_roi.json python3 -m deepstream.main --cameras ...
```

Пример боевого запуска:

```bash
RV_ACTIVE_COLORS=red,green,yellow,blue RV_INVERT_TRACK=1 \
  python3 -m deepstream.main --cameras configs/cameras_live.json
```

---

## 5. Когда пересобирать C++ plugin

Если менялись файлы в `deepstream/src/`:

```bash
cd deepstream/build && cmake --build .
```

Результат: `deepstream/build/libnvdsinfer_racevision.so`.

---

## 6. Когда удалять engine файлы

При **обновлении TRT** или **смене сервера** — TRT engine serialization несовместим между версиями.

```bash
rm models/*.engine
# DS пересоберёт из ONNX при следующем запуске (1-3 мин для PGIE).
```

**Не удалять ONNX** — они portable.

---

## 7. Диагностика если что-то сломалось

### Пайплайн не запускается
```bash
ls -la /dev/shm/ | grep rv       # SHM должен быть
tail -20 /tmp/rv_ds.log          # лог DS если через nohup/background
```

### SGIE `classifier_items=0`
Значит автопарсер DS 9.0 не сработал. Нужен кастомный парсер (уже есть:
`src/color_classifier_parser.cpp`, зарегистрирован в `sgie_color.txt`
через `parse-classifier-func-name`). Проверь что плагин пересобран.

### API не может найти SHM
Сначала запускай DS, потом API. API ретраит `shm_open` каждые 2 сек.

### Rankings не меняются / `fusion_ok=0`
Проверь `_inference_loop` в `api/deepstream_pipeline.py` — есть ли
live-fusion tick (5 Hz, Phase 10). Без него fusion вызывается только
на vote-complete.

### Жокей-лидер не тот
`RV_INVERT_TRACK=0` если они движутся слева→направо на экране.

---

## 8. Каждый запуск сохраняется в `runs/NN_<tag>_<ts>/`

```
runs/NN_main_<cam>_<ts>/
  log.txt              — stdout пайплайна
  meta.txt             — параметры
  rv_pipeline.log      — structured logger
  debug_<cam>/         — frames + crops от debug-скрипта (single-cam only)
```

Сквозная нумерация для debug-скрипта и main.py.

---

## 9. Полезные пути

| Что | Путь |
|---|---|
| SHM | `/dev/shm/rv_detections` + `sem.rv_detections_sem` |
| JSONL детекций (API) | `/tmp/race_analysis/detections.jsonl` |
| Runs | `runs/` в корне проекта |
| Engine cache | `models/*.engine` (auto-rebuild) |
