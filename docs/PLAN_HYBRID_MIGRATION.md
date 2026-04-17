# Race Vision: Полный план — Баг-фиксы + Гибридная архитектура

**Дата:** 2026-04-17
**Сервер:** Intel i7-14700F, RTX 5070 Ti 16GB, 32GB DDR5
**Стек:** Ubuntu 24.04, CUDA 12.8/13.2, TensorRT 10.6, DeepStream 9.0, PyTorch nightly, Ultralytics 8.4.37

---

## Зачем этот план

1. **Баг-фиксы (Phase 1-3):** Аудит показал 13 проблем, из-за которых DeepStream детектирует хуже Python. Главная причина — `model-color-format=1` (BGR вместо RGB). Исправляем все баги.

2. **Гибридная архитектура (Phase 4-8):** Переводим pipeline конфигурацию на Python (pyservicemaker DS 9.0), GPU-критичный код оставляем на C++. Итого: ~600 строк C++ вместо ~1500, плюс ~200 строк Python вместо ~900 строк C++.

---

## Phase 1: Критические баги детекции

> Время: 30 минут. Сразу тестировать — качество должно заметно вырасти.

### 1.1 model-color-format: BGR → RGB

**Файл:** `deepstream/configs/nvinfer_jockey.txt`, строка 35
**Проблема:** YOLO обучен на RGB, DeepStream подаёт BGR. Каналы R и B перевёрнуты.
**Причина:** Комментарий в архивном конфиге `# 0=BGR, 1=RGB` перевёрнут наоборот. В DeepStream: 0=RGB, 1=BGR.

```diff
- model-color-format=1
+ model-color-format=0
```

**Влияние:** Самый большой прирост качества детекции. Один символ.

---

### 1.2 Двойной NMS: убрать cluster-mode=2

**Файл:** `deepstream/configs/nvinfer_jockey.txt`, строка 38
**Проблема:** Кастомный парсер `yolo_parser.cpp:161` делает NMS с IoU=0.45. Потом nvinfer с `cluster-mode=2` делает NMS повторно. Двойное подавление убивает детекции плотно стоящих жокеев.

```diff
- cluster-mode=2
+ cluster-mode=4
```

`cluster-mode=4` = "без кластеризации, парсер сам всё сделал".

---

### 1.3 COLOR_TO_HORSE: 3 цвета → 5 цветов

**Файл:** `api/shared.py`, строки 44-50
**Проблема:** C++ pipeline классифицирует 5 цветов (blue, green, purple, red, yellow), но Python знает только 3 (red, green, yellow). Детекции blue и purple молча отбрасываются в `_build_rankings()`, `FusionEngine`, `VoteEngine`.

```diff
  COLOR_TO_HORSE = {
+     "blue":   {"id": "horse-1", "number": 1, "name": "Blue Storm",     "silkId": 1, "color": "#2563EB", "jockeyName": "Jockey 1"},
-     "red":    {"id": "horse-1", "number": 1, "name": "Red Runner",     "silkId": 1, "color": "#DC2626", "jockeyName": "Jockey 1"},
-     "green":  {"id": "horse-2", "number": 2, "name": "Green Flash",    "silkId": 2, "color": "#16A34A", "jockeyName": "Jockey 2"},
-     "yellow": {"id": "horse-3", "number": 3, "name": "Yellow Thunder", "silkId": 3, "color": "#FBBF24", "jockeyName": "Jockey 3"},
+     "green":  {"id": "horse-2", "number": 2, "name": "Green Flash",    "silkId": 2, "color": "#16A34A", "jockeyName": "Jockey 2"},
+     "purple": {"id": "horse-3", "number": 3, "name": "Purple Knight",  "silkId": 3, "color": "#9333EA", "jockeyName": "Jockey 3"},
+     "red":    {"id": "horse-4", "number": 4, "name": "Red Runner",     "silkId": 4, "color": "#DC2626", "jockeyName": "Jockey 4"},
+     "yellow": {"id": "horse-5", "number": 5, "name": "Yellow Thunder", "silkId": 5, "color": "#FBBF24", "jockeyName": "Jockey 5"},
  }
```

**Примечание:** Имена лошадей, silkId и jockeyName — заглушки. Заменить на реальные данные.

---

### 1.4 visible_colors >= 5: мёртвая логика завершения

**Файл:** `api/deepstream_pipeline.py`, строка 442
**Проблема:** Проверка `len(visible_colors) >= 5` никогда не выполняется при 3 цветах. Условие "confident + grace period" — мёртвый код.

```diff
- if len(visible_colors) >= 5 and cam_id not in self._cam_all_visible_time:
+ if len(visible_colors) >= len(ALL_COLORS) and cam_id not in self._cam_all_visible_time:
```

---

## Phase 2: Высокоприоритетные баги

> Время: 1 час. Улучшает color classification и скорость.

### 2.1 Torso crop: полный bbox → субрегион торса

**Файл:** `deepstream/src/color_infer.h`, строки 77-80
**Проблема:** DeepStream подаёт весь bounding box (включая ноги лошади, землю) в color classifier. Python подаёт только торс жокея (10-40% высоты, 20% боковые отступы). Классификатор обучен на торсах — полный bbox = шум.

```diff
- static constexpr float TORSO_TOP    = 0.0f;
- static constexpr float TORSO_BOTTOM = 1.0f;
- static constexpr float TORSO_LEFT   = 0.0f;
- static constexpr float TORSO_RIGHT  = 0.0f;
+ static constexpr float TORSO_TOP    = 0.10f;
+ static constexpr float TORSO_BOTTOM = 0.40f;
+ static constexpr float TORSO_LEFT   = 0.20f;
+ static constexpr float TORSO_RIGHT  = 0.20f;
```

**Требует перекомпиляции C++.**

---

### 2.2 Fusion speed: last_seen_time обновляется до вычисления dt

**Файл:** `pipeline/fusion.py`, строки 141-158
**Проблема:** `horse.last_seen_time = now` ставится на строке 142, а `dt = max(now - horse.last_seen_time, 0.01)` вычисляется на строке 152. Результат: dt всегда = 0.01, скорость всегда неверна.

```diff
  horse.raw_position_m = raw_pos
- horse.last_seen_time = now
  horse.last_camera = cam_id_str
  horse.observation_count += 1
  horse.missing_frames = 0
  horse.track_confidence = 1.0

  # EMA smoothing
  if horse.observation_count <= 1:
      horse.position_m = raw_pos
  else:
-     dt = max(now - horse.last_seen_time, 0.01)
+     dt = max(now - horse.last_seen_time, 0.01)  # теперь корректно: last_seen_time ещё старое
      delta = abs(raw_pos - horse.position_m)
      alpha = min(self.ema_alpha * (1 + delta / 50.0), 0.8)
      old_pos = horse.position_m
      horse.position_m += alpha * (raw_pos - horse.position_m)
      horse.speed_mps = abs(horse.position_m - old_pos) / dt
+
+ horse.last_seen_time = now  # обновляем ПОСЛЕ вычисления dt
```

---

### 2.3 ColorTracker: подключить EMA-сглаживание

**Файл:** `api/deepstream_pipeline.py`
**Проблема:** Класс `ColorTracker` (строки 27-91) написан, но нигде не вызывается. Транзитные ошибки цвета (yellow→blue на 1-2 кадра) проходят в VoteEngine без фильтрации.

**Что сделать:**
1. В `DeepStreamPipeline.__init__()` создать `self._color_tracker = ColorTracker()`
2. В SHM reader loop: перед записью в DetectionBuffer пропускать каждую детекцию через `self._color_tracker.update(cam_id, track_id, color, conf)`
3. Использовать smoothed color вместо raw color

---

## Phase 3: Средние баги

> Время: 30 минут. Тонкая настройка для паритета с Python.

### 3.1 NMS IoU: 0.45 → 0.50

**Файл:** `deepstream/src/yolo_parser.cpp`, строка 25

```diff
- static constexpr float NMS_IOU_THRESH = 0.45f;
+ static constexpr float NMS_IOU_THRESH = 0.50f;
```

Python (ultralytics) использует `iou=0.5`. Совпадение значений = идентичное поведение NMS.

---

### 3.2 nvinfer interval: 1 → 0 (каждый кадр)

**Файл:** `deepstream/configs/nvinfer_jockey.txt`, строка 20

```diff
- interval=1
+ interval=0
```

`interval=1` пропускает каждый второй кадр (12.5fps вместо 25fps). `interval=0` = inference на каждом кадре.

---

### 3.3 Confidence threshold: 0.15 → 0.35

**Файл:** `deepstream/configs/nvinfer_jockey.txt`, строки 47, 50

```diff
  [class-attrs-all]
- pre-cluster-threshold=0.15
+ pre-cluster-threshold=0.35

  [class-attrs-0]
- pre-cluster-threshold=0.15
+ pre-cluster-threshold=0.35
```

Python detector использует `conf=0.35` (`trt_inference.py:83`). Совпадение порогов = идентичный набор детекций.

---

### 3.4 Aspect ratio: 1.2 → 0.8

**Файл:** `deepstream/src/analysis_pipeline.h`, строка 110

```diff
- static constexpr float MIN_ASPECT_RATIO = 1.2f;
+ static constexpr float MIN_ASPECT_RATIO = 0.8f;
```

1.2 слишком строго — отсекает жокеев в наклонных позах. 0.8 = bbox должен быть хотя бы чуть выше ширины.

---

### 3.5 Speed units: m/s → km/h

**Файл:** `api/deepstream_pipeline.py`, строка 541

```diff
- "speed": round(entry.get("speed_mps", 0), 2),
+ "speed": round(entry.get("speed_mps", 0) * 3.6, 1),
```

Legacy pipeline отправляет km/h (`* 3.6`). DeepStream pipeline должен делать то же самое.

---

### 3.6 Frame skip: 6 → 2

**Файл:** `api/deepstream_pipeline.py`, строка 249

```diff
- self.frame_skip: int = 6
+ self.frame_skip: int = 2
```

С `interval=0` (Phase 3.2) теперь 25fps. `frame_skip=2` даёт ~12fps для voting — достаточно для точности, не перегружает CPU.

---

## Phase 4: Рефакторинг C++ → один плагин

> Время: 2 часа. Собираем всё в один .so, удаляем лишнее.

### Что оставить (GPU-критичный код)

```
deepstream/src/
├── yolo_parser.cpp    — custom bbox parser для nvinfer
├── yolo_parser.h
├── color_infer.cpp    — CUDA kernel: NV12 → crop → resize → normalize → classify
├── color_infer.h
├── shm_writer.cpp     — POSIX SHM binary write
├── shm_writer.h
├── config.h           — SHM struct layout (Detection, CameraSlot, ShmHeader)
└── plugin.cpp         — НОВЫЙ: точка входа, регистрация парсера
```

### Что удалить (уходит в Python)

```
deepstream/src/
├── pipeline.cpp           → deepstream/pipeline.py
├── pipeline.h             → deepstream/pipeline.py
├── analysis_pipeline.cpp  → deepstream/pipeline.py
├── analysis_pipeline.h    → deepstream/pipeline.py
├── dual_pipeline.cpp      → больше не нужен
├── dual_pipeline.h        → больше не нужен
├── trigger_pipeline.cpp   → deepstream/pipeline.py
├── trigger_pipeline.h     → deepstream/pipeline.py
├── main.cpp               → deepstream/main.py
├── diag_logger.cpp        → deepstream/diag.py
└── diag_logger.h          → deepstream/diag.py
```

### Новый CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.18)
project(racevision_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120)  # RTX 5070 Ti Blackwell

find_package(CUDA REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)

# DeepStream paths
set(DS_ROOT /opt/nvidia/deepstream/deepstream)

add_library(nvdsinfer_racevision SHARED
    src/yolo_parser.cpp
    src/color_infer.cpp      # содержит CUDA kernels
    src/shm_writer.cpp
    src/plugin.cpp
)

target_include_directories(nvdsinfer_racevision PRIVATE
    ${DS_ROOT}/sources/includes
    ${CUDA_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    src/
)

target_link_libraries(nvdsinfer_racevision PRIVATE
    ${CUDA_LIBRARIES}
    nvinfer
    nvbufsurface
    pthread
    rt
)

install(TARGETS nvdsinfer_racevision DESTINATION lib)
```

### Сборка

```bash
cd deepstream
mkdir -p build && cd build
cmake ..
make -j$(nproc)
# Результат: build/libnvdsinfer_racevision.so
```

---

## Phase 5: Python pipeline (pyservicemaker)

> Время: 4 часа. Основная работа — заменить 900 строк C++ на ~150 строк Python.

### Новый файл: `deepstream/pipeline.py`

Структура:

```python
# deepstream/pipeline.py — GStreamer pipeline через pyservicemaker

class RaceVisionPipeline:
    """
    Управляет DeepStream pipeline:
    - N камер RTSP → NVDEC → streammux → nvinfer → tracker → probe → fakesink
    - probe callback: фильтрация + color_infer (C++) + shm_writer (C++)
    """

    def __init__(self, config):
        # config: камеры, пути к моделям, пороги

    def build(self):
        # Создать pipeline через pyservicemaker
        # Настроить streammux: batch_size, width, height, padding
        # Подключить nvinfer (nvinfer_jockey.txt → libnvdsinfer_racevision.so)
        # Подключить nvtracker (tracker_iou.yml)
        # Зарегистрировать probe callback

    def start(self):
        # Запустить pipeline

    def stop(self):
        # Остановить pipeline

    def on_detection(self, frame_meta, obj_meta_list):
        # Probe callback:
        # 1. Фильтрация по confidence, aspect ratio, bbox height
        # 2. Вызов color_infer (C++ через ctypes/pybind)
        # 3. Вызов shm_writer (C++ через ctypes/pybind)
        # 4. Диагностика (diag.py)
```

Покрывает функционал:
- `pipeline.cpp` (основной pipeline)
- `analysis_pipeline.cpp` (analysis pipeline)
- `trigger_pipeline.cpp` (trigger pipeline)
- `dual_pipeline.cpp` (менеджер двух pipeline)

---

## Phase 6: Python main + config + diag

> Время: 2 часа.

### `deepstream/main.py` — точка входа

```python
# Заменяет main.cpp (~200 строк)
# argparse: --cameras, --yolo-config, --color-engine, --batch-size,
#           --det-conf, --display, --log-dir, --live/--file
# Загружает config.py → строит pipeline.py → запускает
```

### `deepstream/config.py` — конфигурация

```python
# Заменяет PipelineConfig struct из pipeline.h
# Загружает cameras.json
# Все пороги и параметры в одном месте
# Без перекомпиляции: поменял .py → перезапустил
```

### `deepstream/diag.py` — диагностика

```python
# Заменяет diag_logger.cpp (~150 строк)
# Python logging + cv2.imwrite для снимков
# CSV метрики (FPS, latency, detection count)
```

---

## Phase 7: CMakeLists.txt

> Время: 30 минут.

Обновить CMakeLists.txt (см. Phase 4). Убрать все target для бинарников, оставить только `libnvdsinfer_racevision.so`.

---

## Phase 8: Документация и зависимости

> Время: 1 час.

### `requirements_deepstream.txt`

```
# DeepStream 9.0 bare-metal dependencies
# GPU: NVIDIA RTX 5070 Ti (Blackwell, SM 12.0)
# OS:  Ubuntu 24.04

# Driver: nvidia-driver-570-open (apt)
# CUDA:   cuda-toolkit-13-2 (apt) или cuda-toolkit-12-8
# cuDNN:  cudnn9-cuda-13 (apt) или cudnn9-cuda-12
# TRT:    tensorrt 10.6.0 (apt)
# DS:     deepstream-9.0 (deb)

# Python
torch>=2.5.0              # nightly для sm_120: --index-url .../nightly/cu12
torchvision>=0.26.0
ultralytics>=8.4.37
onnx==1.21.0
onnxruntime-gpu==1.24.4
numpy==2.4.4
opencv-python-headless>=4.12.0
fastapi>=0.115.0
uvicorn>=0.32.0
websockets>=14.0
posix-ipc>=1.1.1
pyservicemaker              # DeepStream 9.0 Python API
```

### `docs/ARCHITECTURE.md` — обновить

Описать гибридную архитектуру:
- Python: pipeline конфигурация, бизнес-логика, API
- C++: GPU kernels (yolo_parser, color_infer, shm_writer)
- Взаимодействие: Python → C++ через libnvdsinfer_racevision.so

---

## Итоговая структура проекта

```
deepstream/
├── main.py                          ← Python: точка входа
├── pipeline.py                      ← Python: pyservicemaker pipeline
├── config.py                        ← Python: конфигурация
├── diag.py                          ← Python: диагностика и логирование
│
├── configs/
│   ├── nvinfer_jockey.txt           ← ИСПРАВЛЕННЫЙ конфиг (Phase 1)
│   └── tracker_iou.yml              ← трекер
│
├── src/
│   ├── config.h                     ← SHM struct layout
│   ├── yolo_parser.cpp / .h         ← C++: custom YOLO bbox parser
│   ├── color_infer.cpp / .h         ← C++: CUDA crop + resize + classify
│   ├── shm_writer.cpp / .h          ← C++: POSIX SHM binary writer
│   └── plugin.cpp                   ← C++: регистрация плагина
│
├── build/
│   └── libnvdsinfer_racevision.so   ← ОДИН скомпилированный плагин
│
└── CMakeLists.txt                   ← упрощённый (только плагин)

api/
├── shared.py                        ← ИСПРАВЛЕН: 5 цветов (Phase 1)
├── deepstream_pipeline.py           ← ИСПРАВЛЕН: visible_colors, frame_skip, speed (Phase 1,3)
├── server.py
├── camera_io.py
└── legacy_pipeline.py

pipeline/
├── fusion.py                        ← ИСПРАВЛЕН: last_seen_time (Phase 2)
├── vote_engine.py
├── shm_reader.py
├── trt_inference.py
├── analyzer.py
├── trigger.py
└── ...
```

---

## Расписание

```
Phase 1  (30 мин)   Критические баги → тест → детекция улучшилась?
Phase 2  (1 час)    Высокие баги → тест → color classification улучшился?
Phase 3  (30 мин)   Средние баги → тест → метрики стабильны?
──── БАГФИКСЫ ЗАВЕРШЕНЫ — качество на уровне Python ────
Phase 4  (2 часа)   C++ → один .so плагин → собирается?
Phase 5  (4 часа)   Python pipeline.py → запускается?
Phase 6  (2 часа)   main.py + config.py + diag.py → полный запуск?
Phase 7  (30 мин)   CMakeLists.txt → make → .so готов?
Phase 8  (1 час)    Документация → requirements → README

Итого: ~12 часов работы
```

---

## Чеклист "готово"

- [ ] Phase 1: `nvidia-smi` показывает RTX 5070 Ti
- [ ] Phase 1: model-color-format=0, cluster-mode=4 применены
- [ ] Phase 1: 5 цветов в COLOR_TO_HORSE
- [ ] Phase 1: visible_colors >= len(ALL_COLORS)
- [ ] Phase 2: Torso crop 0.10/0.40/0.20/0.20
- [ ] Phase 2: fusion speed dt корректный
- [ ] Phase 2: ColorTracker подключён
- [ ] Phase 3: Все средние баги исправлены
- [ ] Phase 3: A/B тест: DeepStream recall >= Python recall
- [ ] Phase 4: libnvdsinfer_racevision.so собирается
- [ ] Phase 5: pipeline.py запускает 25 камер
- [ ] Phase 6: main.py --cameras configs/cameras_live.json работает
- [ ] Phase 7: cmake + make без ошибок
- [ ] Phase 8: Документация обновлена
