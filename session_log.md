# Session Log — Race Vision

Хронологический лог сессий: что менялось, что запускали, результаты экспериментов.

---

## 2026-03-18 (сессия 2) — 1-камерный debug режим, EMA, torso-кроп, mux 1520

### Что сделано
1. **`cameras_1cam.json`** — новый конфиг для 1 камеры (cam-05, file mode)
2. **`nvinfer_jockey_1cam.txt`** — batch=1, interval=0, threshold=0.30
3. **`tracker_iou.yml`** — `probationAge: 3→1` (трек с первого кадра)
4. **`pipeline.h`** — оптимизированы фильтры:
   - `MIN_BBOX_HEIGHT`: 80→35 px
   - `MIN_ASPECT_RATIO`: 0.3→0.25
   - `MIN_CROP_PIXELS`: 400→200, `MAX_CROP_PIXELS`: 15000→20000
   - `STATIC_HISTORY_FRAMES`: 8→0 (отключён — лошади проходят за <8 кадров)
   - Добавлен `ColorSmoother` struct + `color_smooth_` map (EMA)
   - `COLOR_EMA_ALPHA = 0.35f`
5. **`pipeline.cpp`** — исправления:
   - Баг: `STATIC_HISTORY_FRAMES=0` → `0 >= 0` всегда true → все детекции фильтровались. Фикс: `if (STATIC_HISTORY_FRAMES > 0)`
   - Торсо-кроп через `ColorInfer::compute_torso_roi()` (10-40% высоты, 20% margins)
   - EMA сглаживание цвета по track_id
   - `MIN_COLOR_CONF=0.60` — ниже → COLOR_UNKNOWN
   - OSD: `border_width 5→2`, `font_size 16→10`
   - `sync=TRUE` на display sink
6. **`main.cpp`** — `live_source = !file_mode || (cameras.size() > 7)` (FALSE для 1-cam file)

### Ключевые выводы
- **Почему все GREEN/UNKNOWN**: `color_classifier_v2` обучен только на 3 классах (green/red/yellow). Blue/purple жокеев нет в модели → всегда UNKNOWN или misclassified
- **bbox размер** (24×49px) — физическое расстояние камеры от лошади, не зависит от качества модели
- **mux 1520×1520** — торсо-кроп вырастает с ~14×15px до ~28×28px → лучше для классификатора

### Рабочие команды (нативно, без Docker)
```bash
cd /home/user/race_vision

# 1 камера, debug
./deepstream/build/race_vision_deepstream \
  --config cameras_1cam.json \
  --yolo-engine deepstream/configs/nvinfer_jockey_1cam.txt \
  --color-engine models/color_classifier_v2.engine \
  --mux-width 1520 --mux-height 1520 \
  --file-mode --display

# 25 камер
LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH \
./deepstream/build/race_vision_deepstream \
  --config cameras_all_25.json \
  --yolo-engine deepstream/configs/nvinfer_jockey.txt \
  --color-engine models/color_classifier_v2.engine \
  --file-mode --log-dir ds_results
```

### Что нужно завтра
1. **Переобучить `color_classifier_v2`** на все 5 цветов (blue, green, purple, red, yellow)
   - Извлечь кропы жокеев из записей через YOLO
   - Текущий `dataset/` — проверить что есть
2. Возможно поднять mux до 1520×1520 и для 25 камер — замерить FPS
3. Протестировать 1-камерный режим с новым классификатором

---

## 2026-03-18 — Нативный DeepStream на хосте (без Docker)

### Что сделано
1. **Удалён Docker** — освободили ~57 GB (образы + кэш)
2. **DeepStream 8.0 установлен на хост** (Ubuntu 24.04, неофициально — работает)
3. **C++ binary собран нативно** под TRT 10.9 + CUDA 12.8
   - `deepstream/build/race_vision_deepstream` (1.4 MB)
   - `deepstream/build/libnvdsinfer_yolov8_parser.so`
4. **CMakeLists.txt** — фикс JSON_INCLUDE_DIR после FetchContent
5. **Engine пересобран** под TRT 10.9 (~7 мин через nvinfer auto-rebuild из ONNX)
6. **Пути исправлены** с `/app/...` на хостовые в `nvinfer_jockey.txt` и `cameras_all_25.json`

### Результаты
- **38-40 FPS** на 25 камерах (лучше 35-36 в Docker)
- cam-05 yellow(68%), cam-08 green(68%), cam-13 green(98%) — всего 3 детекции за 3.5 мин

### Проблемы для завтра
1. **Мало детекций** — снизить MIN_BBOX_HEIGHT 80→40, убрать interval=1→0
2. **Unknown цвет** каждый 2-й батч — нужен carry-forward по track_id в pipeline.cpp
3. Проверить середину записи где лошади точно видны

### Команда запуска (нативная)
```bash
cd /home/user/race_vision
LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH \
GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins \
./deepstream/build/race_vision_deepstream \
  --config cameras_all_25.json \
  --yolo-engine deepstream/configs/nvinfer_jockey.txt \
  --color-engine models/color_classifier_v2.engine \
  --file-mode --log-dir /home/user/race_vision/ds_results
```

---

## 2026-03-13 (сессия 2) — False positive filtering (static objects)

### Проблема
На фото exp16 видны ложные детекции — YOLO ловит статические объекты (техника, столбы) как person.
Track_id=2 на cam-05 сидел на месте 100+ батчей, координаты почти не менялись (dx ~5px/batch).
Путаницы камер нет — код корректно матчит детекции к кадрам через `source_id`.

### Исправления
1. **`pipeline.h`** — `MIN_BBOX_HEIGHT` 60→80 (отсекает мелкие объекты)
2. **`nvinfer_jockey.txt`** — `pre-cluster-threshold` 0.30→0.50 (YOLO отдаёт только уверенные детекции)
3. **`pipeline.h/cpp`** — фильтр статических объектов:
   - Хранит историю center_x для каждого (cam_index, track_id)
   - После 8 кадров проверяет суммарное перемещение
   - Если < 30px — скипает (стоячий объект, не лошадь на скаку)
   - Очистка каждые 100 батчей для экономии памяти

### Файлы изменены
- `deepstream/src/pipeline.h` — MIN_BBOX_HEIGHT 60→80, добавлен TrackHistory + track_history_
- `deepstream/src/pipeline.cpp` — static member definition, velocity filter в probe, periodic cleanup
- `deepstream/configs/nvinfer_jockey.txt` — threshold 0.30→0.50

### Статус
Код изменён, не скомпилирован, не протестирован. Нужен Docker rebuild.

---

## 2026-03-13 — Fix 25-cam deadlock + diagnostic run

### Проблема
Pipeline зависал на 25 камерах из файлов (State: NULL → READY, дальше не шёл).
На 5-7 камерах работало, на 8+ — deadlock.

### Причина
В `main.cpp` было `pipeline_config.live_source = !file_mode` → `FALSE` для файлов.
`live-source=FALSE` заставляет nvstreammux ждать фреймы от ВСЕХ источников одновременно.
NVDEC на RTX 3060 не может стартовать 25 декодеров разом → deadlock.

### Исправление
- `main.cpp`: `pipeline_config.live_source = true` — ВСЕГДА TRUE (как было в коммите 04626c4)
- `pipeline.cpp`: вернули `attach-sys-ts=TRUE`, `num-surfaces-per-frame=1`

### Рабочая команда запуска (25 камер, файлы)
```bash
docker run --rm --runtime=nvidia --gpus all \
  -v cameras_all_25.json:/app/cameras_all_25.json:ro \
  -v /home/user/recordings:/recordings:ro \
  -v models:/app/models:ro \
  -v deepstream/configs:/app/configs:ro \
  -v /data:/data \
  --network host \
  --entrypoint /app/bin/race_vision_deepstream \
  race-vision:latest \
    --config /app/cameras_all_25.json \
    --yolo-engine /app/configs/nvinfer_jockey.txt \
    --color-engine /app/models/color_classifier_v2.engine \
    --file-mode --log-dir /data/ds_results
```

### Результаты
- **25 камер, 36-39 FPS**
- Детекции: cam-05, cam-01 (начало записи — лошади далеко)
- Диагностика (exp16): 117 строк CSV, 26 JPG снимков
- Снимки не чёрные ✓ (NV12→RGB фикс работает)
- **Проблема**: цвета нестабильны — green/yellow/red скачут на одном track_id

---

## 2026-03-12 — Fix black snapshots + false detection filtering

### Проблема
- Диагностические JPG снимки были **чёрные** с мелкими боксами
- Ложные срабатывания на мелких объектах/шуме

### Причина
1. `diag_logger.cpp` читал surface как **RGBA** (4 байта/пиксель), но nvstreammux выдаёт **NV12**
   - NV12 = Y plane (luminance) + UV plane (chroma, half resolution)
   - При чтении как RGBA → чёрное изображение (Y значения размазаны по 4 каналам)
2. `MIN_BBOX_HEIGHT = 25` — слишком маленький порог, YOLO ловит мелкий шум

### Исправления
1. **`diag_logger.cpp`** — определение формата surface (NV12 vs RGBA), скачивание Y+UV planes отдельно, конвертация NV12→RGB
2. **`pipeline.h`** — `MIN_BBOX_HEIGHT` 25→60, `MIN_CROP_PIXELS` 200→400, добавлен фильтр aspect ratio (0.3–2.5)
3. **`pipeline.cpp`** — фильтр по aspect ratio (отсекает тонкие полоски/артефакты)

---

## 2026-03-10 — Tracker-only fix + Full Pipeline Test

### Изменения
1. **Tracker-only objects скипаются полностью** — `pipeline.cpp`: `if (tracker_only) continue;` до записи детекции
   - Убирает `?(0%)` спам от tracker-interpolated объектов (confidence < 0)
   - Упрощённый маппинг цветов (1:1 correspondence ROI↔mapping)
2. **nvinfer_jockey.txt** → YOLOv11s engine (вместо YOLOv8s)
3. **Docker пересобран** с `--no-cache` для актуализации C++ бинарника

### Результаты теста (25 камер, file mode)
- **Модели**: YOLOv11s (jockey) + EfficientNet-V2-S (3 класса: green/red/yellow)
- **FPS**: 36-37 на 25 камерах
- **Детекции на 10/25 камерах**: cam-05:545, cam-08:63, cam-01:51, cam-13:36, cam-14:11, cam-12:9, cam-03:8, cam-04:4, cam-22:1, cam-02:1
- **Python SHM reader**: РАБОТАЕТ — `CAMERA COMPLETE cam-05 order=[red > green > yellow]`
- **Полный pipeline**: C++ → SHM → Python → Rankings ✓

### Известные проблемы
- `red(47%)` — стабильно 47% confidence, ограничение модели
- GStreamer warnings при старте (безвредные)

---

## 2026-03-04 (сессия 2) — Апгрейд DeepStream 8.0 + пересборка engine

### Что сделано
1. **Dockerfile обновлён** с DeepStream 7.1 → 8.0 (TensorRT 10.9, CUDA 12.8)
2. **yolov8s.onnx исправлен** — был 0 байт, заменён на рабочий 43MB (dynamic batch)
3. **Удалены старые engine** — yolov8s_deepstream.engine (от старого TRT)
4. **Docker image пересобран** — race-vision:latest на DS 8.0 (21.6GB)
5. **test_pipeline_on_files.py** — добавлен ACTIVE_COLORS={green,red,yellow}, --show, horse proximity filter
6. **nvinfer авто-сборка engine** — контейнер запущен с 1 камерой (cam-20), nvinfer строит engine из ONNX
7. **color_classifier.engine** — не загрузился (от старого TRT), нужно пересобрать

### Ошибки
- `yolov8s.onnx_b25_gpu0_fp16.engine` (25MB) — авто-сгенерирован nvinfer от DS 7.1, не удалён с хоста
- `color_classifier.engine` — TRT version mismatch, нужно пересобрать внутри DS 8.0

### Результат: ДЕТЕКЦИИ РАБОТАЮТ!
- Engine собрался из ONNX внутри DS 8.0 (~8 мин)
- Engine сохранён nvinfer как `yolov8s.onnx_b25_gpu0_fp16.engine`, переименован в `yolov8s_deepstream.engine`
- **Парсер исправлен**: координаты были нормализованы [0,1] вместо пиксельных [0,800]
- Тензор: `[84, 13125]` — правильный формат, не транспонирован
- cam-20: ~30 кадров с 1-3 детекциями person (batch 3368-3430)
- `model-color-format=1` добавлен в nvinfer конфиг (RGB для YOLO)
- `color_classifier.engine` не загрузился (TRT version mismatch DS 7.1→8.0, нужно пересобрать)

### Тест 5 камер — РАБОТАЕТ!
- `cameras=5 detections=1-3` — стабильные детекции на 5 камерах
- Лошади проходят через камеры последовательно (волны детекций)
- 25 камер из файлов зависает — NVDEC не справляется с 25 одновременными decode
- `live-source=TRUE` установлен всегда (FALSE блокировал mux на 25 камерах)
- Для живых RTSP камер 25 потоков должно работать (стартуют постепенно)

### Frontend — добавлен ACTIVE индикатор
- `api/server.py` — `live_detections` dict обновляется каждый SHM цикл, broadcast через WebSocket
- `cameraStore.ts` — добавлен `liveDetections: Record<string, number>`, `setLiveDetections()`
- `backendConnection.ts` — обработка `live_detections` сообщения
- `CameraGrid.tsx` — зелёная рамка + пульсирующий "ACTIVE" бейдж когда детекции есть
- Frontend пересобран

### Следующие шаги
- Пересобрать color_classifier.engine внутри DS 8.0
- Тест на живых RTSP камерах (25 штук)
- Проверить ACTIVE индикатор на фронте

---

## 2026-03-04 — Тест pipeline на записях + починка DeepStream

### Диагностика: почему Docker вчера не работал
1. Backend запускался, DeepStream C++ → PLAYING, SHM создан — **всё стартовало**
2. **race_active=False** — Python игнорировал все детекции (строка 831 в server.py)
3. Нужно было либо `--auto-start`, либо отправить `start_race` через WebSocket
4. TRT engine загружался нормально, ошибка `IRuntime::~IRuntime Error Code 3` — не критичная

### Тест YOLO + ColorCNN на записях (Python, без DeepStream)
- Папка: `/home/user/recordings/yaris_20260303_162028/` (25 камер, ~4.5 мин каждая)
- **YOLOv8s детектирует** лошадей и жокеев (class 0 = person, class 17 = horse)
- **ColorCNN работает**: green, yellow, red определяются стабильно с >90% confidence
- **Проблемы**: purple и blue почти не видны; cam-03 ловит людей у здания (false positives); cam-07,09,10,25 — 0 детекций (далеко)
- Скрипт: `tools/test_pipeline_on_files.py`

### Изменения в C++ для поддержки файлов
- `pipeline.h` — добавлен `bool live_source` в PipelineConfig
- `pipeline.cpp` — `live-source` берётся из конфига вместо TRUE; `on_child_added` проверяет наличие свойства перед установкой RTSP-параметров; добавлен лог детекций в probe
- `main.cpp` — добавлен `--file-mode` аргумент (ставит live_source=false)
- `api/server.py` — `--file-mode` CLI аргумент, auto-detect file:// URLs, передаёт в C++
- `nvinfer_yolov8s.txt` — добавлена строка `onnx-file` для автопересборки engine
- Создан `cameras_test_files.json` с `file:///data/kamera_XX.mp4` URLs

### ОШИБКА: перезаписан рабочий TRT engine
- Оригинал `models/yolov8s_deepstream.engine` (Feb 27, 25.9MB) был собран ВНУТРИ Docker
- Я пересобрал engine из Python venv на хосте (другая версия TensorRT!)
- Перезаписал рабочий файл → Docker больше не может его загрузить
- Оригинал не в git, потерян
- **НУЖНО**: собрать engine заново внутри Docker через `trtexec` (~10 мин)

### Важные выводы (запомнить!)
- **TRT engine привязан к версии TensorRT** — собирать ТОЛЬКО внутри Docker
- **race_active=False по умолчанию** — нужен `--auto-start` или WebSocket `start_race`
- **C++ pipeline фильтрует только class 0 (person)** — лошади (class 17) игнорируются

### Статус
Код изменён, Docker пересобран, но engine нужно пересобрать внутри Docker.
Не закоммичено.

---

## 2026-03-03 — Запись + план жокейского триггера

### Что сделано

**Скрипт записи `tools/record_cameras.py`:**
- Каждый запуск → отдельная папка `/home/user/recordings/yaris_YYYYMMDD_HHMMSS/`
- Авторестарт при обрыве камеры
- При CTRL+C — graceful shutdown + время окончания в имени файла
- Пишет с main stream `Channels/101` (не конфликтует с go2rtc/DeepStream)
- Запуск: `python3 /home/user/race_vision/tools/record_cameras.py`
- Скачать на Windows: `scp -r user@100.104.201.30:/home/user/recordings/yaris_... "C:\Users\gghuk\OneDrive\Документы\kabirhan_records\"`

**Исправлен порядок камер (12-25):**
- `cameras_example.json` — cam-12..25 переупорядочены под порядок записи
- `go2rtc.yaml` — cam-12-sub..cam-25-sub переупорядочены
- Теперь cam-18 везде = IP 10.223.70.28

**Тест backend:**
- Запускался, pipeline PLAYING, go2rtc 26/26 online
- Детекций не было — лошадей в кадре не было в момент теста

### План — Жокейский триггер (следующая сессия)

**Цель:** заменить YOLOv8n триггер на легкий детектор жокейских цветов

**Выбранный подход — Движение + цвет (без обучения):**
```
MOG2 background subtraction → маска движения
+ HSV цветовой фильтр (blue/green/red/yellow/purple) → маска цвета
= AND → только движущиеся цветные объекты → trigger
```

**Данные из текущей записи:**
- Нарезать кадры с жокеями → `data/trigger/with_jockey/`
- Нарезать кадры пустого трека → `data/trigger/empty/`
- ~200-500 кадров каждого класса с разных камер

**Что делать следующий раз:**
1. Скачать записи на Windows
2. Написать скрипт нарезки кадров из MP4 (`tools/extract_frames.py`)
3. Разметить по папкам
4. Реализовать MOG2+HSV триггер в Python
5. Протестировать на записях
6. Если нужно точнее → обучить MobileNetV3-Small бинарный классификатор

### Статус
Запись идёт. Код не закоммичен (cameras_example.json, go2rtc.yaml изменены).

---

## 2026-03-02 — Pipeline optimization: resolution, frame skip, one-result-per-camera

### Что сделано

**Часть 1: Resolution 1280 → 800**
- `deepstream/configs/nvinfer_yolov8s_analysis.txt` — infer-dims 1280→800
- `deepstream/configs/nvinfer_yolov8s.txt` — infer-dims 1280→800
- `deepstream/src/analysis_pipeline.h` — mux_width/height 1280→800, MIN_BBOX_HEIGHT 100→65
- `deepstream/src/pipeline.h` — mux_width/height 1280→800, MIN_BBOX_HEIGHT 100→65
- `pipeline/analyzer.py` — MIN_BBOX_HEIGHT 100→65, imgsz 1280→800
- `api/server.py` — imgsz=800 в MultiCameraPipeline
- `tools/export_trt.py` — export analysis+deepstream: imgsz 1280→800

**Часть 2: Frame skip 12fps → 4fps**
- `pipeline/analyzer.py` — analysis_fps default 12.0→4.0
- `api/server.py` — MultiCameraPipeline analysis_fps=4.0
- `api/server.py` — DeepStreamPipeline: frame_skip=6 (~25fps→~4fps), скипает кадры в _reader_loop

**Часть 3: One-result-per-camera**
- `api/server.py` — DeepStreamPipeline: grace period 2s, timeout 8s, _cam_completed, _pending_camera_results
- `api/server.py` — ranking_broadcast_loop: шлет camera_result events, ranking_update только при изменении
- `pipeline/analyzer.py` — AnalysisLoop: та же логика completion (grace/timeout), on_result только при completion
- `Kabirhan-Frontend/src/services/backendConnection.ts` — handler для camera_result event

### Логика завершения камеры
1. Confident + grace 2s (все 5 позиций + ждем отстающих)
2. Confident + 8+ vote frames (no need to wait)
3. Timeout 8s + min 2 vote frames
4. Partial timeout 8s (straggler case — 1-2 лошади, vote_frames=0, всё равно шлём в fusion)

### Статус
Изменения применены, не закоммичено, не протестировано.

---

## 2026-03-01 — Sub-stream migration

### Что сделано
Все 25 аналитических камер переключены с main-stream (Channels/101, 1080p) на sub-stream (Channels/102, D1/CIF).
YOLO все равно ресайзит до 640/1280px — main-stream впустую тратил bandwidth и GPU decode.

### Файлы изменены
- `cameras_example.json` — 25 analytics URLs: `Channels/101` → `Channels/102`
- `cameras_5.json` — 5 analytics URLs: `Channels/101` → `Channels/102`
- `go2rtc.yaml` — удалены 25 `cam-XX` main-stream записей (никто не использовал); `cam-XX-sub` и `ptz-1` сохранены
- `Kabirhan-Frontend/src/config/cameras.ts` — 25 analytics rtspUrl: `Channels/101` → `Channels/102`

### PTZ не тронут
`ptz-1` остается на `Channels/101` (broadcast quality).

### Docker
- go2rtc: volume mount → restart достаточно, rebuild не нужен
- backend: volume mount → restart достаточно, rebuild не нужен
- frontend: `cameras.ts` запекается в бандл → нужен `docker compose build frontend`
- Но `rtspUrl` в cameras.ts чисто информационный — можно обойтись без rebuild

### Статус
Изменения применены локально, не закоммичено, не протестировано на живых камерах.

---

## 2026-03-01 — Lazy loading для CameraGrid

### Что сделано
Operator page (`/operator`) грузил все 25 WebSocket видеопотоков одновременно — браузер лагал.
Добавлена ленивая загрузка через IntersectionObserver в Go2RTCPlayer:
- Новый prop `lazy?: boolean` (default false — обратная совместимость для PTZ)
- Когда камера не видна в viewport → WebSocket закрыт, видео остановлено, статус "Paused"
- Когда камера появляется в viewport (+ 100px margin) → подключается автоматически
- `connect()` и `scheduleReconnect()` проверяют `isVisibleRef` — не переподключаются к невидимым

### Файлы изменены
- `Kabirhan-Frontend/src/components/Go2RTCPlayer.tsx` — IntersectionObserver, `lazy` prop, статус `paused`
- `Kabirhan-Frontend/src/components/operator/CameraGrid.tsx` — передает `lazy` в Go2RTCPlayer

### Результат
Вместо 25 одновременных потоков, браузер держит только видимые (~9 при 3 колонках).
PTZ-плеер не затронут (lazy=false по умолчанию).

### Статус
Закоммичено: `7d62fe1` (47 файлов, +5574/-498). Не запушено (2 коммита ahead of origin).
Frontend нужен rebuild для деплоя: `docker compose build frontend`
Git push не работает — нужна авторизация GitHub (SSH/token/gh auth).

---

## 2026-03-02 — Чтение всей кодовой базы

### Что сделано
Полное чтение и анализ всех файлов проекта:
- Python backend: `api/server.py`, все модули `pipeline/`, `tools/`
- C++ DeepStream: все файлы `deepstream/src/`, `CMakeLists.txt`, конфиги nvinfer
- Конфигурации: `cameras_example.json`, `cameras_5.json`, `go2rtc.yaml`, `docker-compose.yml`, `Dockerfile`
- Документация: `DOCS.md`, `README.md`, `Kabirhan-Frontend/README.md`
- Frontend структура: `Kabirhan-Frontend/src/`
- Извлечён текст из `Camera 1.docx` (25 IP-адресов камер)

### Статус
Только чтение, никаких изменений в коде.

---

## 2026-03-02 — Исследование Triton Inference Server

### Вывод
Triton уже включён в Docker-образ (`deepstream:7.1-triton-multiarch`), но не используется.
Все 3 модели (YOLOv8n ~2.5GB + YOLOv8s ~4.5GB + ColorCNN ~0.3GB) влезают в RTX 3060 12GB (~7.8GB total).

### Рекомендуемый подход — гибридный
- **Trigger pipeline** → оставить на `nvinfer` (простой, batch=25, работает)
- **Analysis pipeline** → перевести на `nvinferserver` (in-process C API, zero-copy)
- Ensemble: YOLOv8s → Python crop → ColorCNN — одним запросом вместо ручного C++ кода
- Убирает POSIX SHM `/rv_detections`, упрощает `api/server.py`
- Overhead in-process C API vs nvinfer: **~1-3%** (подтверждено NVIDIA)

### Ключевые преимущества
- Ensemble pipeline заменяет `color_infer.cpp` + `shm_writer.cpp`
- Приоритеты моделей (trigger priority=1, analysis priority=2)
- Dynamic batching с контролем задержки
- Hot reload моделей без перезапуска

### Что менять при миграции
- `analysis_pipeline.cpp` — `nvinfer` → `nvinferserver`, новые .pbtxt конфиги
- Удалить `color_infer.cpp/h` — Triton ensemble заменяет
- Упростить `shm_writer.cpp/h`
- Добавить `model_repository/` с config.pbtxt для каждой модели
- `CMakeLists.txt` — добавить `tritonserver` в link

### Статус
Исследование завершено. Код не менялся. Миграция возможна поэтапно.

---
