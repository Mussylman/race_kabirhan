# Session Log — Race Vision

Хронологический лог сессий: что менялось, что запускали, результаты экспериментов.

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
