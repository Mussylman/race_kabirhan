# CLAUDE.md — Race Vision

## 1. Project context

Race Vision — реалтайм-трекер позиций жокеев на ипподроме (25 RTSP-камер, full-loop ~800 м). Hybrid-архитектура: DeepStream 9.0 pipeline (C++ writer + Python `pyservicemaker` orchestration) пишет YOLO-детекции и SGIE-классификацию цвета в POSIX SHM (`/rv_detections`); FastAPI-reader отдаёт ranking по WebSocket в React-фронт (Kabirhan-Frontend). Стек: Python 3.12 + torch 2.13 nightly cu130 + ultralytics 8.4.x + TensorRT 10.16 + DeepStream 9.0 + RTX 5070 Ti (sm_120) + React 19 / Vite 7. Production SGIE color classifier (с 2026-04-28) = **OSNet x1.0 market1501** (256×128, 512-dim embedding, prototype-based cosine; запускать с `RV_TIGHT_SGIE=0`); DINOv2 fallback остаётся в `libnvdsinfer_racevision.so` (откат через `tools/rollback_to_dinov2.sh`). Главные директории: `api/` (FastAPI), `deepstream/` (pipeline + C++ src), `pipeline/` (SHM reader, fusion, time_tracker), `tools/` (бенчи, run_record, dataset builders), `Kabirhan-Frontend/src/` (UI), `scripts/rv.sh` (launcher).

---

## 2. The 4 Principles

### 🔪 Surgical Changes (САМЫЙ КРИТИЧНЫЙ для этого проекта)
Менять **только** код, напрямую относящийся к задаче. Не «улучшать» соседние строки попутно. Каждый изменённый файл — обоснован запросом пользователя.

> **Race Vision example:** добавление `source_frame_num` в SHM (`94adbe6`) — surgical: переименовали `_pad → source_frame_num` в `config.h`, синхронизировали 4 точки (ctypes mirror в `rv_plugin.py`, struct format в `shm_reader.py`, поле в `CameraDetections`, write call в `pipeline.py` probe). Размер `CameraSlot` остался байт-в-байт идентичный — `static_assert` не сломался, существующие читатели работают. Если бы попутно «улучшили» соседние поля — каскад регрессий и пересборка `.so` ради косметики.

### 🧠 Think Before Coding
Перед кодом: явно сформулировать assumptions, выявить неясности, предложить альтернативы. Если 2+ блокирующих вопроса — стоп, спросить.

> **Race Vision example:** перед началом работы с SHM-расширением (Phase 2) сначала составили список «что менять в каждой из 4 точек ABI», подтвердили layout assert не сломается, и только потом писали код. Не «попробовали запустить и посмотрели что упадёт».

### ✂️ Simplicity First
Только запрошенное. Без спекулятивных абстракций «на будущее». Самопроверка: «senior engineer назовёт это overengineering?»

> **Race Vision example:** `tools/run_record.py` — один файл, без классов-фабрик. `FrameStore` + `SnapshotPolicy` + `OutputManager` — три простых класса с очевидной ролью. Не плодили `Strategy`/`Builder`/`AbstractFactory` ради «гибкости».

### 🎯 Goal-Driven Execution
Каждая задача формулируется как `Сделать X → проверка Y` с конкретным критерием успеха ДО начала работы.

> **Race Vision example:** seqlock fix → проверка через `tools/test_shm_torn_read.py` (4 сценария: sanity, idempotency, realistic 30 Hz = 0 torn, extreme rate = fallback fires без crash). Без теста задача считалась бы недописанной.

---

## 3. Project Guardrails

### НЕ трогать без явного запроса (HARD)

| Файл / зона | Почему |
|---|---|
| `deepstream/src/config.h` | SHM-протокол C++↔Python. Размер `CameraSlot` зафиксирован `static_assert`. Любая правка = пересборка `.so` + ручная синхронизация `deepstream/rv_plugin.py` (ctypes layout) + `pipeline/shm_reader.py` (struct format) + `pipeline/detections.py` (CameraDetections). |
| `deepstream/src/shm_writer.cpp`, `yolo_parser.cpp`, `color_classifier_parser.cpp`, `dinov2_color_parser.cpp`, `osnet_color_parser.cpp` | Hot-path C++ writer + 3 SGIE classifier парсера. Атомарный seqlock-commit + custom YOLO parser + кастомные color SGIE парсеры (DS 9.0 quirk: автопарсер не работает на 2D output — см. `feedback_ds9_sgie_quirks.md`). Production = OSNet (2026-04-28 deploy); DINOv2 + legacy CNN остаются в `.so` как fallback path. |
| `deepstream/src/osnet_prototypes_v1.h`, `dinov2_prototypes_v1.h` | Auto-generated prototypes headers (4×512 OSNet / 4×768 DINOv2, L2-normalized FP32). DO NOT EDIT — пере-генерировать через `tools/generate_osnet_prototypes_h.py` или `generate_dinov2_prototypes_h.py`. |
| `deepstream/build/libnvdsinfer_racevision.so` | Build artifact. Только пересобирать через `cd deepstream/build && cmake --build .`. Не править. |
| `pipeline/shm_reader.py` `read()` метод | Seqlock retry-loop (88cc2ac). Чувствителен к race conditions. Любое изменение порядка чтения header → данные → header требует регресса через `tools/test_shm_torn_read.py` (4 сценария должны пройти). |
| `models/*.engine`, `models/*.onnx`, `models/*.pt` | Прод-веса. TRT engine привязан к (батч, GPU, FP16) — не пересобирать на ходу. См. `feedback_rules.md`. |
| `configs/cameras_live.json`, `cameras_live_ordered.json`, `deepstream/configs/nvinfer_racevision.txt`, `deepstream/configs/sgie_color.txt` | Прод-конфиги pipeline'а. Меняются с координацией. |
| `scripts/rv.sh` | Прод-launcher (Phase 14). Управляет 3 сервисами + go2rtc. Менять только когда явно про launcher. |
| `Kabirhan-Frontend/Dockerfile`, `nginx.conf`, `server/`, `package-lock.json` | Прод-деплоймент frontend'а. |

### Soft guardrails (правила вместо «не трогать»)

1. **DeepStream pipeline = bare-metal.** Запускается через `scripts/rv.sh start`, требует прямого доступа к CUDA 13 / TensorRT 10.16 / NVIDIA driver 580+ / DeepStream SDK 9.0 / RTX 5070 Ti (sm_120). Docker-инфраструктура в корне (`Dockerfile`, `docker-compose.yml`, `influx.env`) — для **вспомогательных** сервисов (Milvus, monitoring, etc.). **Не предлагать контейнеризацию pipeline без явного запроса пользователя.**

2. **Файлы в `pipeline/`, не упомянутые в `feedback_whitelist.md`, не импортировать в runtime path без подтверждения статуса.** Это касается прежде всего `pipeline/jockey_reid.py`, `pipeline/clip_color_classifier.py` — статус `active / experimental / dead` неясен. Перед использованием в `api/server.py` или `deepstream/pipeline.py` — спросить.

3. **`api/legacy_pipeline.py` + `pipeline/trigger.py` + `pipeline/analyzer.py` + `pipeline/trt_inference.py` = legacy путь.** В активном runtime НЕ задействован (current = DeepStream subprocess + `api/deepstream_pipeline.py`). Можно править, но не реанимировать без обсуждения.

4. **`archive/cpp_legacy/`** — НЕ собирается, **не делать его reference в новом коде**. Только для исторического справочника.

### Зоны где можно работать свободно

- `tools/` — изолированные скрипты (бенчи, dataset builders, run_record.py, тесты)
- `pipeline/` (кроме `shm_reader.py` hot-path) — `time_tracker.py`, `fusion.py`, `vote_engine.py`, `log_utils.py`
- `api/audit_logger.py`, `api/camera_control.py`, `api/camera_io.py`
- `Kabirhan-Frontend/src/` — UI компоненты, store, типы
- `docs/` — markdown
- `~/race_vision_bench/` — отдельная папка вне репо

---

## 4. Goal-Driven Patterns

### 5 типичных задач — переписать в goal-driven форме

| Задача (raw) | Goal-driven форма |
|---|---|
| «Поправить classifier — путает цвета» | **Сделать**: добавить EMA smoothing в `ColorTracker` для (cam, track_id) с alpha=0.5 → **проверка**: на `data/videos/test_full_loop/yaris_20260421_180010/` отношение `flips_per_track / total_obs` ≤ 5% (baseline текущий ≥ 15% по `audit_logger.py` логу). |
| «Замени YOLO11s на новую модель» | **Сделать**: подменить `models/yolo11s_person_960.engine` на `yolo26s_b25_gpu0_fp16.engine` в `deepstream/configs/nvinfer_racevision.txt` и одной строке env-launcher → **проверка**: `scripts/rv.sh start` за 60 с поднимает SHM, `tools/run_record.py --config configs/cameras_yaris.json --run-id post_yolo26s` дает `last_seq ≥ 13800` (≈ baseline) и `torn=0`. |
| «Pipeline ловит зрителей вместо жокеев» | **Сделать**: добавить ROI-маску на cam-01..04 в `configs/camera_roi_normalized.json` → **проверка**: `tools/run_record.py` на `test_full_loop/yaris_20260421_174240` → детекций на cam-01 ≤ 3000 (сейчас 15113), real_jockey survival rate (h≥100 ∧ conf≥0.7) ≥ 5% (сейчас 1.4%). |
| «Frontend не показывает скорость жокея» | **Сделать**: в `api/deepstream_pipeline.py` добавить `speed_mps` в payload WebSocket → **проверка**: при `--video data/videos/test_short/yaris_20260421_141439/` фронт `http://localhost:5173/operator` рисует `km/h` для каждого активного цвета, значения в диапазоне `0..70 km/h`. |
| «Reader иногда отстаёт» | **Сделать**: добавить логирование `reads_total / last_seq` ratio в `pipeline/shm_reader.py.get_stats()` каждые 10 с → **проверка**: `tools/run_record.py` 9-минутного прогона показывает в логах N≥50 строк `STATS_1S shm_catchup=…`, и `last_seq - reads_total ≤ 10%` (текущий 99.8%). |

### Pattern: Dead/Unclear-status code

❌ **Не делать:** удалять обнаруженный «мёртвый» код (legacy modules, старые `.bak` configs, неиспользуемые функции) попутно с основной задачей.

✅ **Делать:** зафиксировать находку в виде TODO с локацией (`file:line`) + 1 строкой почему похож на dead/unclear, и **продолжить основную задачу**. Очистка — отдельный pass с явным согласованием. Это применимо к: `pipeline/jockey_reid.py`, `pipeline/clip_color_classifier.py`, `pipeline/trigger.py`, `pipeline/analyzer.py`, `pipeline/trt_inference.py`, `configs/*.bak*`, top-level `cameras_1cam*.json` / `cameras_5cam.json`.

---

## 5. Notes

### Язык работы
- **Русский** для общения и комментариев в коде.
- **Английский** для технических терминов (SHM, seqlock, batch, throughput, etc.) и идентификаторов в коде.

### Главный файл контекста: `memory/MEMORY.md`
- **Читать перед началом работы.** Содержит индекс всех известных фактов о проекте: архитектура, прошлые сессии, эмпирические находки, quirks DeepStream / SAM 3, активный whitelist файлов, план следующих задач.
- **Обновлять при значимых изменениях.** Новый коммит, новая эмпирика, новый quirk — добавить запись в memory + ссылку в `MEMORY.md`.
- **Различие с CLAUDE.md:** CLAUDE.md описывает **ПРАВИЛА** (как работать), MEMORY.md — **ТЕКУЩЕЕ СОСТОЯНИЕ** (что есть и что было).

### KZ network gotcha
Если нужно достать GitHub raw-файлы и `raw.githubusercontent.com` блокируется казахстанским провайдером — использовать зеркало:
```
https://cdn.jsdelivr.net/gh/{user}/{repo}@{branch}/{path}
```
Пример: `https://cdn.jsdelivr.net/gh/forrestchang/andrej-karpathy-skills@main/CLAUDE.md`. Работает для публичных репо, поддерживает CDN-кэш.

### Open questions — упоминать, но не трогать

Эти аномалии замечены в ходе разведки, статус неясен. **НЕ чистить попутно** — только задокументировать здесь:

- **Q1.** `cameras_1cam.json`, `cameras_1cam_exp10.json`, `cameras_1cam_test.json`, `cameras_5cam.json` лежат в **корне репо** (а не в `configs/`). Назначение неясно — single-cam debug?
- **Q2.** `configs/camera_roi.json.bak`, `camera_roi_normalized.json.bak{,2,3}` — 4 backup'а ROI без явного владельца и порядка.
- **Q3.** 9 разных `cameras_yaris*.json` (yaris/old/new/cam02/cam13/cam14/4/g2rtc) — много вариаций, неочевидно какие активны. Нужен `configs/README.md`.
- **Q4.** `Dockerfile`, `docker-compose.yml`, `influx.env` в корне — для каких сервисов? Pipeline = bare-metal (см. soft guardrail #1).
- **Q5.** `admin/`, `bin/`, `ds_results/`, `logs/`, `results/` — output/aux-папки без README.
- **Q6.** `pipeline/jockey_reid.py`, `pipeline/clip_color_classifier.py` — есть в `pipeline/`, но `feedback_whitelist.md` их не упоминает.
- **Q7.** `runs/` растёт автоматически (каждый `deepstream/main.py` создаёт `exp_NNN_YYYYMMDD_HHMMSS/`). Когда чистить — не определено.
- **Q8.** `archive/cpp_legacy/` + `pipeline/trigger.py`/`analyzer.py`/`trt_inference.py` — статус «есть в коде, не в runtime». Без чтения memory это неочевидно.
