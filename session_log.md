# Session Log — Race Vision

Хронологический лог сессий: что менялось, что запускали, результаты экспериментов.

---

## 2026-03-26 — Capture/Inference/Display разделение

### Архитектура (реализовано)
Разделил `DeepStreamPipeline._reader_loop()` на 3 независимых потока:

```
DeepStream C++ (25 RTSP → YOLO → classify) → SHM /rv_detections
                                                ↓
                                    SHM Reader Thread (~1000hz poll)
                                                ↓
                                          DetectionBuffer (lock, 1 slot)
                                                ↓
                                    Inference Thread (voting/fusion, ~4fps effective)
                                                ↓
                                    WebSocket broadcast (async, 5-10hz)
                                                ↓
                                    Frontend canvas overlay (30fps rAF)

go2rtc WebRTC (25 потоков) ──────→ Frontend <video> (свой FPS)
```

4 независимых скорости. Никто никого не ждёт.

### Что изменено
- `api/server.py` — добавлен класс `DetectionBuffer` (thread-safe single-slot buffer)
- `DeepStreamPipeline` — один `_reader_loop` разделён на:
  - `_shm_reader_loop()` — Thread "SHMReader", читает SHM на макс скорости, пишет в DetectionBuffer
  - `_inference_loop()` — Thread "InferenceWorker", читает DetectionBuffer, делает voting/fusion/ranking
- `ranking_broadcast_loop()` — без изменений (уже был отдельный async loop, 5hz)
- Добавлен `shm_fps` в stats + лог показывает обе скорости: `SHM=Xhz Infer=Yfps`
- `stop()` — join обоих потоков с timeout

### Статус
Код изменён, синтаксис OK. Не протестировано, не закоммичено.

### Следующие шаги (план в memory/project_plan.md)

**Фаза 1: Рефакторинг структуры**
- api/server.py (1880 строк) → 4 файла: server_core, deepstream_sub, camera_io, ws_handlers
- pipeline.cpp (1263 строк) → 3 файла: pipeline, probe_handler, camera_activation
- backendConnection.ts (356 строк) → 2 файла: ws_connection, message_handlers

**Фаза 2: WebRTC production pipeline**
- DeepStream headless → SHM /rv_ws_blob (компактный бинарный формат для браузера)
- Python zero-copy прокси: mmap → websocket.send_bytes (без парсинга)
- go2rtc → 25 камер WebRTC passthrough
- Frontend: Web Worker + OffscreenCanvas + requestVideoFrameCallback + jitter buffer
- Один скрипт: ./scripts/run_all.sh

**Фаза 3: Track voting + Fusion**
- Per-track majority vote за 20+ кадров
- Python fusion → позиции на трассе → WebSocket рейтинг

**Фаза 4: Production**
- RTSP live тест, Docker-compose, мониторинг

**Блокер:** `sudo chown -R user:user Kabirhan-Frontend/node_modules`

---

## 2026-03-24 — v3 classifier + NV12→RGBA fix + mux coord fix

### Что сделано
1. **color_classifier_v3.pt** — SimpleCNN 64×64, 5 классов, обучена на 725 чистых кропах из data/color_dataset/
2. **NV12→RGBA fix** — добавлен nvvideoconvert + capsfilter(RGBA) перед probe в pipeline.cpp. CUDA kernel читал RGBA, а surface был NV12 → мусорная классификация. **Корневая причина плохой работы в DeepStream.**
3. **Mux coord fix** — bbox в mux-пространстве (2880×1620), а crop clamping был к source_frame (1280×720). Объекты с x>1280 получали отрицательный crop → фильтровались. Исправлено: fw/fh = config.mux_width/height.
4. **TRT engine version fix** — venv TRT=10.16, system TRT=10.9. C++ builder `/tmp/build_engine` линкуется с системной libnvinfer.
5. **ONNX export** — PyTorch 2.9 dynamo exporter несовместим с TRT 10.9, использован legacy exporter (opset 17).
6. **Классификация для tracker-only** — убрана проверка `if (!tracker_only)`, классифицируются все объекты.
7. **Чистый лог** — [STAT] каждые 50 кадров: fps, dets(yolo/trk), classify time, active cams.

### Результаты (cam-05, 1 камера, mux 2880×1620)
```
fps=25.0 стабильно
GREEN=100%, RED=100%, YELLOW=100%, BLUE=99%
classify: 55-73ms на 50-163 кропа за 50 кадров
```

### Файлы изменены
- `deepstream/src/pipeline.cpp` — nvvideoconvert RGBA, mux coords, classify all, summary log
- `deepstream/src/color_infer.cpp` — CROP_SIZE=64
- `deepstream/configs/nvinfer_jockey_1cam.txt` — threshold 0.30→0.15
- `models/color_classifier_v3.pt/.engine` — новая модель 5 классов
- `tools/train_color_v3.py`, `tools/build_dataset.py`, `tools/test_color_video.py`

### Результаты 25 камер (mux 1920×1080, exp4)
```
FPS: 31-36 (было 38-40 без nvvideoconvert)
classify: 900-1000ms на 150 кропов за 50 кадров (было 55-73ms на 1 камере)
cams: детекции на cam-05 и cam-01
interval=1 → yolo/trk = 50/50
```

**Проблемы:**
1. **classify 900ms** — nvvideoconvert конвертирует ВСЕ 25 surfaces NV12→RGBA каждый кадр, даже если детекций 0 на 24/25 камер. Главный bottleneck.
2. **Много UNK на 25 камерах** — mux 1920×1080 → кропы мельче чем 2880×1620 → v3 менее уверена (40-59% вместо 100%)
3. FPS упал с 38 до 31 из-за nvvideoconvert overhead

**Решения:**
- Переписать CUDA kernel на NV12 (убрать nvvideoconvert совсем) — 0 overhead, максимальный FPS
- Или снизить mux для восстановления FPS, но кропы будут ещё мельче

### NV12 CUDA kernel (убран nvvideoconvert)
Переписан CUDA kernel в color_infer.cpp — читает NV12 напрямую (Y plane + UV plane), конвертирует в RGB только маленький кроп жокея. Формула BT.601 из diag_logger.cpp. RGBA kernel оставлен как fallback.

**Результаты 25 камер после NV12 kernel (mux 1920×1080):**
```
FPS: 35-38 (было 31-36 с nvvideoconvert) ← улучшение
FPS без детекций: 38.0-38.2 (потолок)
classify: 800-900ms на 150 кропов за 50 кадров (было 900-1000ms)
0 dets: classify=5ms (был 6ms)
```

**Оставшиеся проблемы:**
1. **classify 800ms** — cudaMalloc/cudaFree на каждый crop в цикле. Нужен pre-allocated буфер.
2. **--display лагает** на 25 камерах — тайлер рисует все 25 потоков, тяжело для GPU. Без display = 38fps.
3. **Мало детекций** — cams=1/25 активна, остальные 0 dets (лошади видны на 3-5 камерах за запись)

### Файлы изменены (NV12)
- `deepstream/src/color_infer.cpp` — NV12 kernel + RGBA fallback, авто-определение формата surface
- `deepstream/src/pipeline.cpp` — убран nvvideoconvert/capsfilter, probe на tracker src pad

### Pre-allocate + batch kernel (убрана cudaMalloc per-call)
- d_rois_ выделяется один раз в load() вместо cudaMalloc/cudaFree на каждый кроп
- Кропы с одной камеры батчатся в один kernel launch

**Результаты 25 камер (mux 1920×1080, без display):**
```
FPS: 38.6-39.9 (было 35-38)
classify 150 crops: 340-370ms (было 800-900ms) — 2.7x ускорение
classify 0 crops: 0.4ms (было 5ms)
```

### Camera activation + display analysis (25 марта)

**Valve approach (failed):** Добавили valve элементы на каждый источник — active=3/25 работало, но fps упал до 1.0. nvstreammux с live-source=TRUE ждёт кадры от закрытых valve → stall. Откатили.

**Skip-classify approach (works):** Classify только для камер с недавними детекциями (3 сек cooldown). YOLO batch=25 работает на всех (быстро), classify — только на 1-3 активных. classify_active=1/25.

**Display lag deep analysis:**
```
per-frame avg: work=1.4ms  classify=1.4ms  display=0.0ms  | budget=40ms
GPU: 75%, VRAM: 4.4/12GB, CPU: 11%
```
Pipeline использует 3.5% бюджета, 97% времени спит. Display pipeline = 0ms в нашем коде. Лаг = **X11/EGL рендер** внутри nveglglessink при 25 NV12 surfaces от mux. Не исправить.

**Вывод:** DeepStream display для 25 камер непригоден. Production = headless + go2rtc WebRTC + Frontend. Display = только для отладки 1 камеры.

### Файлы изменены (25 марта)
- `deepstream/src/pipeline.h` — cam_last_det_, CAM_DEACTIVATE_FRAMES, update_camera_activation
- `deepstream/src/pipeline.cpp` — skip-classify для неактивных камер, детальный timing лог, queue для display, throttle 25fps, suppress drop warnings, show-source=0
- `deepstream/src/color_infer.h` — d_rois_ pre-allocated
- `deepstream/src/color_infer.cpp` — NV12 kernel, RGBA fallback, batch kernel launch, pre-alloc
- `deepstream/configs/nvinfer_jockey.txt` — threshold 0.25→0.15

### 5-камерный тест + surface debug (25 марта вечер)

**Конфиг:** cameras_5cam.json (cam-01, 03, 05, 08, 12), nvinfer_jockey_1cam.txt (batch=1, interval=0, threshold=0.15)

**Surface params идентичны 1cam vs 5cam:**
```
2880x1620, pitch=3072, fmt=6 (NV12_709_ER), numPlanes=2
plane[0] (Y):  offset=0, pitch=3072, 2880×1620
plane[1] (UV): offset=4976640, pitch=3072, 1440×810
NV12 kernel UV offset: src + 1620*3072 = src + 4976640 ✓
```

**Результат:**
```
1cam cam-05: GREEN=99%, classify=21ms
5cam cam-05: GREEN=82%→61%, classify=23ms, UNK=55%
```

**4 камеры (01, 03, 08, 12) = 0 детекций** — лошади слишком далеко/быстро на этих камерах.

**Причина деградации 99%→82% (ОШИБОЧНАЯ ГИПОТЕЗА, см. ниже):** предполагали FP16 микроразницы. На самом деле баг был в surface index.

### КРИТИЧЕСКИЙ БАГ: source_id != batch_id (25 марта, найден через crop comparison)

**Симптом:** 1 камера = GREEN 100%, 5+ камер = UNK 55-70%. Bbox координаты идентичны, surface params идентичны. Но crop images **совершенно разные** (mean_diff=61-86 из 255).

**Диагностика:** Сохранили кропы из 1cam и 5cam. На 1cam = жокей в зелёной куртке. На 5cam = **пустое поле с забором** при тех же bbox координатах. CUDA kernel читал пиксели из **чужой камеры**.

**Корневая причина:** `frame_meta->source_id` (номер камеры, постоянный 0-24) использовался как индекс в `surfaceList[]`. Но nvstreammux складывает кадры в batch **в произвольном порядке**. Правильный индекс = `frame_meta->batch_id`.

Пример: cam-05 (source_id=2) лежит в surfaceList[3], но код читал surfaceList[2] = cam-12 → кроп = пустое поле → UNK.

**Фикс:**
1. Добавлен `surface_index` в `TorsoROI` struct
2. `roi.surface_index = frame_meta->batch_id` в pipeline.cpp
3. `surfaceList[surf_idx]` вместо `surfaceList[cam_idx]` в color_infer.cpp

**Также исправлено:** NV12 UV offset теперь через `planeParams.offset[1]` вместо `height * pitch` (правильнее, хотя на практике совпадало).

**Результат:** 5 камер — классификация работает корректно, жокеи определяются правильно.

### Файлы изменены (batch_id fix)
- `deepstream/src/color_infer.h` — добавлен `surface_index` в TorsoROI
- `deepstream/src/color_infer.cpp` — `surfaceList[surf_idx]`, planeParams.offset/pitch
- `deepstream/src/pipeline.cpp` — `roi.surface_index = frame_meta->batch_id`

### WebRTC + Frontend план (26 марта)

**go2rtc тест:** 5 камер через WebRTC — без лагов. go2rtc скачан в /tmp/go2rtc, конфиг configs/go2rtc_files.yaml (файловые источники с -stream_loop -1).

**Архитектура production:**
```
DeepStream (headless, 38fps) → YOLO + classify → SHM
Python server (api/server.py) → SHM reader → WebSocket (bbox + цвета)
go2rtc → 25 камер → WebRTC (видео passthrough, 0 GPU)
Frontend (React) → WebRTC видео + WebSocket bbox → canvas overlay
```

**Что сделано:**
- `api/server.py` — live_detections теперь включает bbox, frame_w, frame_h
- `Kabirhan-Frontend/src/components/operator/CameraGrid.tsx` — добавлен DetectionOverlay canvas компонент (рисует bbox + цвета поверх WebRTC видео)
- `Kabirhan-Frontend/src/store/cameraStore.ts` — тип liveDetections обновлён (bbox + frame размеры)
- `Kabirhan-Frontend/src/services/backendConnection.ts` — убрана старая типизация
- `scripts/run_all.sh` — один скрипт запускает все 4 сервиса (go2rtc + DeepStream + Python + Frontend)

**Блокер:** node_modules в Kabirhan-Frontend принадлежат root. Нужно:
```bash
sudo chown -R user:user /home/user/race_vision/Kabirhan-Frontend/node_modules
```
Потом `./scripts/run_all.sh`

### Следующий шаг (завтра)
1. Починить permissions node_modules
2. Запустить ./scripts/run_all.sh — проверить WebRTC + bbox overlay
3. Track voting (накопление цвета по track_id)
4. Тест 25 камер
5. Тест на живых RTSP камерах

---

## 2026-03-21 — Full bbox + mux 1920, DINOv2 финальная проверка

### Что изменено
1. **color_infer.h** — торсо-кроп убран, передаётся весь bbox:
   - `TORSO_TOP=0.0, TORSO_BOTTOM=1.0, TORSO_LEFT=0.0, TORSO_RIGHT=0.0`
2. **diag_logger.cpp** — добавлено:
   - Сохранение отдельных кропов в `crops/` (JPG, имя = `b{batch}_{cam}_t{track}_{color}_{w}x{h}.jpg`)
   - Колонки `crop_w, crop_h, crop_pixels` в CSV
   - Статистика размеров кропов при завершении (min/avg/max)
   - Размер кропа подписан под bbox на фрейме
3. **mux** поднят до 1920×1080 (CLI `--mux-width 1920 --mux-height 1080`)

### Результаты прогона (cam-05, file mode, mux 1920×1080, full bbox)
```
Total detections: 1859
Crop width:  min=29  avg=59  max=121
Crop height: min=75  avg=117  max=205
Avg crop area: 6903 px
FPS: 324.6 (1 cam, file mode)
```
Кропы теперь чёткие — жокеи различимы визуально (цвет одежды, лошадь, посадка).

### DINOv2 ReID на новых кропах — ПРОВАЛ
Прогнали DINOv2 (dinov2_vits14) по 1859 кропам из ds_results/exp2/crops/:
```
Identified: 329/1859 (17.7%)  ← ХУЖЕ чем раньше
Unknown:    1530/1859 (82.3%)

Per-jockey: blue=167(9%), yellow=160(8.6%), red=1, green=1
Avg score unknown: 0.346 (порог 0.50)

Large crops (>8000px): только 33.2% identified
```
Сравнение:
- Старые торсо-кропы (28×28, mux 1520): 24% identified
- Новые full bbox (~59×117, mux 1920): 17.7% identified

**Вывод: DINOv2 ReID НЕ ПОДХОДИТ для этой задачи.** Проблема не в размере кропа — модель не может сопоставить IP-камерные кадры с галереей. Визуальное сходство (embedding distance) не работает при таком различии ракурсов/освещения/масштаба.

### Файлы результатов
- `ds_results/exp2/detections.csv` — CSV с crop_w, crop_h, crop_pixels
- `ds_results/exp2/frames/` — 965 аннотированных фреймов
- `ds_results/exp2/crops/` — 1859 отдельных кропов
- `tools/test_dinov2_crops.py` — скрипт проверки DINOv2

### Тест color_classifier_v1 (SimpleColorCNN, 5 классов, 64×64)
- `color_classifier.engine` (3.1M, SimpleColorCNN 3 conv, 64×64 input, 5 классов)
- **96% identified** (только 4% unknown), но **всё = YELLOW (87.9%)**
- t0 (green) → YELLOW ❌, t1 (red) → YELLOW ❌, t2 (yellow) → YELLOW ✅
- Модель слабая, обучена на плохих данных — не различает цвета

### Тест HSV классификатора (3 метода: histogram, pixel_vote, dominant_hue)
Пробовали разные варианты:
1. **Базовые HSV диапазоны** — 18.5% accuracy. Green жокей (H~96 teal) попадал в blue
2. **Расширенный green до H=110** — green жокей ловится, но штаны красного жокея (blue H=101-110) тоже = green
3. **Green до H=100, торсо 5-30%** — green ✅ 100%, yellow ✅ 100%, **red ❌** (зелёный фон/трек доминирует на мелких кропах)
4. **Главная проблема**: трекер нестабилен (t0 = сначала green, потом yellow), кропы 50-100px мелкие для HSV торсо-анализа

**Вывод: HSV не решает задачу.** Работает для yellow и green по-отдельности, но red путает с green из-за:
- Маленькие кропы → торсо-область = 30×44px, в ней мало пикселей куртки
- Зелёный фон трека и синие штаны доминируют над красной курткой
- Камерный white balance сдвигает зелёный в teal (H=81-100), что перекрывается с blue

### Выводы и направление
**Что НЕ работает:**
- ❌ DINOv2 ReID — не сопоставляет камерные кропы с галереей (17.7%)
- ❌ color_classifier_v2 (EfficientNet, 3 класса) — не знает blue/purple, путает yellow↔green
- ❌ color_classifier_v1 (SimpleCNN, 5 классов) — всё = yellow (87.9%)
- ❌ HSV histogram — шумно на мелких кропах, red неотличим от green фона

### Тест BoT-SORT-ReID (YOLOv12-BoT-SORT-ReID с HuggingFace)
- Repo: `wish44165/YOLOv12-BoT-SORT-ReID` → `/tmp/YOLOv12-BoT-SORT-ReID/`
- ReID модель: SBS_S50 (ResNeSt-50x, 2048-dim, MOT20), 411MB
- Трекер: BoT-SORT с appearance features
- Наш YOLO jockey_yolov11s + BoT-SORT-ReID на cam-05 (2000 кадров)

**Результат: 65 track ID** на ~3 жокеев (хуже нашего IOU: 10 ID)
- Основные треки: ID1=green(287), ID2=red(140), ID3=yellow(330), ID6=red(313), ID31/32/34=длинные
- ReID обучен на пешеходах (MOT20) — не подходит для жокеев на лошадях
- **Плюс**: кропы из оригинального видео 1280×720 чёткие, жокеи хорошо различимы
- Установленные пакеты: yacs, termcolor, tabulate, scikit-learn, faiss-cpu, tensorboard, lap, filterpy, cython_bbox, gdown

**Что нужно:**
Переобучить EfficientNet (v2 архитектура) на 5 классов используя реальные кропы с камер (1859 штук в ds_results/exp2/crops/).
Или обучить на более крупном датасете с аугментацией.

---

## 2026-03-20 — ReID тест на видео + архитектурное решение

### Что запускали
1. **test_deepstream_reid.py** — DeepStream (YOLO jockey_yolov11s) + DINOv2 ReID на cam-05 (file mode)
2. Добавлены: `--display`, `--log`, `--save-crops`, CSV, per-track stability, crop size distribution

### Результаты ReID (DINOv2, cam-05, 1295 кадров)
- **76% unknown** (2130/2796), только 24% получили ID
- Yellow — лучший: 506 детекций, avg 0.587
- Blue: 76, Green: 24, Purple: 58, Red: 2
- **97% кропов = "small" (500-2000px)** — торсо ~30x25px
- Трекер нестабилен: 11 track_id на ~5 лошадей, track_1 содержит 5 разных "жокеев"

### Диагноз по кропам (визуальный)
- Track 0 (98% unknown) = **человек в красной куртке у ограждения**, не жокей
- Много ложных детекций: люди, техника — YOLO ловит всех как person
- Реальные жокеи видны (yellow, blue), но кропы 30-70px — DINOv2 бесполезен на таком размере
- Галерея (data/reid/) из IP-камерных кропов 80-170px, runtime кропы в 3-5x мельче

### Архитектурное решение: HSV вместо ReID
**ReID — неправильный инструмент.** Задача = определить цвет 5 жокеев в ярких контрастных цветах, не re-identify людей. DINOv2/CLIP overkill, не работает на 30px.

**Новая архитектура (с нуля):**
```
YOLO detect → ROI filter → Velocity filter → HSV color → Track voting → SHM
```

1. **ROI маска** — полигон трека для каждой камеры, детекции вне трека = мусор (убьёт 90% FP)
2. **Velocity filter** — движение > 15px за 3 кадра (отсечёт стоящих людей)
3. **HSV цвет** — торсо-кроп → HSV гистограмма → доминантный Hue → один из 5 цветов
   - Red: H=0-10/170-180, Yellow: H=20-35, Green: H=35-85, Blue: H=100-130, Purple: H=130-170
   - Работает на ЛЮБОМ разрешении, не нужно обучение, не нужен GPU
4. **Track-level voting** — не per-frame, а аккумулятор по track_id, вердикт при >60% и >5 голосов
5. **CNN** — только second opinion когда HSV неуверен

**Не нужно:**
- ❌ DINOv2/CLIP/ReID
- ❌ Телефонные фото
- ❌ Per-frame решения
- ❌ Phase 2 перечитывание видео

### Файлы изменены
- `tools/test_deepstream_reid.py` — добавлены `--display`, `--log`, `--save-crops`, CSV, детальная диагностика
- `pipeline/jockey_reid.py` — незакоммиченный фикс last_hidden_state (от 18 марта)

### Файлы результатов
- `ds_results/reid.log` — полный лог детекций
- `ds_results/reid.csv` — CSV с top-3 similarity, crop sizes
- `ds_results/reid_output.mp4` — аннотированное видео
- `ds_results/crops/` — 500 кропов для визуальной проверки

### Следующие шаги (завтра)
1. Написать HSV классификатор, прогнать по 500 кропам из ds_results/crops/
2. ROI маска для cam-05
3. Velocity filter — включить, подобрать порог
4. Track voting аккумулятор
5. Интегрировать в C++ pipeline

---

## 2026-03-18 — ReID gallery build + test

### Контекст
Прошлая сессия (до отключения света): нарезаны кропы жокеев в `data/reid/` (5 цветов, 137 файлов), но `gallery_embeddings.pkl` не был построен.

### Что сделано
1. **CLIP gallery** — cross-similarity 0.80–0.94 между жокеями → слишком высокая, непригоден
2. **DINOv2 gallery** — cross-similarity 0.63–0.88, лучше. Red отличается лучше всего (0.63–0.79)
3. **Тест accuracy** на 25 кропах из галереи: **92% (23/25)**
   - 2 ошибки = `unknown` (score < 0.50), НЕ перепутанные цвета
   - blue: 5/5, green: 3/5, purple: 5/5, red: 5/5, yellow: 5/5
4. Размер кропов: 75-164px — мелкие, с IP-камер (не телефонные фото)

### Файлы
- `data/reid/gallery_embeddings.pkl` — DINOv2 эмбеддинги, 137 штук, 384-dim
- Backend: dinov2 (ViT-S/14)

### Статус
Галерея готова. Следующий шаг — интеграция в пайплайн или тест на независимых кропах (не из галереи).

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
