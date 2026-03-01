# Race Vision — Архитектура и план системы

## Общая схема

```
  25 аналитических камер (Hikvision, RTSP)        1 PTZ камера (трансляция)
         |                                                 |
    go2rtc (порт 1984)                               go2rtc → MSE
    RTSP → MSE/WebRTC проксирование                        |
         |                                           PublicDisplay
         |                                     (зрители видят чистый эфир)
         |
  ┌──────┴──────────────────────────────────────────┐
  │              Python Backend (GPU)                │
  │                                                  │
  │  ┌─────────────────────┐  ┌───────────────────┐  │
  │  │   TRIGGER LOOP      │  │  ANALYSIS LOOP    │  │
  │  │   (лёгкий)          │  │  (тяжёлый)        │  │
  │  │                     │  │                   │  │
  │  │ YOLOv8n @ 640px     │  │ YOLOv8s @ 1280px  │  │
  │  │ 3 fps × 25 камер    │→→│ 5 fps × 3-5 камер│  │
  │  │ "Есть лошади?"      │  │ + Color CNN       │  │
  │  │ ~50ms на батч        │  │ + VoteEngine      │  │
  │  └─────────┬───────────┘  └──────────┬────────┘  │
  │            │                          │           │
  │     CameraManager              FusionEngine       │
  │     (активация/                (pixel → метры,    │
  │      завершение)                глобальный ранг)  │
  │            │                          │           │
  │            └──────────┬───────────────┘           │
  │                       │                           │
  │                  API Server                       │
  │              (FastAPI + WebSocket)                 │
  └───────────────────────┬───────────────────────────┘
                          │
               WebSocket /ws (позиции, ранги)
                          │
         ┌────────────────┴────────────────┐
         │                                 │
   OperatorPanel (:3000/operator)    PublicDisplay (:3000/)
   - 25 камер через go2rtc (субпотоки)   - PTZ видео
   - 2D трек с позициями                 - Таблица рангов
   - Настройки гонки                     - Анимированные иконки жокеев
```

---

## Два пайплайна — ключевая идея

### Зачем

RTX 3060 (12GB) не потянет тяжёлую модель на 25 камерах одновременно.
Тест показал: YOLOv8s @ 1280px = 100% GPU на 3 камерах.

**Решение:** две модели с разной нагрузкой.

### Pipeline 1: Trigger (лёгкий, все камеры)

| Параметр | Значение |
|----------|----------|
| Модель | YOLOv8n (nano, 6MB) |
| Размер | 640px |
| FPS | 3 fps на камеру |
| Камеры | Все 25 одновременно |
| Батч | ~50ms на 4 камеры (тест) |
| Задача | Ответ на вопрос: "Есть ли лошади на этой камере?" |

Файл: `pipeline/trigger.py`

### Pipeline 2: Analyzer (тяжёлый, только активные камеры)

| Параметр | Значение |
|----------|----------|
| Модель YOLO | YOLOv8s (small, 22MB) |
| Размер | 1280px |
| FPS | 5 fps на камеру |
| Камеры | Только активные (3-5 макс.) |
| + Классификатор | SimpleColorCNN (torso crop → 5 цветов) |
| + Голосование | VoteEngine (позиции по X-координате) |
| Задача | Определить цвета жокеев и их порядок |

Файл: `pipeline/analyzer.py`

---

## Жизненный цикл камеры

```
        TRIGGER видит лошадей
                │
                ▼
    ┌─── ACTIVATE камеру ───┐
    │                        │
    │  ANALYZER начинает     │
    │  обработку (5 fps)     │
    │                        │
    │  VoteEngine копит      │
    │  голоса по позициям    │
    │                        │
    │  Через 3 сек ИЛИ      │
    │  все 5 позиций есть:   │
    │                        │
    ▼                        │
  RESULT READY               │
  (order=[blue>red>...])     │
                │
                ▼
       COMPLETED камеру
       (больше не активируется
        триггером в этой гонке)
                │
                ▼
       Следующая камера...
```

### Тест на exp11 (4 камеры)

| Камера | Активация | Результат | Время | Порядок |
|--------|-----------|-----------|-------|---------|
| cam1 | 17:52:09 | 17:52:13 | **4 сек** | blue > red > purple |
| cam2 | 17:52:56 | 17:52:58 | **2 сек** | green > red > yellow > blue > purple |
| cam3 | 17:53:04 | 17:53:07 | **3 сек** | red > yellow > blue |
| cam4 | 17:53:19 | 17:53:23 | **4 сек** | green > blue > purple |

cam2 увидела все 5 жокеев → завершилась раньше лимита.

---

## Состояния камеры (CameraManager)

```
   IDLE                 Камера зарегистрирована, триггер сканирует
    │
    ▼  (trigger: count > 0)
  ACTIVE                Анализатор обрабатывает кадры
    │
    │  (result ready ИЛИ timeout 3 сек)
    ▼
  COMPLETED             Результат получен, триггер игнорирует
    │
    │  (новая гонка → reset)
    ▼
   IDLE
```

Файл: `pipeline/camera_manager.py`

- `max_active` — макс. одновременно активных камер (ограничение GPU)
- `cooldown_sec` — сколько держать камеру активной после последнего триггера
- `completed` — множество камер с финальным результатом (не реактивируются)

---

## Модули pipeline/

### trigger.py — TriggerLoop
- Фоновый поток, сканирует все 25 камер
- YOLOv8n @ 640px, батч-детекция
- Фильтры: мин. высота bbox (50px), aspect ratio (0.8)
- Если >= 1 детекция → `camera_manager.activate(cam_id)`
- Логирует ACTIVATE / DEACTIVATE

### analyzer.py — AnalysisLoop
- Фоновый поток, обрабатывает только активные камеры
- YOLOv8s @ 1280px, батч-детекция
- Фильтры: мин. высота (100px), aspect ratio (1.2), edge margin (10px)
- Извлечение торса (top 10-40%, left/right 20%)
- SimpleColorCNN → 5 цветов + HSV кросс-чек
- Отправка в VoteEngine
- Проверка завершения: all positions filled ИЛИ timeout 3 сек
- Пометка `camera_manager.mark_completed(cam_id)`

### vote_engine.py — VoteEngine
- По одному на камеру
- Сортировка детекций по X (правее = впереди)
- Уникальность цветов через softmax reassignment
- Взвешенное голосование: 5 видимых → вес 5, 4 → 2, 3 → 1
- 3-проходное вычисление результата (strict → relaxed → fallback)
- `is_result_ready()` — все позиции заполнены с достаточными голосами

### camera_manager.py — CameraManager
- Регистрация камер (analytics / display)
- Активация/деактивация с cooldown
- Состояние completed (камера отработала)
- Ограничение max_active (GPU бюджет)
- Потокобезопасный (threading.Lock)

### fusion.py — FusionEngine
- Объединяет результаты всех камер в глобальную позицию
- pixel_x → track_m через TrackTopology
- EMA-сглаживание позиций
- Глобальный ранг: выше position_m → впереди

### track_topology.py — TrackTopology
- Карта: (camera_id, pixel_x) → метры на трассе
- 25 камер покрывают 2500м овал
- Зоны перекрытия (~10м) между соседними камерами
- Взвешенное слияние (центр FOV → больший вес)

### trt_inference.py — YOLODetector, ColorClassifierInfer
- Обёртка для YOLO и Color CNN
- TensorRT engine (если есть) или PyTorch fallback
- Батч-детекция и батч-классификация

---

## Камеры (25 Hikvision)

### Конфигурация

| Файл | Назначение |
|------|-----------|
| `cameras_example.json` | Бэкенд — все камеры + track_start/track_end |
| `go2rtc.yaml` | go2rtc — RTSP потоки (main + sub) |
| `Kabirhan-Frontend/src/config/cameras.ts` | Фронтенд — cam IDs + go2rtc IDs |

### Потоки

| Тип | URL паттерн | Разрешение | Для чего |
|-----|------------|------------|----------|
| Main (101) | `.../Channels/101` | 1080p+ | Аналитика (YOLO) |
| Sub (102) | `.../Channels/102` | D1/CIF | Превью в браузере |

- Оператор видит **субпотоки** через go2rtc (MSE) — браузер тянет 25 штук
- Аналитика работает с **основными** потоками (1080p) для точности детекции

### PTZ камера

1 PTZ камера для трансляции зрителям (ptz-1). Отдельный пароль, не участвует в аналитике.

---

## Инфраструктура (Docker)

```
docker-compose.yml
├── go2rtc        (alexxit/go2rtc, порт 1984, host network)
│   └── RTSP → MSE/WebRTC проксирование для всех камер
├── frontend      (React + Nginx, порт 3000)
│   ├── /operator  — панель оператора
│   └── /          — публичный экран
└── backend       (Python + GPU, порт 8000)  ← TODO: запустить с новым pipeline
    ├── Trigger + Analyzer + Fusion
    └── WebSocket API → фронтенд
```

---

## Цвета и жокеи

5 цветов жокейской формы:

| Цвет | ID | Иконка | HEX |
|------|----|--------|-----|
| Красный | horse-1 | silk_1.svg | #DC2626 |
| Синий | horse-2 | silk_2.svg | #2563EB |
| Зелёный | horse-3 | silk_3.svg | #16A34A |
| Жёлтый | horse-4 | silk_4.svg | #FBBF24 |
| Фиолетовый | horse-5 | silk_5.svg | #9333EA |

---

## WebSocket протокол

### Backend → Frontend

**Список жокеев:**
```json
{"type": "horses_detected", "horses": [{"id": "horse-1", "silkId": 1, "color": "#DC2626", ...}]}
```

**Старт гонки:**
```json
{"type": "race_start", "race": {"trackLength": 2500, "status": "racing"}}
```

**Обновление позиций (каждые 200 мс):**
```json
{"type": "ranking_update", "rankings": [
  {"id": "horse-5", "silkId": 5, "position": 1, "distanceCovered": 1850.5, "gapToLeader": 0},
  {"id": "horse-1", "silkId": 1, "position": 2, "distanceCovered": 1780.2, "gapToLeader": 1.5}
]}
```

**Переключение камеры:**
```json
{"type": "camera_switch", "cameraId": "ptz-1"}
```

### Frontend → Backend

| Сообщение | Описание |
|-----------|----------|
| `{"type": "ping"}` | Heartbeat (каждые 5 сек) |
| `{"type": "start_race"}` | Старт гонки |
| `{"type": "stop_race"}` | Остановка гонки |
| `{"type": "get_state"}` | Запрос текущего состояния |

---

## Фронтенд

### Страницы

| URL | Назначение |
|-----|-----------|
| `/` | PublicDisplay — PTZ видео + анимированные иконки жокеев |
| `/operator` | OperatorPanel — 25 камер + трек + настройки |

### Видеоплеер — Go2RTCPlayer (MSE)

- WebSocket → MediaSource Extensions (не WebRTC)
- Субпотоки (Channels/102) для оператора — 25 камер без нагрузки
- Exponential backoff при реконнекте (5s → 10s → max 60s)
- Staggered connection delays (300ms между камерами)

### Анимация

- Плавное перемещение иконок (3 сек, tween)
- Дуга вверх (обгон) / вниз (отставание)
- Подсветка: зелёная (обогнал) / красная (отстал)
- Стрелки: +N / -N позиций

---

## Что сделано

- [x] 25 камер настроены в go2rtc (main + sub потоки)
- [x] Фронтенд: оператор видит 25 камер через MSE субпотоки
- [x] Фронтенд: публичный экран с PTZ + рангами
- [x] Pipeline: TriggerLoop (YOLOv8n @ 640)
- [x] Pipeline: AnalysisLoop (YOLOv8s @ 1280 + Color CNN)
- [x] Pipeline: VoteEngine (голосование по позициям)
- [x] Pipeline: CameraManager (activate / completed / max_active)
- [x] Pipeline: FusionEngine (multi-camera → global position)
- [x] Pipeline: TrackTopology (pixel → метры)
- [x] Логика быстрого завершения (3 сек timeout или all positions filled)
- [x] Тест на exp11: 4 камеры, каждая завершается за 2-4 сек

## Что нужно доделать

- [ ] **Backend API сервер** — запустить FastAPI с новым двухпайплайновым pipeline (trigger + analyzer + fusion), отправлять ranking_update через WebSocket
- [ ] **RTSP → Pipeline** — подключить реальные камеры к trigger/analyzer (сейчас работает на видеофайлах)
- [ ] **TensorRT** — конвертировать .pt модели в .engine для ускорения (сейчас PyTorch fallback)
- [ ] **GPU бюджет** — протестировать сколько камер потянет RTX 3060 с TensorRT (ожидание: 5-8 активных)
- [ ] **Fusion → WebSocket** — соединить FusionEngine с API для отправки ranking_update на фронтенд
- [ ] **Оператор: статус камер** — показывать на фронте какие камеры active/completed/idle
- [ ] **PTZ пароль** — проверить пароль для ptz-1 (10.223.70.43)

---

## Запуск

### Текущий (видео из фронтенда)

```bash
docker compose up -d          # go2rtc + frontend
# http://localhost:3000/operator  — оператор (25 камер)
# http://localhost:3000/          — публичный экран
```

### Тест pipeline на видеофайлах

```bash
./venv/bin/python test_pipeline_exp11.py
```

### Полная система (TODO)

```bash
docker compose --profile gpu up -d   # go2rtc + frontend + backend (с GPU)
```

---

## Требования

- **GPU**: NVIDIA RTX 3060 12GB (или выше)
- **CUDA**: 11.8+
- **Python**: 3.10+ с PyTorch + ultralytics
- **Node.js**: 18+
- **Docker**: с nvidia-container-toolkit
- **Камеры**: 25x Hikvision (RTSP, admin/Zxcv2Zxcv2)
