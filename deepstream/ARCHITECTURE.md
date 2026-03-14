# DeepStream — Архитектура системы Race Vision

> **Про Python DeepStream SDK**: У NVIDIA есть `pyds` (Python bindings для DeepStream).
> Тот же пайплайн можно написать на Python через `Gst.ElementFactory.make()` + `pyds.gst_buffer_get_nvds_batch_meta()`.
> Но C++ даёт прямой доступ к CUDA-ядрам и TensorRT без обёрток — критично для кастомного
> CUDA kernel кропа торса и прямого TensorRT inference.

---

## Как используется DeepStream C++ SDK (GStreamer API)

DeepStream SDK — надстройка над **GStreamer**. Весь код написан на C++ и использует GStreamer C API + DeepStream плагины + NVIDIA-специфичные элементы.

### Инициализация GStreamer

```cpp
// pipeline.cpp:33-36
gst_init(nullptr, nullptr);                           // инициализация GStreamer
loop_ = g_main_loop_new(nullptr, FALSE);              // event loop GLib
pipeline_ = gst_pipeline_new("race-vision-pipeline"); // контейнер-пайплайн
```

`GMainLoop` крутится в `main()` через `g_main_loop_run(loop)` — обрабатывает все события.

### Создание элементов — `gst_element_factory_make()`

```cpp
// DeepStream-элементы (NVIDIA):
gst_element_factory_make("nvstreammux", "streammux");      // батчинг N камер
gst_element_factory_make("nvinfer", "yolo-infer");         // TRT-инференс
gst_element_factory_make("nvtracker", "tracker");          // IOU-трекинг
gst_element_factory_make("nvmultistreamtiler", "tiler");   // мозаика камер
gst_element_factory_make("nvdsosd", "osd");                // отрисовка bbox
gst_element_factory_make("nvvideoconvert", "conv");        // конвертер
gst_element_factory_make("nv3dsink", "sink");              // вывод на экран

// Стандартные GStreamer-элементы:
gst_element_factory_make("uridecodebin", "source-0");     // RTSP-источник
gst_element_factory_make("fakesink", "fakesink");          // /dev/null
```

### Настройка свойств — `g_object_set()`

```cpp
// nvstreammux — батчит кадры со всех камер в один тензор
g_object_set(G_OBJECT(streammux_),
    "batch-size",  25,     "width",  800,     "height",  800,
    "batched-push-timeout", 40000,
    "live-source", TRUE,   "enable-padding", TRUE,   NULL);

// nvinfer — TensorRT-инференс (YOLOv8)
g_object_set(G_OBJECT(nvinfer),
    "config-file-path", "configs/nvinfer_yolov8s.txt",
    "unique-id", 1,   NULL);

// nvtracker — IOU-трекер
g_object_set(G_OBJECT(tracker),
    "tracker-width", 800,  "tracker-height", 800,
    "ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
    "ll-config-file", "configs/tracker_iou.yml",   NULL);
```

### Сборка пайплайна — `gst_bin_add()` + `gst_element_link()`

```cpp
gst_bin_add(GST_BIN(pipeline_), streammux_);
gst_bin_add(GST_BIN(pipeline_), nvinfer);
gst_bin_add(GST_BIN(pipeline_), tracker);

gst_element_link(streammux_, nvinfer);    // streammux → nvinfer
gst_element_link(nvinfer, tracker);       // nvinfer → tracker
// или одной строкой:
gst_element_link_many(link_from, osd, tiler, conv, sink, NULL);
```

### Динамическая линковка (pad-added)

`uridecodebin` создаёт pad'ы **после** подключения к RTSP. Линковка через callback:

```cpp
g_signal_connect(source, "pad-added", G_CALLBACK(on_pad_added), this);

void Pipeline::on_pad_added(GstElement* src, GstPad* pad, gpointer data) {
    // Только видео pad'ы
    if (!g_str_has_prefix(name, "video/")) return;

    // Запрашиваем sink-pad у muxer: "sink_0", "sink_1", ... "sink_24"
    GstPad* mux_pad = gst_element_request_pad_simple(streammux_, "sink_0");
    gst_pad_link(pad, mux_pad);  // uridecodebin:src → nvstreammux:sink_0
}
```

### Probe — Сердце всей логики

Callback на каждом буфере, проходящем через pad:

```cpp
GstPad* src_pad = gst_element_get_static_pad(tracker, "src");
gst_pad_add_probe(src_pad, GST_PAD_PROBE_TYPE_BUFFER, inference_probe, this, nullptr);

GstPadProbeReturn Pipeline::inference_probe(GstPad*, GstPadProbeInfo* info, gpointer data) {
    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);

    // 1. Batch-метаданные (все камеры)
    NvDsBatchMeta* batch = gst_buffer_get_nvds_batch_meta(buf);

    // 2. GPU-поверхность (для CUDA-кропов)
    NvBufSurface* surface = ...;

    // 3. Итерация по кадрам (frame = камера)
    for (NvDsMetaList* l = batch->frame_meta_list; l; l = l->next) {
        NvDsFrameMeta* frame = static_cast<NvDsFrameMeta*>(l->data);
        int cam_idx = frame->source_id;

        // 4. Итерация по детекциям
        for (NvDsMetaList* lo = frame->obj_meta_list; lo; lo = lo->next) {
            NvDsObjectMeta* obj = static_cast<NvDsObjectMeta*>(lo->data);
            float x1   = obj->rect_params.left;
            float conf = obj->confidence;
            uint64_t track_id = obj->object_id;  // от nvtracker!
        }
    }
    return GST_PAD_PROBE_OK;
}
```

### Доступ к GPU-памяти (NvBufSurface) — Zero-Copy

```cpp
const auto& params = surface->surfaceList[cam_idx];
const uint8_t* gpu_ptr = static_cast<const uint8_t*>(params.dataPtr);  // GPU!
// CUDA-ядро читает ПРЯМО из буфера DeepStream:
crop_resize_normalize_kernel<<<blocks, threads, 0, stream>>>(gpu_ptr, ...);
```

### Кастомный YOLO-парсер

```cpp
// Компилируется в libnvdsinfer_yolov8_parser.so
extern "C" bool NvDsInferParseYoloV8(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    const NvDsInferNetworkInfo& networkInfo,
    const NvDsInferParseDetectionParams& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList);
// Подключается через: custom-lib-path + parse-bbox-func-name в конфиге nvinfer
```

### Какие API используются

| Слой | API | Для чего |
|------|-----|----------|
| **GStreamer** | `gst_element_factory_make`, `gst_bin_add`, `gst_element_link`, `gst_pad_add_probe` | Пайплайн |
| **GLib** | `g_main_loop_run`, `g_object_set`, `g_signal_connect` | Event loop, свойства |
| **DeepStream** | `gst_buffer_get_nvds_batch_meta`, `NvDsBatchMeta/FrameMeta/ObjectMeta` | Метаданные |
| **NvBufSurface** | `surfaceList[i].dataPtr` | GPU-кадры |
| **TensorRT** | `createInferRuntime`, `deserializeCudaEngine`, `enqueueV3` | Цвет-inference |
| **CUDA** | `cudaMalloc`, kernel `<<<>>>` | Crop/resize/normalize |
| **POSIX IPC** | `shm_open`, `mmap`, `sem_post` | Данные → Python |

---

## Общая картина

Система анализирует **25 камер** на скаковой дорожке в реальном времени.
Цель — обнаружить жокеев, присвоить каждому стабильный ID (трекинг) и определить **цвет формы** (зелёный, красный, жёлтый, синий, фиолетовый).

Результаты передаются в Python через **POSIX shared memory** — без сетевого оверхеда, с задержкой ~микросекунды.

---

## Двойной пайплайн (Dual Pipeline)

Ключевая идея — **два конвейера** с разной задачей:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRIGGER PIPELINE                            │
│                                                                     │
│  25 камер ─► nvstreammux (640×640) ─► YOLOv8n ─► trigger_probe     │
│                                                      │              │
│                                          подсчёт людей на кадре     │
│                                          решение: какие камеры      │
│                                          сейчас «активны»           │
│                                                      │              │
│                                                      ▼              │
│                                              activation_callback    │
└──────────────────────────────────────────────────────┬──────────────┘
                                                       │
                                          activate() / deactivate()
                                                       │
                                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        ANALYSIS PIPELINE                            │
│                                                                     │
│  25 камер ─► valve (drop=TRUE/FALSE) ─► nvstreammux (800×800)      │
│                                             │                       │
│                                       YOLOv8s (тяжёлая модель)     │
│                                             │                       │
│                                        IOU Tracker                  │
│                                             │                       │
│                                      inference_probe                │
│                                             │                       │
│                                    ┌────────┴────────┐              │
│                                    │  Фильтрация     │              │
│                                    │  Crop торса      │              │
│                                    │  Color classify  │              │
│                                    │  Запись в SHM    │              │
│                                    └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

### Почему два пайплайна?

| | Trigger | Analysis |
|---|---------|----------|
| **Модель** | YOLOv8**n** (nano) | YOLOv8**s** (small) |
| **Разрешение** | 640×640 | 800×800 |
| **Камер одновременно** | Все 25 | 3–8 активных |
| **Задача** | «Где сейчас лошади?» | «Кто именно и какого цвета?» |
| **Нагрузка** | Лёгкая | Тяжёлая |

Trigger работает **постоянно** на всех камерах. Как только он видит людей на камере — открывает «вентиль» (valve) для Analysis. Это экономит GPU: тяжёлая модель работает только там, где есть действие.

---

## Trigger Pipeline — Подробности

**Файлы**: `src/trigger_pipeline.cpp`, `src/trigger_pipeline.h`

### Probe-функция (trigger_probe)

На каждом кадре:
1. Итерирует по батчу из 25 фреймов
2. Для каждой камеры считает детекции с `class_id=0` (person)
3. Фильтрует по:
   - Минимальная высота bbox ≥ 50 px
   - Aspect ratio ≥ 0.8 (чтобы отсечь горизонтальные артефакты)
4. Передаёт счётчики в `update_activations()`

### Логика активации (update_activations)

```
Для каждой камеры:
  ├─ Есть детекции? → Обновить last_seen_time
  └─ Нет детекций дольше cooldown (3 сек)? → Деактивировать

Если свободных слотов нет (max_active=8):
  └─ Вытесняется самая «старая» активная камера (LRU)

Результат: bitmask активных камер → записывается в Trigger SHM
           callback → analysis_pipeline.activate(cam) / deactivate(cam)
```

---

## Analysis Pipeline — Подробности

**Файлы**: `src/analysis_pipeline.cpp`, `src/analysis_pipeline.h`

### Valve (вентиль) — Главная хитрость

Все 25 RTSP-сессий **подключены постоянно** (чтобы не терять время на переподключение).
Каждая камера проходит через GStreamer-элемент `valve`:
- `drop=TRUE` → кадры отбрасываются, не попадают в inference
- `drop=FALSE` → кадры идут в muxer → inference

Переключение вентиля — **мгновенное**, без задержки на RTSP handshake.

### Inference Probe (inference_probe)

На каждом кадре для каждой детекции:

```
1. ФИЛЬТРАЦИЯ
   ├─ Не у края кадра (margin ≥ 10 px)
   ├─ Высота bbox ≥ 65 px
   ├─ Aspect ratio ≥ 1.2 (выше, чем шире — фигура человека)
   └─ Площадь торса: 400–15000 px

2. CROP ТОРСА (CUDA)
   ├─ Берём 10–40% высоты bbox (зона торса жокея)
   ├─ Убираем 20% по краям (руки)
   ├─ Resize до 64×64 на GPU
   └─ ImageNet-нормализация (mean/std)

3. COLOR CLASSIFICATION (TensorRT)
   ├─ Батч до 128 торсов
   ├─ Модель выдаёт 3 класса: green, red, yellow
   └─ Softmax → маппинг в 5-слотовый формат

4. ЗАПИСЬ В SHM
   ├─ bbox (x1, y1, x2, y2)
   ├─ center_x (для ранжирования)
   ├─ det_conf (уверенность YOLO)
   ├─ track_id (от IOU-трекера)
   ├─ color_id + color_conf
   └─ color_probs[5] (все вероятности)
```

---

## Трекинг (IOU Tracker)

**Конфиг**: `configs/tracker_iou.yml`

Простой, но эффективный IOU-based трекер от NVIDIA:
- Ассоциация по IoU (пересечение bbox между кадрами)
- `maxShadowTrackingAge=30` — помнит объект 30 кадров после потери
- `probationAge=3` — назначает стабильный ID после 3 кадров
- До 30 объектов на камеру

Каждый жокей получает **уникальный track_id**, который сохраняется между кадрами.

---

## Shared Memory — Протокол обмена данными

### Detection SHM (`/rv_detections`)

```
ShmHeader
├─ write_seq (uint64)          ← атомарный счётчик записей
├─ num_cameras (uint32)
└─ cameras[25]
    └─ CameraSlot
        ├─ cam_id[16]          (например "cam-01")
        ├─ timestamp_us        (микросекунды)
        ├─ frame_width/height
        ├─ num_detections      (0..20)
        └─ detections[20]
            └─ Detection (56 байт)
                ├─ x1, y1, x2, y2
                ├─ center_x
                ├─ det_conf
                ├─ track_id
                ├─ color_id
                ├─ color_conf
                └─ color_probs[5]
```

**Размер**: ~31 KB на весь SHM (25 камер × 20 детекций × 56 байт)

### Протокол записи (C++)
1. Lock mutex
2. `memcpy()` каждый слот камеры
3. Atomic increment `write_seq`
4. Memory fence
5. `sem_post()` — будим читателя

### Протокол чтения (Python)
1. `sem_timedwait(200ms)` — ждём сигнала
2. Проверяем `write_seq` — изменился?
3. Читаем все слоты
4. Парсим структуры → Python-объекты

### Trigger SHM (`/rv_trigger`)

Компактная структура:
```
TriggerShmHeader
├─ write_seq
├─ active_mask        ← битовая маска активных камер
├─ num_cameras
├─ detection_counts[25]  ← кол-во людей на каждой камере
└─ timestamp_us
```

---

## Классификация цвета — GPU-конвейер

**Файлы**: `src/color_infer.h`, `src/color_infer.cpp`

```
Входной кадр (RGBA, GPU)
        │
        ▼
┌───────────────────┐
│  compute_torso_roi │  ← Вырезаем зону торса жокея
│  (10-40% высоты,  │     (убираем голову, ноги, руки)
│   -20% по краям)  │
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  CUDA kernel      │  ← Crop + Resize + Normalize
│  crop_resize_norm │     Bilinear sampling
│  RGBA→RGB         │     ImageNet нормализация
│  → NCHW 64×64     │     Всё на GPU, zero-copy
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  TensorRT engine  │  ← Batch до 128
│  3 класса на выход│     FP16 inference
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  Softmax + Map    │  green(0) → slot 1
│                   │  red(1)   → slot 3
│                   │  yellow(2)→ slot 4
└───────────────────┘
```

---

## YOLO Parser — Кастомный парсер

**Файлы**: `src/yolo_parser.h`, `src/yolo_parser.cpp`

DeepStream требует кастомный парсер для YOLOv8 (не поддерживается из коробки):

1. **Автоопределение формата** — определяет ориентацию тензора:
   - `[84, 8400]` (нормальный) или `[8400, 84]` (транспонированный)
   - 84 = 4 (bbox) + 80 (COCO classes) или 5 = 4 + 1 (custom)

2. **Декодирование** — `cx, cy, w, h` → `x1, y1, x2, y2`

3. **NMS** — Non-Maximum Suppression с IoU threshold = 0.45

4. **Регистрация**:
   ```cpp
   extern "C" bool NvDsInferParseYoloV8(...);
   ```
   Подключается как `libnvdsinfer_yolov8_parser.so` через конфиг nvinfer.

---

## RTSP Reconnect

**В `pipeline.cpp`** (строки 305–390):

При обрыве RTSP-потока:
- Пробует переподключиться с **экспоненциальным backoff**
- Не блокирует остальные камеры
- Каждый uridecodebin работает независимо

---

## Запуск

### Single Pipeline (все камеры через одну модель)
```bash
./race_vision_deepstream \
    --config cameras.json \
    --yolo-engine models/yolov8s.engine \
    --color-engine models/color_classifier.engine
```

### Dual Pipeline (trigger + analysis)
```bash
./race_vision_deepstream --dual \
    --config cameras.json \
    --trigger-conf configs/nvinfer_yolov8n_trigger.txt \
    --yolo-engine configs/nvinfer_yolov8s_analysis.txt \
    --color-engine models/color_classifier.engine \
    --cooldown 3.0 \
    --max-active 8
```

---

## Карта файлов

| Компонент | Файл | Ключевые строки |
|-----------|------|-----------------|
| Single Pipeline | `src/pipeline.cpp` | 475–810 (probe), 100–124 (sources) |
| Trigger Pipeline | `src/trigger_pipeline.cpp` | 202–250 (probe), 255–328 (активация) |
| Analysis Pipeline | `src/analysis_pipeline.cpp` | 278–410 (probe), 195–218 (valve) |
| Dual Coordinator | `src/dual_pipeline.cpp` | 17–46 (сборка + связка) |
| YOLO Parser | `src/yolo_parser.cpp` | 67–176 (parseYoloV8) |
| Color Classifier | `src/color_infer.cpp` | 33–84 (CUDA), 224–264 (softmax) |
| SHM Writer | `src/shm_writer.cpp` | 85–106 (протокол записи) |
| SHM Structs | `src/config.h` | 43–80 (структуры данных) |
| Main Entry | `src/main.cpp` | 162–252 (оба режима) |
| Inference Configs | `configs/nvinfer_yolov8*.txt` | Параметры моделей |
| Tracker Config | `configs/tracker_iou.yml` | Параметры трекера |
| Python Reader | `pipeline/shm_reader.py` | 96–150 (чтение SHM) |
