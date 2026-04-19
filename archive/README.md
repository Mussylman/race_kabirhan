# archive/

Не трогать. Сюда складываются файлы, которые **больше не используются**
активным пайплайном, но сохраняются для истории / возможного возврата.

- `cpp_legacy/` — pre-Phase 4 C++ бинарь (main/pipeline/trigger/analysis/dual).
  Весь функционал перенесён в Python `deepstream/main.py + pipeline.py`.
- `root_models/`  — старые `.pt/.onnx` weights (yolo11s, yolo26n, yolov8s).
  В production используются `models/yolo11s_person_960.onnx` и
  `models/color_classifier_v4.onnx`.
- `notes/` — старые session-логи. Актуальная документация — `docs/RUNBOOK.md`
  и файлы в `~/.claude/.../memory/`.

См. также: `models/archive/`, `configs/archive/`,
`deepstream/configs/archive/`, `tools/archive/`.
