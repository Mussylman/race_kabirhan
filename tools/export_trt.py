"""
export_trt.py — Export YOLO and color classifier models to TensorRT FP16.

Exports:
    1. YOLOv8n → yolov8n_trigger.engine (640px, FP16, batch=25)
    2. YOLOv8s → yolov8s_analysis.engine (800px, FP16, batch=8)
    3. YOLOv8s → yolov8s_deepstream.engine (800px, FP16, batch=25) — for DeepStream
    4. SimpleColorCNN → color_classifier.engine (64px, FP16, dynamic batch 1-128)

Requirements:
    - NVIDIA GPU with TensorRT support
    - ultralytics (for YOLO export)
    - torch + tensorrt

Usage:
    python tools/export_trt.py                     # export all
    python tools/export_trt.py --model trigger     # export trigger only
    python tools/export_trt.py --model analysis    # export analysis only
    python tools/export_trt.py --model deepstream  # export YOLOv8s for DeepStream (batch=25)
    python tools/export_trt.py --model classifier  # export classifier only
"""

import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("export_trt")

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

MODELS_DIR = Path("models")


def export_yolo_trigger():
    """Export YOLOv8n for trigger (640px, batch=25)."""
    from ultralytics import YOLO

    pt_path = "yolov8n.pt"
    output = MODELS_DIR / "yolov8n_trigger.engine"

    log.info("=== Exporting YOLOv8n trigger ===")
    log.info("  Source: %s", pt_path)
    log.info("  Target: %s", output)
    log.info("  imgsz=640, half=True, batch=25")

    # Download if needed
    model = YOLO(pt_path)
    model.export(
        format="engine",
        imgsz=640,
        half=True,
        batch=25,
        device=0,
        simplify=True,
    )

    # ultralytics saves next to the .pt file; move to models/
    exported = Path(pt_path).with_suffix(".engine")
    if exported.exists():
        exported.rename(output)
        log.info("  Saved: %s (%.1f MB)", output, output.stat().st_size / 1024 / 1024)
    else:
        log.warning("  Export may have saved to a different location, check yolov8n.engine")


def export_yolo_analysis():
    """Export YOLOv8s for analysis (800px, batch=8)."""
    from ultralytics import YOLO

    pt_path = "yolov8s.pt"
    output = MODELS_DIR / "yolov8s_analysis.engine"

    log.info("=== Exporting YOLOv8s analysis ===")
    log.info("  Source: %s", pt_path)
    log.info("  Target: %s", output)
    log.info("  imgsz=800, half=True, batch=8")

    model = YOLO(pt_path)
    model.export(
        format="engine",
        imgsz=800,
        half=True,
        batch=8,
        device=0,
        simplify=True,
    )

    exported = Path(pt_path).with_suffix(".engine")
    if exported.exists():
        exported.rename(output)
        log.info("  Saved: %s (%.1f MB)", output, output.stat().st_size / 1024 / 1024)


def export_yolo_deepstream():
    """Export YOLOv8s for DeepStream (800px, batch=25, FP16).

    This engine runs ALL 25 cameras simultaneously in the DeepStream C++
    pipeline. No trigger/analysis split needed — TRT FP16 is fast enough.
    """
    from ultralytics import YOLO

    pt_path = "yolov8s.pt"
    output = MODELS_DIR / "yolov8s_deepstream.engine"

    log.info("=== Exporting YOLOv8s for DeepStream ===")
    log.info("  Source: %s", pt_path)
    log.info("  Target: %s", output)
    log.info("  imgsz=800, half=True, batch=25")

    model = YOLO(pt_path)
    model.export(
        format="engine",
        imgsz=800,
        half=True,
        batch=25,
        device=0,
        simplify=True,
    )

    exported = Path(pt_path).with_suffix(".engine")
    if exported.exists():
        exported.rename(output)
        log.info("  Saved: %s (%.1f MB)", output, output.stat().st_size / 1024 / 1024)
    else:
        log.warning("  Export may have saved to a different location, check yolov8s.engine")


def export_color_classifier():
    """Export SimpleColorCNN to TensorRT via ONNX → TRT."""
    import torch
    import torch.onnx

    pt_path = MODELS_DIR / "color_classifier.pt"
    onnx_path = MODELS_DIR / "color_classifier.onnx"
    engine_path = MODELS_DIR / "color_classifier.engine"

    log.info("=== Exporting SimpleColorCNN ===")
    log.info("  Source: %s", pt_path)
    log.info("  Target: %s", engine_path)

    # Load PyTorch model
    from pipeline.trt_inference import SimpleColorCNN

    ckpt = torch.load(pt_path, map_location="cuda:0", weights_only=False)
    model = SimpleColorCNN(num_classes=len(ckpt['classes'])).cuda()
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Step 1: Export to ONNX
    log.info("  Step 1: PyTorch → ONNX")
    dummy = torch.randn(1, 3, 64, 64).cuda()
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    log.info("  ONNX saved: %s", onnx_path)

    # Step 2: ONNX → TensorRT
    log.info("  Step 2: ONNX → TensorRT")
    try:
        import tensorrt as trt

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # parse_from_file resolves external weights relative to ONNX path
        if not parser.parse_from_file(str(onnx_path)):
            for i in range(parser.num_errors):
                log.error("  ONNX parse error: %s", parser.get_error(i))
            return

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256MB
        config.set_flag(trt.BuilderFlag.FP16)

        # Set optimization profile for dynamic batch
        profile = builder.create_optimization_profile()
        profile.set_shape("input",
                          min=(1, 3, 64, 64),
                          opt=(32, 3, 64, 64),
                          max=(128, 3, 64, 64))
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized:
            with open(engine_path, "wb") as f:
                f.write(serialized)
            log.info("  Saved: %s (%.1f MB)",
                     engine_path, engine_path.stat().st_size / 1024 / 1024)
        else:
            log.error("  TensorRT build failed")

    except ImportError:
        log.warning("  TensorRT not available — ONNX exported only")
        log.info("  Convert on Linux: trtexec --onnx=%s --saveEngine=%s --fp16", onnx_path, engine_path)


def main():
    parser = argparse.ArgumentParser(description="Export models to TensorRT")
    parser.add_argument("--model", choices=["trigger", "analysis", "deepstream", "classifier", "all"],
                        default="all", help="Which model to export")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)

    if args.model in ("trigger", "all"):
        export_yolo_trigger()

    if args.model in ("analysis", "all"):
        export_yolo_analysis()

    if args.model in ("deepstream", "all"):
        export_yolo_deepstream()

    if args.model in ("classifier", "all"):
        export_color_classifier()

    log.info("Done!")


if __name__ == "__main__":
    main()
