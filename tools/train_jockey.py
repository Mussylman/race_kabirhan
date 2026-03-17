#!/usr/bin/env python3
"""
Train YOLOv11s with custom 'jockey' class for horse race detection.

Usage:
    python tools/train_jockey.py --data dataset/jockey_v2/data.yaml

    # Custom settings:
    python tools/train_jockey.py \
        --data dataset/jockey_v2/data.yaml \
        --model yolo11s.pt \
        --epochs 150 \
        --imgsz 800 \
        --batch 16

After training:
    # Export to ONNX for DeepStream:
    python tools/train_jockey.py --export runs/detect/jockey_v2/weights/best.pt
"""

import argparse
import sys


def train(args):
    from ultralytics import YOLO

    model = YOLO(args.model)
    print(f"Training {args.model} on {args.data}")
    print(f"  epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name="jockey_v2",
        patience=25,
        save=True,
        plots=True,

        # NMS — less suppression for crowded scenes
        iou=0.5,

        # Augmentation tuned for horse racing
        mosaic=0.3,
        mixup=0.0,
        copy_paste=0.3,       # simulate crowded groups
        degrees=5.0,
        scale=0.3,
        translate=0.1,
        fliplr=0.0,           # horses run in one direction
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,

        # LR & scheduling
        cos_lr=True,
        close_mosaic=20,      # disable mosaic last 20 epochs
    )

    best = results.save_dir / "weights" / "best.pt"
    print(f"\nTraining complete!")
    print(f"Best model: {best}")
    print(f"\nExport to ONNX:")
    print(f"  python tools/train_jockey.py --export {best}")


def export(model_path, imgsz):
    from ultralytics import YOLO

    model = YOLO(model_path)
    print(f"Exporting {model_path} to ONNX (imgsz={imgsz})")

    model.export(
        format="onnx",
        imgsz=imgsz,
        opset=16,
        simplify=True,
        dynamic=True,
    )

    onnx_path = model_path.replace(".pt", ".onnx")
    print(f"\nExported: {onnx_path}")
    print(f"\nNext: copy to models/ and update nvinfer config:")
    print(f"  cp {onnx_path} models/jockey_yolov11s.onnx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/export jockey detector")
    parser.add_argument("--data", help="Path to data.yaml")
    parser.add_argument("--model", default="yolo11s.pt", help="Base model (default: yolo11s.pt)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--export", metavar="MODEL_PATH", help="Export trained .pt to ONNX")
    args = parser.parse_args()

    if args.export:
        export(args.export, args.imgsz)
    elif args.data:
        train(args)
    else:
        parser.print_help()
