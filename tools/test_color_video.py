#!/usr/bin/env python3
"""Test color classifiers on video with display window."""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from ultralytics import YOLO
from pipeline.trt_inference import ColorClassifierInfer

COLORS_BGR = {
    "blue":    (255, 100, 0),
    "green":   (0, 200, 0),
    "purple":  (200, 0, 200),
    "red":     (0, 0, 255),
    "yellow":  (0, 230, 230),
    "unknown": (128, 128, 128),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="models/color_classifier_v3.pt", help="Color classifier .pt")
    parser.add_argument("--yolo", default="models/jockey_yolov11s.pt")
    parser.add_argument("--conf", type=float, default=0.20, help="YOLO confidence")
    parser.add_argument("--save", default=None, help="Save output video")
    args = parser.parse_args()

    print(f"Loading YOLO: {args.yolo}")
    yolo = YOLO(args.yolo)

    print(f"Loading classifier: {args.model}")
    clf = ColorClassifierInfer(device="cuda:0")
    clf._load_pytorch(args.model)
    print(f"  Classes: {clf.classes}, Input: {clf.INPUT_SIZE}x{clf.INPUT_SIZE}")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {vid_w}x{vid_h} @ {fps:.0f}fps, {total} frames")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (vid_w, vid_h))

    fnum = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fnum += 1

        results = yolo(frame, imgsz=800, conf=args.conf, verbose=False)
        boxes = results[0].boxes if results else []

        crops = []
        coords = []
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            crop = frame[y1:y2, x1:x2]
            if crop.size > 200:
                crops.append(crop)
                coords.append((x1, y1, x2, y2, box.conf.item()))

        if crops:
            preds = clf.classify_batch(crops)
            for (x1, y1, x2, y2, yolo_conf), (color, conf, _) in zip(coords, preds):
                bgr = COLORS_BGR.get(color, (128, 128, 128))
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                label = f"{color} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), bgr, -1)
                cv2.putText(frame, label, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Color Classifier", frame)
        if writer:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Done")


if __name__ == "__main__":
    main()
