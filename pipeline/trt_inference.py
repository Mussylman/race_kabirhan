"""
trt_inference.py — TensorRT inference wrapper with PyTorch fallback.

Provides unified API for:
    - YOLOv8 detection (trigger + analysis)
    - SimpleColorCNN classification

When TensorRT is available, loads .engine files for maximum throughput.
Falls back to PyTorch (.pt) when TRT is not available (Windows dev, etc.).

Usage:
    # Detection
    detector = YOLODetector(
        engine_path="models/yolov8n_trigger.engine",
        fallback_pt="yolov8n.pt",
        imgsz=640,
    )
    results = detector.detect_batch(frames)  # list of np arrays

    # Classification
    classifier = ColorClassifierInfer(
        engine_path="models/color_classifier.engine",
        fallback_pt="models/color_classifier.pt",
    )
    colors, confs, prob_dicts = classifier.classify_batch(crops)
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

log = logging.getLogger("pipeline.trt_inference")

# Try to import TensorRT
_TRT_AVAILABLE = False
try:
    import tensorrt as trt
    _TRT_AVAILABLE = True
except ImportError:
    pass


# ── Color classifier model (must match training) ─────────────────────

class SimpleColorCNN(nn.Module):
    """Same architecture as tools/test_race_count.py."""

    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 5),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── YOLO Detector ────────────────────────────────────────────────────

class YOLODetector:
    """Batch YOLO detection using TensorRT or ultralytics fallback.

    Returns per-frame detection results: list of dicts with
    bbox, conf, center_x.
    """

    def __init__(
        self,
        engine_path: Optional[str] = None,
        fallback_pt: str = "yolov8s.pt",
        imgsz: int = 1280,
        conf: float = 0.35,
        iou: float = 0.3,
        device: str = "cuda:0",
        half: bool = True,
    ):
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.half = half
        self._trt_engine = None
        self._yolo = None

        engine = Path(engine_path) if engine_path else None

        if engine and engine.exists() and _TRT_AVAILABLE:
            log.info("Loading TensorRT engine: %s", engine)
            self._load_trt(str(engine))
        else:
            if engine and not engine.exists():
                log.info("TRT engine not found (%s), falling back to PyTorch", engine)
            log.info("Loading YOLO PyTorch: %s (imgsz=%d)", fallback_pt, imgsz)
            from ultralytics import YOLO
            self._yolo = YOLO(fallback_pt)

    def _load_trt(self, engine_path: str):
        """Load TensorRT engine for YOLO."""
        # TRT engine loading will be used on Linux production
        # For now, this is a placeholder — full TRT integration in Phase 3
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self._trt_engine = runtime.deserialize_cuda_engine(f.read())
        log.info("TRT engine loaded: %s", engine_path)

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[dict]]:
        """Run detection on a batch of frames.

        Args:
            frames: list of BGR numpy arrays (can be different sizes).

        Returns:
            list of detection lists, one per frame.
            Each detection: {"bbox": (x1,y1,x2,y2), "conf": float, "center_x": float}
        """
        if not frames:
            return []

        if self._yolo is not None:
            return self._detect_batch_ultralytics(frames)
        else:
            return self._detect_batch_trt(frames)

    def _detect_batch_ultralytics(self, frames: list[np.ndarray]) -> list[list[dict]]:
        """Batch detection using ultralytics YOLO."""
        results = self._yolo(
            frames,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=[0],  # person only
            device=self.device,
            half=self.half,
            verbose=False,
        )

        batch_dets = []
        for result in results:
            dets = []
            if result.boxes is not None and len(result.boxes) > 0:
                bboxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                for bbox, conf in zip(bboxes, confs):
                    x1, y1, x2, y2 = bbox
                    dets.append({
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "conf": float(conf),
                        "center_x": float((x1 + x2) / 2),
                    })
            batch_dets.append(dets)

        return batch_dets

    def _detect_batch_trt(self, frames: list[np.ndarray]) -> list[list[dict]]:
        """Batch detection using TensorRT engine (placeholder for Phase 3)."""
        # Will be fully implemented when deploying on Linux with TensorRT
        log.warning("TRT batch detection not yet implemented, returning empty")
        return [[] for _ in frames]


# ── Color Classifier ─────────────────────────────────────────────────

class ColorClassifierInfer:
    """Batch color classification using TensorRT or PyTorch fallback."""

    CLASSES = ["blue", "green", "purple", "red", "yellow"]
    INPUT_SIZE = 64
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        engine_path: Optional[str] = None,
        fallback_pt: str = "models/color_classifier.pt",
        device: str = "cuda:0",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._trt_engine = None
        self._model = None
        self.classes = list(self.CLASSES)

        engine = Path(engine_path) if engine_path else None

        if engine and engine.exists() and _TRT_AVAILABLE:
            log.info("Loading TRT color classifier: %s", engine)
            self._load_trt(str(engine))
        else:
            log.info("Loading PyTorch color classifier: %s", fallback_pt)
            self._load_pytorch(fallback_pt)

    def _load_pytorch(self, pt_path: str):
        ckpt = torch.load(pt_path, map_location=self.device, weights_only=False)
        self.classes = ckpt['classes']
        self._model = SimpleColorCNN(num_classes=len(self.classes)).to(self.device)
        self._model.load_state_dict(ckpt['model_state_dict'])
        self._model.eval()

    def _load_trt(self, engine_path: str):
        """Load TensorRT engine (placeholder for Phase 3)."""
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self._trt_engine = runtime.deserialize_cuda_engine(f.read())

    def _preprocess_crop(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """Preprocess a single BGR crop to normalized tensor."""
        import cv2
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        # Resize to 64x64
        rgb = cv2.resize(rgb, (self.INPUT_SIZE, self.INPUT_SIZE))
        # To float [0, 1]
        t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        # Normalize with ImageNet stats
        for c in range(3):
            t[c] = (t[c] - self.MEAN[c]) / self.STD[c]
        return t

    def classify_batch(
        self,
        crops: list[np.ndarray],
    ) -> list[tuple[str, float, dict]]:
        """Classify a batch of BGR crops.

        Args:
            crops: list of BGR numpy arrays (torso crops).

        Returns:
            list of (color, confidence, prob_dict) tuples.
        """
        if not crops:
            return []

        if self._model is not None:
            return self._classify_batch_pytorch(crops)
        else:
            return self._classify_batch_trt(crops)

    def _classify_batch_pytorch(
        self,
        crops: list[np.ndarray],
    ) -> list[tuple[str, float, dict]]:
        """Batch classification using PyTorch."""
        tensors = []
        valid_indices = []
        for i, crop in enumerate(crops):
            if crop is not None and crop.size > 0:
                tensors.append(self._preprocess_crop(crop))
                valid_indices.append(i)

        if not tensors:
            return [("unknown", 0.0, {})] * len(crops)

        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            logits = self._model(batch)
            probs = torch.softmax(logits, dim=1)

        # Build results for all inputs (including invalid ones)
        results = [("unknown", 0.0, {})] * len(crops)

        for batch_idx, orig_idx in enumerate(valid_indices):
            p = probs[batch_idx]
            prob_dict = {
                self.classes[j]: round(p[j].item(), 4)
                for j in range(len(self.classes))
            }
            best_idx = p.argmax().item()
            color = self.classes[best_idx]
            conf = p[best_idx].item()
            results[orig_idx] = (color, conf, prob_dict)

        return results

    def _classify_batch_trt(
        self,
        crops: list[np.ndarray],
    ) -> list[tuple[str, float, dict]]:
        """Batch classification using TensorRT (placeholder for Phase 3)."""
        log.warning("TRT color classification not yet implemented")
        return [("unknown", 0.0, {})] * len(crops)
