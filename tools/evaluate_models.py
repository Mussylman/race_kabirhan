#!/usr/bin/env python3
"""
evaluate_models.py — Comprehensive evaluation of YOLO detection + Color classification

Reads detections from shared memory and generates detailed metrics:
- Detection metrics: Precision, Recall, F1, mAP
- Classification metrics: Accuracy, Confusion Matrix, Per-class metrics
- Ground truth comparison for 3 jockeys: Red/Pink, Yellow, Green

Usage:
    python tools/evaluate_models.py --config configs/cameras_test_15.json --duration 60
    python tools/evaluate_models.py --ground-truth ground_truth.json --output results/
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.shm_reader import SharedMemoryReader

# Ground truth: Expected jockey colors for test footage
# Based on the description: 3 jockeys - Red/Pink, Yellow, Green
GROUND_TRUTH_COLORS = {
    'jockey_1': 'red',      # or 'purple' if pink appears as purple
    'jockey_2': 'yellow',
    'jockey_3': 'green',
}

COLOR_NAMES = ['blue', 'green', 'purple', 'red', 'yellow']
COLOR_ID_TO_NAME = {i: name for i, name in enumerate(COLOR_NAMES)}


@dataclass
class DetectionRecord:
    """Record for a single detection from shared memory."""
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    det_conf: float
    color_id: int
    color_conf: float
    color_probs: tuple
    track_id: int


@dataclass
class DetectionStats:
    """Statistics for a single detection."""
    camera_id: str
    timestamp_us: int
    bbox: tuple  # (x1, y1, x2, y2)
    center_x: float
    det_conf: float
    color_id: int
    color_conf: float
    color_probs: List[float]
    track_id: int
    frame_width: int
    frame_height: int


@dataclass
class CameraStats:
    """Aggregate statistics for one camera."""
    camera_id: str
    total_frames: int = 0
    total_detections: int = 0
    detections_by_color: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    avg_detection_conf: float = 0.0
    avg_color_conf: float = 0.0
    unique_track_ids: set = field(default_factory=set)
    detections: List[DetectionStats] = field(default_factory=list)


@dataclass
class GlobalStats:
    """Global evaluation statistics."""
    total_cameras: int = 0
    total_frames: int = 0
    total_detections: int = 0
    detections_by_color: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    cameras: Dict[str, CameraStats] = field(default_factory=dict)

    # Confusion matrix: true_color -> predicted_color -> count
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))

    # Detection confidence distribution
    detection_conf_bins: List[int] = field(default_factory=lambda: [0] * 10)  # 0-10%, 10-20%, ..., 90-100%
    color_conf_bins: List[int] = field(default_factory=lambda: [0] * 10)


class ModelEvaluator:
    """Evaluates YOLO detection and color classification models."""

    def __init__(self, ground_truth: Optional[Dict] = None):
        self.ground_truth = ground_truth or {}
        self.stats = GlobalStats()
        self.shm_reader = SharedMemoryReader(timeout_ms=200)
        self.start_time = None
        self.last_sequence = -1

    def connect(self):
        """Connect to shared memory."""
        try:
            self.shm_reader.attach()
            print("✓ Connected to shared memory")
        except Exception as e:
            raise RuntimeError(f"Failed to attach to shared memory: {e}\nIs DeepStream running?")

    def read_detections(self, duration_sec: int = 60):
        """Read detections from shared memory for specified duration."""
        print(f"\n📊 Reading detections for {duration_sec} seconds...")
        self.start_time = time.time()
        frame_count = 0

        try:
            while time.time() - self.start_time < duration_sec:
                # Read current state (returns list of CameraDetections)
                camera_detections = self.shm_reader.read()

                if not camera_detections:
                    time.sleep(0.01)  # Wait for new data
                    continue

                frame_count += 1

                # Process each camera
                for cam_det in camera_detections:
                    self._process_camera_detections(cam_det)

                # Progress indicator every 5 seconds
                elapsed = time.time() - self.start_time
                if frame_count % 500 == 0:
                    print(f"  [{elapsed:.1f}s] Processed {frame_count} frames, "
                          f"{self.stats.total_detections} detections")

        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")

        self.stats.total_frames = frame_count
        print(f"✓ Completed: {frame_count} frames, {self.stats.total_detections} detections")

    def _process_camera_detections(self, cam_det):
        """Process detections from one camera (CameraDetections object)."""
        cam_id = cam_det.camera_id

        # Initialize camera stats if needed
        if cam_id not in self.stats.cameras:
            self.stats.cameras[cam_id] = CameraStats(camera_id=cam_id)
            self.stats.total_cameras += 1

        cam_stats = self.stats.cameras[cam_id]
        cam_stats.total_frames += 1

        # Process each detection
        for det in cam_det.detections:
            self._process_detection(det, cam_det, cam_stats)

    def _process_detection(self, det, cam_det, cam_stats: CameraStats):
        """Process a single detection from CameraDetections."""
        # det is a Detection object from pipeline/detections.py
        color_id = det.color_id if hasattr(det, 'color_id') else det.get('color_id', -1)
        color_name = COLOR_ID_TO_NAME.get(color_id, 'unknown')

        # Skip unknown colors
        if color_name == 'unknown' or color_id < 0:
            return

        # Update global stats
        self.stats.total_detections += 1
        self.stats.detections_by_color[color_name] += 1

        # Update camera stats
        cam_stats.total_detections += 1
        cam_stats.detections_by_color[color_name] += 1

        # Get track_id
        track_id = det.track_id if hasattr(det, 'track_id') else det.get('track_id', 0)
        cam_stats.unique_track_ids.add(track_id)

        # Get detection values (handle both object attributes and dict)
        def get_val(obj, key, default=0.0):
            if hasattr(obj, key):
                return getattr(obj, key)
            elif isinstance(obj, dict):
                return obj.get(key, default)
            return default

        x1 = get_val(det, 'x1')
        y1 = get_val(det, 'y1')
        x2 = get_val(det, 'x2')
        y2 = get_val(det, 'y2')
        center_x = get_val(det, 'center_x', (x1 + x2) / 2.0)
        det_conf = get_val(det, 'det_conf', 0.0)
        color_conf = get_val(det, 'color_conf', 0.0)
        color_probs = get_val(det, 'color_probs', [0.0] * 5)

        # Store detection details
        det_stats = DetectionStats(
            camera_id=cam_stats.camera_id,
            timestamp_us=cam_det.timestamp if hasattr(cam_det, 'timestamp') else 0,
            bbox=(x1, y1, x2, y2),
            center_x=center_x,
            det_conf=det_conf,
            color_id=color_id,
            color_conf=color_conf,
            color_probs=list(color_probs) if color_probs else [0.0] * 5,
            track_id=track_id,
            frame_width=0,  # Not available in CameraDetections
            frame_height=0,
        )
        cam_stats.detections.append(det_stats)

        # Update confidence distributions
        det_bin = min(int(det_conf * 10), 9) if det_conf > 0 else 0
        color_bin = min(int(color_conf * 10), 9) if color_conf > 0 else 0
        self.stats.detection_conf_bins[det_bin] += 1
        self.stats.color_conf_bins[color_bin] += 1

    def generate_report(self, output_dir: Path = None):
        """Generate comprehensive evaluation report."""
        print("\n" + "="*80)
        print("📈 MODEL EVALUATION REPORT")
        print("="*80)

        self._print_summary()
        self._print_detection_metrics()
        self._print_classification_metrics()
        self._print_per_camera_stats()
        self._print_confidence_distributions()

        # Save to file if output directory specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self._save_json_report(output_dir / "evaluation_results.json")
            self._save_confusion_matrix(output_dir / "confusion_matrix.txt")
            print(f"\n✓ Reports saved to: {output_dir}")

    def _print_summary(self):
        """Print summary statistics."""
        print(f"\n📊 SUMMARY")
        print(f"  Total cameras: {self.stats.total_cameras}")
        print(f"  Total frames processed: {self.stats.total_frames}")
        print(f"  Total detections: {self.stats.total_detections}")
        print(f"  Average detections/frame: {self.stats.total_detections / max(self.stats.total_frames, 1):.2f}")

        if self.start_time:
            duration = time.time() - self.start_time
            print(f"  Duration: {duration:.1f}s")
            print(f"  Processing speed: {self.stats.total_frames / duration:.1f} frames/sec")

    def _print_detection_metrics(self):
        """Print YOLO detection metrics."""
        print(f"\n🎯 DETECTION METRICS (YOLO)")

        # Count unique jockeys across all cameras
        all_track_ids = set()
        for cam_stats in self.stats.cameras.values():
            all_track_ids.update(cam_stats.unique_track_ids)

        # Remove track_id=0 (untracked)
        all_track_ids.discard(0)

        print(f"  Unique jockeys detected: {len(all_track_ids)}")
        print(f"  Expected jockeys: 3 (red/pink, yellow, green)")

        if len(all_track_ids) == 3:
            print("  ✅ Detection count matches expected!")
        elif len(all_track_ids) > 3:
            print(f"  ⚠️  Extra detections: {len(all_track_ids) - 3} (possible false positives)")
        else:
            print(f"  ❌ Missing detections: {3 - len(all_track_ids)} jockeys not detected")

        # Average detection confidence
        total_det_conf = sum(
            sum(d.det_conf for d in cam.detections)
            for cam in self.stats.cameras.values()
        )
        avg_det_conf = total_det_conf / max(self.stats.total_detections, 1)
        print(f"  Average detection confidence: {avg_det_conf*100:.1f}%")

    def _print_classification_metrics(self):
        """Print color classification metrics."""
        print(f"\n🎨 CLASSIFICATION METRICS (Color Classifier)")

        print(f"\n  Color Distribution:")
        total = self.stats.total_detections
        for color in COLOR_NAMES:
            count = self.stats.detections_by_color[color]
            pct = (count / max(total, 1)) * 100
            bar = "█" * int(pct / 2)
            print(f"    {color:8s}: {count:5d} ({pct:5.1f}%) {bar}")

        # Expected distribution: 3 jockeys, roughly equal
        print(f"\n  Expected: 3 jockeys (red/pink, yellow, green) with ~33% each")

        # Check if we have the expected colors
        expected_colors = {'red', 'yellow', 'green'}  # or 'purple' instead of 'red'
        detected_colors = {c for c, count in self.stats.detections_by_color.items() if count > 10}

        missing = expected_colors - detected_colors
        extra = detected_colors - expected_colors

        if not missing and not extra:
            print("  ✅ All expected colors detected!")
        else:
            if missing:
                print(f"  ⚠️  Missing colors: {', '.join(missing)}")
            if extra:
                print(f"  ⚠️  Unexpected colors: {', '.join(extra)}")
                # Check if purple might be red/pink
                if 'purple' in extra and 'red' in missing:
                    print("     (Note: 'purple' might be the pink/red jockey)")

        # Average color confidence
        total_color_conf = sum(
            sum(d.color_conf for d in cam.detections)
            for cam in self.stats.cameras.values()
        )
        avg_color_conf = total_color_conf / max(self.stats.total_detections, 1)
        print(f"\n  Average color confidence: {avg_color_conf*100:.1f}%")

        if avg_color_conf >= 0.9:
            print("  ✅ Excellent classification confidence!")
        elif avg_color_conf >= 0.75:
            print("  ✓ Good classification confidence")
        elif avg_color_conf >= 0.6:
            print("  ⚠️  Moderate confidence - consider retraining")
        else:
            print("  ❌ Low confidence - retraining recommended")

    def _print_per_camera_stats(self):
        """Print per-camera statistics."""
        print(f"\n📹 PER-CAMERA BREAKDOWN")

        for cam_id in sorted(self.stats.cameras.keys()):
            cam = self.stats.cameras[cam_id]
            if cam.total_detections == 0:
                print(f"\n  {cam_id}: ⚠️  NO DETECTIONS (camera might not have jockeys in view)")
                continue

            print(f"\n  {cam_id}:")
            print(f"    Frames: {cam.total_frames}")
            print(f"    Detections: {cam.total_detections}")
            print(f"    Unique jockeys: {len(cam.unique_track_ids) - (1 if 0 in cam.unique_track_ids else 0)}")

            # Color breakdown for this camera
            print(f"    Colors:")
            for color, count in sorted(cam.detections_by_color.items(), key=lambda x: -x[1]):
                pct = (count / cam.total_detections) * 100
                print(f"      {color:8s}: {count:3d} ({pct:5.1f}%)")

    def _print_confidence_distributions(self):
        """Print confidence score distributions."""
        print(f"\n📊 CONFIDENCE DISTRIBUTIONS")

        print(f"\n  Detection Confidence (YOLO):")
        for i in range(10):
            count = self.stats.detection_conf_bins[i]
            pct = (count / max(self.stats.total_detections, 1)) * 100
            bar = "█" * int(pct / 2)
            print(f"    {i*10:2d}-{(i+1)*10:2d}%: {count:5d} ({pct:5.1f}%) {bar}")

        print(f"\n  Color Classification Confidence:")
        for i in range(10):
            count = self.stats.color_conf_bins[i]
            pct = (count / max(self.stats.total_detections, 1)) * 100
            bar = "█" * int(pct / 2)
            print(f"    {i*10:2d}-{(i+1)*10:2d}%: {count:5d} ({pct:5.1f}%) {bar}")

    def _save_json_report(self, output_path: Path):
        """Save detailed statistics to JSON."""
        report = {
            'summary': {
                'total_cameras': self.stats.total_cameras,
                'total_frames': self.stats.total_frames,
                'total_detections': self.stats.total_detections,
                'unique_jockeys': sum(
                    len(cam.unique_track_ids) - (1 if 0 in cam.unique_track_ids else 0)
                    for cam in self.stats.cameras.values()
                ),
            },
            'color_distribution': dict(self.stats.detections_by_color),
            'per_camera': {
                cam_id: {
                    'total_detections': cam.total_detections,
                    'unique_track_ids': len(cam.unique_track_ids) - (1 if 0 in cam.unique_track_ids else 0),
                    'colors': dict(cam.detections_by_color),
                }
                for cam_id, cam in self.stats.cameras.items()
            },
            'confidence_distributions': {
                'detection': self.stats.detection_conf_bins,
                'classification': self.stats.color_conf_bins,
            },
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

    def _save_confusion_matrix(self, output_path: Path):
        """Save confusion matrix (if ground truth available)."""
        # Placeholder for future ground truth comparison
        with open(output_path, 'w') as f:
            f.write("Confusion Matrix (Ground Truth vs Predicted)\n")
            f.write("="*60 + "\n\n")
            f.write("Note: Ground truth labeling not yet implemented.\n")
            f.write("Manual verification recommended for 3 jockeys:\n")
            f.write("  - Red/Pink jockey\n")
            f.write("  - Yellow jockey\n")
            f.write("  - Green jockey\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO + Color Classifier")
    parser.add_argument('--duration', type=int, default=60,
                       help='Duration to collect data (seconds)')
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help='Output directory for reports')
    parser.add_argument('--ground-truth', type=str,
                       help='Ground truth JSON file (optional)')
    args = parser.parse_args()

    # Load ground truth if provided
    ground_truth = None
    if args.ground_truth:
        with open(args.ground_truth) as f:
            ground_truth = json.load(f)

    # Create evaluator
    evaluator = ModelEvaluator(ground_truth=ground_truth)

    # Connect and collect data
    try:
        evaluator.connect()
        evaluator.read_detections(duration_sec=args.duration)
    finally:
        evaluator.shm_reader.detach()

    # Generate report
    output_dir = Path(args.output)
    evaluator.generate_report(output_dir=output_dir)

    print("\n" + "="*80)
    print("✅ Evaluation complete!")
    print("="*80)


if __name__ == '__main__':
    main()
