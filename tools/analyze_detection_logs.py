#!/usr/bin/env python3
"""
analyze_detection_logs.py — Analyze DeepStream detection logs

Parses terminal output from DeepStream to generate statistics about:
- Detection counts and confidence
- Color classification accuracy
- Per-camera breakdowns

Usage:
    python tools/analyze_detection_logs.py /tmp/deepstream_eval.log
    python tools/analyze_detection_logs.py /tmp/performance_test.log --output results/
"""

import re
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List

COLOR_NAMES = ['blue', 'green', 'purple', 'red', 'yellow']


@dataclass
class DetectionRecord:
    """Single detection parsed from log."""
    camera_id: str
    track_id: int
    det_conf: float
    color_name: str
    color_conf: float
    bbox: tuple


@dataclass
class Stats:
    """Aggregate statistics."""
    total_batches: int = 0
    total_detections: int = 0
    fps_values: List[float] = field(default_factory=list)
    detections_by_color: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    detections_by_camera: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    track_ids_by_camera: Dict[str, set] = field(default_factory=lambda: defaultdict(set))
    all_track_ids: set = field(default_factory=set)
    det_conf_values: List[float] = field(default_factory=list)
    color_conf_values: List[float] = field(default_factory=list)
    detections: List[DetectionRecord] = field(default_factory=list)


def parse_log_file(log_path: Path) -> Stats:
    """Parse DeepStream log file."""
    stats = Stats()

    # Regex patterns for ANSI color-coded output
    batch_pattern = re.compile(r'\[BATCH #(\d+)\].*?(\d+) detections.*?(\d+\.?\d*) FPS')
    camera_pattern = re.compile(r'├─ \x1b\[1m(cam-\d+) \[(\d+) objects\]')
    detection_pattern = re.compile(
        r'\[#(\d+)\].*?YOLO:.*?(\d+)%.*?Color:.*?\x1b\[[\d;]+m(\w+).*?\((\d+)%\).*?bbox=\[([0-9,]+)\]'
    )

    print(f"📖 Reading log file: {log_path}")

    with open(log_path, 'r', errors='ignore') as f:
        content = f.read()

    # Find all batch blocks
    batch_blocks = re.split(r'━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', content)

    current_camera = None

    for block in batch_blocks:
        # Extract batch info
        batch_match = batch_pattern.search(block)
        if batch_match:
            batch_num = int(batch_match.group(1))
            num_dets = int(batch_match.group(2))
            fps = float(batch_match.group(3))

            stats.total_batches += 1
            stats.total_detections += num_dets
            stats.fps_values.append(fps)

        # Extract camera
        cam_match = camera_pattern.search(block)
        if cam_match:
            current_camera = cam_match.group(1)
            num_objs = int(cam_match.group(2))
            stats.detections_by_camera[current_camera] += num_objs

        # Extract detections
        for det_match in detection_pattern.finditer(block):
            track_id = int(det_match.group(1))
            det_conf = int(det_match.group(2))
            color_name = det_match.group(3).lower()
            color_conf = int(det_match.group(4))
            bbox_str = det_match.group(5)

            if current_camera:
                stats.track_ids_by_camera[current_camera].add(track_id)
                stats.all_track_ids.add(track_id)

            stats.detections_by_color[color_name] += 1
            stats.det_conf_values.append(det_conf / 100.0)
            stats.color_conf_values.append(color_conf / 100.0)

            # Parse bbox
            try:
                bbox = tuple(map(int, bbox_str.split(',')))
            except:
                bbox = (0, 0, 0, 0)

            det_rec = DetectionRecord(
                camera_id=current_camera or 'unknown',
                track_id=track_id,
                det_conf=det_conf / 100.0,
                color_name=color_name,
                color_conf=color_conf / 100.0,
                bbox=bbox,
            )
            stats.detections.append(det_rec)

    # Remove track_id=0 (untracked)
    stats.all_track_ids.discard(0)
    for cam, tids in stats.track_ids_by_camera.items():
        tids.discard(0)

    print(f"✓ Parsed {stats.total_batches} batches, {len(stats.detections)} detections")
    return stats


def print_report(stats: Stats):
    """Print comprehensive analysis report."""
    print("\n" + "="*80)
    print("📈 DETECTION & CLASSIFICATION ANALYSIS")
    print("="*80)

    # Summary
    print(f"\n📊 SUMMARY")
    print(f"  Total batches: {stats.total_batches}")
    print(f"  Total detections: {len(stats.detections)}")
    print(f"  Unique jockeys (track IDs): {len(stats.all_track_ids)}")
    print(f"  Expected jockeys: 3 (red/pink, yellow, green)")

    if len(stats.all_track_ids) == 3:
        print("  ✅ Detection count matches expected!")
    elif len(stats.all_track_ids) > 3:
        print(f"  ⚠️  Extra detections: {len(stats.all_track_ids) - 3} (possible false positives)")
    else:
        print(f"  ⚠️  Missing detections: {3 - len(stats.all_track_ids)} jockeys")

    # FPS statistics
    if stats.fps_values:
        avg_fps = sum(stats.fps_values) / len(stats.fps_values)
        min_fps = min(stats.fps_values)
        max_fps = max(stats.fps_values)
        print(f"\n⚡ PERFORMANCE")
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  Min FPS: {min_fps:.1f}")
        print(f"  Max FPS: {max_fps:.1f}")

    # Color distribution
    print(f"\n🎨 COLOR CLASSIFICATION")
    total_dets = len(stats.detections)

    print(f"\n  Color Distribution:")
    for color in ['red', 'purple', 'yellow', 'green', 'blue']:  # Ordered by expected
        count = stats.detections_by_color.get(color, 0)
        pct = (count / max(total_dets, 1)) * 100
        bar = "█" * int(pct / 2)
        status = ""
        if color in ['red', 'purple'] and count > 10:
            status = " ← likely RED/PINK jockey"
        elif color == 'yellow' and count > 10:
            status = " ← YELLOW jockey"
        elif color == 'green' and count > 10:
            status = " ← GREEN jockey"

        print(f"    {color:8s}: {count:5d} ({pct:5.1f}%) {bar}{status}")

    # Validate expected colors
    print(f"\n  Expected: 3 jockeys (red/pink, yellow, green) with ~33% each")

    expected_colors = {'red', 'yellow', 'green'}  # or 'purple' instead of 'red'
    detected_colors = {c for c, count in stats.detections_by_color.items() if count > 50}

    if 'purple' in detected_colors and count > stats.detections_by_color.get('red', 0):
        print("  ℹ️  Note: 'purple' likely represents the pink/red jockey")
        detected_colors.add('red')  # Count purple as red

    missing = expected_colors - detected_colors
    if not missing:
        print("  ✅ All expected colors detected!")
    else:
        print(f"  ⚠️  Low/missing colors: {', '.join(missing)}")

    # Confidence statistics
    if stats.det_conf_values:
        avg_det_conf = sum(stats.det_conf_values) / len(stats.det_conf_values)
        print(f"\n  Average YOLO confidence: {avg_det_conf*100:.1f}%")

    if stats.color_conf_values:
        avg_color_conf = sum(stats.color_conf_values) / len(stats.color_conf_values)
        print(f"  Average color confidence: {avg_color_conf*100:.1f}%")

        if avg_color_conf >= 0.9:
            print("  ✅ Excellent classification confidence!")
        elif avg_color_conf >= 0.75:
            print("  ✓ Good classification confidence")
        elif avg_color_conf >= 0.6:
            print("  ⚠️  Moderate confidence - consider retraining")
        else:
            print("  ❌ Low confidence - retraining recommended")

    # Per-camera breakdown
    print(f"\n📹 PER-CAMERA BREAKDOWN")
    for cam_id in sorted(stats.detections_by_camera.keys()):
        count = stats.detections_by_camera[cam_id]
        unique_jockeys = len(stats.track_ids_by_camera.get(cam_id, set()))

        if count == 0:
            print(f"  {cam_id}: ⚠️  NO DETECTIONS (jockeys not in view)")
        else:
            print(f"  {cam_id}: {count:4d} detections, {unique_jockeys} unique jockeys")

    # Track ID analysis
    print(f"\n🎯 TRACK ID ANALYSIS")
    print(f"  Total unique track IDs: {len(stats.all_track_ids)}")
    if stats.all_track_ids:
        print(f"  Track IDs detected: {sorted(stats.all_track_ids)}")

        # Count detections per track ID
        track_counts = Counter(d.track_id for d in stats.detections if d.track_id > 0)
        print(f"\n  Detections per jockey:")
        for track_id, count in sorted(track_counts.items()):
            pct = (count / max(total_dets, 1)) * 100
            # Find dominant color for this track
            track_dets = [d for d in stats.detections if d.track_id == track_id]
            colors = Counter(d.color_name for d in track_dets)
            dominant_color = colors.most_common(1)[0][0] if colors else 'unknown'
            print(f"    Jockey #{track_id}: {count:4d} detections ({pct:5.1f}%) - {dominant_color}")


def save_json_report(stats: Stats, output_path: Path):
    """Save statistics to JSON."""
    report = {
        'summary': {
            'total_batches': stats.total_batches,
            'total_detections': len(stats.detections),
            'unique_jockeys': len(stats.all_track_ids),
        },
        'performance': {
            'avg_fps': sum(stats.fps_values) / len(stats.fps_values) if stats.fps_values else 0,
            'min_fps': min(stats.fps_values) if stats.fps_values else 0,
            'max_fps': max(stats.fps_values) if stats.fps_values else 0,
        },
        'color_distribution': dict(stats.detections_by_color),
        'per_camera': {
            cam_id: {
                'detections': count,
                'unique_track_ids': len(stats.track_ids_by_camera.get(cam_id, set())),
            }
            for cam_id, count in stats.detections_by_camera.items()
        },
        'confidence': {
            'avg_detection_conf': sum(stats.det_conf_values) / len(stats.det_conf_values) if stats.det_conf_values else 0,
            'avg_color_conf': sum(stats.color_conf_values) / len(stats.color_conf_values) if stats.color_conf_values else 0,
        },
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ JSON report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze DeepStream detection logs")
    parser.add_argument('logfile', type=str, help='Path to DeepStream log file')
    parser.add_argument('--output', type=str, default='log_analysis',
                       help='Output directory for reports')
    args = parser.parse_args()

    log_path = Path(args.logfile)
    if not log_path.exists():
        print(f"❌ Log file not found: {log_path}")
        return 1

    # Parse log
    stats = parse_log_file(log_path)

    # Print report
    print_report(stats)

    # Save JSON
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json_report(stats, output_dir / "analysis_results.json")

    print("\n" + "="*80)
    print("✅ Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    sys.exit(main())
