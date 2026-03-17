#!/usr/bin/env python3
"""
track_jockey_timeline.py — Track when each jockey appears on each camera

Analyzes DeepStream logs to create a timeline showing:
- When each jockey (track_id) first appears on each camera
- Which color each jockey is classified as
- Timeline of movement through cameras

This helps validate track_start/track_end configuration.

Usage:
    python tools/track_jockey_timeline.py /tmp/deepstream_25cam.log
"""

import re
import sys
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class JockeyAppearance:
    """When a jockey appeared on a camera."""
    camera_id: str
    track_id: int
    first_batch: int
    last_batch: int
    color: str
    detection_count: int = 0


@dataclass
class Timeline:
    """Complete timeline of jockey movements."""
    # camera_id -> track_id -> JockeyAppearance
    appearances: Dict[str, Dict[int, JockeyAppearance]] = field(
        default_factory=lambda: defaultdict(dict)
    )

    # Track dominant color for each track_id
    track_colors: Dict[int, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )

    all_cameras: Set[str] = field(default_factory=set)
    all_track_ids: Set[int] = field(default_factory=set)


def parse_log(log_path: Path) -> Timeline:
    """Parse DeepStream log to extract jockey timeline."""
    timeline = Timeline()

    batch_pattern = re.compile(r'\[BATCH #(\d+)\]')
    camera_pattern = re.compile(r'├─ \x1b\[1m(cam-\d+) \[(\d+) objects\]')
    detection_pattern = re.compile(
        r'\[#(\d+)\].*?YOLO:.*?(\d+)%.*?Color:.*?\x1b\[[\d;]+m(\w+).*?\((\d+)%\)'
    )

    print(f"📖 Parsing log: {log_path}")

    with open(log_path, 'r', errors='ignore') as f:
        content = f.read()

    # Split by batch separator
    blocks = re.split(r'━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', content)

    current_batch = 0
    current_camera = None

    for block in blocks:
        # Get batch number
        batch_match = batch_pattern.search(block)
        if batch_match:
            current_batch = int(batch_match.group(1))

        # Get camera
        cam_match = camera_pattern.search(block)
        if cam_match:
            current_camera = cam_match.group(1)
            timeline.all_cameras.add(current_camera)

        # Get detections
        for det_match in detection_pattern.finditer(block):
            track_id = int(det_match.group(1))
            color = det_match.group(3).lower()

            if track_id == 0 or not current_camera:
                continue

            timeline.all_track_ids.add(track_id)
            timeline.track_colors[track_id][color] += 1

            # Update or create appearance
            if track_id in timeline.appearances[current_camera]:
                app = timeline.appearances[current_camera][track_id]
                app.last_batch = current_batch
                app.detection_count += 1
            else:
                timeline.appearances[current_camera][track_id] = JockeyAppearance(
                    camera_id=current_camera,
                    track_id=track_id,
                    first_batch=current_batch,
                    last_batch=current_batch,
                    color=color,
                    detection_count=1,
                )

    # Determine dominant color for each track
    for track_id in timeline.all_track_ids:
        colors = timeline.track_colors[track_id]
        if colors:
            dominant = max(colors.items(), key=lambda x: x[1])[0]
            # Update all appearances with dominant color
            for cam_apps in timeline.appearances.values():
                if track_id in cam_apps:
                    cam_apps[track_id].color = dominant

    print(f"✓ Found {len(timeline.all_track_ids)} unique track IDs across {len(timeline.all_cameras)} cameras")
    return timeline


def print_timeline(timeline: Timeline):
    """Print jockey timeline report."""
    print("\n" + "="*100)
    print("🏇 JOCKEY MOVEMENT TIMELINE")
    print("="*100)

    # Group jockeys by color (expected: red/pink, yellow, green)
    color_groups = defaultdict(list)
    for track_id in sorted(timeline.all_track_ids):
        dominant_color = max(
            timeline.track_colors[track_id].items(),
            key=lambda x: x[1]
        )[0] if timeline.track_colors[track_id] else 'unknown'
        color_groups[dominant_color].append(track_id)

    print(f"\n📊 JOCKEYS BY COLOR:")
    for color in ['red', 'purple', 'yellow', 'green', 'blue']:
        if color in color_groups:
            track_ids = color_groups[color]
            print(f"  {color.upper():8s}: {len(track_ids):2d} track IDs - {track_ids}")

    # Expected: 3 jockeys (red/pink, yellow, green)
    print(f"\n  Expected: 3 jockeys total")
    print(f"  Note: Multiple track IDs may represent same jockey (tracker ID reassignment)")

    # Print timeline per jockey
    print(f"\n" + "="*100)
    print("📹 CAMERA APPEARANCE TIMELINE (by jockey)")
    print("="*100)

    # For each color group, show camera progression
    for color in ['red', 'purple', 'yellow', 'green']:
        if color not in color_groups:
            continue

        print(f"\n{'='*100}")
        print(f"🎨 {color.upper()} JOCKEY/JOCKEYS")
        print(f"{'='*100}")

        for track_id in sorted(color_groups[color]):
            print(f"\n  Track ID #{track_id}:")

            # Get all cameras this track appeared on
            cameras_with_track = []
            for cam_id in sorted(timeline.all_cameras):
                if track_id in timeline.appearances[cam_id]:
                    app = timeline.appearances[cam_id][track_id]
                    cameras_with_track.append((cam_id, app))

            if not cameras_with_track:
                print(f"    (no appearances)")
                continue

            # Sort by first appearance
            cameras_with_track.sort(key=lambda x: x[1].first_batch)

            for cam_id, app in cameras_with_track:
                duration = app.last_batch - app.first_batch + 1
                print(f"    {cam_id}: batch {app.first_batch:5d} → {app.last_batch:5d} "
                      f"({duration:4d} batches, {app.detection_count:3d} detections)")

    # Camera-centric view
    print(f"\n" + "="*100)
    print("📹 CAMERA-CENTRIC VIEW (which jockeys appeared on each camera)")
    print("="*100)

    for cam_id in sorted(timeline.all_cameras):
        print(f"\n  {cam_id}:")

        appearances = timeline.appearances[cam_id]
        if not appearances:
            print(f"    ⚠️  NO DETECTIONS")
            continue

        # Group by color
        cam_color_groups = defaultdict(list)
        for track_id, app in appearances.items():
            cam_color_groups[app.color].append((track_id, app))

        for color in ['red', 'purple', 'yellow', 'green', 'blue']:
            if color not in cam_color_groups:
                continue

            tracks = cam_color_groups[color]
            total_dets = sum(app.detection_count for _, app in tracks)
            track_ids = [tid for tid, _ in sorted(tracks)]

            print(f"    {color:8s}: {len(tracks):2d} track IDs ({total_dets:3d} detections) - {track_ids}")


def save_json_timeline(timeline: Timeline, output_path: Path):
    """Save timeline to JSON for further analysis."""
    data = {
        'summary': {
            'total_cameras': len(timeline.all_cameras),
            'total_track_ids': len(timeline.all_track_ids),
            'cameras': sorted(timeline.all_cameras),
            'track_ids': sorted(timeline.all_track_ids),
        },
        'jockey_colors': {
            str(tid): max(timeline.track_colors[tid].items(), key=lambda x: x[1])[0]
            for tid in timeline.all_track_ids
            if timeline.track_colors[tid]
        },
        'timeline': {}
    }

    # Build timeline data
    for cam_id in sorted(timeline.all_cameras):
        data['timeline'][cam_id] = {
            str(track_id): {
                'first_batch': app.first_batch,
                'last_batch': app.last_batch,
                'color': app.color,
                'detection_count': app.detection_count,
            }
            for track_id, app in timeline.appearances[cam_id].items()
        }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n✓ Timeline saved to: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/track_jockey_timeline.py <deepstream_log_file>")
        return 1

    log_path = Path(sys.argv[1])
    if not log_path.exists():
        print(f"❌ Log file not found: {log_path}")
        return 1

    # Parse log
    timeline = parse_log(log_path)

    # Print report
    print_timeline(timeline)

    # Save JSON
    output_path = log_path.parent / f"{log_path.stem}_timeline.json"
    save_json_timeline(timeline, output_path)

    print("\n" + "="*100)
    print("✅ Timeline analysis complete!")
    print("="*100)

    return 0


if __name__ == '__main__':
    sys.exit(main())
