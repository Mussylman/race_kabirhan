"""
make_test_config.py — Generate cameras.json for multi-camera test.

Takes a folder of video files and creates a config where each video
is treated as a separate camera on the track.

Usage:
    # All .mp4 files in a folder → cameras.json
    python tools/make_test_config.py data/videos/test_25cam/

    # Specific files in order
    python tools/make_test_config.py video1.mp4 video2.mp4 ... video25.mp4

    # Mix: 21 empty + 4 with horses (specify horse videos explicitly)
    python tools/make_test_config.py data/videos/test_25cam/ \
        --horse-cams 7 12 18 23 \
        --horse-videos data/videos/exp10_cam1.mp4 data/videos/exp10_cam2.mp4 \
                       data/videos/exp10_cam3.mp4 data/videos/exp10_cam4.mp4
"""

import sys
import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate cameras.json for testing")
    parser.add_argument("sources", nargs="+",
                        help="Video files or a directory containing .mp4 files")
    parser.add_argument("--track-length", type=float, default=2500,
                        help="Total track length in meters (default: 2500)")
    parser.add_argument("--overlap", type=float, default=10,
                        help="Camera overlap in meters (default: 10)")
    parser.add_argument("--max-active", type=int, default=8,
                        help="Max simultaneously active cameras (default: 8)")
    parser.add_argument("--horse-cams", type=int, nargs="*", default=[],
                        help="Camera numbers (1-based) to replace with horse videos")
    parser.add_argument("--horse-videos", nargs="*", default=[],
                        help="Horse video files to use for replacement")
    parser.add_argument("--output", default="configs/cameras_test.json",
                        help="Output config file (default: configs/cameras_test.json)")
    args = parser.parse_args()

    # Collect video files
    videos = []
    for src in args.sources:
        p = Path(src)
        if p.is_dir():
            found = sorted(p.glob("*.mp4"))
            if not found:
                found = sorted(p.glob("*.MP4"))
            videos.extend([str(f) for f in found])
        elif p.is_file():
            videos.append(str(p))
        else:
            print(f"WARNING: {src} not found, skipping")

    if not videos:
        print("ERROR: No video files found")
        return 1

    n = len(videos)
    print(f"Found {n} videos")

    # Replace horse cameras
    if args.horse_cams and args.horse_videos:
        if len(args.horse_cams) != len(args.horse_videos):
            print(f"ERROR: --horse-cams ({len(args.horse_cams)}) and "
                  f"--horse-videos ({len(args.horse_videos)}) must have same count")
            return 1

        for cam_num, horse_video in zip(args.horse_cams, args.horse_videos):
            if 1 <= cam_num <= n:
                old = videos[cam_num - 1]
                videos[cam_num - 1] = horse_video
                print(f"  cam-{cam_num:02d}: {Path(old).name} → {Path(horse_video).name} (HORSES)")
            else:
                print(f"WARNING: cam {cam_num} out of range 1..{n}")

    # Build config
    segment = args.track_length / n
    config = {
        "track_length": args.track_length,
        "max_active": args.max_active,
        "analytics": [],
        "display": [],
    }

    for i, video in enumerate(videos):
        start = i * segment
        end = min(start + segment + args.overlap, args.track_length)
        cam_id = f"cam-{i+1:02d}"

        config["analytics"].append({
            "id": cam_id,
            "url": str(Path(video).resolve()),
            "track_start": round(start, 1),
            "track_end": round(end, 1),
        })

        print(f"  {cam_id}: {Path(video).name} → {start:.0f}–{end:.0f}m")

    # Save
    output = Path(args.output)
    with open(output, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved: {output}")
    print(f"\nRun with:")
    print(f"  python -m api.server --config {output} --auto-start")

    return 0


if __name__ == "__main__":
    sys.exit(main())
