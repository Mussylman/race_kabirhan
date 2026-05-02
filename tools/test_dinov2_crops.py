#!/usr/bin/env python3
"""
Quick test: run DINOv2 ReID on saved crops from ds_results/exp2/crops/
to see if larger full-bbox crops fix the UNKNOWN problem.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from pipeline.jockey_reid import JockeyReID

CROPS_DIR = Path("ds_results/exp2/crops")
GALLERY_DIR = Path("data/reid")

def main():
    # Load ReID
    print("Loading DINOv2 ReID...")
    reid = JockeyReID(gallery_dir=str(GALLERY_DIR), backend="dinov2", device="cuda:0")
    print(f"Gallery: {len(reid.jockey_names)} jockeys, {len(reid.gallery_embeddings)} embeddings")

    # Load all crops
    crop_files = sorted(CROPS_DIR.glob("*.jpg"))
    print(f"Found {len(crop_files)} crops")

    # Process in batches
    BATCH = 64
    results = []  # (filename, jockey_name, score, crop_w, crop_h)

    for i in range(0, len(crop_files), BATCH):
        batch_files = crop_files[i:i+BATCH]
        crops = []
        for f in batch_files:
            img = cv2.imread(str(f))
            if img is not None:
                crops.append(img)
            else:
                crops.append(np.zeros((10, 10, 3), dtype=np.uint8))

        batch_results = reid.identify_batch(crops, top_k=5)

        for f, (name, score, details) in zip(batch_files, batch_results):
            h, w = cv2.imread(str(f)).shape[:2]
            results.append((f.name, name, score, w, h))

        if (i // BATCH) % 5 == 0:
            print(f"  processed {min(i+BATCH, len(crop_files))}/{len(crop_files)}")

    # === Statistics ===
    print(f"\n{'='*60}")
    print(f"RESULTS: DINOv2 ReID on {len(results)} full-bbox crops (mux 1920x1080)")
    print(f"{'='*60}")

    # Overall
    id_counts = Counter(name for _, name, _, _, _ in results)
    total = len(results)
    unknown = id_counts.get("unknown", 0)
    identified = total - unknown
    print(f"\nIdentified: {identified}/{total} ({100*identified/total:.1f}%)")
    print(f"Unknown:    {unknown}/{total} ({100*unknown/total:.1f}%)")

    print(f"\nPer-jockey breakdown:")
    for name, count in sorted(id_counts.items(), key=lambda x: -x[1]):
        avg_score = np.mean([s for _, n, s, _, _ in results if n == name])
        print(f"  {name:12s}: {count:5d} ({100*count/total:5.1f}%)  avg_score={avg_score:.3f}")

    # Per-track analysis (from filename: b00011_cam-05_t1_yellow_51x106.jpg)
    print(f"\nPer-track analysis:")
    track_results = defaultdict(list)
    for fname, name, score, w, h in results:
        parts = fname.split("_")
        track_id = [p for p in parts if p.startswith("t") and p[1:].isdigit()]
        if track_id:
            track_results[track_id[0]].append((name, score))

    for tid in sorted(track_results.keys()):
        entries = track_results[tid]
        id_dist = Counter(n for n, _ in entries)
        total_t = len(entries)
        dominant = id_dist.most_common(1)[0]
        avg_s = np.mean([s for _, s in entries])
        print(f"  {tid}: {total_t:4d} dets, dominant={dominant[0]}({dominant[1]}/{total_t}), "
              f"avg_score={avg_s:.3f}, ids={dict(id_dist)}")

    # Crop size vs identification
    print(f"\nCrop size vs identification rate:")
    size_buckets = {"tiny(<1000px)": [], "small(1-3k)": [], "medium(3-8k)": [], "large(>8k)": []}
    for _, name, score, w, h in results:
        px = w * h
        if px < 1000:
            size_buckets["tiny(<1000px)"].append(name != "unknown")
        elif px < 3000:
            size_buckets["small(1-3k)"].append(name != "unknown")
        elif px < 8000:
            size_buckets["medium(3-8k)"].append(name != "unknown")
        else:
            size_buckets["large(>8k)"].append(name != "unknown")

    for bucket, identified_list in size_buckets.items():
        if identified_list:
            rate = 100 * sum(identified_list) / len(identified_list)
            print(f"  {bucket:18s}: {len(identified_list):5d} crops, identified {rate:.1f}%")

    # Compare with old results (session_log said 24% identified with torso crops)
    print(f"\n{'='*60}")
    print(f"COMPARISON:")
    print(f"  Old (torso 28x28, mux 1520): 24% identified, 76% unknown")
    print(f"  New (full bbox ~59x117, mux 1920): {100*identified/total:.1f}% identified, {100*unknown/total:.1f}% unknown")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
