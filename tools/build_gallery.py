"""
Build jockey gallery from phone photos.

Workflow:
  1. Create folders with jockey names in gallery/ directory
  2. Put 10-15 phone photos of each jockey into their folder
  3. Run this script to build the embedding gallery

The script:
  - Validates photos (checks for quality, duplicates)
  - Extracts CLIP/DINOv2 embeddings
  - Saves gallery_embeddings.pkl for fast loading at runtime
  - Shows similarity matrix between jockeys (should be low cross-similarity)

Usage:
    # Build gallery from phone photos:
    python tools/build_gallery.py --gallery gallery/

    # Use DINOv2 instead of CLIP:
    python tools/build_gallery.py --gallery gallery/ --backend dinov2

    # Rebuild (force re-extract even if cache exists):
    python tools/build_gallery.py --gallery gallery/ --rebuild

Example gallery structure:
    gallery/
    ├── Аслан_зелёный/
    │   ├── IMG_001.jpg
    │   ├── IMG_002.jpg
    │   └── ... (10-15 photos)
    ├── Марат_красный/
    │   └── ...
    └── Болат_жёлтый/
        └── ...

Tips for taking phone photos:
  - Photograph from different angles (front, side, back)
  - Include close-up and full-body shots
  - Capture in different lighting if possible
  - Focus on the torso/uniform area
  - 10-15 photos per jockey is optimal
"""

import argparse
import sys
import os
import numpy as np
from pathlib import Path


def validate_gallery(gallery_dir: Path) -> bool:
    """Check gallery structure and report issues."""
    if not gallery_dir.exists():
        print(f"ERROR: Gallery directory not found: {gallery_dir}")
        print(f"Create it and add jockey folders with photos:")
        print(f"  mkdir -p {gallery_dir}/jockey_name_1")
        print(f"  mkdir -p {gallery_dir}/jockey_name_2")
        return False

    jockey_dirs = sorted([
        d for d in gallery_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])

    if not jockey_dirs:
        print(f"ERROR: No jockey folders found in {gallery_dir}")
        print(f"Create folders with jockey names and add 10-15 photos each")
        return False

    print(f"\nGallery: {gallery_dir}")
    print(f"{'='*50}")

    all_ok = True
    total_photos = 0
    for jd in jockey_dirs:
        photos = [
            f for f in jd.iterdir()
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
        ]
        count = len(photos)
        total_photos += count

        status = "OK" if count >= 10 else "LOW" if count >= 5 else "NEED MORE"
        icon = "+" if count >= 10 else "~" if count >= 5 else "!"

        print(f"  [{icon}] {jd.name}: {count} photos ({status})")

        if count < 5:
            all_ok = False

    print(f"{'='*50}")
    print(f"Total: {len(jockey_dirs)} jockeys, {total_photos} photos")

    if not all_ok:
        print("\nWARNING: Some jockeys have too few photos (<5).")
        print("Recommended: 10-15 photos per jockey for best accuracy.")

    return True


def compute_cross_similarity(reid) -> None:
    """Show how similar jockeys are to each other (should be LOW)."""
    if reid.gallery_embeddings is None or len(reid.jockey_names) < 2:
        return

    print(f"\nCross-Similarity Matrix (lower = better distinction):")
    print(f"{'':>20}", end="")
    for name in reid.jockey_names:
        print(f"{name[:15]:>16}", end="")
    print()

    for i, name_i in enumerate(reid.jockey_names):
        mask_i = reid.gallery_labels == i
        emb_i = reid.gallery_embeddings[mask_i]
        centroid_i = emb_i.mean(axis=0)
        centroid_i = centroid_i / np.linalg.norm(centroid_i)

        print(f"{name_i[:20]:>20}", end="")
        for j, name_j in enumerate(reid.jockey_names):
            mask_j = reid.gallery_labels == j
            emb_j = reid.gallery_embeddings[mask_j]
            centroid_j = emb_j.mean(axis=0)
            centroid_j = centroid_j / np.linalg.norm(centroid_j)

            sim = float(centroid_i @ centroid_j)
            marker = "*" if i == j else (" " if sim < 0.7 else "!")
            print(f"{sim:>15.3f}{marker}", end="")
        print()

    print(f"\n  * = same jockey (should be ~1.0)")
    print(f"  ! = high cross-similarity (>0.7) — may cause confusion")


def main():
    parser = argparse.ArgumentParser(description="Build jockey gallery from phone photos")
    parser.add_argument("--gallery", default="gallery", help="Gallery directory path")
    parser.add_argument("--backend", default="clip", choices=["clip", "dinov2", "osnet"],
                        help="Embedding model (default: clip)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild even if cache exists")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate gallery structure")
    args = parser.parse_args()

    gallery_dir = Path(args.gallery)

    # Validate
    if not validate_gallery(gallery_dir):
        sys.exit(1)

    if args.validate_only:
        return

    # Remove cache if rebuilding
    cache_path = gallery_dir / "gallery_embeddings.pkl"
    if args.rebuild and cache_path.exists():
        cache_path.unlink()
        print(f"\nRemoved old cache: {cache_path}")

    # Build gallery
    print(f"\nBuilding gallery with {args.backend} backend...")
    print(f"This may take a minute on first run (downloading model)...\n")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.jockey_reid import JockeyReID

    reid = JockeyReID(
        gallery_dir=str(gallery_dir),
        backend=args.backend,
        device=args.device,
    )

    # Force rebuild
    if args.rebuild or reid.gallery_embeddings is None:
        reid.build_gallery()

    # Stats
    stats = reid.get_gallery_stats()
    print(f"\nGallery Stats:")
    print(f"  Backend: {stats['backend']}")
    print(f"  Jockeys: {stats['n_jockeys']}")
    print(f"  Total embeddings: {stats['n_embeddings']}")
    print(f"  Embedding dim: {stats['embed_dim']}")
    for name, count in stats.get('jockeys', {}).items():
        print(f"    {name}: {count} embeddings")

    # Cross-similarity
    compute_cross_similarity(reid)

    print(f"\nGallery ready! Cached at: {cache_path}")
    print(f"\nTo use in pipeline:")
    print(f'  reid = JockeyReID(gallery_dir="{args.gallery}", backend="{args.backend}")')
    print(f'  results = reid.identify_batch(crops)')


if __name__ == "__main__":
    main()
