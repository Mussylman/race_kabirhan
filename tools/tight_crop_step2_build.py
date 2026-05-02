#!/usr/bin/env python3
"""tight_crop_step2_build.py — ШАГ 2:
  - tight crop из raw_frames для 752 multicam labeled crops
  - DINOv2 embeddings через HF processor + FP16 model
  - build prototypes_v2_multicam_tight.npz
  - stats comparison vs v2_multicam (без tight)
"""
from __future__ import annotations
import json, sys, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from tight_crop_utils import tight_crop_xyxy, TIGHT_X_LO, TIGHT_X_HI, TIGHT_Y_LO, TIGHT_Y_HI

WORK = Path("/home/ipodrom/race_vision_bench/dinov2_prod/multicam_collection")
OUT  = Path("/home/ipodrom/race_vision_bench/dinov2_prod/v2_tight_build")
TIGHT_CROPS = OUT / "crops_tight"
TIGHT_CROPS.mkdir(parents=True, exist_ok=True)

V2_BUILD = Path("/home/ipodrom/race_vision_bench/dinov2_prod/v2_build")

CLASS_ORDER = ["green", "yellow", "red", "blue"]


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"ШАГ 2 — tight crop build (X {TIGHT_X_LO:.0%}-{TIGHT_X_HI:.0%}, "
          f"Y {TIGHT_Y_LO:.0%}-{TIGHT_Y_HI:.0%})")
    print("=" * 78)

    inv = json.loads((WORK / "crops_index.json").read_text())
    by_id = {e["id"]: e for e in inv}
    labels = json.loads((WORK / "labels.json").read_text())

    # Build flat list — multicam only (color_dataset has no raw_frame + bbox)
    items = []
    skipped_no_bbox = 0
    for cls in CLASS_ORDER:
        for cid in labels.get(cls, []):
            e = by_id.get(cid)
            if not e: continue
            if e.get("origin") == "color_dataset":
                skipped_no_bbox += 1; continue
            if not e.get("frame_path") or not e.get("bbox"):
                skipped_no_bbox += 1; continue
            items.append({
                "id": cid, "label": cls,
                "frame_path": e["frame_path"], "bbox": e["bbox"],
            })
    print(f"  multicam labeled items: {len(items)}")
    print(f"  skipped (color_dataset/no bbox): {skipped_no_bbox}")
    by_class = defaultdict(int)
    for it in items: by_class[it["label"]] += 1
    print(f"    per class: {dict(by_class)}")
    print()

    # 1. Tight-crop and save
    print("=== generating tight crops from raw_frames ===")
    t1 = time.time()
    last_path = None; cached = None
    for i, it in enumerate(items):
        if it["frame_path"] != last_path:
            cached = cv2.imread(it["frame_path"]); last_path = it["frame_path"]
            if cached is None:
                print(f"  ERROR: cannot read {it['frame_path']}"); continue
        tight = tight_crop_xyxy(cached, it["bbox"])
        if tight.size == 0:
            print(f"  WARN: empty tight {it['id']}"); continue
        cv2.imwrite(str(TIGHT_CROPS / f"{it['id']}.jpg"), tight,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        if (i+1) % 200 == 0:
            print(f"  {i+1}/{len(items)}")
    print(f"  done: {time.time()-t1:.1f}s, "
          f"{len(list(TIGHT_CROPS.glob('*.jpg')))} files in {TIGHT_CROPS}")
    print()

    # 2. Extract DINOv2 embeddings via HF processor
    print("=== extracting DINOv2 embeddings (FP16, batch=16) ===")
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base",
                                      dtype=torch.float16).to("cuda").eval()
    BATCH = 16
    embs = np.empty((len(items), 768), dtype=np.float32)
    t2 = time.time()
    for i in range(0, len(items), BATCH):
        batch_items = items[i:i+BATCH]
        pils = [Image.open(TIGHT_CROPS / f"{it['id']}.jpg").convert("RGB")
                for it in batch_items]
        inputs = processor(images=pils, return_tensors="pt").to("cuda", torch.float16)
        with torch.inference_mode():
            out = model(**inputs)
        embs[i:i+len(batch_items)] = out.pooler_output.float().cpu().numpy()
    print(f"  done: {embs.shape}, {time.time()-t2:.1f}s "
          f"({len(items)/(time.time()-t2):.0f} crops/s)")
    print()

    # 3. Save embeddings_tight.npz
    embs_path = OUT / "embeddings_tight.npz"
    np.savez(embs_path,
             embeddings=embs,
             ids=np.array([it["id"] for it in items]),
             labels=np.array([it["label"] for it in items]),
             tight_x_lo=TIGHT_X_LO, tight_x_hi=TIGHT_X_HI,
             tight_y_lo=TIGHT_Y_LO, tight_y_hi=TIGHT_Y_HI)
    print(f"  saved: {embs_path}")

    # 4. Build prototypes (multicam only, tight)
    protos = np.zeros((4, 768), dtype=np.float32)
    counts = {}
    intra_stats = {}
    for ci, cls in enumerate(CLASS_ORDER):
        mask = np.array([it["label"] == cls for it in items])
        cls_embs = embs[mask]
        counts[cls] = int(mask.sum())
        if counts[cls] == 0: continue
        m = cls_embs.mean(axis=0)
        protos[ci] = m / (np.linalg.norm(m) + 1e-12)
        # Intra-class stats
        embs_n = cls_embs / (np.linalg.norm(cls_embs, axis=1, keepdims=True) + 1e-12)
        sims = embs_n @ protos[ci]
        intra_stats[cls] = {
            "n":      counts[cls],
            "median": float(np.median(sims)),
            "mean":   float(np.mean(sims)),
            "p25":    float(np.percentile(sims, 25)),
            "p75":    float(np.percentile(sims, 75)),
            "min":    float(np.min(sims)),
            "max":    float(np.max(sims)),
        }
    proto_path = OUT / "prototypes_v2_multicam_tight.npz"
    np.savez(proto_path,
             prototypes=protos,
             classes=np.array(CLASS_ORDER),
             n_per_class=np.array([counts[c] for c in CLASS_ORDER]),
             source="multicam_only_tight",
             tight_x_lo=TIGHT_X_LO, tight_x_hi=TIGHT_X_HI,
             tight_y_lo=TIGHT_Y_LO, tight_y_hi=TIGHT_Y_HI)
    inter = protos @ protos.T
    print(f"  saved: {proto_path}")
    print()

    # 5. Compare with existing v2_multicam (no tight)
    print("=" * 78)
    print("=== COMPARISON: v2_multicam (full bbox) vs v2_multicam_tight ===")
    print("=" * 78)

    # Load v2_multicam (no-tight) prototypes + intra stats
    v2_data = np.load(V2_BUILD / "prototypes_v2_multicam.npz", allow_pickle=False)
    protos_full = v2_data["prototypes"].astype(np.float32)
    inter_full = protos_full @ protos_full.T

    # Recompute intra for v2_multicam (no tight) using its embeddings
    v2_embs = np.load(V2_BUILD / "embeddings_all.npz", allow_pickle=False)
    e_all = v2_embs["embeddings"]
    l_all = v2_embs["labels"]
    s_all = v2_embs["sources"]
    intra_full = {}
    for ci, cls in enumerate(CLASS_ORDER):
        mask = (l_all == cls) & (s_all == "multicam")
        if not mask.any(): continue
        ce = e_all[mask]
        ce_n = ce / (np.linalg.norm(ce, axis=1, keepdims=True) + 1e-12)
        sims = ce_n @ protos_full[ci]
        intra_full[cls] = {
            "n": int(mask.sum()),
            "median": float(np.median(sims)),
            "mean":   float(np.mean(sims)),
        }

    print()
    print(f"  --- INTRA-class median similarity (выше = классы плотнее) ---")
    print(f"    {'class':<8s} {'n':>4s} {'full bbox':>11s} {'tight':>9s} {'Δ':>8s}")
    for cls in CLASS_ORDER:
        f = intra_full.get(cls, {}); t = intra_stats.get(cls, {})
        if not f or not t: continue
        d = t['median'] - f['median']
        marker = "↑" if d > 0.01 else ("↓" if d < -0.01 else "·")
        print(f"    {cls:<8s} {t['n']:>4d} {f['median']:>11.3f} {t['median']:>9.3f} "
              f"{d:>+8.3f} {marker}")
    print()

    print(f"  --- INTER-prototype 4×4 cosine (off-diagonal: ниже = лучше separation) ---")
    print(f"    full bbox:               tight:")
    print(f"            {' '.join(f'{c[:4]:>6s}' for c in CLASS_ORDER)}        "
          f"      {' '.join(f'{c[:4]:>6s}' for c in CLASS_ORDER)}")
    for i, ri in enumerate(CLASS_ORDER):
        f_row = " ".join(f"{inter_full[i][j]:>6.3f}" for j in range(4))
        t_row = " ".join(f"{inter[i][j]:>6.3f}" for j in range(4))
        print(f"    {ri:<5s} {f_row}      {ri:<5s} {t_row}")
    print()

    print(f"  --- Δ off-diagonal pairs ---")
    print(f"    {'pair':<22s} {'full':>8s} {'tight':>8s} {'Δ':>8s}")
    n_better = 0; n_worse = 0
    for i in range(4):
        for j in range(i+1, 4):
            d = inter[i][j] - inter_full[i][j]
            marker = "↓ better" if d < -0.01 else ("↑ worse" if d > 0.01 else "≈")
            if d < -0.01: n_better += 1
            elif d > 0.01: n_worse += 1
            print(f"    {CLASS_ORDER[i]:<7s} ↔ {CLASS_ORDER[j]:<8s} "
                  f"{inter_full[i][j]:>8.3f} {inter[i][j]:>8.3f} {d:>+8.3f}  {marker}")
    print()
    print(f"    pairs better-separated (Δ<-0.01): {n_better}/6")
    print(f"    pairs worse-separated (Δ>+0.01):  {n_worse}/6")
    print()

    # Save report
    report = {
        "tight_params": {
            "x_lo": TIGHT_X_LO, "x_hi": TIGHT_X_HI,
            "y_lo": TIGHT_Y_LO, "y_hi": TIGHT_Y_HI,
        },
        "n_items": len(items),
        "by_class": dict(by_class),
        "intra_full_bbox": intra_full,
        "intra_tight":     intra_stats,
        "inter_full_bbox": inter_full.tolist(),
        "inter_tight":     inter.tolist(),
        "n_better_pairs": n_better,
        "n_worse_pairs":  n_worse,
        "elapsed_sec":    time.time() - t0,
    }
    rpath = OUT / "step2_report.json"
    rpath.write_text(json.dumps(report, indent=2))
    print(f"  saved report: {rpath}")
    print(f"  total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
