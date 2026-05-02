#!/usr/bin/env python3
"""multicam_step5_build_prototypes_v2.py — ШАГ 5:
  - извлекаем DINOv2 embeddings для всех 1444 размеченных crops
  - строим 2 варианта prototypes (multicam-only vs combined)
  - intra-class similarity + inter-prototype matrix отчёт

Pipeline (REF: тот же что строил v1):
  HF AutoImageProcessor (resize-256 + center-crop-224 + ImageNet)
  → DINOv2-base FP16 GPU
  → pooler_output [768]
  → save FP32

Class order: [green, yellow, red, blue]  (как в v1, для совместимости header'а)
"""
from __future__ import annotations
import json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

WORK = Path("/home/ipodrom/race_vision_bench/dinov2_prod/multicam_collection")
OUT  = Path("/home/ipodrom/race_vision_bench/dinov2_prod/v2_build")
OUT.mkdir(parents=True, exist_ok=True)

CLASS_ORDER = ["green", "yellow", "red", "blue"]   # same as v1


def main():
    t0 = time.time()
    labels    = json.loads((WORK / "labels.json").read_text())
    inv_index = {e["id"]: e for e in
                 json.loads((WORK / "crops_index.json").read_text())}
    print("=" * 78)
    print("ШАГ 5 — embedding extraction + 2 prototype builds")
    print("=" * 78)

    # 1. Build flat list (id, label, source, path) — only items in 4 classes
    items = []
    for cls in CLASS_ORDER:
        for cid in labels.get(cls, []):
            e = inv_index.get(cid)
            if not e:
                print(f"  WARN: {cid} not in crops_index, skip"); continue
            src = "color_dataset" if e.get("origin") == "color_dataset" else "multicam"
            items.append({
                "id":        cid,
                "label":     cls,
                "source":    src,
                "crop_path": e["crop_path"],
            })
    print(f"  loaded {len(items)} labeled items")
    by_class = defaultdict(int); by_src = defaultdict(int)
    for it in items:
        by_class[it["label"]] += 1; by_src[it["source"]] += 1
    print(f"    per class:  {dict(by_class)}")
    print(f"    per source: {dict(by_src)}")
    print()

    # 2. Load HF processor + model
    print("=== loading HF DINOv2-base FP16 + processor ===")
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base",
                                      dtype=torch.float16).to("cuda").eval()

    # 3. Batch embedding extraction (batch=16)
    print("=== extracting embeddings (batch=16) ===")
    BATCH = 16
    embeddings = np.empty((len(items), 768), dtype=np.float32)
    t1 = time.time()
    for i in range(0, len(items), BATCH):
        batch_items = items[i:i+BATCH]
        pils = []
        for it in batch_items:
            try:
                pils.append(Image.open(it["crop_path"]).convert("RGB"))
            except Exception as ex:
                print(f"  ERROR: {it['crop_path']}: {ex}"); raise
        inputs = processor(images=pils, return_tensors="pt").to("cuda", torch.float16)
        with torch.inference_mode():
            out = model(**inputs)
        embs = out.pooler_output.float().cpu().numpy()
        embeddings[i:i+len(batch_items)] = embs
        if (i // BATCH) % 10 == 0:
            done = i + len(batch_items)
            eta = (time.time() - t1) / max(done, 1) * (len(items) - done)
            print(f"  {done}/{len(items)}  ({time.time()-t1:.1f}s, ETA {eta:.0f}s)")
    print(f"  done: embeddings shape={embeddings.shape}, "
          f"{time.time()-t1:.1f}s total")
    print()

    # 4. Save embeddings_all.npz
    embeddings_path = OUT / "embeddings_all.npz"
    np.savez(
        embeddings_path,
        embeddings=embeddings,
        ids=np.array([it["id"] for it in items]),
        labels=np.array([it["label"] for it in items]),
        sources=np.array([it["source"] for it in items]),
        paths=np.array([it["crop_path"] for it in items]),
    )
    print(f"  saved: {embeddings_path}")
    print()

    # 5. Build 2 prototype variants
    def build_protos(mask_name: str, mask):
        protos = np.zeros((4, 768), dtype=np.float32)
        counts = {}
        intra_stats = {}
        for ci, cls in enumerate(CLASS_ORDER):
            cls_mask = mask & np.array([it["label"] == cls for it in items])
            embs = embeddings[cls_mask]
            counts[cls] = int(cls_mask.sum())
            if counts[cls] == 0:
                print(f"  WARN: {mask_name} has 0 crops for {cls}!")
                continue
            mean = embs.mean(axis=0)
            mean = mean / (np.linalg.norm(mean) + 1e-12)
            protos[ci] = mean
            # Intra-class: cos sim of each emb to mean
            embs_n = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
            sims = embs_n @ mean
            intra_stats[cls] = {
                "n":      int(cls_mask.sum()),
                "mean":   float(np.mean(sims)),
                "p25":    float(np.percentile(sims, 25)),
                "median": float(np.median(sims)),
                "p75":    float(np.percentile(sims, 75)),
                "min":    float(np.min(sims)),
                "max":    float(np.max(sims)),
            }
        # Inter-prototype 4×4 cosine
        inter = protos @ protos.T
        return protos, counts, intra_stats, inter

    print("=== building prototypes_v2_multicam (multicam-only) ===")
    mask_mc = np.array([it["source"] == "multicam" for it in items])
    print(f"  mask: {mask_mc.sum()} crops")
    protos_mc, counts_mc, intra_mc, inter_mc = build_protos("multicam", mask_mc)
    np.savez(
        OUT / "prototypes_v2_multicam.npz",
        prototypes=protos_mc,
        classes=np.array(CLASS_ORDER),
        n_per_class=np.array([counts_mc[c] for c in CLASS_ORDER]),
        source="multicam_only",
    )
    print(f"  saved: {OUT}/prototypes_v2_multicam.npz")
    print()

    print("=== building prototypes_v2_combined (multicam + color_dataset) ===")
    mask_all = np.ones(len(items), dtype=bool)
    print(f"  mask: {mask_all.sum()} crops (all)")
    protos_cb, counts_cb, intra_cb, inter_cb = build_protos("combined", mask_all)
    np.savez(
        OUT / "prototypes_v2_combined.npz",
        prototypes=protos_cb,
        classes=np.array(CLASS_ORDER),
        n_per_class=np.array([counts_cb[c] for c in CLASS_ORDER]),
        source="multicam_plus_color_dataset",
    )
    print(f"  saved: {OUT}/prototypes_v2_combined.npz")
    print()

    # 6. Print comparison report
    def print_intra(title, intra, counts):
        print(f"\n  === {title} ===")
        print(f"    {'class':<8s} {'n':>4s} {'min':>6s} {'p25':>6s} {'median':>7s} "
              f"{'p75':>6s} {'max':>6s} {'mean':>6s}")
        for cls in CLASS_ORDER:
            s = intra[cls]
            print(f"    {cls:<8s} {counts[cls]:>4d} "
                  f"{s['min']:>6.3f} {s['p25']:>6.3f} {s['median']:>7.3f} "
                  f"{s['p75']:>6.3f} {s['max']:>6.3f} {s['mean']:>6.3f}")

    print("=" * 78)
    print("=== INTRA-CLASS SIMILARITY (cos sim к mean prototype) ===")
    print("=" * 78)
    print_intra("v2_multicam (752 crops)", intra_mc, counts_mc)
    print_intra("v2_combined (1444 crops)", intra_cb, counts_cb)

    print("\n" + "=" * 78)
    print("=== INTER-PROTOTYPE 4×4 COSINE SIMILARITY ===")
    print("=" * 78)
    def print_inter(title, inter):
        print(f"\n  --- {title} ---")
        hdr = "         " + "".join(f"{c:>9s}" for c in CLASS_ORDER)
        print(hdr)
        for i, ri in enumerate(CLASS_ORDER):
            row = f"  {ri:<7s}" + "".join(f"{inter[i][j]:>9.3f}" for j in range(4))
            print(row)
    print_inter("v2_multicam", inter_mc)
    print_inter("v2_combined", inter_cb)

    # 7. Compare which prototypes shifted "closer" combined vs multicam
    # — признак color_dataset noise добавляет/убирает class separation
    print("\n" + "=" * 78)
    print("=== СДВИГ COMBINED vs MULTICAM (off-diagonal Δ) ===")
    print("=" * 78)
    print("  Положительный Δ = в combined пара prototypes СБЛИЖАЕТСЯ "
          "(хуже разделимость)")
    print("  Отрицательный Δ = в combined пара ОТДАЛЯЕТСЯ (лучше)")
    print()
    print(f"    {'pair':<20s} {'multicam':>10s} {'combined':>10s} {'Δ':>8s}")
    pair_diffs = []
    for i in range(4):
        for j in range(i+1, 4):
            d = inter_cb[i][j] - inter_mc[i][j]
            pair_diffs.append((CLASS_ORDER[i], CLASS_ORDER[j],
                               inter_mc[i][j], inter_cb[i][j], d))
            print(f"    {CLASS_ORDER[i]:<7s} ↔ {CLASS_ORDER[j]:<8s} "
                  f"{inter_mc[i][j]:>10.3f} {inter_cb[i][j]:>10.3f} "
                  f"{d:>+8.3f}")
    n_closer = sum(1 for _,_,_,_,d in pair_diffs if d > 0.01)
    n_farther = sum(1 for _,_,_,_,d in pair_diffs if d < -0.01)
    print(f"\n    pairs closer (Δ>0.01) in combined: {n_closer}/6")
    print(f"    pairs farther (Δ<-0.01)           : {n_farther}/6")

    # 8. Save report
    report = {
        "n_items": len(items),
        "by_class":  dict(by_class),
        "by_source": dict(by_src),
        "intra_multicam": intra_mc,
        "intra_combined": intra_cb,
        "inter_multicam": inter_mc.tolist(),
        "inter_combined": inter_cb.tolist(),
        "pair_diffs": [
            {"a": a, "b": b, "multicam": float(m), "combined": float(c),
             "delta": float(d)} for a, b, m, c, d in pair_diffs
        ],
        "elapsed_sec": time.time() - t0,
    }
    (OUT / "step5_report.json").write_text(json.dumps(report, indent=2))

    print()
    print("=" * 78)
    print(f"  artifacts:")
    for p in [embeddings_path, OUT/"prototypes_v2_multicam.npz",
              OUT/"prototypes_v2_combined.npz", OUT/"step5_report.json"]:
        print(f"    {p}  ({p.stat().st_size/1024:.1f} KB)")
    print(f"  total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
