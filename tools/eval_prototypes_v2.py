#!/usr/bin/env python3
"""eval_prototypes_v2.py — ШАГ 6: regression test на golden cam-13 video.

Сравнение 3 prototypes:
  v1_prod      = ~/race_vision_bench/dinov2_prod/prototypes_dinov2_v1.npz
  v2_multicam  = .../v2_build/prototypes_v2_multicam.npz
  v2_combined  = .../v2_build/prototypes_v2_combined.npz

Test set: 420 размеченных crops в output/diagnose_v4_2026-04-25/labels.json
  (170 jockey color + 250 not_jockey + 23 HARD cases в frame_002310..370)

Pipeline (REF, как в v1 build):
  HF AutoImageProcessor (resize-256 + center-crop-224 + ImageNet)
  → DINOv2-base FP16 GPU
  → cosine sim ко всем 4 prototypes
  → argmax + threshold reject (0.55 default)

Output:
  - сравнительная таблица 3 prototypes на 6 metrics
  - threshold sweep (0.45..0.75 step 0.05) для v2_multicam и v2_combined
  - verdict какой prototype set выигрывает
"""
from __future__ import annotations
import json, sys, time
from collections import Counter
from pathlib import Path

import numpy as np

REPO     = Path(__file__).resolve().parent.parent
DIAGNOSE = REPO / "output" / "diagnose_v4_2026-04-25"
LABELS_JSON = DIAGNOSE / "labels.json"
CROPS_INDEX = DIAGNOSE / "crops_index.json"
RAW_FRAMES  = DIAGNOSE / "raw_frames"
CROPS_DIR_MARGIN = DIAGNOSE / "crops_raw"   # с 5% margin

WORK_BENCH = Path("/home/ipodrom/race_vision_bench/dinov2_prod")
PROTOS_V1  = WORK_BENCH / "prototypes_dinov2_v1.npz"
PROTOS_MC  = WORK_BENCH / "v2_build" / "prototypes_v2_multicam.npz"
PROTOS_CB  = WORK_BENCH / "v2_build" / "prototypes_v2_combined.npz"

OUT_DIR = WORK_BENCH / "v2_build"
COLOR_UNKNOWN_ID = 255
LABELS_COLOR_TXT = {"blue": 0, "green": 1, "purple": 2, "red": 3, "yellow": 4}

# 23 HARD cases — frame_002310..002370, NJ
HARD_FRAMES = {f"frame_{f:06d}" for f in range(2310, 2371, 30)}


def load_protos(path):
    d = np.load(path, allow_pickle=False)
    return d["prototypes"].astype(np.float32), [str(c) for c in d["classes"]]


def parser(emb, protos, classes, thr):
    norm = float(np.linalg.norm(emb)) + 1e-12
    sims = (protos @ emb) / norm
    best = int(np.argmax(sims)); ms = float(sims[best])
    if ms < thr: return "unknown", ms
    return classes[best], ms


def compute_metrics(predictions, true_labels, hard_ids):
    """predictions: [{name, sim}] aligned to true_labels."""
    n_jockey_true = sum(1 for t in true_labels if t != "not_jockey")
    n_nj_true     = sum(1 for t in true_labels if t == "not_jockey")
    n_jockey_correct = 0; n_jockey_wrong = 0; n_jockey_reject = 0
    n_nj_reject = 0; n_nj_passed = 0; n_nj_conf_wrong = 0
    n_yellow_to_red = 0
    n_hard_rejected = 0; n_hard_total = 0
    confusion = {r: Counter() for r in ["blue","green","red","yellow","not_jockey"]}
    for i, (p, t) in enumerate(zip(predictions, true_labels)):
        pred = p["name"]; sim = p["sim"]
        if t in confusion: confusion[t][pred] += 1
        if t != "not_jockey":
            if pred == t: n_jockey_correct += 1
            elif pred == "unknown": n_jockey_reject += 1
            else: n_jockey_wrong += 1
            if t == "yellow" and pred == "red": n_yellow_to_red += 1
        else:
            if pred == "unknown": n_nj_reject += 1
            else:
                n_nj_passed += 1
                if sim > 0.9: n_nj_conf_wrong += 1
        if i in hard_ids:
            n_hard_total += 1
            if pred == "unknown": n_hard_rejected += 1
    return {
        "n_total":            len(predictions),
        "n_jockey_true":      n_jockey_true,
        "n_nj_true":          n_nj_true,
        "jockey_acc":         n_jockey_correct / max(n_jockey_true, 1),
        "jockey_wrong":       n_jockey_wrong,
        "jockey_rejected":    n_jockey_reject,
        "nj_reject_rate":     n_nj_reject / max(n_nj_true, 1),
        "nj_passed":          n_nj_passed,
        "nj_conf_wrong":      n_nj_conf_wrong,
        "nj_conf_wrong_rate": n_nj_conf_wrong / max(n_nj_true, 1),
        "yellow_to_red":      n_yellow_to_red,
        "hard_rejected":      n_hard_rejected,
        "hard_total":         n_hard_total,
        "confusion":          {k: dict(v) for k, v in confusion.items()},
    }


def main():
    t0 = time.time()
    print("=" * 84)
    print("ШАГ 6 — regression test on golden cam-13 (3-way: v1 / v2_multicam / v2_combined)")
    print("=" * 84)

    # 1. Load golden labels
    labels = json.loads(LABELS_JSON.read_text())
    crops_index = json.loads(CROPS_INDEX.read_text())
    by_id = {e["crop"].replace(".jpg",""): e for e in crops_index}
    label_map = {cid: cls for cls, ids in labels.items() for cid in ids}

    # 2. Build flat list (id, label, frame_path) — only labeled
    items = []
    missing = 0
    for cid, true_lbl in label_map.items():
        e = by_id.get(cid)
        if not e:
            missing += 1; continue
        # Re-crop from raw frame at exact bbox (no margin) — production-faithful
        items.append({
            "id":         cid,
            "label":      true_lbl,
            "frame":      e["frame"],
            "bbox":       e["bbox"],
            "h":          e.get("h", 0),
            "w":          e.get("w", 0),
        })
    print(f"  loaded {len(items)} labeled crops "
          f"({sum(1 for it in items if it['label']!='not_jockey')} jockey, "
          f"{sum(1 for it in items if it['label']=='not_jockey')} not_jockey)")
    if missing: print(f"  missing in crops_index: {missing}")

    # 3. Load all 3 prototype sets
    print()
    print("=== loading prototypes ===")
    p_v1, c_v1 = load_protos(PROTOS_V1)
    p_mc, c_mc = load_protos(PROTOS_MC)
    p_cb, c_cb = load_protos(PROTOS_CB)
    print(f"  v1_prod:     shape={p_v1.shape}, order={c_v1}")
    print(f"  v2_multicam: shape={p_mc.shape}, order={c_mc}")
    print(f"  v2_combined: shape={p_cb.shape}, order={c_cb}")
    assert c_v1 == c_mc == c_cb, "Class order differs across prototypes!"

    # 4. Extract embeddings for 420 crops via HF (REF preprocessing)
    print()
    print("=== extracting embeddings (420 crops, HF processor) ===")
    import cv2
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base",
                                      dtype=torch.float16).to("cuda").eval()
    BATCH = 16
    embs = np.empty((len(items), 768), dtype=np.float32)
    pils = []
    last_frame = None; cached = None
    pre_t = time.time()
    for it in items:
        fp = str(RAW_FRAMES / it["frame"])
        if fp != last_frame:
            cached = cv2.imread(fp); last_frame = fp
        x1,y1,x2,y2 = it["bbox"]
        h_img,w_img = cached.shape[:2]
        x1=max(0,int(x1)); y1=max(0,int(y1))
        x2=min(w_img,int(x2)); y2=min(h_img,int(y2))
        crop = cached[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"  WARN: empty crop {it['id']}"); pils.append(None); continue
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pils.append(Image.fromarray(rgb))
    print(f"  re-crop time: {time.time()-pre_t:.1f}s")
    inf_t = time.time()
    for i in range(0, len(items), BATCH):
        batch = pils[i:i+BATCH]
        inputs = processor(images=batch, return_tensors="pt").to("cuda", torch.float16)
        with torch.inference_mode():
            out = model(**inputs)
        embs[i:i+len(batch)] = out.pooler_output.float().cpu().numpy()
    print(f"  embedding inference: {time.time()-inf_t:.1f}s "
          f"({len(items)/(time.time()-inf_t):.0f} crops/s)")
    print()

    # 5. Mark HARD case indices
    hard_idx = set()
    for i, it in enumerate(items):
        frame_stem = it["frame"].replace(".jpg", "")
        if it["label"] == "not_jockey" and frame_stem in HARD_FRAMES:
            hard_idx.add(i)
    print(f"  HARD NJ cases: {len(hard_idx)} (target ≥18/23 reject)")
    print()

    # 6. Predict with each prototype + threshold 0.55 (default)
    THR_DEFAULT = 0.55
    true_labels = [it["label"] for it in items]
    def predict_all(p, c, thr):
        preds = []
        for i in range(len(items)):
            name, sim = parser(embs[i], p, c, thr)
            preds.append({"name": name, "sim": sim})
        return preds
    preds_v1 = predict_all(p_v1, c_v1, THR_DEFAULT)
    preds_mc = predict_all(p_mc, c_mc, THR_DEFAULT)
    preds_cb = predict_all(p_cb, c_cb, THR_DEFAULT)
    m_v1 = compute_metrics(preds_v1, true_labels, hard_idx)
    m_mc = compute_metrics(preds_mc, true_labels, hard_idx)
    m_cb = compute_metrics(preds_cb, true_labels, hard_idx)

    # 7. Print main comparison table
    print("=" * 84)
    print("=== MAIN COMPARISON TABLE (threshold = 0.55) ===")
    print("=" * 84)
    print(f"  {'metric':<32s} {'v1_prod':>12s} {'v2_multicam':>14s} {'v2_combined':>14s}")
    print("  " + "-" * 76)
    rows = [
        ("jockey color accuracy %",
         100*m_v1['jockey_acc'], 100*m_mc['jockey_acc'], 100*m_cb['jockey_acc']),
        ("  → wrong color",
         m_v1['jockey_wrong'], m_mc['jockey_wrong'], m_cb['jockey_wrong']),
        ("  → rejected (unknown)",
         m_v1['jockey_rejected'], m_mc['jockey_rejected'], m_cb['jockey_rejected']),
        ("not_jockey reject rate %",
         100*m_v1['nj_reject_rate'], 100*m_mc['nj_reject_rate'], 100*m_cb['nj_reject_rate']),
        ("  → NJ passed as color",
         m_v1['nj_passed'], m_mc['nj_passed'], m_cb['nj_passed']),
        ("confident wrong on NJ %",
         100*m_v1['nj_conf_wrong_rate'], 100*m_mc['nj_conf_wrong_rate'], 100*m_cb['nj_conf_wrong_rate']),
        (f"23 HARD NJ rejected ({m_v1['hard_total']})",
         m_v1['hard_rejected'], m_mc['hard_rejected'], m_cb['hard_rejected']),
        ("yellow → red errors",
         m_v1['yellow_to_red'], m_mc['yellow_to_red'], m_cb['yellow_to_red']),
    ]
    for name, a, b, c in rows:
        if isinstance(a, float):
            print(f"  {name:<32s} {a:>12.1f} {b:>14.1f} {c:>14.1f}")
        else:
            print(f"  {name:<32s} {a:>12d} {b:>14d} {c:>14d}")
    print()

    # 8. Per-class confusion
    def print_cm(name, m):
        print(f"  --- {name} confusion (rows=true, cols=pred) ---")
        cols = ["blue","green","red","yellow","unknown"]
        rows = ["blue","green","red","yellow","not_jockey"]
        print(f"    {'':<13s}" + "".join(f"{c:>10s}" for c in cols))
        for r in rows:
            print(f"    {r:<13s}" + "".join(
                f"{m['confusion'].get(r,{}).get(c,0):>10d}" for c in cols))
    print_cm("v1_prod", m_v1)
    print_cm("v2_multicam", m_mc)
    print_cm("v2_combined", m_cb)
    print()

    # 9. Threshold sweep for v2_multicam and v2_combined
    print("=" * 84)
    print("=== THRESHOLD SWEEP (v2_multicam, v2_combined) ===")
    print("=" * 84)
    sweep_results = {"multicam": [], "combined": []}
    for variant, p, c in [("multicam", p_mc, c_mc), ("combined", p_cb, c_cb)]:
        print(f"\n  ─── {variant} ───")
        print(f"    {'thr':>5s} {'jockey%':>9s} {'NJ_rej%':>9s} "
              f"{'NJ_cw%':>8s} {'HARD':>5s} {'composite':>10s}")
        for thr in np.arange(0.45, 0.76, 0.05):
            thr_val = round(float(thr), 2)
            preds = predict_all(p, c, thr_val)
            m = compute_metrics(preds, true_labels, hard_idx)
            # Composite: weighted score that prefers high jockey_acc + high NJ_reject
            # + small NJ_conf_wrong + good HARD reject
            comp = (
                m['jockey_acc'] * 0.35
              + m['nj_reject_rate'] * 0.35
              + (1.0 - m['nj_conf_wrong_rate']) * 0.15
              + (m['hard_rejected'] / max(m['hard_total'], 1)) * 0.15
            )
            sweep_results[variant].append({
                "thr":             thr_val,
                "jockey_acc":      m['jockey_acc'],
                "nj_reject_rate":  m['nj_reject_rate'],
                "nj_conf_wrong":   m['nj_conf_wrong_rate'],
                "hard_rejected":   m['hard_rejected'],
                "hard_total":      m['hard_total'],
                "composite":       comp,
            })
            print(f"    {thr_val:>5.2f} {100*m['jockey_acc']:>8.1f}% "
                  f"{100*m['nj_reject_rate']:>8.1f}% "
                  f"{100*m['nj_conf_wrong_rate']:>7.1f}% "
                  f"{m['hard_rejected']:>2d}/{m['hard_total']:<2d} "
                  f"{comp:>10.3f}")
        # Top-3
        top3 = sorted(sweep_results[variant], key=lambda r: -r['composite'])[:3]
        print(f"    TOP-3 by composite:")
        for r in top3:
            print(f"      thr={r['thr']:.2f}  composite={r['composite']:.3f} "
                  f"(jockey={100*r['jockey_acc']:.1f}%  NJ_rej={100*r['nj_reject_rate']:.1f}% "
                  f"HARD={r['hard_rejected']}/{r['hard_total']})")
    print()

    # 10. Verdict
    print("=" * 84)
    print("=== VERDICT ===")
    print("=" * 84)
    # Compare best of each at default thr=0.55
    candidates = [
        ("v1_prod",     m_v1, 0.55),
        ("v2_multicam", m_mc, 0.55),
        ("v2_combined", m_cb, 0.55),
    ]
    print(f"  composite @ thr=0.55:")
    for name, m, t in candidates:
        comp = (m['jockey_acc']*0.35 + m['nj_reject_rate']*0.35
              + (1-m['nj_conf_wrong_rate'])*0.15
              + (m['hard_rejected']/max(m['hard_total'],1))*0.15)
        print(f"    {name:<14s}: {comp:.3f}  "
              f"(jockey={100*m['jockey_acc']:.1f}%  NJ={100*m['nj_reject_rate']:.1f}%)")
    # Also report best from each sweep
    best_mc = max(sweep_results['multicam'], key=lambda r: r['composite'])
    best_cb = max(sweep_results['combined'], key=lambda r: r['composite'])
    print()
    print(f"  best v2_multicam: thr={best_mc['thr']:.2f} composite={best_mc['composite']:.3f}")
    print(f"  best v2_combined: thr={best_cb['thr']:.2f} composite={best_cb['composite']:.3f}")

    # Save full report
    full = {
        "n_items": len(items),
        "n_jockey_true": m_v1['n_jockey_true'],
        "n_nj_true": m_v1['n_nj_true'],
        "n_hard": len(hard_idx),
        "default_thr": THR_DEFAULT,
        "metrics_at_default": {
            "v1_prod":     m_v1,
            "v2_multicam": m_mc,
            "v2_combined": m_cb,
        },
        "sweep": sweep_results,
        "elapsed_sec": time.time() - t0,
    }
    out_path = OUT_DIR / "step6_regression_report.json"
    out_path.write_text(json.dumps(full, indent=2, default=str))
    print()
    print(f"  saved: {out_path}")
    print(f"  total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
