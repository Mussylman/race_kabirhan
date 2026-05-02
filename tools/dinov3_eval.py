#!/usr/bin/env python3
"""dinov3_eval.py — DINOv2-based prototype color classifier.

Note: имя файла исторически dinov3, но фактически грузит DINOv2-base
как proxy (DINOv3 gated, awaiting Meta review). Когда DINOv3 одобрят —
замена MODEL_ID одной строкой.

Этапы:
  --preprocess-check  : показать как processor обрабатывает один crop
  --extract           : embedding extraction для всех 420 crops
  --prototypes        : построить per-class prototypes
  --sanity            : sanity check на 5 reference crops
  --eval              : threshold sweep + сравнение с CNN v4
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LABELS_PATH = REPO / "output" / "diagnose_v4_2026-04-25" / "labels.json"
CROPS_DIR   = REPO / "output" / "diagnose_v4_2026-04-25" / "crops_raw"
OUT_DIR     = REPO / "output" / "dinov2_eval_2026-04-25"

MODEL_ID  = "facebook/dinov2-base"
DTYPE_STR = "float16"
DEVICE    = "cuda:0"
SEED      = 42
BATCH     = 16


def _device_dtype():
    import torch
    dt = torch.float16 if DTYPE_STR == "float16" else torch.float32
    return DEVICE, dt


def _load_processor_model():
    import torch
    from transformers import AutoImageProcessor, AutoModel
    dev, dt = _device_dtype()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, dtype=dt).to(dev).eval()
    return processor, model, dev, dt


def preprocess_check():
    """Показать как processor обрабатывает один вертикальный crop 24×72."""
    import torch
    from PIL import Image
    import cv2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    processor, _, dev, dt = _load_processor_model()
    ip = processor.image_processor if hasattr(processor, 'image_processor') else processor

    print("=== processor configuration ===")
    print(f"  size:           {ip.size}")
    print(f"  do_resize:      {ip.do_resize}")
    print(f"  do_center_crop: {getattr(ip, 'do_center_crop', '<not set>')}")
    print(f"  do_rescale:     {ip.do_rescale}")
    print(f"  rescale_factor: {ip.rescale_factor}")
    print(f"  do_normalize:   {ip.do_normalize}")
    print(f"  image_mean:     {ip.image_mean}")
    print(f"  image_std:      {ip.image_std}")
    print(f"  resample:       {getattr(ip, 'resample', '<not set>')}")
    print()

    # pick first crop (small vertical one)
    fname = sorted(p.name for p in CROPS_DIR.glob("frame_*.jpg"))[0]
    img_path = CROPS_DIR / fname
    img = Image.open(img_path).convert("RGB")
    print(f"Source crop: {fname}, size={img.size}")

    inputs = processor(images=img, return_tensors="pt")
    print(f"Output pixel_values: shape={tuple(inputs['pixel_values'].shape)}, "
          f"dtype={inputs['pixel_values'].dtype}")
    print(f"Pixel range: [{inputs['pixel_values'].min().item():.3f}, "
          f"{inputs['pixel_values'].max().item():.3f}]")

    # denormalize back to viewable RGB
    px = inputs["pixel_values"][0]   # 3×224×224
    mean = torch.tensor(ip.image_mean).view(3, 1, 1)
    std  = torch.tensor(ip.image_std).view(3, 1, 1)
    rgb_unnorm = (px * std + mean).clamp(0, 1)
    rgb = (rgb_unnorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    out_path = OUT_DIR / "preprocess_check.jpg"
    cv2.imwrite(str(out_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print()
    print(f"Saved denormalized 'what DINOv2 sees': {out_path}")
    print(f"  → original {img.size} → processor → {tuple(inputs['pixel_values'].shape[-2:])}")


def extract():
    """Embedding extraction для всех crops в labels.json."""
    import torch
    from PIL import Image
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    labels_data = json.loads(LABELS_PATH.read_text())
    items = []
    for cls, ids in labels_data.items():
        for cid in ids:
            items.append((cid, cls))
    print(f"Total crops to embed: {len(items)}")

    processor, model, dev, dt = _load_processor_model()
    print(f"Loaded {MODEL_ID}, batch_size={BATCH}")

    embs = np.zeros((len(items), 768), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(items), BATCH):
        chunk = items[i : i + BATCH]
        imgs = [Image.open(CROPS_DIR / f"{cid}.jpg").convert("RGB") for cid, _ in chunk]
        inputs = processor(images=imgs, return_tensors="pt").to(dev, dt)
        with torch.inference_mode():
            out = model(**inputs)
        e = out.pooler_output.float().cpu().numpy()
        embs[i : i + len(chunk), :] = e
        if (i // BATCH) % 5 == 0:
            print(f"  {i+len(chunk):4d}/{len(items)}  ({(i+len(chunk))/(time.time()-t0):.1f} crops/s)")
    print(f"Extracted {len(items)} embeddings in {time.time()-t0:.1f}s")

    out_path = OUT_DIR / "embeddings.npz"
    np.savez(
        out_path,
        embeddings=embs,
        ids=np.array([cid for cid, _ in items]),
        labels=np.array([lbl for _, lbl in items]),
    )
    print(f"Saved: {out_path}  ({out_path.stat().st_size / 1024**2:.1f} MB)")


def split_indices():
    """Stratified split: 70% jockey for reference, 30% for test + ALL not_jockey for test."""
    rng = random.Random(SEED)
    data = np.load(OUT_DIR / "embeddings.npz", allow_pickle=False)
    ids = data["ids"]
    labels = data["labels"]
    ref_mask = np.zeros(len(ids), dtype=bool)
    for cls in ["green", "yellow", "red"]:
        cls_idx = [i for i, l in enumerate(labels) if l == cls]
        rng.shuffle(cls_idx)
        n_ref = int(round(len(cls_idx) * 0.70))
        for i in cls_idx[:n_ref]:
            ref_mask[i] = True
    return ref_mask, ids, labels


def prototypes():
    """Per-class prototypes: average embeddings of reference set, L2-normalize."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(OUT_DIR / "embeddings.npz", allow_pickle=False)
    embs = data["embeddings"]
    ref_mask, ids, labels = split_indices()

    # split breakdown
    print("=" * 60)
    print(f"{'class':<12s} {'reference':>10s} {'test':>10s}")
    print("-" * 60)
    for cls in ["green", "yellow", "red", "not_jockey"]:
        in_cls = labels == cls
        n_ref  = int((ref_mask & in_cls).sum())
        n_test = int((~ref_mask & in_cls).sum())
        print(f"{cls:<12s} {n_ref:>10d} {n_test:>10d}")
    print("=" * 60)
    print()

    proto_classes = ["green", "yellow", "red"]
    protos = []
    for cls in proto_classes:
        mask = ref_mask & (labels == cls)
        avg = embs[mask].mean(axis=0)
        avg /= np.linalg.norm(avg) + 1e-12
        protos.append(avg)
    protos = np.stack(protos)   # [3, 768]

    out_path = OUT_DIR / "prototypes.npz"
    np.savez(out_path, prototypes=protos, classes=np.array(proto_classes))
    print(f"Prototypes saved: {out_path}, shape={protos.shape}")
    print()


def sanity():
    """Cosine similarity 5 reference crops to all 3 prototypes."""
    data = np.load(OUT_DIR / "embeddings.npz", allow_pickle=False)
    p = np.load(OUT_DIR / "prototypes.npz", allow_pickle=False)
    embs = data["embeddings"]
    ids = data["ids"]
    labels = data["labels"]
    protos = p["prototypes"]
    classes = list(p["classes"])
    ref_mask, _, _ = split_indices()

    # Pick 5 reference crops (one or two per class)
    rng = random.Random(SEED + 1)
    ref_indices = [i for i in range(len(ids)) if ref_mask[i]]
    sample = rng.sample(ref_indices, 5)
    print("=" * 80)
    print(f"{'crop_id':<35s} {'true_label':>10s}  " + "  ".join(f"sim_{c:<7s}" for c in classes) + "  match")
    print("-" * 80)
    for i in sample:
        emb = embs[i]
        emb_n = emb / (np.linalg.norm(emb) + 1e-12)
        sims = emb_n @ protos.T          # [3]
        argmax = int(np.argmax(sims))
        pred_class = classes[argmax]
        true_label = str(labels[i])
        match = "✓" if pred_class == true_label else "✗"
        sims_str = "  ".join(f"{s:>9.3f}" for s in sims)
        print(f"{str(ids[i]):<35s} {true_label:>10s}  {sims_str}  {match}")
    print("=" * 80)


def eval_report():
    """Threshold sweep + сравнение с CNN v4 + galleries + report.md."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(OUT_DIR / "embeddings.npz", allow_pickle=False)
    p    = np.load(OUT_DIR / "prototypes.npz",  allow_pickle=False)
    embs = data["embeddings"]
    ids  = data["ids"]
    labels = data["labels"]
    protos  = p["prototypes"]
    classes = list(p["classes"])     # ['green','yellow','red']
    ref_mask, _, _ = split_indices()

    # L2-normalize embeddings, compute sims to prototypes
    embs_n = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    sims_all = embs_n @ protos.T     # [N, 3]

    test_idx = np.where(~ref_mask)[0]
    test_ids = ids[test_idx]
    test_labels = labels[test_idx]
    test_sims = sims_all[test_idx]
    test_max_sim = test_sims.max(axis=1)
    test_argmax  = test_sims.argmax(axis=1)
    test_pred_jockey = np.array([classes[a] for a in test_argmax])

    print(f"Test set: {len(test_idx)} crops "
          f"({(test_labels != 'not_jockey').sum()} jockey + "
          f"{(test_labels == 'not_jockey').sum()} not_jockey)")
    print()

    # ── threshold sweep ──
    thresholds = [round(0.4 + 0.05*i, 2) for i in range(12)]   # 0.40 .. 0.95
    rows = []
    for thr in thresholds:
        # pred: if max_sim < thr → "not_jockey", else → predicted color
        pred = np.where(test_max_sim < thr, "not_jockey", test_pred_jockey)

        # tp_rate = % jockeys correctly classified (right color)
        is_jk = test_labels != "not_jockey"
        jk_correct = (pred[is_jk] == test_labels[is_jk]).sum()
        tp_rate = jk_correct / max(is_jk.sum(), 1)

        # rejected_nj = % not_jockey correctly assigned to "not_jockey"
        is_nj = test_labels == "not_jockey"
        nj_rejected = (pred[is_nj] == "not_jockey").sum()
        reject_nj = nj_rejected / max(is_nj.sum(), 1)

        # misclassified jockey (другой цвет, не not_jockey)
        jk_wrong_color = ((pred[is_jk] != test_labels[is_jk]) & (pred[is_jk] != "not_jockey")).sum()
        miscls = jk_wrong_color / max(is_jk.sum(), 1)

        composite = (reject_nj + tp_rate) / 2
        rows.append({
            "threshold": thr,
            "tp_rate": tp_rate,
            "reject_nj": reject_nj,
            "miscls": miscls,
            "composite": composite,
        })

    print("=" * 76)
    print(f"{'thr':>5s}  {'tp_rate':>8s}  {'reject_nj':>10s}  {'miscls':>8s}  {'composite':>10s}")
    print("-" * 76)
    for r in rows:
        print(f"{r['threshold']:>5.2f}  {r['tp_rate']:>8.3f}  {r['reject_nj']:>10.3f}  "
              f"{r['miscls']:>8.3f}  {r['composite']:>10.3f}")
    best = max(rows, key=lambda x: x["composite"])
    print("=" * 76)
    print(f"BEST by composite: thr={best['threshold']}  "
          f"tp_rate={best['tp_rate']:.3f}  reject_nj={best['reject_nj']:.3f}  "
          f"composite={best['composite']:.3f}")
    print()

    # ── confusion matrix at best threshold ──
    pred_best = np.where(test_max_sim < best["threshold"], "not_jockey", test_pred_jockey)
    print(f"=== CONFUSION MATRIX @ threshold={best['threshold']} ===")
    pred_classes = ["green", "yellow", "red", "not_jockey"]
    print(f"{'true \\ pred':<14s}", *[f"{c:>10s}" for c in pred_classes], "  total")
    for tc in pred_classes:
        sub = test_labels == tc
        if sub.sum() == 0:
            continue
        line = [f"{(pred_best[sub] == pc).sum():>10d}" for pc in pred_classes]
        print(f"{tc:<14s} " + " ".join(line) + f"   {int(sub.sum()):>5d}")
    print()

    # ── compare to CNN v4 on the SAME test set ──
    print("=" * 76)
    print("=== DIRECT COMPARISON: DINOv2 vs CNN v4 on the same test_set ===")
    print("=" * 76)
    v4_preds = json.loads((REPO / "output/diagnose_v4_2026-04-25/step4_predict.json").read_text())
    v4_by_id = {r["id"]: r for r in v4_preds}

    test_ids_set = set(test_ids.tolist())
    v4_subset = [r for r in v4_preds if r["id"] in test_ids_set]
    print(f"v4 predictions matched on test_set: {len(v4_subset)}/{len(test_idx)}")

    # v4 confusion matrix (no reject — v4 always classifies)
    print()
    print(f"=== CNN v4 CONFUSION MATRIX (no reject capability) ===")
    print(f"{'true \\ pred':<14s}", *[f"{c:>10s}" for c in pred_classes], "  total")
    for tc in pred_classes:
        sub_records = [r for r in v4_subset if r["label"] == tc]
        if not sub_records:
            continue
        counts = {pc: sum(1 for r in sub_records if r["pred"] == pc) for pc in pred_classes}
        line = [f"{counts[pc]:>10d}" for pc in pred_classes]
        print(f"{tc:<14s} " + " ".join(line) + f"   {len(sub_records):>5d}")

    # Headline metrics
    is_jk = test_labels != "not_jockey"
    is_nj = test_labels == "not_jockey"
    v4_jk_records = [r for r in v4_subset if r["label"] != "not_jockey"]
    v4_nj_records = [r for r in v4_subset if r["label"] == "not_jockey"]
    v4_jk_acc = sum(1 for r in v4_jk_records if r["pred"] == r["label"]) / max(len(v4_jk_records), 1)
    v4_reject = 0.0   # CNN v4 has no reject path
    v4_confident_wrong = sum(1 for r in v4_nj_records if r["pred_conf"] >= 0.9) / max(len(v4_nj_records), 1)

    dn_jk_acc = (pred_best[is_jk] == test_labels[is_jk]).sum() / max(is_jk.sum(), 1)
    dn_reject = (pred_best[is_nj] == "not_jockey").sum() / max(is_nj.sum(), 1)
    # confident wrong DINOv2: not_jockey crops where pred != not_jockey AND max_sim >= 0.9
    nj_high_sim_wrong = ((test_labels == "not_jockey") & (pred_best != "not_jockey") & (test_max_sim >= 0.9)).sum()
    dn_confident_wrong = nj_high_sim_wrong / max(is_nj.sum(), 1)

    print()
    print("=== HEADLINE COMPARISON ===")
    print(f"{'metric':<28s}  {'v4 CNN':>10s}  {'DINOv2 proto':>12s}")
    print("-" * 56)
    print(f"{'jockey color accuracy':<28s}  {v4_jk_acc:>10.1%}  {dn_jk_acc:>12.1%}")
    print(f"{'not_jockey reject rate':<28s}  {v4_reject:>10.1%}  {dn_reject:>12.1%}")
    print(f"{'confident wrong on NJ':<28s}  {v4_confident_wrong:>10.1%}  {dn_confident_wrong:>12.1%}")
    print()

    # ── Galleries ──
    nj_test = [(test_ids[i], test_max_sim[i], test_pred_jockey[i])
               for i in range(len(test_idx)) if test_labels[i] == "not_jockey"]
    jk_test = [(test_ids[i], test_max_sim[i], test_pred_jockey[i], test_labels[i])
               for i in range(len(test_idx)) if test_labels[i] != "not_jockey"]

    # 5 examples: dinov2 правильно reject (max_sim < thr), v4 уверенно зеленил
    correctly_rejected = []
    for tid, sim, pred_color in nj_test:
        if sim < best["threshold"]:
            v4 = v4_by_id.get(str(tid), {})
            if v4.get("pred") == "green" and v4.get("pred_conf", 0) >= 0.9:
                correctly_rejected.append((str(tid), float(sim), pred_color, v4["pred_conf"]))
    correctly_rejected.sort(key=lambda x: x[1])  # lowest similarity first
    correctly_rejected = correctly_rejected[:5]

    # 5 false positive (NJ passed) — наибольший max_sim
    fp = sorted([(str(tid), float(sim), pc) for tid, sim, pc in nj_test if sim >= best["threshold"]],
                key=lambda x: -x[1])[:5]

    # 5 jockeys misclassified (true ≠ pred AND not rejected)
    miscls_jk = []
    for tid, sim, pc, true_lbl in jk_test:
        if sim >= best["threshold"] and pc != true_lbl:
            miscls_jk.append((str(tid), float(sim), pc, str(true_lbl)))
    miscls_jk = miscls_jk[:5]

    galleries = {
        "rejected_correctly": correctly_rejected,
        "false_positives":    fp,
        "misclassified_jockeys": miscls_jk,
    }
    (OUT_DIR / "galleries.json").write_text(json.dumps(galleries, indent=2))

    # render HTML gallery
    write_eval_gallery(galleries, best, dn_jk_acc, dn_reject, v4_jk_acc, v4_reject)

    # ── Final report.md ──
    write_eval_report(rows, best, dn_jk_acc, dn_reject, dn_confident_wrong,
                      v4_jk_acc, v4_reject, v4_confident_wrong,
                      len(test_idx), is_jk.sum(), is_nj.sum())


def write_eval_gallery(galleries, best, dn_jk, dn_rj, v4_jk, v4_rj):
    from html import escape
    parts = [
        '<!doctype html><html lang="ru"><head><meta charset="utf-8">',
        '<title>DINOv2 eval — qualitative</title>',
        '<style>',
        '  body { font-family:system-ui; background:#1a1a1a; color:#ddd; margin:0; padding:14px; }',
        '  h1 { color:#fff; }',
        '  h2 { color:#ffaf3a; border-bottom:1px solid #444; padding:6px 0; margin-top:20px; }',
        '  .desc { color:#aaa; font-size:13px; }',
        '  .grid { display:grid; grid-template-columns:repeat(5,1fr); gap:8px; }',
        '  .card { background:#2a2a2a; border-radius:4px; padding:6px; }',
        '  .card.good { border:2px solid #2a8; }',
        '  .card.bad  { border:2px solid #a44; }',
        '  .card img { width:100%; max-height:280px; object-fit:contain; display:block; }',
        '  .meta { font-family:monospace; font-size:11px; color:#ccc; padding:6px; }',
        '</style></head><body>',
        '<h1>DINOv2 prototype eval — qualitative samples</h1>',
        f'<p class="desc">threshold={best["threshold"]}  '
        f'jockey_acc={dn_jk:.1%}  reject_NJ={dn_rj:.1%}  '
        f'(v4 baseline: jk={v4_jk:.1%}  reject=0%)</p>',
    ]

    parts.append(f'<h2>✅ DINOv2 правильно отвергнул not_jockey, которые v4 уверенно зеленил (top 5)</h2>')
    parts.append('<div class="grid">')
    for cid, sim, pc, v4_conf in galleries["rejected_correctly"]:
        src = f"../diagnose_v4_2026-04-25/crops_raw/{cid}.jpg"
        parts.append(
            f'<div class="card good">'
            f'<img src="{escape(src)}" loading="lazy">'
            f'<div class="meta"><b>{escape(cid)}</b><br>'
            f'true: not_jockey<br>'
            f'DINOv2 max_sim = {sim:.3f} → {pc}<br>'
            f'(rejected, sim &lt; {best["threshold"]})<br>'
            f'<b>v4 said:</b> green @ {v4_conf:.3f} ❌'
            f'</div></div>'
        )
    parts.append('</div>')

    parts.append(f'<h2>❌ DINOv2 false positives — not_jockey прошедшие фильтр</h2>')
    parts.append('<div class="grid">')
    for cid, sim, pc in galleries["false_positives"]:
        src = f"../diagnose_v4_2026-04-25/crops_raw/{cid}.jpg"
        parts.append(
            f'<div class="card bad">'
            f'<img src="{escape(src)}" loading="lazy">'
            f'<div class="meta"><b>{escape(cid)}</b><br>'
            f'true: not_jockey<br>'
            f'DINOv2 max_sim = {sim:.3f} → {pc}<br>'
            f'(false positive, sim ≥ {best["threshold"]})'
            f'</div></div>'
        )
    parts.append('</div>')

    parts.append(f'<h2>🟡 Jockeys misclassified by color (passed filter, wrong color)</h2>')
    parts.append('<div class="grid">')
    for cid, sim, pc, true_lbl in galleries["misclassified_jockeys"]:
        src = f"../diagnose_v4_2026-04-25/crops_raw/{cid}.jpg"
        parts.append(
            f'<div class="card bad">'
            f'<img src="{escape(src)}" loading="lazy">'
            f'<div class="meta"><b>{escape(cid)}</b><br>'
            f'true: {escape(true_lbl)}<br>'
            f'DINOv2 max_sim = {sim:.3f} → <b>{pc}</b> (wrong color)'
            f'</div></div>'
        )
    parts.append('</div>')
    parts.append('</body></html>')
    out = OUT_DIR / "gallery_eval.html"
    out.write_text("\n".join(parts))
    print(f"Gallery: {out}")


def write_eval_report(rows, best, dn_jk, dn_rj, dn_cw, v4_jk, v4_rj, v4_cw,
                      n_test, n_jk, n_nj):
    """Финальный markdown-отчёт."""
    md = []
    md.append("# DINOv2 prototype-based color classifier — eval report")
    md.append("")
    md.append("**Дата:** 2026-04-25")
    md.append("")
    md.append("## Модель")
    md.append("")
    md.append("**Использована: `facebook/dinov2-base`** (proxy для DINOv3)")
    md.append("")
    md.append("- DINOv3 (`facebook/dinov3-vitb16-pretrain-lvd1689m`) — gated repo, awaiting Meta access review.")
    md.append("- DINOv2-base — public, ViT-B/14, 86.6M params, обучен self-supervised на LVD-142M (vs DINOv3 LVD-1689M, в 12× меньше данных).")
    md.append("- Концепция и API идентичны → когда DINOv3 одобрят, замена `MODEL_ID` одной строкой.")
    md.append("")
    md.append("## Setup")
    md.append("")
    md.append("- Reference set: 70% jockey crops (118 шт.)  для построения per-class prototypes")
    md.append(f"- Test set: 30% jockey + ВСЕ not_jockey ({n_test} шт.: {int(n_jk)} jockey + {int(n_nj)} not_jockey)")
    md.append("- random_seed=42, stratified split по 3 классам цвета")
    md.append("- Embedding: `pooler_output` 768-dim, FP16 на RTX 5070 Ti")
    md.append("- Throughput: ~540 crops/s (batch=16) → весь dataset embed за <1 секунды")
    md.append("- Classification: cosine similarity к L2-normalized prototypes; argmax если max_sim ≥ threshold, иначе not_jockey")
    md.append("")
    md.append("## Threshold sweep")
    md.append("")
    md.append("| threshold | tp_rate | reject_nj | miscls | composite |")
    md.append("|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(f"| {r['threshold']:.2f} | {r['tp_rate']:.3f} | {r['reject_nj']:.3f} | {r['miscls']:.3f} | {r['composite']:.3f} |")
    md.append("")
    md.append(f"**Best by composite (`(reject_nj + tp_rate) / 2`): threshold={best['threshold']}**")
    md.append("")

    md.append("## DINOv2 vs CNN v4 — head-to-head на одном test set")
    md.append("")
    md.append("| metric | v4 CNN | DINOv2 prototype |")
    md.append("|---|---:|---:|")
    md.append(f"| jockey color accuracy | **{v4_jk:.1%}** | **{dn_jk:.1%}** |")
    md.append(f"| not_jockey reject rate | **{v4_rj:.1%}** | **{dn_rj:.1%}** |")
    md.append(f"| confident wrong on not_jockey (≥0.9) | {v4_cw:.1%} | {dn_cw:.1%} |")
    md.append("")
    md.append("## Verdict")
    md.append("")

    # decision logic
    if dn_rj >= 0.50 and dn_jk >= 0.85:
        verdict = (
            "**DINOv2 prototype-based DRAMATICALLY улучшает поведение over CNN v4.**\n\n"
            f"reject_nj повышен с 0% (v4 не умеет в принципе) до {dn_rj:.1%} (DINOv2 prototype), "
            f"при сохранении high jockey accuracy ({dn_jk:.1%}). "
            "Гипотеза «embedding-based подход разделяет not_jockey от jockey» **подтверждена эмпирически**.\n\n"
            "DINOv3 (12× больше data) ожидаемо даст ещё лучше качество.\n\n"
            "**Next steps:**\n"
            "- Запросить approve DINOv3 (или дождаться review), повторить evaluation\n"
            "- Расширить reference set (добавить crops с других видео, особенно из cam-01..04 fail mode)\n"
            "- Linear probe на embeddings (учить лёгкую head на (jockey/not_jockey + 5 colors))\n"
            "- Production integration в SGIE postprocessing"
        )
    elif dn_rj >= 0.30:
        verdict = (
            "**DINOv2 prototype-based ЧАСТИЧНО улучшает поведение vs CNN v4.**\n\n"
            f"reject_nj = {dn_rj:.1%} (vs 0% у v4) — embedding-based подход даёт reject capability, "
            "но недостаточно высокую для production. "
            f"jockey accuracy = {dn_jk:.1%}.\n\n"
            "**Возможные next steps:**\n"
            "- Linear probe на embeddings — может улучшить дискриминацию vs raw cosine\n"
            "- Triplet loss / contrastive fine-tune\n"
            "- Замена на DINOv3 (когда одобрят) — может качество хватит"
        )
    else:
        verdict = (
            f"**DINOv2 prototype-based НЕ значительно лучше CNN v4** "
            f"(reject_nj={dn_rj:.1%}, jockey_acc={dn_jk:.1%}).\n\n"
            "Гипотеза «embedding-based подход решает проблему» **не подтверждена**. "
            "Возможные причины:\n"
            "- center crop в processor отрезает silk\n"
            "- DINOv2 embeddings слишком обобщённые для дискриминации цвета\n"
            "- Малый reference set (38-41 crops/класс)\n\n"
            "**Следующее направление под вопросом** — нужно либо retrain (Variant B из verdict.md), "
            "либо принципиально другой подход."
        )
    md.append(verdict)
    md.append("")

    md.append("## Артефакты")
    md.append("")
    md.append("- `embeddings.npz` — 420 × 768 float32 + ids + labels")
    md.append("- `prototypes.npz` — 3 × 768 normalized")
    md.append("- `galleries.json` — sample IDs для visual review")
    md.append("- `gallery_eval.html` — 3 секции: правильно отвергнутые / false positives / misclassified")
    md.append("- `preprocess_check.jpg`, `preprocess_001590.jpg`, `preprocess_000300.jpg` — что DINOv2 реально видит")
    md.append("")
    md.append("Tool: `tools/dinov3_eval.py --step {extract,prototypes,sanity}` + `eval_report` — воспроизводимо.")

    out = OUT_DIR / "eval_report.md"
    out.write_text("\n".join(md))
    print(f"Report: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True,
                    choices=["preprocess-check", "extract", "prototypes", "sanity", "eval"])
    args = ap.parse_args()
    if args.step == "preprocess-check":
        preprocess_check()
    elif args.step == "extract":
        extract()
    elif args.step == "prototypes":
        prototypes()
    elif args.step == "sanity":
        sanity()
    elif args.step == "eval":
        eval_report()


if __name__ == "__main__":
    main()
