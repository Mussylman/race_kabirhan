#!/usr/bin/env python3
"""diagnose_v2_combined_failures.py — 3 диагностических вопроса
до решения о Phase 7 / альтернативах.

Q1: Сколько cam-13 crops в combined training set? Per-class breakdown.
Q2: Где теряются жокеи на cam-13 test (170 crops)? Reject vs wrong-color.
    Top-10 worst cases с max_sim/second_sim/winning_proto + gallery.
Q3: Inherent cross-camera limitation? Leave-cam-out evaluation:
    re-build prototypes WITHOUT cam-01..05, test на тех же cam-01..05.
"""
from __future__ import annotations
import json, re, sys, time
from collections import defaultdict, Counter
from html import escape
from pathlib import Path

import cv2
import numpy as np

REPO  = Path(__file__).resolve().parent.parent
WORK  = Path("/home/ipodrom/race_vision_bench/dinov2_prod/multicam_collection")
V2    = Path("/home/ipodrom/race_vision_bench/dinov2_prod/v2_build")
OUT   = V2 / "diagnose_failures"
OUT.mkdir(parents=True, exist_ok=True)

DIAG  = REPO / "output" / "diagnose_v4_2026-04-25"

CLASS_ORDER = ["green", "yellow", "red", "blue"]


def parse_camera_from_id(cid: str) -> tuple[str, str]:
    """Return (camera, source_family).
    multicam:        135242_kamera24_720p_t055000_det0      → ('cam-24', 'multicam')
    color_old:       colordata_kamera13_163303_f00536_d2    → ('cam-13', 'color_old_session')
    color_exp:       colordata_exp10cam2_f00096_d3          → ('cam-02', 'color_exp10')
    """
    m = re.match(r"^colordata_kamera(\d+)_(\d+)_", cid)
    if m: return f"cam-{m.group(1).zfill(2)}", "color_old_session"
    m = re.match(r"^colordata_(exp\d+)cam(\d+)_", cid)
    if m: return f"cam-{m.group(2).zfill(2)}", f"color_{m.group(1)}"
    m = re.match(r"^\d+_kamera(\d+)_", cid)
    if m: return f"cam-{m.group(1).zfill(2)}", "multicam"
    return ("cam-??", "unknown")


def parser(emb, protos, classes, thr):
    norm = float(np.linalg.norm(emb)) + 1e-12
    sims = (protos @ emb) / norm
    order = np.argsort(-sims)
    best = int(order[0]); ms = float(sims[best])
    second = int(order[1]); ss = float(sims[second])
    if ms < thr: return "unknown", ms, classes[best], ss, classes[second], sims
    return classes[best], ms, classes[best], ss, classes[second], sims


# ────────── Q1 ──────────

def question_1(labels, inv_index):
    print("=" * 84)
    print("Q1: cam-13 crops в combined training set (per source)")
    print("=" * 84)

    # Parse per (color, camera, source_family)
    full = defaultdict(int)
    cam13_total = 0
    cam13_by_color = defaultdict(int)
    cam13_by_color_src = defaultdict(int)

    for color in CLASS_ORDER:
        for cid in labels.get(color, []):
            cam, src = parse_camera_from_id(cid)
            full[(color, cam, src)] += 1
            if cam == "cam-13":
                cam13_total += 1
                cam13_by_color[color] += 1
                cam13_by_color_src[(color, src)] += 1

    # Total per color
    print(f"  {'color':<8s} {'cam-13 (color_dataset)':>26s} {'TOTAL_in_class':>18s}")
    for c in CLASS_ORDER:
        n_c13 = cam13_by_color[c]
        n_total = len(labels.get(c, []))
        pct = 100 * n_c13 / max(n_total, 1)
        print(f"  {c:<8s} {n_c13:>26d} {n_total:>18d}  ({pct:>4.1f}% of class)")
    print(f"  {'TOTAL':<8s} {cam13_total:>26d} {sum(len(labels[c]) for c in CLASS_ORDER):>18d}")
    print()

    print("  cam-13 breakdown by (color × source):")
    for (color, src), n in sorted(cam13_by_color_src.items()):
        print(f"    {color:<7s} {src:<22s}: {n}")
    print()

    print("  comparison vs cam-13 TEST SET (golden video):")
    test_labels = json.loads((DIAG / "labels.json").read_text())
    test_counts = {c: len(test_labels.get(c, [])) for c in CLASS_ORDER + ["not_jockey"]}
    print(f"    test set (golden cam-13 video): "
          f"green={test_counts.get('green',0)}  yellow={test_counts.get('yellow',0)}  "
          f"red={test_counts.get('red',0)}  blue={test_counts.get('blue',0)}  "
          f"NJ={test_counts.get('not_jockey',0)}  total={sum(test_counts.values())}")
    print(f"    cam-13 in TRAIN set:            "
          f"green={cam13_by_color.get('green',0):>3d}  "
          f"yellow={cam13_by_color.get('yellow',0):>3d}  "
          f"red={cam13_by_color.get('red',0):>3d}  "
          f"blue={cam13_by_color.get('blue',0):>3d}")

    return cam13_total, cam13_by_color


# ────────── Q2 ──────────

def question_2(emb_data, protos_cb, classes, thr=0.55):
    print()
    print("=" * 84)
    print("Q2: Где именно теряются жокеи на cam-13 test (170 jockey crops)?")
    print("=" * 84)

    # Load cam-13 test items
    test_labels = json.loads((DIAG / "labels.json").read_text())
    crops_idx   = json.loads((DIAG / "crops_index.json").read_text())
    by_id = {e["crop"].replace(".jpg",""): e for e in crops_idx}
    items = []
    for cls, ids in test_labels.items():
        if cls == "not_jockey": continue
        for cid in ids:
            e = by_id.get(cid)
            if not e: continue
            items.append({
                "id": cid, "label": cls,
                "frame": e["frame"], "bbox": e["bbox"],
                "h": e.get("h",0), "w": e.get("w",0),
            })
    print(f"  jockey crops in test: {len(items)}")

    # Re-extract embeddings (we already did this in step6 but inline for self-contained)
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    print("  loading HF model + processor (FP16)...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base",
                                      dtype=torch.float16).to("cuda").eval()
    BATCH = 16
    embs = np.empty((len(items), 768), dtype=np.float32)
    pils = []
    last_path = None; cached = None
    for it in items:
        fp = str(DIAG / "raw_frames" / it["frame"])
        if fp != last_path: cached = cv2.imread(fp); last_path = fp
        x1,y1,x2,y2 = it["bbox"]
        h_img,w_img = cached.shape[:2]
        x1=max(0,int(x1)); y1=max(0,int(y1)); x2=min(w_img,int(x2)); y2=min(h_img,int(y2))
        crop = cached[y1:y2, x1:x2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pils.append(Image.fromarray(rgb))
    for i in range(0, len(items), BATCH):
        batch = pils[i:i+BATCH]
        inputs = processor(images=batch, return_tensors="pt").to("cuda", torch.float16)
        with torch.inference_mode():
            out = model(**inputs)
        embs[i:i+len(batch)] = out.pooler_output.float().cpu().numpy()
    print(f"  embeddings extracted ({embs.shape})")

    # Predict + analyze each
    rejected = []
    wrong = []
    correct = 0
    for i, it in enumerate(items):
        pred, ms, win_name, ss, second_name, sims = parser(
            embs[i], protos_cb, classes, thr)
        rec = {**it, "pred": pred, "max_sim": ms,
               "winning_proto": win_name, "second_sim": ss,
               "second_proto": second_name,
               "all_sims": {classes[k]: float(sims[k]) for k in range(4)}}
        if pred == it["label"]: correct += 1
        elif pred == "unknown": rejected.append(rec)
        else: wrong.append(rec)

    print()
    print(f"  CORRECT:           {correct}/{len(items)}  ({100*correct/len(items):.1f}%)")
    print(f"  REJECTED (unknown): {len(rejected)}/{len(items)}  ({100*len(rejected)/len(items):.1f}%)")
    print(f"  WRONG COLOR:       {len(wrong)}/{len(items)}  ({100*len(wrong)/len(items):.1f}%)")
    print()

    # Patterns within REJECTED: which prototype was the closest (just barely below thr)
    print(f"  REJECTED: distribution of winning_proto (which prototype almost matched)")
    rej_winner = Counter(r["winning_proto"] for r in rejected)
    rej_label  = Counter(r["label"] for r in rejected)
    print(f"    by true label:    {dict(rej_label)}")
    print(f"    by winning proto: {dict(rej_winner)}")
    sims_rejected = [r["max_sim"] for r in rejected]
    if sims_rejected:
        print(f"    max_sim stats: min={min(sims_rejected):.3f} "
              f"median={np.median(sims_rejected):.3f} max={max(sims_rejected):.3f}")
    print()

    # Patterns within WRONG: true → pred
    print(f"  WRONG: true → pred frequency")
    wrong_pairs = Counter((r["label"], r["pred"]) for r in wrong)
    for (true, pred), n in sorted(wrong_pairs.items(), key=lambda x: -x[1]):
        avg_sim = np.mean([r["max_sim"] for r in wrong if r["label"]==true and r["pred"]==pred])
        print(f"    {true:<7s} → {pred:<7s}: {n}  (avg sim={avg_sim:.3f})")
    print()

    # 10 worst rejected (lowest max_sim) for gallery
    worst_rejected = sorted(rejected, key=lambda r: r["max_sim"])[:10]
    # 10 wrong with HIGHEST max_sim (model very confident in wrong color)
    confident_wrong = sorted(wrong, key=lambda r: -r["max_sim"])[:10]

    # Build small gallery
    crops_dir = OUT / "q2_failure_gallery"
    crops_dir.mkdir(exist_ok=True)
    print(f"  saving 20 failure example crops + gallery...")
    for r in worst_rejected + confident_wrong:
        # Re-cut the same crop
        fp = str(DIAG / "raw_frames" / r["frame"])
        img = cv2.imread(fp)
        x1,y1,x2,y2 = r["bbox"]
        h_img,w_img = img.shape[:2]
        x1=max(0,int(x1)); y1=max(0,int(y1)); x2=min(w_img,int(x2)); y2=min(h_img,int(y2))
        crop = img[y1:y2, x1:x2]
        cv2.imwrite(str(crops_dir / f"{r['id']}.jpg"), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    parts = ['<!doctype html><html><head><meta charset="utf-8">',
             '<title>Q2 — failure cases on cam-13</title>',
             '<style>body{font-family:system-ui;background:#1a1a1a;color:#ddd;padding:14px;}',
             'h2{color:#fff;font-size:14px;margin:18px 0 6px 0;padding:6px 10px;background:#333;border-radius:4px;}',
             '.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:18px;}',
             '.cell{background:#2a2a2a;border-radius:4px;padding:6px;}',
             '.cell img{max-width:100%;display:block;margin:0 auto;max-height:280px;}',
             '.meta{font-size:11px;color:#aaa;padding:6px 0;line-height:1.5;}',
             '.true{color:#88dd88;}.pred{color:#ff8888;}',
             '</style></head><body>',
             f'<h1 style="color:#fff">Q2 failure analysis (v2_combined @ thr=0.55)</h1>',
             ]
    for title, group in [("10 worst REJECTED (lowest max_sim — модель не уверена ни в чём)", worst_rejected),
                          ("10 worst WRONG (highest max_sim — модель уверенно ошибается)", confident_wrong)]:
        parts.append(f'<h2>{title}</h2><div class="grid">')
        for r in group:
            sims_str = " · ".join(f"{k}={v:.2f}" for k,v in sorted(r["all_sims"].items(), key=lambda x: -x[1]))
            parts.append(
                f'<div class="cell">'
                f'<img src="q2_failure_gallery/{escape(r["id"])}.jpg">'
                f'<div class="meta">{escape(r["id"])}<br>'
                f'<span class="true">true: {r["label"]}</span> · '
                f'<span class="pred">pred: {r["pred"]} (sim={r["max_sim"]:.3f})</span><br>'
                f'all sims: {sims_str}</div></div>')
        parts.append('</div>')
    parts.append('</body></html>')
    (OUT / "q2_failure_gallery.html").write_text("\n".join(parts))
    print(f"    gallery: {OUT / 'q2_failure_gallery.html'}")

    return rejected, wrong


# ────────── Q3 ──────────

def question_3(emb_data, classes):
    print()
    print("=" * 84)
    print("Q3: Cross-camera generalization (leave-cam-out)")
    print("=" * 84)
    print("  Procedure:")
    print("    1. Hold out all multicam crops from cam-01..05 (excl cam-13, cam-24)")
    print("    2. Rebuild prototypes from remaining (multicam cam-24 + all color_dataset)")
    print("    3. Predict held-out cam-01..05")
    print("    4. Compare with golden cam-13 (39.4%) and v2_combined-as-trained")
    print()

    embs   = emb_data["embeddings"]
    ids    = emb_data["ids"]
    labels_arr = emb_data["labels"]
    sources = emb_data["sources"]

    # Determine camera per item
    cams = np.array([parse_camera_from_id(str(c))[0] for c in ids])
    held_cams = {f"cam-{n:02d}" for n in range(1,6)}     # cam-01..05

    # Held-out indices = multicam from cam-01..05
    holdout_mask = np.array([
        cams[i] in held_cams and sources[i] == "multicam"
        for i in range(len(ids))
    ])
    train_mask = ~holdout_mask
    print(f"  held-out items (multicam cam-01..05): {holdout_mask.sum()}")
    print(f"  training items (rest):                {train_mask.sum()}")
    held_label_counts = Counter(labels_arr[holdout_mask].tolist())
    print(f"  held-out label distribution: {dict(held_label_counts)}")

    # Build prototypes from train_mask
    leave_out_protos = np.zeros((4, 768), dtype=np.float32)
    for ci, cls in enumerate(classes):
        cls_mask = train_mask & (labels_arr == cls)
        if cls_mask.sum() == 0:
            print(f"  WARN: 0 train items for {cls}!"); continue
        m = embs[cls_mask].mean(axis=0)
        leave_out_protos[ci] = m / (np.linalg.norm(m) + 1e-12)

    # Predict held-out
    THR = 0.55
    preds = []
    for i in np.where(holdout_mask)[0]:
        p, ms, *_ = parser(embs[i], leave_out_protos, classes, THR)
        preds.append((str(labels_arr[i]), p, ms, str(cams[i])))
    correct = sum(1 for true, pred, _, _ in preds if pred == true)
    rejected = sum(1 for _, pred, _, _ in preds if pred == "unknown")
    wrong    = sum(1 for true, pred, _, _ in preds if pred != true and pred != "unknown")
    print()
    print(f"  HELD-OUT cam-01..05 (rebuilt prototypes WITHOUT them) @ thr=0.55:")
    print(f"    correct:  {correct}/{len(preds)}  ({100*correct/max(len(preds),1):.1f}%)")
    print(f"    rejected: {rejected}/{len(preds)}  ({100*rejected/max(len(preds),1):.1f}%)")
    print(f"    wrong:    {wrong}/{len(preds)}  ({100*wrong/max(len(preds),1):.1f}%)")
    print()

    # Per-camera breakdown
    by_cam = defaultdict(lambda: {"n":0, "correct":0, "rej":0, "wrong":0})
    for true, pred, ms, cam in preds:
        by_cam[cam]["n"] += 1
        if pred == true: by_cam[cam]["correct"] += 1
        elif pred == "unknown": by_cam[cam]["rej"] += 1
        else: by_cam[cam]["wrong"] += 1
    print(f"  per camera:")
    print(f"    {'cam':<7s} {'n':>4s} {'correct':>9s} {'reject':>7s} {'wrong':>6s} {'acc%':>6s}")
    for cam in sorted(by_cam.keys()):
        s = by_cam[cam]
        print(f"    {cam:<7s} {s['n']:>4d} {s['correct']:>9d} {s['rej']:>7d} "
              f"{s['wrong']:>6d} {100*s['correct']/max(s['n'],1):>5.1f}%")
    print()

    # Per-color breakdown
    by_color = defaultdict(lambda: {"n":0, "correct":0, "rej":0, "wrong":0})
    for true, pred, _, _ in preds:
        by_color[true]["n"] += 1
        if pred == true: by_color[true]["correct"] += 1
        elif pred == "unknown": by_color[true]["rej"] += 1
        else: by_color[true]["wrong"] += 1
    print(f"  per color:")
    print(f"    {'color':<8s} {'n':>4s} {'correct':>9s} {'reject':>7s} {'wrong':>6s} {'acc%':>6s}")
    for cls in CLASS_ORDER:
        s = by_color[cls]
        print(f"    {cls:<8s} {s['n']:>4d} {s['correct']:>9d} {s['rej']:>7d} "
              f"{s['wrong']:>6d} {100*s['correct']/max(s['n'],1):>5.1f}%")
    print()

    overall = correct / max(len(preds), 1)
    cam13_baseline = 0.394   # from step 6
    print(f"  ─── COMPARISON ───")
    print(f"    cam-13 (in-test, NOT in train):     39.4% jockey acc")
    print(f"    cam-01..05 (in-test, NOT in train): {100*overall:.1f}% jockey acc")
    print()
    if overall >= 0.65:
        print(f"  → Generalization works on cam-01..05 ({100*overall:.0f}%). "
              f"cam-13 is HARD CASE (warm-shift lighting).")
    elif overall >= 0.4:
        print(f"  → Mediocre generalization. Both cam-13 and cam-01..05 fail similarly.")
    else:
        print(f"  → System failure: prototypes don't generalize cross-camera at all.")

    return overall


def main():
    t0 = time.time()
    labels = json.loads((WORK / "labels.json").read_text())
    inv_index = {e["id"]: e for e in json.loads((WORK / "crops_index.json").read_text())}

    # Q1
    cam13_total, cam13_by_color = question_1(labels, inv_index)

    # Load v2_combined prototypes + embeddings for Q2/Q3
    proto_data = np.load(V2 / "prototypes_v2_combined.npz", allow_pickle=False)
    protos_cb = proto_data["prototypes"].astype(np.float32)
    classes   = [str(c) for c in proto_data["classes"]]
    emb_data = np.load(V2 / "embeddings_all.npz", allow_pickle=False)

    # Q2
    rejected, wrong = question_2(emb_data, protos_cb, classes)

    # Q3
    held_acc = question_3({
        "embeddings": emb_data["embeddings"],
        "ids":        emb_data["ids"],
        "labels":     emb_data["labels"],
        "sources":    emb_data["sources"],
    }, classes)

    # Save report
    summary = {
        "q1_cam13_in_train": {
            "total": int(cam13_total),
            "by_color": {k: int(v) for k, v in cam13_by_color.items()},
        },
        "q2_failures_on_cam13": {
            "rejected": len(rejected),
            "wrong":    len(wrong),
        },
        "q3_held_out_cam01_05_acc": float(held_acc),
        "elapsed_sec": time.time() - t0,
    }
    (OUT / "diagnose_summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print(f"  saved summary: {OUT / 'diagnose_summary.json'}")
    print(f"  total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
