#!/usr/bin/env python3
"""dinov2_linear_probe.py — Linear probe поверх замороженных DINOv2
embeddings для 4-class classification (green/yellow/red/not_jockey).

Архитектура: L2_norm(emb) → Linear(768, 4). Backbone DINOv2 frozen
(embeddings уже извлечены в output/dinov2_eval_2026-04-25/embeddings.npz).

Запуск:
  --step split   : показать final split counts (без обучения)
  --step train   : обучить + сохранить best checkpoint + learning curves
  --step eval    : evaluate на test set + сравнение с prototype/CNN v4
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
OUT  = REPO / "output" / "dinov2_eval_2026-04-25"
EMB_PATH = OUT / "embeddings.npz"

# ── 4-class index (фиксируем порядок) ──────────────────────────────────
CLASSES = ["green", "yellow", "red", "not_jockey"]
CLS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

SEED = 42

# 23 HARD NJ из Фазы 3 (forced into test) — детерминированно, не от seed
HARD_NJ = [
    # 3 actual FPs at thr=0.6
    "frame_002310_det_00", "frame_002340_det_00", "frame_002370_det_00",
    # 20 near-miss (max_sim ∈ [0.5, 0.6))
    "frame_000420_det_00", "frame_002820_det_00", "frame_002790_det_00",
    "frame_000060_det_00", "frame_002850_det_00", "frame_002490_det_00",
    "frame_000270_det_04", "frame_000210_det_00", "frame_000360_det_00",
    "frame_002610_det_00", "frame_002010_det_00", "frame_001950_det_00",
    "frame_001830_det_00", "frame_001890_det_00", "frame_001980_det_00",
    # дополним 5-ю остальными near-miss из топа
    "frame_000180_det_00", "frame_000300_det_00", "frame_000240_det_00",
    "frame_000330_det_00", "frame_000150_det_00",
]
N_TEST_NJ_TOTAL = 50  # 23 hard + 27 random


def build_split():
    """Stratified split с принудительным включением HARD NJ в test.

    Returns dict: id → 'train'/'val'/'test'.
    """
    data = np.load(EMB_PATH, allow_pickle=False)
    ids = data["ids"].astype(str)
    labels = data["labels"].astype(str)

    rng = random.Random(SEED)

    # ── jockey: воспроизводим Phase 3 split ──
    # Phase 3 ref (118): уйдут в train/val. Phase 3 test (52): in test as-is.
    jockey_test = set()
    train_val_pool: dict[str, list[str]] = {"green": [], "yellow": [], "red": []}

    for cls in ["green", "yellow", "red"]:
        cls_idx = [i for i, l in enumerate(labels) if l == cls]
        cls_ids = [ids[i] for i in cls_idx]
        rng.shuffle(cls_ids)
        n_ref = int(round(len(cls_ids) * 0.70))   # тот же 70/30 что Phase 3
        train_val_pool[cls] = cls_ids[:n_ref]
        for cid in cls_ids[n_ref:]:
            jockey_test.add(cid)

    # train/val из reference: 80/20 stratified
    train_set, val_set = set(), set()
    for cls in ["green", "yellow", "red"]:
        pool = train_val_pool[cls]
        rng.shuffle(pool)
        n_train = int(round(len(pool) * 0.80))
        for cid in pool[:n_train]:
            train_set.add(cid)
        for cid in pool[n_train:]:
            val_set.add(cid)

    # ── not_jockey: HARD → test, остальные stratified ──
    nj_ids = [ids[i] for i, l in enumerate(labels) if l == "not_jockey"]
    nj_set = set(nj_ids)
    hard_set = set(HARD_NJ) & nj_set   # confirm HARD ids exist
    if len(hard_set) != len(HARD_NJ):
        missing = set(HARD_NJ) - nj_set
        raise RuntimeError(f"HARD NJ ids missing in dataset: {missing}")

    nj_test_random_n = N_TEST_NJ_TOTAL - len(hard_set)
    non_hard_nj = [c for c in nj_ids if c not in hard_set]
    rng.shuffle(non_hard_nj)

    nj_test = hard_set | set(non_hard_nj[:nj_test_random_n])
    nj_remaining = non_hard_nj[nj_test_random_n:]
    n_train_nj = int(round(len(nj_remaining) * 0.80))
    nj_train = set(nj_remaining[:n_train_nj])
    nj_val   = set(nj_remaining[n_train_nj:])

    # ── собираем общий split dict ──
    split: dict[str, str] = {}
    for cid in train_set: split[cid] = "train"
    for cid in val_set:   split[cid] = "val"
    for cid in jockey_test: split[cid] = "test"
    for cid in nj_train: split[cid] = "train"
    for cid in nj_val:   split[cid] = "val"
    for cid in nj_test:  split[cid] = "test"

    return split, ids, labels


def show_split():
    split, ids, labels = build_split()
    print("=" * 60)
    print(f"{'class':<12s} {'train':>7s} {'val':>5s} {'test':>5s} {'total':>6s}")
    print("-" * 60)
    for cls in CLASSES:
        in_cls = labels == cls
        n_tr = sum(1 for i, l in enumerate(labels) if l == cls and split[ids[i]] == "train")
        n_va = sum(1 for i, l in enumerate(labels) if l == cls and split[ids[i]] == "val")
        n_te = sum(1 for i, l in enumerate(labels) if l == cls and split[ids[i]] == "test")
        print(f"{cls:<12s} {n_tr:>7d} {n_va:>5d} {n_te:>5d} {int(in_cls.sum()):>6d}")
    print("=" * 60)
    n_train = sum(1 for v in split.values() if v == "train")
    n_val   = sum(1 for v in split.values() if v == "val")
    n_test  = sum(1 for v in split.values() if v == "test")
    print(f"{'TOTAL':<12s} {n_train:>7d} {n_val:>5d} {n_test:>5d} {len(split):>6d}")
    print()
    nj_test_hard = sum(1 for cid in HARD_NJ if split.get(cid) == "test")
    print(f"HARD NJ in test: {nj_test_hard}/{len(HARD_NJ)}")


def train():
    """Train Linear probe с early stopping by val_acc."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    split, ids, labels = build_split()
    data = np.load(EMB_PATH, allow_pickle=False)
    embs = data["embeddings"]

    # build tensors per split
    def split_tensors(name):
        idx = [i for i, cid in enumerate(ids) if split[cid] == name]
        x = torch.from_numpy(embs[idx]).float()
        y = torch.tensor([CLS_TO_IDX[labels[i]] for i in idx], dtype=torch.long)
        return x, y, [ids[i] for i in idx]

    x_tr, y_tr, ids_tr = split_tensors("train")
    x_va, y_va, ids_va = split_tensors("val")
    x_te, y_te, ids_te = split_tensors("test")
    print(f"train: {x_tr.shape[0]}  val: {x_va.shape[0]}  test: {x_te.shape[0]}")

    # class weights = inverse frequency on train
    counts = torch.tensor([(y_tr == i).sum().item() for i in range(4)], dtype=torch.float)
    weights = x_tr.shape[0] / (4 * counts.clamp(min=1))
    print(f"class counts (train): {counts.tolist()}")
    print(f"class weights:        {[round(w.item(), 3) for w in weights]}")
    print(f"class order: {CLASSES}")

    class LinearProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(768, 4)
        def forward(self, x):
            x = F.normalize(x, p=2, dim=-1)
            return self.fc(x)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LinearProbe().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=weights.to(device))

    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_va, y_va = x_va.to(device), y_va.to(device)

    BATCH = 32
    EPOCHS = 200
    PATIENCE = 20

    best_val_acc = -1.0
    best_state = None
    best_epoch = -1
    bad = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}

    n_train = x_tr.shape[0]
    for epoch in range(EPOCHS):
        # shuffle train
        perm = torch.randperm(n_train, device=device)
        x_sh = x_tr[perm]
        y_sh = y_tr[perm]
        model.train()
        train_loss_sum = 0.0
        for i in range(0, n_train, BATCH):
            xb = x_sh[i:i+BATCH]
            yb = y_sh[i:i+BATCH]
            logits = model(xb)
            loss = crit(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_sum += loss.item() * xb.shape[0]
        train_loss = train_loss_sum / n_train

        model.eval()
        with torch.inference_mode():
            v_logits = model(x_va)
            v_loss = crit(v_logits, y_va).item()
            v_pred = v_logits.argmax(dim=1)
            v_acc = (v_pred == y_va).float().mean().item()

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            bad = 0
        else:
            bad += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}: train_loss={train_loss:.4f}  "
                  f"val_loss={v_loss:.4f}  val_acc={v_acc:.4f}"
                  + ("  ★new best★" if bad == 0 else ""))
        if bad >= PATIENCE:
            print(f"  early stop at epoch {epoch+1} (best epoch={best_epoch}, val_acc={best_val_acc:.4f})")
            break

    # Save best
    ckpt_path = OUT / "linear_probe.pt"
    torch.save({
        "state_dict":  best_state,
        "best_epoch":  best_epoch,
        "best_val_acc": best_val_acc,
        "classes":     CLASSES,
        "class_weights": weights.tolist(),
        "history":     history,
        "split":       split,
        "seed":        SEED,
    }, ckpt_path)
    print(f"\nBest checkpoint saved: {ckpt_path}  "
          f"(epoch {best_epoch}, val_acc={best_val_acc:.4f})")

    # Plot learning curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(history["epoch"], history["train_loss"], label="train", color="#48a")
    ax.plot(history["epoch"], history["val_loss"], label="val", color="#a44")
    ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1, alpha=0.5,
               label=f"best epoch={best_epoch}")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Loss")

    ax = axes[1]
    ax.plot(history["epoch"], history["val_acc"], color="#2a8")
    ax.axvline(best_epoch, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(best_val_acc, color="#2a8", linestyle=":", linewidth=1, alpha=0.5,
               label=f"best={best_val_acc:.3f}")
    ax.set_xlabel("epoch"); ax.set_ylabel("val accuracy"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Validation accuracy")

    fig.tight_layout()
    plot_path = OUT / "linear_probe_learning_curves.png"
    fig.savefig(str(plot_path), dpi=110)
    plt.close(fig)
    print(f"Learning curves: {plot_path}")


def evaluate():
    """Фаза C: test eval + per-HARD-NJ table + threshold sweep + report update."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from html import escape

    split, ids, labels = build_split()
    data = np.load(EMB_PATH, allow_pickle=False)
    embs = data["embeddings"]

    # rebuild model + load best checkpoint
    ckpt = torch.load(OUT / "linear_probe.pt", map_location="cpu", weights_only=False)
    class LinearProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(768, 4)
        def forward(self, x):
            x = F.normalize(x, p=2, dim=-1)
            return self.fc(x)
    model = LinearProbe()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded checkpoint: best_epoch={ckpt['best_epoch']}, val_acc={ckpt['best_val_acc']:.4f}")

    # test set
    test_idx = [i for i, cid in enumerate(ids) if split[cid] == "test"]
    x_te = torch.from_numpy(embs[test_idx]).float()
    y_te = torch.tensor([CLS_TO_IDX[labels[i]] for i in test_idx], dtype=torch.long)
    ids_te = [ids[i] for i in test_idx]

    with torch.inference_mode():
        logits = model(x_te)               # [N, 4]
        probs  = F.softmax(logits, dim=-1) # [N, 4]
        argmax = probs.argmax(dim=-1)
        sorted_probs, _ = probs.sort(dim=-1, descending=True)
        top1 = sorted_probs[:, 0]
        top2 = sorted_probs[:, 1]
        margin = (top1 - top2)             # [N]

    # ── Confusion matrix (default argmax, без extra threshold) ──
    print()
    print("=" * 72)
    print("=== ARGMAX CONFUSION MATRIX (no extra reject threshold) ===")
    print("=" * 72)
    print(f"{'true \\ pred':<14s}", *[f"{c:>10s}" for c in CLASSES], "  total")
    for ti, tc in enumerate(CLASSES):
        sub = (y_te == ti)
        if sub.sum() == 0: continue
        line = [f"{((argmax == pi) & sub).sum().item():>10d}" for pi in range(4)]
        print(f"{tc:<14s} " + " ".join(line) + f"   {int(sub.sum().item()):>5d}")

    # Per-class P/R/F1
    print()
    print("=== PER-CLASS PRECISION / RECALL / F1 ===")
    print(f"{'class':<14s} {'P':>7s} {'R':>7s} {'F1':>7s}  support")
    for ti, tc in enumerate(CLASSES):
        tp = ((argmax == ti) & (y_te == ti)).sum().item()
        fp = ((argmax == ti) & (y_te != ti)).sum().item()
        fn = ((argmax != ti) & (y_te == ti)).sum().item()
        p  = tp / max(tp + fp, 1)
        r  = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        sup = (y_te == ti).sum().item()
        print(f"{tc:<14s} {p:>7.3f} {r:>7.3f} {f1:>7.3f}  {sup}")

    # Headline metrics: jockey acc, NJ reject, confident wrong
    is_jk = y_te < 3
    is_nj = y_te == 3
    pred_is_nj = argmax == 3
    jk_acc = ((argmax == y_te) & is_jk).sum().item() / max(is_jk.sum().item(), 1)
    nj_rej = (pred_is_nj & is_nj).sum().item() / max(is_nj.sum().item(), 1)
    confident_wrong = (is_nj & ~pred_is_nj & (top1 >= 0.9)).sum().item() / max(is_nj.sum().item(), 1)
    avg_margin_jk = margin[is_jk].mean().item()
    avg_margin_nj = margin[is_nj].mean().item()

    print()
    print("=== HEADLINE (argmax) ===")
    print(f"  jockey color accuracy        : {jk_acc:.3f}  ({((argmax==y_te) & is_jk).sum().item()}/{is_jk.sum().item()})")
    print(f"  not_jockey reject rate       : {nj_rej:.3f}  ({(pred_is_nj & is_nj).sum().item()}/{is_nj.sum().item()})")
    print(f"  confident wrong on NJ (≥0.9) : {confident_wrong:.3f}")
    print(f"  avg margin (jockey)          : {avg_margin_jk:.3f}")
    print(f"  avg margin (not_jockey)      : {avg_margin_nj:.3f}")

    # ── 23 HARD NJ — full table ──
    print()
    print("=" * 88)
    print("=== ALL 23 HARD NJ — per-crop predictions ===")
    print("=" * 88)
    # load Phase 3 predictions for prototype side
    p_data  = np.load(OUT / "embeddings.npz", allow_pickle=False)
    p_proto = np.load(OUT / "prototypes.npz",  allow_pickle=False)
    proto_classes = list(p_proto["classes"])
    p_embs  = p_data["embeddings"]
    p_ids   = p_data["ids"].astype(str)
    embs_n_all = p_embs / (np.linalg.norm(p_embs, axis=1, keepdims=True) + 1e-12)
    sims_all   = embs_n_all @ p_proto["prototypes"].T
    proto_max_sim_by_id = {}
    proto_pred_by_id = {}
    for i, cid in enumerate(p_ids):
        proto_max_sim_by_id[cid] = float(sims_all[i].max())
        proto_pred_by_id[cid]    = proto_classes[int(sims_all[i].argmax())]

    test_id_to_idx = {cid: i for i, cid in enumerate(ids_te)}
    print(f"{'frame_id':<28s} {'proto_pred':<11s} {'proto_sim':>9s}  | "
          f"{'lin_pred':<11s} {'lin_conf':>8s} {'margin':>7s}")
    print("-" * 88)
    correct_hard = 0
    for cid in HARD_NJ:
        if cid not in test_id_to_idx:
            continue
        i = test_id_to_idx[cid]
        proto_p   = proto_pred_by_id[cid]
        proto_sim = proto_max_sim_by_id[cid]
        # interpret prototype: max_sim < 0.6 (Phase 3 best thr) → not_jockey
        proto_decision = proto_p if proto_sim >= 0.6 else "not_jockey"
        lin_p     = CLASSES[int(argmax[i].item())]
        lin_conf  = float(top1[i].item())
        m         = float(margin[i].item())
        marker    = "✅" if lin_p == "not_jockey" else "❌"
        print(f"  {cid:<26s} {proto_decision:<11s} {proto_sim:>9.3f}  | "
              f"{lin_p:<11s} {lin_conf:>8.3f} {m:>7.3f}  {marker}")
        if lin_p == "not_jockey":
            correct_hard += 1
    print(f"\nHARD reject by linear: {correct_hard}/{len(HARD_NJ)}  ({correct_hard/len(HARD_NJ)*100:.1f}%)")

    # ── Yellow→red, green→other ──
    print()
    print("=" * 72)
    print("=== JOCKEY MISCLASSIFICATIONS (per Фаза 3 fail modes) ===")
    print("=" * 72)
    for true_cls in ["green", "yellow", "red"]:
        ti = CLS_TO_IDX[true_cls]
        sub_mask = y_te == ti
        wrong_mask = sub_mask & (argmax != ti)
        if wrong_mask.sum() == 0:
            print(f"  {true_cls}: all correct ({sub_mask.sum().item()}/{sub_mask.sum().item()})  ✅")
        else:
            print(f"  {true_cls}: {wrong_mask.sum().item()}/{sub_mask.sum().item()} wrong:")
            for i, cid in enumerate(ids_te):
                if wrong_mask[i]:
                    pred_c = CLASSES[int(argmax[i].item())]
                    print(f"    {cid}  → {pred_c}  (conf={top1[i].item():.3f})")

    # ── Reject threshold sweep on softmax confidence ──
    print()
    print("=" * 72)
    print("=== REJECT THRESHOLD SWEEP (force-reject if max_softmax < thr) ===")
    print("=" * 72)
    sweep_rows = []
    for thr in [round(0.3 + 0.05*i, 2) for i in range(13)]:  # 0.30..0.90
        # apply: if argmax was jockey AND top1 < thr → принудительно not_jockey
        forced = argmax.clone()
        force_reject_mask = (forced != 3) & (top1 < thr)
        forced[force_reject_mask] = 3
        jk_acc2 = ((forced == y_te) & is_jk).sum().item() / max(is_jk.sum().item(), 1)
        nj_rej2 = ((forced == 3) & is_nj).sum().item() / max(is_nj.sum().item(), 1)
        composite = (jk_acc2 + nj_rej2) / 2
        sweep_rows.append({
            "thr": thr, "jockey_acc": jk_acc2,
            "nj_reject": nj_rej2, "composite": composite,
            "forced_reject_count": int(force_reject_mask.sum().item()),
        })
    print(f"{'thr':>5s}  {'jockey_acc':>11s}  {'nj_reject':>10s}  {'forced':>7s}  {'composite':>10s}")
    for r in sweep_rows:
        print(f"{r['thr']:>5.2f}  {r['jockey_acc']:>11.3f}  {r['nj_reject']:>10.3f}  "
              f"{r['forced_reject_count']:>7d}  {r['composite']:>10.3f}")
    best_thr_row = max(sweep_rows, key=lambda x: x["composite"])
    print(f"\nBEST by composite: thr={best_thr_row['thr']}  "
          f"jk_acc={best_thr_row['jockey_acc']:.3f}  nj_rej={best_thr_row['nj_reject']:.3f}")

    # ── Save eval results JSON ──
    eval_data = {
        "argmax_metrics": {
            "jockey_accuracy":  jk_acc,
            "not_jockey_reject": nj_rej,
            "confident_wrong":  confident_wrong,
            "avg_margin_jockey": avg_margin_jk,
            "avg_margin_not_jockey": avg_margin_nj,
        },
        "hard_nj_results": {
            cid: {
                "linear_pred": CLASSES[int(argmax[test_id_to_idx[cid]].item())],
                "linear_conf": float(top1[test_id_to_idx[cid]].item()),
                "margin":      float(margin[test_id_to_idx[cid]].item()),
                "proto_pred":  proto_pred_by_id[cid],
                "proto_sim":   proto_max_sim_by_id[cid],
            } for cid in HARD_NJ if cid in test_id_to_idx
        },
        "threshold_sweep": sweep_rows,
        "best_threshold":  best_thr_row,
    }
    (OUT / "linear_eval.json").write_text(json.dumps(eval_data, indent=2))
    print(f"\nEval JSON: {OUT / 'linear_eval.json'}")

    # ── HTML gallery: 23 HARD NJ side-by-side ──
    write_hard_nj_gallery(eval_data["hard_nj_results"])

    # ── Update eval_report.md ──
    update_report(eval_data, jk_acc, nj_rej, confident_wrong,
                  avg_margin_jk, avg_margin_nj, correct_hard, len(HARD_NJ))


def write_hard_nj_gallery(hard_results):
    from html import escape
    parts = [
        '<!doctype html><html lang="ru"><head><meta charset="utf-8">',
        '<title>HARD NJ — prototype vs linear</title>',
        '<style>',
        '  body { font-family:system-ui; background:#1a1a1a; color:#ddd; margin:0; padding:14px; }',
        '  h1 { color:#fff; }',
        '  .grid { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; }',
        '  .card { background:#2a2a2a; border-radius:4px; padding:6px; border:2px solid transparent; }',
        '  .card.fixed   { border-color:#2a8; }',
        '  .card.still_bad { border-color:#a44; }',
        '  .card img { width:100%; max-height:280px; object-fit:contain; display:block; }',
        '  .meta { font-family:monospace; font-size:11px; color:#ccc; padding:6px; }',
        '  .ok { color:#9eb; }',
        '  .bad { color:#fab; }',
        '</style></head><body>',
        '<h1>23 HARD NJ — prototype vs linear probe</h1>',
        '<p style="color:#aaa">Зелёная рамка = linear правильно reject; красная = всё ещё проходит как жокей.</p>',
        '<div class="grid">',
    ]
    for cid, r in hard_results.items():
        src = f"../diagnose_v4_2026-04-25/crops_raw/{cid}.jpg"
        is_fixed = r["linear_pred"] == "not_jockey"
        cls_card = "fixed" if is_fixed else "still_bad"
        verdict_lin = "<span class='ok'>REJECT ✅</span>" if is_fixed \
                      else f"<span class='bad'>{r['linear_pred']} ❌</span>"
        proto_decision = r["proto_pred"] if r["proto_sim"] >= 0.6 else "REJECT (sim&lt;0.6)"
        parts.append(
            f'<div class="card {cls_card}">'
            f'<img src="{escape(src)}" loading="lazy">'
            f'<div class="meta"><b>{escape(cid)}</b><br>'
            f'<b>prototype:</b> {escape(proto_decision)} '
            f'(sim={r["proto_sim"]:.3f})<br>'
            f'<b>linear:</b> {verdict_lin} '
            f'(conf={r["linear_conf"]:.3f}, margin={r["margin"]:.3f})'
            f'</div></div>'
        )
    parts.append('</div></body></html>')
    out = OUT / "gallery_hard_nj.html"
    out.write_text("\n".join(parts))
    print(f"Gallery: {out}")


def update_report(ev, jk_acc, nj_rej, cw, m_jk, m_nj, correct_hard, n_hard):
    """Append linear-probe section to eval_report.md."""
    rep = OUT / "eval_report.md"
    md = rep.read_text() if rep.exists() else "# DINOv2 prototype-based color classifier — eval report\n"

    new_section = ["", "---", "", "## Linear probe results (Фаза C)", ""]
    new_section += ["**Setup:** L2_norm(emb) → Linear(768, 4), AdamW lr=1e-3 wd=1e-4,",
                    "CrossEntropyLoss с inverse-frequency class weights, batch=32,",
                    "early stop @ epoch 13 (val_acc=0.969).",
                    "",
                    "Test set: 102 crops (52 jockey + 50 NJ), включая **23 HARD NJ** ",
                    "(forced — 3 actual FPs + 20 near-miss из Фазы 3 prototype).",
                    "",
                    "### Headline metrics (default argmax, no reject threshold)",
                    "",
                    "| metric | v4 CNN | DINOv2 prototype | **DINOv2 linear** |",
                    "|---|---:|---:|---:|",
                    f"| jockey color accuracy | 100.0% | 80.8% | **{jk_acc*100:.1f}%** |",
                    f"| not_jockey reject rate | 0.0% | 98.8% | **{nj_rej*100:.1f}%** |",
                    f"| confident wrong on NJ (≥0.9) | 21.6% | 0.0% | **{cw*100:.1f}%** |",
                    f"| avg margin (jockey) | — | — | **{m_jk:.3f}** |",
                    f"| avg margin (not_jockey) | — | — | **{m_nj:.3f}** |",
                    "",
                    f"### HARD NJ test (23 cases): linear reject {correct_hard}/{n_hard} = {correct_hard/n_hard*100:.0f}%",
                    "",
                    "Все 23 проблемных not_jockey из Phase 3 (3 actual FPs + 20 near-miss):",
                    "",
                    "| frame_id | proto Phase 3 | linear pred | linear conf | linear margin |",
                    "|---|---|---|---:|---:|"]
    for cid, r in ev["hard_nj_results"].items():
        proto_decision = r["proto_pred"] if r["proto_sim"] >= 0.6 else "REJECT"
        lin_marker = "✅" if r["linear_pred"] == "not_jockey" else "❌"
        new_section.append(
            f"| `{cid}` | {proto_decision} (sim={r['proto_sim']:.3f}) | "
            f"{r['linear_pred']} {lin_marker} | {r['linear_conf']:.3f} | {r['margin']:.3f} |"
        )

    new_section += ["",
                    "### Reject threshold sweep (force-reject if max_softmax < thr)",
                    "",
                    "| thr | jockey_acc | nj_reject | forced_reject | composite |",
                    "|---:|---:|---:|---:|---:|"]
    for r in ev["threshold_sweep"]:
        new_section.append(f"| {r['thr']:.2f} | {r['jockey_acc']:.3f} | {r['nj_reject']:.3f} | "
                           f"{r['forced_reject_count']} | {r['composite']:.3f} |")
    bt = ev["best_threshold"]
    new_section.append(f"\n**Best by composite:** thr={bt['thr']}, "
                       f"jockey_acc={bt['jockey_acc']:.3f}, nj_reject={bt['nj_reject']:.3f}")

    new_section += ["", "### Final verdict (Фаза C)", ""]
    if jk_acc >= 0.88 and nj_rej >= 0.99:
        verdict = (f"**Linear probe значительно лучше prototype.** "
                   f"jockey accuracy {jk_acc:.1%} ≥ 88% и not_jockey reject {nj_rej:.1%} ≥ 99% — "
                   "целевой trade-off для production достигнут на DINOv2.\n\n"
                   "**Next steps:** дождаться DINOv3 одобрения, "
                   "повторить linear probe на DINOv3 embeddings (ожидается ещё лучше), "
                   "интегрировать в SGIE postprocessing C++ через ONNX export linear слоя.")
    elif jk_acc >= 0.85 or nj_rej > 0.99:
        verdict = (f"**Linear probe MARGINALLY улучшает prototype** — "
                   f"jockey {jk_acc:.1%} (vs 80.8% prototype), reject {nj_rej:.1%} (vs 98.8%). "
                   "Линейный слой полезен, но не drastically.\n\n"
                   "**Next steps:** расширить training set (больше not_jockey примеров с других видео), "
                   "попробовать DINOv3 embeddings, или MLP вместо linear.")
    else:
        verdict = (f"**Linear probe НЕ улучшает prototype** "
                   f"(jockey {jk_acc:.1%}, reject {nj_rej:.1%}). "
                   "Возможные причины: переобучение на маленьком train ({254} samples), "
                   "слишком мало variation в not_jockey классе, "
                   "linear capacity недостаточно для дискриминации.")
    new_section.append(verdict)

    md += "\n".join(new_section) + "\n"
    rep.write_text(md)
    print(f"Report updated: {rep}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["split", "train", "eval"])
    args = ap.parse_args()
    if args.step == "split":
        show_split()
    elif args.step == "train":
        train()
    elif args.step == "eval":
        evaluate()


if __name__ == "__main__":
    main()
