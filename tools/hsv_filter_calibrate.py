#!/usr/bin/env python3
"""hsv_filter_calibrate.py — калибровка HSV agreement filter (Вариант D).

Идея: после CNN softmax считать долю пикселей crop'а попадающих в HSV-
диапазон ПРЕДСКАЗАННОГО класса. Если доля < threshold — отбрасывать
классификацию.

Источники правды:
  output/diagnose_v4_2026-04-25/labels.json      — ground truth
  output/diagnose_v4_2026-04-25/step4_predict.json — CNN predictions
  output/diagnose_v4_2026-04-25/crops_raw/        — сами crops

Запуск:
  python tools/hsv_filter_calibrate.py --show-ranges
  python tools/hsv_filter_calibrate.py --calibrate
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT  = REPO / "output" / "diagnose_v4_2026-04-25"
CROPS_DIR = OUT / "crops_raw"
CALIB_DIR = OUT / "hsv_calibration"

# ── Grid search axes ──────────────────────────────────────────────────
S_MIN_GRID = [30, 40, 50, 60]
V_MIN = 50          # фиксирован в этой калибровке (отсекает чёрный/тени)

# Per-class hue ranges. Список = (h_lo, h_hi) для wrap-around.
# Для green даём ДВА варианта (wide и narrow).
GREEN_VARIANTS = {
    "wide":   [(35, 85)],
    "narrow": [(40, 75)],
}

HSV_BASE: dict[str, list[tuple[int, int]]] = {
    "blue":   [(95, 130)],
    "purple": [(130, 160)],
    "red":    [(0, 10), (170, 179)],
    "yellow": [(20, 35)],
}

THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 13)]   # 0.05..0.60


def hsv_color_ratio(crop_bgr: np.ndarray, predicted_class: str,
                    s_min: int, v_min: int,
                    hsv_ranges: dict[str, list[tuple[int, int]]]) -> float:
    """Доля пикселей crop'а в HSV-диапазоне предсказанного класса.

    Только пиксели где saturation>=s_min И value>=v_min считаются "цветными".
    Возвращает float в [0..1] от площади всего crop'а.
    """
    if predicted_class not in hsv_ranges:
        return 0.0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sat_ok = (s >= s_min) & (v >= v_min)
    hue_ok = np.zeros_like(h, dtype=bool)
    for lo, hi in hsv_ranges[predicted_class]:
        hue_ok |= (h >= lo) & (h <= hi)
    match = sat_ok & hue_ok
    return float(match.sum()) / float(h.size)


def load_data():
    labels = json.loads((OUT / "labels.json").read_text())
    preds  = json.loads((OUT / "step4_predict.json").read_text())
    return labels, preds


def show_ranges():
    print("=" * 72)
    print(f"S_MIN grid: {S_MIN_GRID}, V_MIN fixed: {V_MIN}")
    print("Per-class hue ranges (HSV OpenCV: H∈[0,179], S∈[0,255], V∈[0,255])")
    print("=" * 72)
    for cls, ranges in HSV_BASE.items():
        ranges_s = " ∪ ".join(f"H∈[{lo},{hi}]" for lo, hi in ranges)
        print(f"  {cls:7s}  {ranges_s}")
    print()
    print("Green has TWO variants (compared in grid search):")
    for name, ranges in GREEN_VARIANTS.items():
        ranges_s = " ∪ ".join(f"H∈[{lo},{hi}]" for lo, hi in ranges)
        print(f"  green/{name:6s}  {ranges_s}")
    print()
    print(f"Threshold sweep: {THRESHOLDS}")


def calibrate():
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    preds = json.loads((OUT / "step4_predict.json").read_text())
    print(f"Loaded {len(preds)} predictions")

    # Pre-load all crops (BGR)
    crops_cache: dict[str, np.ndarray] = {}
    for r in preds:
        cp = CROPS_DIR / f"{r['id']}.jpg"
        img = cv2.imread(str(cp))
        if img is not None:
            crops_cache[r['id']] = img
    print(f"Loaded {len(crops_cache)} crop images")

    # ── 1. Red distribution check (do we need to widen red range?) ──
    print()
    print("=" * 72)
    print("RED RANGE DISTRIBUTION CHECK (на 59 настоящих red-jockey crops)")
    print("=" * 72)
    red_jockeys = [r for r in preds if r['label'] == 'red']
    red_ratios_default = []
    for r in red_jockeys:
        img = crops_cache.get(r['id'])
        if img is not None:
            red_ratios_default.append(hsv_color_ratio(
                img, 'red', s_min=50, v_min=V_MIN, hsv_ranges=HSV_BASE
            ))
    median_red = float(np.median(red_ratios_default))
    print(f"Red range default H∈[0,10]∪[170,179] @ S=50,V={V_MIN}:")
    print(f"  median ratio: {median_red:.3f}, min: {min(red_ratios_default):.3f}, max: {max(red_ratios_default):.3f}")
    if median_red < 0.10:
        print(f"  → median < 0.10 → расширяю red до H∈[0,12]∪[165,179]")
        HSV_BASE['red'] = [(0, 12), (165, 179)]
        red_ratios_widened = []
        for r in red_jockeys:
            img = crops_cache.get(r['id'])
            if img is not None:
                red_ratios_widened.append(hsv_color_ratio(
                    img, 'red', s_min=50, v_min=V_MIN, hsv_ranges=HSV_BASE
                ))
        new_median = float(np.median(red_ratios_widened))
        print(f"  after widening: median {new_median:.3f}")
    else:
        print(f"  → median >= 0.10, red range OK as-is")

    # ── 2. Grid search: (S_MIN, green_variant, threshold) ──
    print()
    print("=" * 72)
    print("GRID SEARCH: (S_MIN × green_variant × threshold)")
    print("=" * 72)

    rows = []   # for CSV
    composite_rows = []
    detailed_per_combo: dict[tuple, dict] = {}

    real_jockeys = [r for r in preds if r['label'] != 'not_jockey']
    not_jockeys  = [r for r in preds if r['label'] == 'not_jockey']
    nj_by_pred = {
        'green':  [r for r in not_jockeys if r['pred'] == 'green'],
        'yellow': [r for r in not_jockeys if r['pred'] == 'yellow'],
        'purple': [r for r in not_jockeys if r['pred'] == 'purple'],
    }
    print(f"  real_jockeys: {len(real_jockeys)}  |  not_jockeys: {len(not_jockeys)}")
    print(f"    nj_by_pred: green={len(nj_by_pred['green'])}, "
          f"yellow={len(nj_by_pred['yellow'])}, purple={len(nj_by_pred['purple'])}")
    print()

    for s_min in S_MIN_GRID:
        for green_name, green_range in GREEN_VARIANTS.items():
            ranges = dict(HSV_BASE)
            ranges['green'] = green_range

            # compute ratio per crop using PREDICTED class
            ratio_by_id: dict[str, float] = {}
            for r in preds:
                img = crops_cache.get(r['id'])
                if img is None:
                    continue
                ratio_by_id[r['id']] = hsv_color_ratio(
                    img, r['pred'], s_min=s_min, v_min=V_MIN, hsv_ranges=ranges
                )

            # per-threshold metrics
            for thr in THRESHOLDS:
                # real_jockey TP rate: ratio >= thr (passes filter)
                rj_pass = sum(1 for r in real_jockeys if ratio_by_id.get(r['id'], 0) >= thr)
                tp_rate = rj_pass / len(real_jockeys)

                # not_jockey rejection: ratio < thr (отсеян)
                nj_reject = sum(1 for r in not_jockeys if ratio_by_id.get(r['id'], 1) < thr)
                rejected_nj = nj_reject / len(not_jockeys)

                # per-pred class breakdown
                per_pred_reject = {}
                for pred_cls, items in nj_by_pred.items():
                    if items:
                        rej = sum(1 for r in items if ratio_by_id.get(r['id'], 1) < thr)
                        per_pred_reject[pred_cls] = rej / len(items)

                # composite: rejected_nj - 2 * (1 - tp_rate)
                composite = rejected_nj - 2 * (1 - tp_rate)

                row = {
                    'S_MIN':       s_min,
                    'green':       green_name,
                    'threshold':   thr,
                    'tp_rate':     round(tp_rate, 3),
                    'rejected_nj': round(rejected_nj, 3),
                    'composite':   round(composite, 3),
                    'reject_nj_green':  round(per_pred_reject.get('green', 0), 3),
                    'reject_nj_yellow': round(per_pred_reject.get('yellow', 0), 3),
                    'reject_nj_purple': round(per_pred_reject.get('purple', 0), 3),
                }
                rows.append(row)
                composite_rows.append((composite, row, ratio_by_id))

    # save full CSV
    csv_path = CALIB_DIR / "grid_search.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Full grid → {csv_path} ({len(rows)} combinations)")

    # ── Top-5 by composite ──
    composite_rows.sort(key=lambda x: -x[0])
    top5 = composite_rows[:5]
    print()
    print("=" * 72)
    print("TOP 5 COMBINATIONS (by composite = rejected_nj - 2*(1-tp_rate))")
    print("=" * 72)
    headers = ['S_MIN', 'green', 'thr', 'tp_rate', 'rej_nj', 'rej_g', 'rej_y', 'rej_p', 'composite']
    print(f"  {headers[0]:>5s} {headers[1]:>7s} {headers[2]:>5s} "
          f"{headers[3]:>8s} {headers[4]:>7s} "
          f"{headers[5]:>6s} {headers[6]:>6s} {headers[7]:>6s} "
          f"{headers[8]:>9s}")
    print("  " + "-" * 70)
    for composite, row, _ in top5:
        print(f"  {row['S_MIN']:>5d} {row['green']:>7s} {row['threshold']:>5.2f} "
              f"{row['tp_rate']:>8.3f} {row['rejected_nj']:>7.3f} "
              f"{row['reject_nj_green']:>6.3f} {row['reject_nj_yellow']:>6.3f} "
              f"{row['reject_nj_purple']:>6.3f} {composite:>9.3f}")
    print()

    # ── Critical checkpoint: any combo passes acceptance? ──
    accepted = [(c, r, rid) for c, r, rid in composite_rows
                if r['rejected_nj'] >= 0.70 and r['tp_rate'] >= 0.90]
    print("=" * 72)
    print("ACCEPTANCE CRITERIA: rejected_nj >= 0.70  AND  tp_rate >= 0.90")
    print("=" * 72)
    if accepted:
        print(f"✅ {len(accepted)} combinations meet criteria. Top 5 below:")
        for composite, row, _ in accepted[:5]:
            print(f"  S_MIN={row['S_MIN']} green={row['green']} thr={row['threshold']} "
                  f"→ tp={row['tp_rate']:.1%}  reject_nj={row['rejected_nj']:.1%}  "
                  f"composite={composite:.3f}")
    else:
        print("❌ NO COMBINATION meets both criteria.")
        print("Best by composite (top 5 above) — see for partial trade-offs.")

    # ── Distribution histograms (matplotlib) for the BEST combination ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        best_composite, best_row, best_ratios = composite_rows[0]
        rj_ratios = [best_ratios[r['id']] for r in real_jockeys if r['id'] in best_ratios]
        nj_ratios = [best_ratios[r['id']] for r in not_jockeys  if r['id'] in best_ratios]

        fig, ax = plt.subplots(figsize=(10, 5))
        bins = np.linspace(0, max(max(rj_ratios), max(nj_ratios)), 40)
        ax.hist(rj_ratios, bins=bins, alpha=0.6, color='#2a8', label=f'real jockeys (n={len(rj_ratios)})')
        ax.hist(nj_ratios, bins=bins, alpha=0.6, color='#a44', label=f'not_jockey (n={len(nj_ratios)})')
        ax.axvline(best_row['threshold'], color='black', linestyle='--',
                   label=f"best threshold = {best_row['threshold']}")
        ax.set_xlabel('HSV color ratio (predicted class)')
        ax.set_ylabel('count')
        ax.set_title(f"Distribution at S_MIN={best_row['S_MIN']}, green={best_row['green']}, "
                     f"V_MIN={V_MIN}")
        ax.legend()
        ax.grid(alpha=0.3)
        plot_path = CALIB_DIR / "distribution_best.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=100)
        plt.close(fig)
        print(f"\nDistribution plot (best combo): {plot_path}")

        # second plot: stacked per-class for not_jockey, at best combo
        fig, ax = plt.subplots(figsize=(10, 5))
        for pred_cls, color in [('green', '#2a8'), ('yellow', '#aa3'), ('purple', '#a4a')]:
            items = nj_by_pred[pred_cls]
            ratios = [best_ratios[r['id']] for r in items if r['id'] in best_ratios]
            ax.hist(ratios, bins=bins, alpha=0.6, label=f"not_jockey → {pred_cls} (n={len(ratios)})",
                    color=color)
        ax.axvline(best_row['threshold'], color='black', linestyle='--',
                   label=f"thr={best_row['threshold']}")
        ax.set_xlabel('HSV color ratio (predicted class)')
        ax.set_ylabel('count')
        ax.set_title(f"not_jockey distribution by predicted class @ best combo")
        ax.legend()
        ax.grid(alpha=0.3)
        plot_path2 = CALIB_DIR / "distribution_not_jockey_per_pred.png"
        fig.tight_layout()
        fig.savefig(plot_path2, dpi=100)
        plt.close(fig)
        print(f"Per-pred-class plot:           {plot_path2}")
    except ImportError:
        print("(matplotlib not installed — skipping plots)")


def gallery_lost_greens():
    """gallery_lost_greens.html — визуально посмотреть 27 lost green jockeys
    + 5 passed-with-highest-ratio для контраста.

    Применяет winning combination: S_MIN=30, V_MIN=50, green=wide [35,85], thr=0.05.
    """
    from html import escape
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    preds = json.loads((OUT / "step4_predict.json").read_text())

    S_MIN, V_MIN_local = 30, 50
    HSV_LOCAL = dict(HSV_BASE)
    HSV_LOCAL["green"] = [(35, 85)]    # wide
    THR = 0.05

    # compute ratios for label=green
    greens = [r for r in preds if r["label"] == "green"]
    for r in greens:
        img = cv2.imread(str(CROPS_DIR / f"{r['id']}.jpg"))
        r["_ratio"] = (hsv_color_ratio(img, "green", S_MIN, V_MIN_local, HSV_LOCAL)
                       if img is not None else 0.0)

    lost = sorted([r for r in greens if r["_ratio"] < THR],
                  key=lambda x: x["_ratio"])
    passed = sorted([r for r in greens if r["_ratio"] >= THR],
                    key=lambda x: -x["_ratio"])
    top_passed = passed[:5]

    print(f"Lost greens (ratio < {THR}): {len(lost)}")
    print(f"Passed greens for contrast (top 5 by ratio): {len(top_passed)}")

    parts = [
        '<!doctype html><html lang="ru"><head><meta charset="utf-8">',
        '<title>HSV filter — lost green jockeys review</title>',
        '<style>',
        '  body { font-family:system-ui; background:#1a1a1a; color:#ddd; margin:0; padding:14px; }',
        '  h1 { color:#fff; }',
        '  h2 { color:#ffaf3a; border-bottom:1px solid #444; padding:6px 0; margin-top:20px; }',
        '  .desc { color:#aaa; font-size:13px; margin:0 0 10px 0; }',
        '  .grid { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; }',
        '  .card { background:#2a2a2a; border-radius:4px; padding:6px; }',
        '  .card.lost   { border:2px solid #a44; }',
        '  .card.passed { border:2px solid #2a8; }',
        '  .card img { width:100%; max-height:280px; object-fit:contain; display:block; '
        '              cursor:zoom-in; }',
        '  .meta { font-family:monospace; font-size:11px; color:#ccc; padding:6px; }',
        '  .meta b { color:#fff; }',
        '  .ratio { color:#fea; font-weight:bold; font-size:13px; }',
        '  nav { position:sticky; top:0; background:#1a1a1a; padding:8px 0; z-index:50;'
        '        border-bottom:1px solid #444; margin-bottom:10px; }',
        '  nav a { color:#88ddff; margin-right:14px; text-decoration:none; }',
        '</style></head><body>',
        '<h1>HSV filter — green jockeys review</h1>',
        '<p class="desc">'
        f'Combo: S_MIN={S_MIN}, V_MIN={V_MIN_local}, green=H∈[35,85], threshold={THR}. '
        'Все crops размечены как <b>green</b> в labels.json.'
        '</p>',
        '<nav>',
        f'<a href="#lost">🔴 Lost ({len(lost)})</a>',
        f'<a href="#passed">🟢 Passed contrast ({len(top_passed)})</a>',
        '</nav>',
    ]

    # ── LOST section ──
    parts.append(f'<h2 id="lost">🔴 Lost greens — ratio &lt; {THR} (отбракованы фильтром)</h2>')
    parts.append('<p class="desc">'
                 'Это reference set для решения: реальные жокеи в плохих условиях / mislabel / corner cases?'
                 '</p>')
    parts.append('<div class="grid">')
    for r in lost:
        src = f"../crops_raw/{r['id']}.jpg"
        parts.append(
            f'<div class="card lost">'
            f'<a href="{escape(src)}" target="_blank"><img src="{escape(src)}" loading="lazy"></a>'
            f'<div class="meta"><b>{escape(r["id"])}</b><br>'
            f'<span class="ratio">HSV ratio = {r["_ratio"]:.4f}</span><br>'
            f'CNN: <b>{r["pred"]}</b> conf={r["pred_conf"]:.3f}<br>'
            f'true label: green</div>'
            f'</div>'
        )
    parts.append('</div>')

    # ── PASSED contrast ──
    parts.append(f'<h2 id="passed">🟢 Passed greens (top 5 by ratio) — для сравнения</h2>')
    parts.append('<p class="desc">Это «эталонные» зелёные жокеи где silk занимает заметную долю crop\'а.</p>')
    parts.append('<div class="grid">')
    for r in top_passed:
        src = f"../crops_raw/{r['id']}.jpg"
        parts.append(
            f'<div class="card passed">'
            f'<a href="{escape(src)}" target="_blank"><img src="{escape(src)}" loading="lazy"></a>'
            f'<div class="meta"><b>{escape(r["id"])}</b><br>'
            f'<span class="ratio">HSV ratio = {r["_ratio"]:.4f}</span><br>'
            f'CNN: <b>{r["pred"]}</b> conf={r["pred_conf"]:.3f}<br>'
            f'true label: green</div>'
            f'</div>'
        )
    parts.append('</div>')
    parts.append('</body></html>')

    out_path = CALIB_DIR / "gallery_lost_greens.html"
    out_path.write_text("\n".join(parts))
    print(f"\nGallery: {out_path}")
    print(f"  → http://100.122.27.103:8767/hsv_calibration/gallery_lost_greens.html")


def gallery_lost_greens_diag():
    """gallery_lost_greens_diag.html — per-crop диагностика что HSV видит:
    оригинал | бинарная маска прошедших пикселей | Hue histogram | Sat histogram.

    Применяет winning combo: S_MIN=30, V_MIN=50, green=wide [35,85], thr=0.05.
    """
    from html import escape
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    DIAG_DIR = CALIB_DIR / "diag"
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    preds = json.loads((OUT / "step4_predict.json").read_text())

    S_MIN, V_MIN_local = 30, 50
    GREEN_RANGE = [(35, 85)]
    THR = 0.05

    greens = [r for r in preds if r["label"] == "green"]
    for r in greens:
        img = cv2.imread(str(CROPS_DIR / f"{r['id']}.jpg"))
        r["_ratio"] = (hsv_color_ratio(img, "green", S_MIN, V_MIN_local,
                                       {"green": GREEN_RANGE})
                       if img is not None else 0.0)

    lost   = sorted([r for r in greens if r["_ratio"] <  THR],  key=lambda x: x["_ratio"])
    passed = sorted([r for r in greens if r["_ratio"] >= THR],  key=lambda x: -x["_ratio"])[:5]

    print(f"Generating diag for {len(lost)} lost + {len(passed)} passed crops")

    def per_crop_artifacts(rec):
        cid = rec["id"]
        img = cv2.imread(str(CROPS_DIR / f"{cid}.jpg"))
        if img is None:
            return None
        h_img, w_img = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        # binary mask: pixels passing FULL filter (S>=S_MIN AND V>=V_MIN AND H in green range)
        sat_ok = (s >= S_MIN) & (v >= V_MIN_local)
        hue_ok = np.zeros_like(h, dtype=bool)
        for lo, hi in GREEN_RANGE:
            hue_ok |= (h >= lo) & (h <= hi)
        mask = sat_ok & hue_ok
        # mask png: white where matched, black else
        mask_png = (mask.astype(np.uint8) * 255)
        mask_path = DIAG_DIR / f"{cid}_mask.png"
        cv2.imwrite(str(mask_path), mask_png)

        # Hue histogram: только пиксели с sat_ok (S>=S_MIN AND V>=V_MIN)
        h_filtered = h[sat_ok]
        fig, ax = plt.subplots(figsize=(4, 2.5))
        if h_filtered.size > 0:
            ax.hist(h_filtered, bins=180, range=(0, 180), color='#48a', alpha=0.85)
            median_h = float(np.median(h_filtered))
            ax.axvline(median_h, color='red', linestyle='--', linewidth=1, label=f'median={median_h:.0f}')
        else:
            median_h = float('nan')
            ax.text(90, 0.5, 'no pixels passed S filter', ha='center', va='center')
        # green range overlay
        ax.axvspan(GREEN_RANGE[0][0], GREEN_RANGE[0][1], color='green', alpha=0.15, label='green range')
        ax.set_xlim(0, 180)
        ax.set_xlabel('Hue (S>=' + str(S_MIN) + ' only)')
        ax.set_ylabel('count')
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=7)
        fig.tight_layout()
        hue_path = DIAG_DIR / f"{cid}_hue.png"
        fig.savefig(str(hue_path), dpi=80)
        plt.close(fig)

        # Saturation histogram: ВСЕ пиксели crop'а
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.hist(s.flatten(), bins=64, range=(0, 256), color='#a48', alpha=0.85)
        ax.axvline(S_MIN, color='red', linestyle='--', linewidth=1, label=f'S_MIN={S_MIN}')
        s_above = float((s >= S_MIN).sum()) / float(s.size)
        ax.set_title(f'{s_above*100:.1f}% pixels above S_MIN', fontsize=8)
        ax.set_xlabel('Saturation (all pixels)')
        ax.set_ylabel('count')
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=7)
        fig.tight_layout()
        sat_path = DIAG_DIR / f"{cid}_sat.png"
        fig.savefig(str(sat_path), dpi=80)
        plt.close(fig)

        s_filtered = s[sat_ok]
        median_s = float(np.median(s_filtered)) if s_filtered.size else float('nan')

        return {
            "id":         cid,
            "w":          w_img,
            "h":          h_img,
            "ratio":      rec["_ratio"],
            "cnn_pred":   rec["pred"],
            "cnn_conf":   rec["pred_conf"],
            "median_h":   median_h,
            "median_s":   median_s,
            "s_above":    s_above,
        }

    lost_data   = [per_crop_artifacts(r) for r in lost]
    passed_data = [per_crop_artifacts(r) for r in passed]
    lost_data   = [d for d in lost_data   if d is not None]
    passed_data = [d for d in passed_data if d is not None]

    parts = [
        '<!doctype html><html lang="ru"><head><meta charset="utf-8">',
        '<title>HSV diag — lost greens</title>',
        '<style>',
        '  body { font-family:system-ui; background:#1a1a1a; color:#ddd; margin:0; padding:14px; }',
        '  h1 { color:#fff; }',
        '  h2 { color:#ffaf3a; border-bottom:1px solid #444; padding:6px 0; margin-top:20px; }',
        '  .desc { color:#aaa; font-size:13px; }',
        '  table { border-collapse:collapse; width:100%; }',
        '  th { background:#2a2a2a; color:#88ddff; padding:6px; text-align:center; font-size:12px;'
        '       position:sticky; top:0; z-index:5; }',
        '  td { padding:4px; vertical-align:top; border:1px solid #333; background:#1f1f1f; }',
        '  td img { width:100%; max-height:280px; object-fit:contain; display:block; cursor:zoom-in; }',
        '  .meta { font-family:monospace; font-size:11px; color:#ccc; padding:4px; }',
        '  .meta b { color:#fff; }',
        '  .ratio { color:#fea; font-weight:bold; }',
        '  tr.lost   td:first-child { border-left:4px solid #a44; }',
        '  tr.passed td:first-child { border-left:4px solid #2a8; }',
        '  nav { position:sticky; top:0; background:#1a1a1a; padding:8px 0; z-index:50;'
        '        border-bottom:1px solid #444; margin-bottom:10px; }',
        '  nav a { color:#88ddff; margin-right:14px; text-decoration:none; }',
        '</style></head><body>',
        '<h1>HSV diagnostic — что фильтр реально видит</h1>',
        f'<p class="desc">Combo: S_MIN={S_MIN}, V_MIN={V_MIN_local}, green=H∈[35,85], threshold={THR}.</p>',
        f'<nav>'
        f'<a href="#lost">🔴 Lost ({len(lost_data)})</a> '
        f'<a href="#passed">🟢 Passed contrast ({len(passed_data)})</a>'
        f'</nav>',
    ]

    def section(title, anchor, klass, data):
        parts.append(f'<h2 id="{anchor}">{escape(title)}</h2>')
        parts.append('<table>')
        parts.append('<thead><tr>'
                     '<th style="width:20%">crop</th>'
                     '<th style="width:20%">HSV mask (white = passed)</th>'
                     '<th style="width:25%">Hue histogram (S≥' + str(S_MIN) + ')</th>'
                     '<th style="width:25%">Saturation histogram (all px)</th>'
                     '<th style="width:10%">meta</th>'
                     '</tr></thead><tbody>')
        for d in data:
            cid = d["id"]
            crop_src = f"../crops_raw/{cid}.jpg"
            mask_src = f"diag/{cid}_mask.png"
            hue_src  = f"diag/{cid}_hue.png"
            sat_src  = f"diag/{cid}_sat.png"
            mh = f'{d["median_h"]:.1f}' if d["median_h"] == d["median_h"] else 'NaN'
            ms = f'{d["median_s"]:.0f}' if d["median_s"] == d["median_s"] else 'NaN'
            parts.append(
                f'<tr class="{klass}">'
                f'<td><a href="{crop_src}" target="_blank"><img src="{crop_src}" loading="lazy"></a></td>'
                f'<td><a href="{mask_src}" target="_blank"><img src="{mask_src}" loading="lazy"></a></td>'
                f'<td><a href="{hue_src}"  target="_blank"><img src="{hue_src}"  loading="lazy"></a></td>'
                f'<td><a href="{sat_src}"  target="_blank"><img src="{sat_src}"  loading="lazy"></a></td>'
                f'<td class="meta"><b>{escape(cid)}</b><br>'
                f'bbox: {d["w"]}×{d["h"]} px<br>'
                f'<span class="ratio">ratio = {d["ratio"]:.4f}</span><br>'
                f'CNN: {d["cnn_pred"]} ({d["cnn_conf"]:.3f})<br><br>'
                f'median Hue (S≥{S_MIN}): {mh}<br>'
                f'median Sat (S≥{S_MIN}): {ms}<br>'
                f'% px above S_MIN: {d["s_above"]*100:.1f}%</td>'
                f'</tr>'
            )
        parts.append('</tbody></table>')

    section("🔴 Lost greens — ratio < 0.05", "lost", "lost", lost_data)
    section("🟢 Passed greens — top 5 by ratio (контраст)", "passed", "passed", passed_data)
    parts.append('</body></html>')

    out_path = CALIB_DIR / "gallery_lost_greens_diag.html"
    out_path.write_text("\n".join(parts))
    print(f"\nGallery: {out_path}")
    print(f"  → http://100.122.27.103:8767/hsv_calibration/gallery_lost_greens_diag.html")


def hue_distribution():
    """Per-label distribution of median Hue (S>=S_MIN) across all crops.
    Проверяем гипотезу: green-jockey median hue ≠ [35-85]."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    preds = json.loads((OUT / "step4_predict.json").read_text())

    S_MIN_local = 30
    V_MIN_local = 50
    GROUPS = ['green', 'yellow', 'red', 'not_jockey']
    COLORS = {'green': '#2a8', 'yellow': '#cb3', 'red': '#c33', 'not_jockey': '#888'}

    medians_by_group: dict[str, list[float]] = {g: [] for g in GROUPS}
    pct_above_s_by_group: dict[str, list[float]] = {g: [] for g in GROUPS}

    for r in preds:
        if r['label'] not in GROUPS:
            continue
        img = cv2.imread(str(CROPS_DIR / f"{r['id']}.jpg"))
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        sat_ok = (s >= S_MIN_local) & (v >= V_MIN_local)
        h_filtered = h[sat_ok]
        if h_filtered.size > 0:
            medians_by_group[r['label']].append(float(np.median(h_filtered)))
        pct_above_s_by_group[r['label']].append(float(sat_ok.sum()) / float(h.size))

    # ── console stats ──
    print("=" * 78)
    print(f"Median Hue per crop (только S≥{S_MIN_local} ∧ V≥{V_MIN_local}), per label group")
    print("=" * 78)
    print(f"{'group':<12s} {'n':>5s} {'min':>6s} {'p25':>6s} {'median':>8s} "
          f"{'p75':>6s} {'max':>6s}  {'avg %above_S':>14s}")
    print("-" * 78)
    for g in GROUPS:
        meds = medians_by_group[g]
        if not meds:
            continue
        meds_arr = np.array(meds)
        pct_above = np.array(pct_above_s_by_group[g])
        print(f"{g:<12s} {len(meds):>5d} {meds_arr.min():>6.1f} "
              f"{np.percentile(meds_arr, 25):>6.1f} {np.median(meds_arr):>8.1f} "
              f"{np.percentile(meds_arr, 75):>6.1f} {meds_arr.max():>6.1f} "
              f"{pct_above.mean()*100:>13.1f}%")
    print()

    # ── overlaid histogram ──
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    bins = np.arange(0, 181, 3)

    # top: real jockeys
    ax = axes[0]
    for g in ['green', 'yellow', 'red']:
        ax.hist(medians_by_group[g], bins=bins,
                alpha=0.55, color=COLORS[g],
                label=f"{g} (n={len(medians_by_group[g])})", edgecolor='black', linewidth=0.3)
    # current ranges overlay
    ax.axvspan(35, 85, color='#2a8', alpha=0.10)
    ax.axvspan(20, 35, color='#cb3', alpha=0.10)
    ax.axvspan(0, 10, color='#c33', alpha=0.10)
    ax.axvspan(170, 179, color='#c33', alpha=0.10)
    ax.set_title('Per-crop median Hue — real jockeys (background = current ranges)')
    ax.set_ylabel('crop count')
    ax.legend()
    ax.grid(alpha=0.3)

    # bottom: not_jockey
    ax = axes[1]
    ax.hist(medians_by_group['not_jockey'], bins=bins,
            alpha=0.7, color=COLORS['not_jockey'],
            label=f"not_jockey (n={len(medians_by_group['not_jockey'])})",
            edgecolor='black', linewidth=0.3)
    ax.axvspan(35, 85, color='#2a8', alpha=0.10)
    ax.axvspan(20, 35, color='#cb3', alpha=0.10)
    ax.axvspan(0, 10, color='#c33', alpha=0.10)
    ax.axvspan(170, 179, color='#c33', alpha=0.10)
    ax.set_title('Per-crop median Hue — not_jockey')
    ax.set_xlabel('Hue (OpenCV H ∈ [0, 179])')
    ax.set_ylabel('crop count')
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    plot_path = CALIB_DIR / "hue_distribution_per_label.png"
    fig.savefig(str(plot_path), dpi=110)
    plt.close(fig)
    print(f"Plot: {plot_path}")
    print(f"  → http://100.122.27.103:8767/hsv_calibration/hue_distribution_per_label.png")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    if sys.argv[1] == "--show-ranges":
        show_ranges()
    elif sys.argv[1] == "--calibrate":
        calibrate()
    elif sys.argv[1] == "--gallery-lost":
        gallery_lost_greens()
    elif sys.argv[1] == "--gallery-diag":
        gallery_lost_greens_diag()
    elif sys.argv[1] == "--hue-distribution":
        hue_distribution()
    else:
        print(f"Unknown arg: {sys.argv[1]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
