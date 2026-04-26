#!/usr/bin/env python3
"""diagnose_color_classifier.py — посмотреть, куда смотрит
color_classifier_v4 на проблемных кропах из видео cam-13.

ШАГ 2 (sample frames): извлечь кадры с шагом N, сохранить в
output/diagnose_v4_<date>/raw_frames/. ROI cam-13 загружается, но
применяется на ШАГ 3 (фильтр детекций), не здесь.

Запуск шагов:
    python tools/diagnose_color_classifier.py --step 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent
VIDEO = ("/home/ipodrom/Рабочий стол/Ipodrom-Project/user/recordings/"
         "yaris_20260303_164835/kamera_13_164841_END165759.mp4")
ROI_JSON = REPO / "configs" / "camera_roi_normalized.json"
CAM_ID = "cam-13"
OUT_ROOT = REPO / "output" / "diagnose_v4_2026-04-25"
SAMPLE_STRIDE = 30   # каждый 30-й кадр (~1 fps на 30fps видео)

# YOLO для детекции — берём прод-веса из models/ (ONNX, person-only export)
YOLO_ONNX = REPO / "models" / "yolo11s_person_960.onnx"
DET_CONF = 0.25
DET_IOU  = 0.5
DET_IMGSZ = 960
CROP_MARGIN = 0.05   # 5% margin вокруг bbox (без сильного фона, но не впритык)
MIN_BBOX_H = 25      # фильтр совсем мелких боксов чтобы не шумить


def load_roi_cam13() -> list[tuple[float, float]] | None:
    if not ROI_JSON.is_file():
        return None
    data = json.loads(ROI_JSON.read_text())
    polys = data.get(CAM_ID)
    if not polys:
        return None
    poly = polys[0]
    return [(float(p["x"]), float(p["y"])) for p in poly]


def step2_sample_frames():
    out_dir = OUT_ROOT / "raw_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"FATAL: cannot open {VIDEO}", file=sys.stderr)
        sys.exit(2)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps if fps else 0
    print(f"Video: {VIDEO}")
    print(f"  total frames: {total}")
    print(f"  fps: {fps:.2f}")
    print(f"  resolution: {w}x{h}")
    print(f"  duration: {duration:.1f} s ({duration/60:.1f} min)")
    print(f"  expected samples (stride={SAMPLE_STRIDE}): ~{total // SAMPLE_STRIDE}")

    saved = 0
    frame_idx = 0
    saved_paths: list[Path] = []
    while True:
        ok = cap.grab()
        if not ok:
            break
        if frame_idx % SAMPLE_STRIDE == 0:
            ok2, frame = cap.retrieve()
            if ok2 and frame is not None:
                fp = out_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(fp), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                saved_paths.append(fp)
                saved += 1
        frame_idx += 1
    cap.release()

    # ROI hint
    roi = load_roi_cam13()
    print()
    print(f"Saved: {saved} frames into {out_dir}")
    print(f"ROI cam-13: {'loaded, ' + str(len(roi)) + ' points' if roi else 'NOT FOUND'}")

    # 10 примеров: 5 равномерно из всего диапазона + 5 случайных
    print()
    print("=== sample previews (10 frames spread across video) ===")
    if saved >= 10:
        idxs = [int(i * (saved - 1) / 9) for i in range(10)]
        for i in idxs:
            print(f"  {saved_paths[i]}")
    else:
        for fp in saved_paths:
            print(f"  {fp}")
    return saved, saved_paths


def step3_detect_and_crop():
    """Run YOLO person detection on all sampled frames, save crops with margin,
    generate gallery_all.html for manual labeling. NO ROI filter — все детекции."""
    raw_dir   = OUT_ROOT / "raw_frames"
    crops_dir = OUT_ROOT / "crops_raw"
    crops_dir.mkdir(parents=True, exist_ok=True)
    if not raw_dir.is_dir():
        print(f"FATAL: {raw_dir} missing. Run --step 2 first.", file=sys.stderr)
        sys.exit(2)
    if not YOLO_ONNX.is_file():
        print(f"FATAL: YOLO model not found at {YOLO_ONNX}", file=sys.stderr)
        sys.exit(2)

    from ultralytics import YOLO
    print(f"Loading YOLO: {YOLO_ONNX.name}")
    model = YOLO(str(YOLO_ONNX), task="detect")

    frame_paths = sorted(raw_dir.glob("frame_*.jpg"))
    print(f"Frames to scan: {len(frame_paths)}")

    rows: list[dict] = []
    for idx, fp in enumerate(frame_paths):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]
        results = model.predict(source=str(fp), imgsz=DET_IMGSZ,
                                conf=DET_CONF, iou=DET_IOU,
                                classes=[0], device=0, half=False, verbose=False)
        if not results:
            continue
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for det_i, (box, conf) in enumerate(zip(boxes, confs)):
            x1, y1, x2, y2 = box.tolist()
            bbox_h = y2 - y1
            if bbox_h < MIN_BBOX_H:
                continue
            # Margin вокруг bbox
            mw = (x2 - x1) * CROP_MARGIN
            mh = (y2 - y1) * CROP_MARGIN
            cx1 = max(0, int(x1 - mw))
            cy1 = max(0, int(y1 - mh))
            cx2 = min(w_img, int(x2 + mw))
            cy2 = min(h_img, int(y2 + mh))
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            crop = img[cy1:cy2, cx1:cx2]
            frame_id = fp.stem.split("_")[1]
            crop_name = f"frame_{frame_id}_det_{det_i:02d}.jpg"
            cv2.imwrite(str(crops_dir / crop_name), crop,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            rows.append({
                "frame": fp.name,
                "crop":  crop_name,
                "bbox":  [int(x1), int(y1), int(x2), int(y2)],
                "det_conf": float(conf),
                "h": int(bbox_h),
                "w": int(x2 - x1),
            })
        if (idx + 1) % 50 == 0:
            print(f"  processed {idx+1}/{len(frame_paths)} frames, crops so far: {len(rows)}")

    # Сохраняем индекс для последующих шагов
    (OUT_ROOT / "crops_index.json").write_text(json.dumps(rows, indent=2))

    # Generate gallery_all.html (4 колонки)
    gallery = OUT_ROOT / "gallery_all.html"
    write_gallery(gallery, rows, crops_dir.name)

    print()
    print(f"=== ШАГ 3 готово ===")
    print(f"  Crops saved: {len(rows)} → {crops_dir}")
    print(f"  Index:       {OUT_ROOT}/crops_index.json")
    print(f"  Gallery:     {gallery}")


def write_gallery(out_path: Path, rows: list[dict], crops_subdir: str):
    """Интерактивная HTML-сетка: 4 колонки, под каждым crop кнопки-метки.
    Метки сохраняются в localStorage; кнопка Export даёт готовый блок
    для копи-паста."""
    from html import escape
    LABELS = ["not_jockey", "blue", "green", "purple", "red", "yellow", "skip"]
    parts = [
        '<!doctype html><html lang="ru"><head><meta charset="utf-8">',
        '<title>diagnose v4 — label crops</title>',
        '<style>',
        '  body { font-family: system-ui, monospace; background:#1a1a1a; color:#ddd; margin:0; padding:12px; }',
        '  h1 { color:#fff; font-size:14px; margin:8px 0; }',
        '  .toolbar { position:sticky; top:0; background:#1a1a1a; padding:10px 0; z-index:50; '
        '             border-bottom:1px solid #444; margin-bottom:10px; }',
        '  .toolbar button { padding:6px 12px; margin-right:6px; cursor:pointer; '
        '                    background:#333; color:#fff; border:1px solid #555; border-radius:4px; }',
        '  .toolbar button:hover { background:#444; }',
        '  .stats { display:inline-block; margin-left:14px; color:#88ddff; font-size:12px; }',
        '  .grid { display:grid; grid-template-columns: repeat(4, 1fr); gap:8px; }',
        '  .card { background:#2a2a2a; border-radius:4px; padding:4px; transition:border 0.1s; '
        '          border:2px solid transparent; }',
        '  .card.labeled { border-color:#4a4; }',
        '  .card.not_jockey { border-color:#a44; }',
        '  .card img { max-width:100%; max-height:280px; display:block; margin:0 auto; }',
        '  .meta { font-size:11px; color:#aaa; padding:4px; word-break:break-all; }',
        '  .name { color:#88ddff; }',
        '  .labels { display:flex; flex-wrap:wrap; gap:3px; padding:4px; }',
        '  .labels label { font-size:11px; padding:3px 6px; background:#333; border-radius:3px; cursor:pointer; }',
        '  .labels label:hover { background:#555; }',
        '  .labels input { display:none; }',
        '  .labels input:checked + span { color:#fff; }',
        '  .labels input[value="not_jockey"]:checked + span { background:#a44; padding:2px 5px; border-radius:3px; }',
        '  .labels input[value="blue"]:checked    + span { background:#4af; padding:2px 5px; border-radius:3px; }',
        '  .labels input[value="green"]:checked   + span { background:#3a3; padding:2px 5px; border-radius:3px; }',
        '  .labels input[value="purple"]:checked  + span { background:#a4a; padding:2px 5px; border-radius:3px; }',
        '  .labels input[value="red"]:checked     + span { background:#a33; padding:2px 5px; border-radius:3px; }',
        '  .labels input[value="yellow"]:checked  + span { background:#aa3; padding:2px 5px; border-radius:3px; color:#000; }',
        '  .labels input[value="skip"]:checked    + span { background:#666; padding:2px 5px; border-radius:3px; }',
        '  textarea#export { width:100%; min-height:200px; background:#000; color:#0f0; '
        '                    font-family:monospace; padding:8px; }',
        '  .modal { display:none; position:fixed; top:5%; left:5%; right:5%; bottom:5%; '
        '           background:#222; padding:16px; border:1px solid #555; z-index:100; overflow:auto; }',
        '  .modal.active { display:block; }',
        '  .modal-close { position:absolute; top:8px; right:12px; cursor:pointer; color:#aaa; font-size:20px; }',
        '</style></head><body>',
        '<div class="toolbar">',
        '  <button onclick="exportLabels()">📋 Export labels</button>',
        '  <button onclick="clearAll()">🗑 Clear all</button>',
        '  <span class="stats" id="stats"></span>',
        '</div>',
        f'<h1>diagnose v4 — {len(rows)} crops</h1>',
        '<p style="color:#aaa;font-size:12px">Клик по метке = сохранение в браузере (localStorage). '
        'F5 не сбрасывает. Кнопка Export → готовый блок для копи-паста.</p>',
        '<div class="grid">',
    ]
    for r in rows:
        src = f"{crops_subdir}/{r['crop']}"
        crop_id = r['crop'].replace('.jpg', '')
        meta = (f"<span class='name'>{escape(r['crop'])}</span><br>"
                f"conf={r['det_conf']:.2f} · {r['h']}×{r['w']} px")
        labels_html = ''.join(
            f'<label><input type="radio" name="lbl_{crop_id}" value="{lbl}" '
            f'onchange="setLabel(\'{crop_id}\', \'{lbl}\')"><span>{lbl}</span></label>'
            for lbl in LABELS
        )
        parts.append(
            f'<div class="card" id="card_{crop_id}">'
            f'<img src="{escape(src)}" loading="lazy">'
            f'<div class="meta">{meta}</div>'
            f'<div class="labels">{labels_html}</div>'
            f'</div>'
        )
    parts.append('</div>')

    # Modal for export
    parts.append(
        '<div class="modal" id="exportModal">'
        '  <span class="modal-close" onclick="document.getElementById(\'exportModal\').classList.remove(\'active\')">✕</span>'
        '  <h2 style="color:#fff">Labels (copy this block)</h2>'
        '  <textarea id="export" readonly></textarea>'
        '</div>'
    )

    # Inline JS
    parts.append('''<script>
const STORAGE_KEY = 'diagnose_v4_labels';
function loadLabels() { try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}'); } catch(e){return{};} }
function saveLabels(o) { localStorage.setItem(STORAGE_KEY, JSON.stringify(o)); }

function setLabel(cropId, lbl) {
  const labels = loadLabels();
  labels[cropId] = lbl;
  saveLabels(labels);
  refreshCard(cropId, lbl);
  refreshStats();
}

function refreshCard(cropId, lbl) {
  const card = document.getElementById('card_' + cropId);
  if (!card) return;
  card.classList.remove('labeled', 'not_jockey');
  if (lbl === 'not_jockey') card.classList.add('not_jockey');
  else if (lbl) card.classList.add('labeled');
}

function refreshStats() {
  const labels = loadLabels();
  const counts = {};
  for (const v of Object.values(labels)) counts[v] = (counts[v] || 0) + 1;
  const total = Object.keys(labels).length;
  const grandTotal = document.querySelectorAll('.card').length;
  const parts = [`labeled ${total}/${grandTotal}`];
  for (const [k, v] of Object.entries(counts).sort()) parts.push(`${k}=${v}`);
  document.getElementById('stats').textContent = parts.join(' · ');
}

function exportLabels() {
  const labels = loadLabels();
  const groups = {};
  for (const [crop, lbl] of Object.entries(labels)) {
    if (lbl === 'skip') continue;
    if (!groups[lbl]) groups[lbl] = [];
    groups[lbl].push(crop);
  }
  let txt = '';
  for (const [lbl, crops] of Object.entries(groups).sort()) {
    crops.sort();
    txt += `${lbl}: [\\n  ${crops.join(',\\n  ')}\\n]\\n\\n`;
  }
  if (!txt) txt = '(no labels yet)';
  document.getElementById('export').value = txt;
  document.getElementById('exportModal').classList.add('active');
}

function clearAll() {
  if (!confirm('Удалить ВСЕ метки в localStorage?')) return;
  localStorage.removeItem(STORAGE_KEY);
  for (const card of document.querySelectorAll('.card')) {
    card.classList.remove('labeled', 'not_jockey');
    for (const r of card.querySelectorAll('input[type=radio]')) r.checked = false;
  }
  refreshStats();
}

// Init: restore labels from storage
window.addEventListener('DOMContentLoaded', () => {
  const labels = loadLabels();
  for (const [crop, lbl] of Object.entries(labels)) {
    const radios = document.getElementsByName('lbl_' + crop);
    for (const r of radios) if (r.value === lbl) r.checked = true;
    refreshCard(crop, lbl);
  }
  refreshStats();
});
</script>''')
    parts.append('</body></html>')
    out_path.write_text("\n".join(parts))


def step4_classifier_predict():
    """Прогнать color_classifier_v4.pt через ColorClassifierInfer на размеченных
    crops. Вывести таблицу + проверить гипотезу 'not_jockey → green > 0.9'."""
    sys.path.insert(0, str(REPO))
    import numpy as np

    labels_path = OUT_ROOT / "labels.json"
    crops_dir = OUT_ROOT / "crops_raw"
    if not labels_path.is_file():
        print(f"FATAL: {labels_path} missing — paste labels first.", file=sys.stderr)
        sys.exit(2)
    labels = json.loads(labels_path.read_text())

    # use the same loader the legacy/prod side uses
    from pipeline.trt_inference import ColorClassifierInfer
    pt_path = REPO / "models" / "color_classifier_v4.pt"
    print(f"Loading: {pt_path}")
    # CPU: системный cuDNN 9.21 не совместим с torch nightly bundled 9.20
    # для 420 crops через SimpleColorCNN это секунды, GPU не нужен
    cls = ColorClassifierInfer(fallback_pt=str(pt_path), device="cpu")
    print(f"Classes order: {cls.classes}")
    print(f"INPUT_SIZE:    {cls.INPUT_SIZE}")
    print()

    # collect all crops with labels
    rows = []
    for label, crop_ids in labels.items():
        for cid in crop_ids:
            crop_path = crops_dir / f"{cid}.jpg"
            if not crop_path.is_file():
                continue
            img_bgr = cv2.imread(str(crop_path))
            if img_bgr is None:
                continue
            rows.append({"id": cid, "label": label, "img": img_bgr})

    print(f"Loaded {len(rows)} labeled crops, running classifier ...")

    # batch through classifier
    BATCH = 64
    results = []
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i+BATCH]
        crops = [r["img"] for r in chunk]
        out = cls.classify_batch(crops)  # list of (color, conf%, prob_dict)
        for r, (pred_color, pred_conf, prob_dict) in zip(chunk, out):
            results.append({
                "id":         r["id"],
                "label":      r["label"],
                "pred":       pred_color,
                "pred_conf":  pred_conf,   # % per loader convention
                "probs":      prob_dict,   # {color: prob 0-1}
            })

    # Save raw results
    out_json = OUT_ROOT / "step4_predict.json"
    serializable = [{
        "id": r["id"], "label": r["label"], "pred": r["pred"],
        "pred_conf": float(r["pred_conf"]),
        "probs": {k: float(v) for k, v in r["probs"].items()},
    } for r in results]
    out_json.write_text(json.dumps(serializable, indent=2))

    # ── Аналитика ──
    print()
    print("=" * 80)
    print(f"=== ШАГ 4: classifier predictions vs my labels ({len(results)} crops) ===")
    print("=" * 80)

    # confusion matrix
    classes_pred = ["blue", "green", "purple", "red", "yellow"]
    classes_true = sorted(set(r["label"] for r in results))
    print()
    print(f"{'true \\ pred':14s}", *[f"{c:>9s}" for c in classes_pred])
    for tlabel in classes_true:
        subset = [r for r in results if r["label"] == tlabel]
        counts = {c: sum(1 for r in subset if r["pred"] == c) for c in classes_pred}
        line = f"{tlabel:14s}" + " ".join(f"{counts[c]:>9d}" for c in classes_pred)
        line += f"   total={len(subset)}"
        print(line)

    # CRITICAL: not_jockey → conf>0.9 distribution
    print()
    print("=" * 80)
    print("=== HYPOTHESIS: 'not_jockey crops get green > 0.9' ===")
    print("=" * 80)
    # ColorClassifierInfer возвращает softmax probability в [0..1]
    nj = [r for r in results if r["label"] == "not_jockey"]
    nj_green = [r for r in nj if r["pred"] == "green"]
    nj_green_high = [r for r in nj if r["pred"] == "green" and r["pred_conf"] >= 0.9]
    nj_green_med  = [r for r in nj if r["pred"] == "green" and r["pred_conf"] >= 0.7]
    print(f"not_jockey crops:                        {len(nj)}")
    print(f"  → predicted as green (any conf):       {len(nj_green)} ({len(nj_green)/len(nj)*100:.1f}%)")
    print(f"  → predicted as green AND conf >= 0.7:  {len(nj_green_med)} ({len(nj_green_med)/len(nj)*100:.1f}%)")
    print(f"  → predicted as green AND conf >= 0.9:  {len(nj_green_high)} ({len(nj_green_high)/len(nj)*100:.1f}%)")
    print()
    pred_distribution = {}
    for r in nj:
        pred_distribution[r["pred"]] = pred_distribution.get(r["pred"], 0) + 1
    print("Distribution of predictions on not_jockey:")
    for c, n in sorted(pred_distribution.items(), key=lambda x: -x[1]):
        print(f"  {c:8s}  {n:4d}  ({n/len(nj)*100:5.1f}%)")

    print()
    print("Confidence buckets for not_jockey (softmax prob 0..1):")
    buckets = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.001)]
    for lo, hi in buckets:
        n = sum(1 for r in nj if lo <= r["pred_conf"] < hi)
        marker = " ← high-conf gradcam target" if lo >= 0.9 else ""
        print(f"  [{lo:.2f}..{hi:.2f})  {n:4d}{marker}")

    # Verdict
    print()
    print("=" * 80)
    pct_green_high = len(nj_green_high) / len(nj) * 100
    pct_green_any  = len(nj_green) / len(nj) * 100
    if pct_green_high >= 10 or pct_green_any >= 50:
        print(f"VERDICT: hypothesis CONFIRMED")
        print(f"  {pct_green_any:.0f}% of not_jockey predicted as 'green'")
        print(f"  {pct_green_high:.0f}% of not_jockey predicted as 'green' with conf>=0.9")
        print("→ proceed to ШАГ 5 (Grad-CAM on green-predicted not_jockey crops)")
    else:
        print(f"VERDICT: hypothesis NOT CONFIRMED ({pct_green_any:.0f}% any-conf, {pct_green_high:.0f}% high-conf)")
    print("=" * 80)
    print()
    print(f"Raw predictions saved: {out_json}")


def step5_gradcam():
    """Grad-CAM heatmaps на проблемных crops (not_jockey → конкретный цвет)
    + reference real jockeys для сравнения. Hooks на SimpleColorCNN.features[6]
    (последний Conv2d 64→128). Без сторонних либ."""
    sys.path.insert(0, str(REPO))
    import numpy as np
    import torch
    import torch.nn.functional as F

    pred_path = OUT_ROOT / "step4_predict.json"
    if not pred_path.is_file():
        print(f"FATAL: {pred_path} missing — run --step 4 first", file=sys.stderr)
        sys.exit(2)
    predictions = json.loads(pred_path.read_text())

    # --- load model in eval mode на CPU (cuDNN mismatch на GPU)
    from pipeline.trt_inference import SimpleColorCNN
    pt_path = REPO / "models" / "color_classifier_v4.pt"
    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    classes = ckpt['classes']
    model = SimpleColorCNN(num_classes=len(classes))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded model. Classes: {classes}")

    # --- pick the last conv as Grad-CAM target ---
    target_layer = model.features[6]   # Conv2d 64→128
    print(f"Grad-CAM target: features[6] = {target_layer}")

    # --- pick crops to visualize ---
    by_id = {r["id"]: r for r in predictions}
    sections: list[tuple[str, str, list[str]]] = []  # (title, description, [crop_ids])

    # 1) not_jockey → green высокая уверенность (top 16 by conf)
    nj_green_high = sorted(
        [r for r in predictions if r["label"] == "not_jockey"
         and r["pred"] == "green" and r["pred_conf"] >= 0.9],
        key=lambda r: -r["pred_conf"],
    )[:16]
    sections.append((
        "🔴 not_jockey → GREEN с conf ≥ 0.9 (top 16 по уверенности)",
        "Если heatmap на одежде/теле — модель не видит цвет; на фоне — модель смотрит на газон.",
        [r["id"] for r in nj_green_high],
    ))

    # 2) not_jockey → yellow (всё, это КАМАЗ)
    nj_yellow = [r for r in predictions if r["label"] == "not_jockey" and r["pred"] == "yellow"]
    sections.append((
        "🚛 not_jockey → YELLOW (КАМАЗ-эпизод)",
        "Heatmap должен явно подсвечивать кузов машины — это финальная проверка теории.",
        [r["id"] for r in nj_yellow],
    ))

    # 3) not_jockey → purple
    nj_purple = sorted(
        [r for r in predictions if r["label"] == "not_jockey" and r["pred"] == "purple"],
        key=lambda r: -r["pred_conf"],
    )[:6]
    sections.append((
        "🟣 not_jockey → PURPLE (top 6)",
        "Покажет на какие тёмные/фиолетовые объекты модель залипает.",
        [r["id"] for r in nj_purple],
    ))

    # 4) reference: real jockeys (control group)
    for color in ["green", "yellow", "red"]:
        crops = sorted(
            [r for r in predictions if r["label"] == color and r["pred"] == color],
            key=lambda r: -r["pred_conf"],
        )[:3]
        sections.append((
            f"✅ REAL {color} jockey (control, top 3)",
            f"Heatmap здесь должен показать «правильное» поведение — фокус на торсе с {color} silk.",
            [r["id"] for r in crops],
        ))

    # --- compute Grad-CAM for each crop ---
    crops_dir = OUT_ROOT / "crops_raw"
    gradcam_dir = OUT_ROOT / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    INPUT_SIZE = 128
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    activations = {}
    gradients = {}
    def fwd_hook(_, __, output):  activations["x"] = output.detach()
    def bwd_hook(_, grad_in, grad_out): gradients["x"] = grad_out[0].detach()
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    n_done = 0
    for title, desc, ids in sections:
        for cid in ids:
            crop_path = crops_dir / f"{cid}.jpg"
            if not crop_path.is_file():
                continue
            bgr = cv2.imread(str(crop_path))
            if bgr is None:
                continue

            # preprocess EXACTLY как ColorClassifierInfer._preprocess_crop
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
            t = rgb_resized.astype(np.float32) / 255.0
            t = (t - MEAN) / STD
            x = torch.from_numpy(t).permute(2, 0, 1).unsqueeze(0)  # 1×3×128×128

            # forward + backward от логита predicted класса
            x.requires_grad_(False)
            model.zero_grad()
            logits = model(x)
            pred_idx = int(logits.argmax(dim=1).item())
            target_score = logits[0, pred_idx]
            target_score.backward()

            acts = activations["x"][0]   # [128, 32, 32]
            grads = gradients["x"][0]    # [128, 32, 32]
            weights = grads.mean(dim=(1, 2))      # [128]
            cam = (weights[:, None, None] * acts).sum(dim=0)   # [32, 32]
            cam = F.relu(cam)
            # normalize 0..1
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros_like(cam)
            cam_np = cam.cpu().numpy()

            # upscale до RAW crop size
            h_raw, w_raw = bgr.shape[:2]
            cam_full = cv2.resize(cam_np, (w_raw, h_raw))
            heatmap_color = cv2.applyColorMap((cam_full * 255).astype(np.uint8),
                                              cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(bgr, 0.55, heatmap_color, 0.45, 0)
            cv2.imwrite(str(gradcam_dir / f"{cid}.jpg"), overlay,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            n_done += 1
    h1.remove()
    h2.remove()
    print(f"Grad-CAM heatmaps written: {n_done} → {gradcam_dir}")

    # --- generate gallery_gradcam.html ---
    write_gradcam_gallery(
        OUT_ROOT / "gallery_gradcam.html",
        sections, by_id,
    )
    print(f"Gallery: {OUT_ROOT / 'gallery_gradcam.html'}")


def write_gradcam_gallery(out_path: Path, sections, predictions_by_id):
    from html import escape
    parts = [
        '<!doctype html><html lang="ru"><head><meta charset="utf-8">',
        '<title>diagnose v4 — Grad-CAM</title>',
        '<style>',
        '  body { font-family:system-ui; background:#1a1a1a; color:#ddd; margin:0; padding:14px; }',
        '  h1 { color:#fff; }',
        '  h2 { color:#ffaf3a; border-bottom:1px solid #444; padding:6px 0; margin-top:24px; }',
        '  .desc { color:#aaa; font-size:13px; margin:0 0 10px 0; }',
        '  table { border-collapse:collapse; width:100%; margin-bottom:10px; }',
        '  th { background:#2a2a2a; color:#88ddff; padding:6px; position:sticky; top:0; z-index:5;'
        '       font-size:12px; }',
        '  td { padding:4px; border:1px solid #333; vertical-align:top; background:#1f1f1f; }',
        '  td img { width:100%; max-height:300px; object-fit:contain; display:block; cursor:zoom-in; }',
        '  .meta { font-family:monospace; font-size:11px; color:#ccc; padding:6px; }',
        '  .meta b { color:#fff; }',
        '  .true { color:#aef; }',
        '  .pred { color:#fea; }',
        '  nav { position:sticky; top:0; background:#1a1a1a; padding:8px 0; z-index:50;'
        '        border-bottom:1px solid #444; margin-bottom:10px; }',
        '  nav a { color:#88ddff; margin-right:14px; text-decoration:none; font-size:13px; }',
        '</style></head><body>',
        '<h1>diagnose v4 — Grad-CAM heatmaps</h1>',
    ]
    parts.append('<nav>')
    for i, (title, _, _) in enumerate(sections):
        parts.append(f'<a href="#sec_{i}">{escape(title)}</a>')
    parts.append('</nav>')

    for i, (title, desc, ids) in enumerate(sections):
        parts.append(f'<h2 id="sec_{i}">{escape(title)}</h2>')
        parts.append(f'<p class="desc">{escape(desc)}</p>')
        parts.append('<table>')
        parts.append('<thead><tr>'
                     '<th style="width:30%">crop</th>'
                     '<th style="width:30%">Grad-CAM overlay</th>'
                     '<th style="width:40%">метаданные</th>'
                     '</tr></thead><tbody>')
        for cid in ids:
            r = predictions_by_id.get(cid)
            if not r:
                continue
            crop_src = f"crops_raw/{cid}.jpg"
            cam_src  = f"gradcam/{cid}.jpg"
            probs = r["probs"]
            probs_html = "<br>".join(
                f"{c}: {probs[c]:.3f}" for c in ["blue","green","purple","red","yellow"]
            )
            parts.append(
                f'<tr>'
                f'<td><a href="{crop_src}" target="_blank"><img src="{crop_src}" loading="lazy"></a></td>'
                f'<td><a href="{cam_src}"  target="_blank"><img src="{cam_src}"  loading="lazy"></a></td>'
                f'<td class="meta"><b>{cid}</b><br>'
                f'<span class="true">label = {r["label"]}</span><br>'
                f'<span class="pred">pred  = {r["pred"]} (conf={r["pred_conf"]:.3f})</span><br><br>'
                f'<span style="color:#888">all softmax probs:</span><br>'
                f'{probs_html}</td>'
                f'</tr>'
            )
        parts.append('</tbody></table>')
    parts.append('</body></html>')
    out_path.write_text("\n".join(parts))


def regen_gallery_from_index():
    """Перегенерация HTML из существующего crops_index.json (без re-run YOLO)."""
    idx_path = OUT_ROOT / "crops_index.json"
    if not idx_path.is_file():
        print(f"FATAL: {idx_path} missing — run --step 3 first.", file=sys.stderr)
        sys.exit(2)
    rows = json.loads(idx_path.read_text())
    out = OUT_ROOT / "gallery_all.html"
    write_gallery(out, rows, "crops_raw")
    print(f"Gallery regenerated: {out} ({len(rows)} crops)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True, choices=[2, 3, 4, 5, 99])
    args = ap.parse_args()
    if args.step == 2:
        step2_sample_frames()
    elif args.step == 3:
        step3_detect_and_crop()
    elif args.step == 4:
        step4_classifier_predict()
    elif args.step == 5:
        step5_gradcam()
    elif args.step == 99:
        regen_gallery_from_index()
    else:
        print(f"step {args.step} not implemented yet")
        sys.exit(1)


if __name__ == "__main__":
    main()
