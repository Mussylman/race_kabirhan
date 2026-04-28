"""
test_osnet_sgie_parity.py — ЭТАП 3.4 — C++ parser logic vs Python reference.

Цель: убедиться что parser, как он сейчас собран в libnvdsinfer_racevision.so,
выдаёт тот же class_id что и Python reference на тех же embeddings.

Стратегия (т.к. NvDsInferClassifierParseCustomOsnet использует std::vector
в сигнатуре, ctypes-вызов невозможен напрямую):
  1. Парсим kOsnetProtos[4][512] напрямую из osnet_prototypes_v1.h
     → сравниваем поэлементно с prototypes_osnet_combined.npz
     (это гарантирует что C++ parser работает с теми же числами)
  2. Faithful Python port логики C++ parser (L2-norm + cosine + argmax +
     threshold + class_id mapping) — это ровно то что выполняется в .so.
     Запускаем на 100 TRT embeddings.
  3. Reference Python: ту же math но через np.linalg + npz prototypes.
  4. Сравниваем class_ids: agreement >= 99% (target).

Эмбеддинги генерируются через тот же TRT engine
(osnet_x1_0_market_b1_gpu0_fp16.engine) — production-realistic FP16 path.
"""
from __future__ import annotations
import json
import random
import re
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

REPO = Path(__file__).resolve().parent.parent
ROOT_BENCH = Path('/home/ipodrom/race_vision_bench')
DATA = ROOT_BENCH / 'dinov2_prod' / 'multicam_collection'
HEADER = REPO / 'deepstream' / 'src' / 'osnet_prototypes_v1.h'
NPZ = ROOT_BENCH / 'osnet_test' / 'prototypes_osnet_combined.npz'
ENGINE = ROOT_BENCH / 'osnet_test' / 'integration' / 'osnet_x1_0_market_b1_gpu0_fp16.engine'

SEED = 42
N_CROPS = 100  # balanced ~25 per class
PER_CLASS = 25

# labels_color.txt: blue=0, green=1, purple=2, red=3, yellow=4
LABELS_COLOR_TXT_INDEX = {"blue": 0, "green": 1, "purple": 2, "red": 3, "yellow": 4}

# Parser env defaults (from osnet_color_parser.cpp)
RV_OSNET_MIN_SIM = 0.75
COLOR_UNKNOWN_ID = 255


# ============================================================
#                    1. Parse the C++ header
# ============================================================
def parse_header(path: Path) -> tuple[list[str], list[int], np.ndarray]:
    """Extract kOsnetProtoLabels, kOsnetClassMapping, kOsnetProtos from .h."""
    text = path.read_text()

    # Labels
    m = re.search(r'kOsnetProtoLabels\[OSNET_NUM_PROTOS\]\s*=\s*\{([^}]+)\}', text, re.S)
    assert m, 'failed to find kOsnetProtoLabels'
    labels = [s.strip().strip('"') for s in m.group(1).split(',') if s.strip().strip('"')]

    # Class mapping
    m = re.search(r'kOsnetClassMapping\[OSNET_NUM_PROTOS\]\s*=\s*\{([^}]+)\}', text, re.S)
    assert m, 'failed to find kOsnetClassMapping'
    body = m.group(1)
    mapping = [int(x) for x in re.findall(r'(\d+)u', body)]

    # Prototypes — extract floats inside kOsnetProtos { ... };
    m = re.search(r'kOsnetProtos\[OSNET_NUM_PROTOS\]\[OSNET_DIM\]\s*=\s*\{(.+?)\};', text, re.S)
    assert m, 'failed to find kOsnetProtos'
    floats_text = m.group(1)
    float_pat = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?f')
    raw = float_pat.findall(floats_text)
    vals = [float(s.rstrip('f')) for s in raw]
    assert len(vals) == 4 * 512, f'expected {4*512} floats, got {len(vals)}'
    arr = np.array(vals, dtype=np.float32).reshape(4, 512)

    return labels, mapping, arr


# ============================================================
#               2. C++ parser logic — faithful port
# ============================================================
def cpp_parser_logic(emb: np.ndarray, protos: np.ndarray, mapping: list[int],
                     min_sim: float = RV_OSNET_MIN_SIM,
                     classifier_threshold: float = 0.0) -> tuple[int, float, int]:
    """
    Faithful port of NvDsInferClassifierParseCustomOsnet from
    deepstream/src/osnet_color_parser.cpp.

    Returns (class_id, max_sim, best_proto_idx).
    """
    # Step 1: L2 norm
    sumsq = float((emb * emb).sum())
    norm = float(np.sqrt(sumsq) + 1e-12)
    # Step 2: cosine sim to 4 prototypes (already L2-normalized)
    sims = []
    for p in range(protos.shape[0]):
        dot = float((emb * protos[p]).sum())
        sims.append(dot / norm)
    # Step 3: argmax
    best = 0
    for p in range(1, len(sims)):
        if sims[p] > sims[best]:
            best = p
    max_sim = sims[best]
    # Step 4: reject path
    effective_thr = max(min_sim, classifier_threshold)
    if max_sim < effective_thr:
        return (COLOR_UNKNOWN_ID, max_sim, best)
    # Step 5: map to labels_color.txt id
    return (mapping[best], max_sim, best)


# ============================================================
#               3. Python reference (npz prototypes)
# ============================================================
def python_reference(emb: np.ndarray, protos_npz: np.ndarray, classes_npz: list[str],
                     min_sim: float = RV_OSNET_MIN_SIM) -> tuple[int, float]:
    """Same math but via np.linalg directly with npz protos.
    Returns (class_id, max_sim)."""
    e = emb / max(np.linalg.norm(emb), 1e-12)
    sims = protos_npz @ e   # protos already L2-normalized
    best = int(sims.argmax())
    max_sim = float(sims[best])
    if max_sim < min_sim:
        return (COLOR_UNKNOWN_ID, max_sim)
    cls = classes_npz[best]
    return (LABELS_COLOR_TXT_INDEX[cls], max_sim)


# ============================================================
#                       4. TRT runner
# ============================================================
class TRTRunner:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.io = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            self.io.append({'name': name, 'mode': mode, 'shape': tuple(shape), 'dtype': dtype})
        self.host = {}
        self.dev = {}
        for io in self.io:
            n_elems = int(np.prod(io['shape']))
            self.host[io['name']] = cuda.pagelocked_empty(n_elems, io['dtype'])
            self.dev[io['name']] = cuda.mem_alloc(self.host[io['name']].nbytes)
            self.ctx.set_tensor_address(io['name'], int(self.dev[io['name']]))
        self.stream = cuda.Stream()
        self.in_name = next(io['name'] for io in self.io if io['mode'] == trt.TensorIOMode.INPUT)
        self.out_name = next(io['name'] for io in self.io if io['mode'] == trt.TensorIOMode.OUTPUT)
        self.out_shape = next(io['shape'] for io in self.io if io['mode'] == trt.TensorIOMode.OUTPUT)

    def infer(self, x: np.ndarray) -> np.ndarray:
        flat = np.ascontiguousarray(x, dtype=np.float32).ravel()
        np.copyto(self.host[self.in_name], flat)
        cuda.memcpy_htod_async(self.dev[self.in_name], self.host[self.in_name], self.stream)
        self.ctx.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.host[self.out_name], self.dev[self.out_name], self.stream)
        self.stream.synchronize()
        return self.host[self.out_name].reshape(self.out_shape).copy()


# ============================================================
#                            Main
# ============================================================
def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print('=== STEP 1: parse osnet_prototypes_v1.h ===')
    h_labels, h_mapping, h_protos = parse_header(HEADER)
    print(f'  labels: {h_labels}')
    print(f'  mapping: {h_mapping}')
    print(f'  protos.shape: {h_protos.shape}')
    norms = np.linalg.norm(h_protos, axis=1)
    print(f'  norms: min={norms.min():.6f} max={norms.max():.6f}')

    npz = np.load(NPZ)
    npz_protos_raw = npz['prototypes']
    npz_classes = [str(c) for c in npz['classes']]
    print(f'  npz classes: {npz_classes}')

    # Reorder npz to match header order
    src_idx = [npz_classes.index(c) for c in h_labels]
    npz_protos_reordered = npz_protos_raw[src_idx]

    diff = np.abs(h_protos - npz_protos_reordered)
    print(f'  header vs npz: max abs = {diff.max():.4e}, mean = {diff.mean():.4e}')
    assert diff.max() <= 1e-6, f'header drift too big: {diff.max()}'
    print('  → header values match npz within FP32 print precision\n')

    print('=== STEP 2: load TRT engine ===')
    runner = TRTRunner(str(ENGINE))
    print(f'  engine input: {runner.in_name}, output: {runner.out_name} {runner.out_shape}\n')

    print(f'=== STEP 3: build {N_CROPS} balanced TRT embeddings ===')
    with open(DATA / 'labels.json') as f:
        labels = json.load(f)
    chosen = []
    for c in ['blue', 'green', 'red', 'yellow']:
        ids = list(labels[c])
        random.shuffle(ids)
        for cid in ids[:PER_CLASS]:
            chosen.append({'id': cid, 'gt': c})
    random.shuffle(chosen)
    print(f'  {len(chosen)} crops')

    preprocess = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embs = []
    for s in chosen:
        img = Image.open(DATA / 'crops_raw' / f'{s["id"]}.jpg').convert('RGB')
        x = preprocess(img).unsqueeze(0).numpy().astype(np.float32)
        emb = runner.infer(x)[0]
        embs.append(emb.astype(np.float32))
    embs = np.stack(embs)
    print(f'  embeddings shape: {embs.shape}\n')

    print('=== STEP 4: Run C++-port parser vs Python-reference ===')
    cpp_ids, ref_ids = [], []
    cpp_sims, ref_sims = [], []
    for emb in embs:
        cpp_id, cpp_sim, _ = cpp_parser_logic(emb, h_protos, h_mapping)
        ref_id, ref_sim = python_reference(emb, npz_protos_raw, npz_classes)
        cpp_ids.append(cpp_id)
        ref_ids.append(ref_id)
        cpp_sims.append(cpp_sim)
        ref_sims.append(ref_sim)
    cpp_ids = np.array(cpp_ids)
    ref_ids = np.array(ref_ids)
    same = int((cpp_ids == ref_ids).sum())
    pct = 100 * same / len(embs)
    max_sim_diff = float(np.abs(np.array(cpp_sims) - np.array(ref_sims)).max())
    print(f'  agreement: {same}/{len(embs)} = {pct:.2f}%')
    print(f'  max sim diff (cpp_logic vs ref): {max_sim_diff:.6e}')

    # Class distribution
    print('\n=== Class distribution (C++ parser logic on TRT embeddings) ===')
    from collections import Counter
    cnt = Counter(cpp_ids)
    inv = {v: k for k, v in LABELS_COLOR_TXT_INDEX.items()}
    inv[COLOR_UNKNOWN_ID] = 'unknown'
    for cid in sorted(cnt.keys()):
        n = cnt[cid]
        print(f'  class_id={cid} ({inv.get(cid, "?")}): {n} ({100*n/len(embs):.1f}%)')

    # Per-GT accuracy
    print('\n=== Per-GT accuracy (C++ logic) ===')
    for gt_color in ['blue', 'green', 'red', 'yellow']:
        idx = [i for i, s in enumerate(chosen) if s['gt'] == gt_color]
        gt_id = LABELS_COLOR_TXT_INDEX[gt_color]
        correct = sum(1 for i in idx if cpp_ids[i] == gt_id)
        rejected = sum(1 for i in idx if cpp_ids[i] == COLOR_UNKNOWN_ID)
        print(f'  GT {gt_color:<7}: {correct}/{len(idx)} correct ({100*correct/len(idx):.1f}%), '
              f'{rejected} rejected')

    print('\n=== Verdict ===')
    pass_parity = pct >= 99.0
    pass_sim = max_sim_diff <= 1e-5
    print(f'  Agreement >= 99%:  {"PASS" if pass_parity else "FAIL"} ({pct:.2f}%)')
    print(f'  Sim diff <= 1e-5:  {"PASS" if pass_sim else "FAIL"} ({max_sim_diff:.4e})')

    out = ROOT_BENCH / 'osnet_test' / 'integration' / 'sgie_parity_results.json'
    with open(out, 'w') as f:
        json.dump({
            'header_npz_max_diff': float(diff.max()),
            'n_crops': len(embs),
            'agreement_pct': pct,
            'agreement_count': same,
            'max_sim_diff': max_sim_diff,
            'class_distribution': {inv.get(int(k), '?'): int(v) for k, v in cnt.items()},
            'verdict_parity_pass': pass_parity,
            'verdict_sim_pass': pass_sim,
        }, f, indent=2)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
