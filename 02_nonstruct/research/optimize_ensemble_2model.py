"""
Optimize 2-model ensembling (Nearest + Model B) on OOF.

Goal
----
Move closer to Oracle-2 using ONLY the existing two models.

We implement, evaluate, and export:
1) Fixed blend (wN/wB)
2) Class-wise logit-channel weights (wB per class channel)
3) Disagree-only gating (rule-based v0)

All metrics use the SAME definition:
- global confusion over all OOF pixels (ignore=255 masked)
- per-class IoU -> nanmean (classes with union==0 are NaN and excluded)

Outputs
-------
Writes to: 00_data/output/e3_nearest_plus_b_opt/
- metrics.csv                      (global mIoU for each method)
- selection_stats.csv              (how often gating selects B and its "good vs bad" split)
- per_class_iou_<method>.csv       (per-class IoU for each method)
- notes.md                         (copy-paste friendly summary)

Data contract (this repo's current layout)
------------------------------------------
- Nearest:
  - 01_nearest/golden_artifacts/oof_logits.npy          (N,13,480,640) float32
  - 01_nearest/golden_artifacts/oof_file_ids.npy        (N,) strings like "000002.png"
- Model B:
  - 00_data/02_nonstruct_frozen/golden_artifacts/oof/oof_fold{k}_logits.npy
- Splits:
  - 00_data/splits/folds_v1.json (manifest listing fold*.json)
  - 00_data/splits/fold{k}.json contain train_ids/val_ids (ids without .png)
- GT:
  - 00_data/train/label/{id}.png
  - (optional) 00_data/train/depth/{id}.png for depth-aware gating
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


NUM_CLASSES = 13
IGNORE_INDEX = 255
CLASS_NAMES = [
    "bed",
    "books",
    "ceiling",
    "chair",
    "floor",
    "furniture",
    "objects",
    "picture",
    "sofa",
    "table",
    "tv",
    "wall",
    "window",
]

CID_BOOKS = 1
CID_TABLE = 9
CID_OBJECTS = 6
CID_WALL = 11
CID_FURNITURE = 5


def _die(msg: str) -> None:
    raise SystemExit(f"[E3-OPT][FATAL] {msg}")


def _ensure(path: str, desc: str) -> None:
    if not os.path.exists(path):
        _die(f"Missing {desc}: {path}")


def _maybe_cv2() -> None:
    if cv2 is None:
        _die("OpenCV (cv2) is required. Install opencv-python.")


def norm_id(x: str) -> str:
    s = str(x).strip()
    return s if s.endswith(".png") else s + ".png"


def load_label(label_dir: str, file_id: str) -> np.ndarray:
    _maybe_cv2()
    p = os.path.join(label_dir, file_id)
    gt = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if gt is None:
        _die(f"Failed to read GT: {p}")
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    return gt.astype(np.int64, copy=False)


def load_depth_m(depth_dir: Optional[str], file_id: str) -> Optional[np.ndarray]:
    if not depth_dir:
        return None
    _maybe_cv2()
    p = os.path.join(depth_dir, file_id)
    d = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    return (d.astype(np.float32) / 1000.0).astype(np.float32, copy=False)


def confusion_update(cm: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> None:
    gt_i = gt.astype(np.int64, copy=False)
    pr_i = pred.astype(np.int64, copy=False)
    m = (gt_i != IGNORE_INDEX) & (gt_i >= 0) & (gt_i < NUM_CLASSES)
    if not np.any(m):
        return
    idx = gt_i[m] * NUM_CLASSES + pr_i[m]
    cm += np.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)


def iou_from_cm(cm: np.ndarray) -> List[float]:
    ious: List[float] = []
    for c in range(NUM_CLASSES):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - cm[c, c])
        fn = float(cm[c, :].sum() - cm[c, c])
        u = tp + fp + fn
        ious.append(float("nan") if u <= 0 else tp / u)
    return ious


def miou_from_cm(cm: np.ndarray) -> float:
    vals = [x for x in iou_from_cm(cm) if not (math.isnan(x) or math.isinf(x))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def write_csv(path: str, header: List[str], rows: List[List[object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


@dataclass
class OOFIndexB:
    fold_logits: Dict[int, np.ndarray]
    id_to_pos: Dict[str, Tuple[int, int]]  # file_id -> (fold, idx_in_fold)


def load_splits_val_ids_by_fold(splits_manifest: str) -> List[List[str]]:
    with open(splits_manifest, "r", encoding="utf-8") as f:
        obj = json.load(f)
    folds = obj.get("folds")
    if not isinstance(folds, list):
        _die(f"Expected splits manifest format with 'folds': {splits_manifest}")
    base = os.path.dirname(splits_manifest)
    out: List[List[str]] = []
    for fname in folds:
        fpath = os.path.join(base, str(fname))
        with open(fpath, "r", encoding="utf-8") as ff:
            fd = json.load(ff)
        val_ids = fd.get("val_ids")
        if not isinstance(val_ids, list):
            _die(f"Missing val_ids in {fpath}")
        out.append([norm_id(x) for x in val_ids])
    if len(out) != 5:
        _die(f"Expected 5 folds, got {len(out)}")
    return out


def build_b_index(b_oof_dir: str, val_ids_by_fold: List[List[str]]) -> OOFIndexB:
    fold_logits: Dict[int, np.ndarray] = {}
    id_to_pos: Dict[str, Tuple[int, int]] = {}
    for fold in range(5):
        p = os.path.join(b_oof_dir, f"oof_fold{fold}_logits.npy")
        _ensure(p, f"Model B fold{fold} logits")
        arr = np.load(p, mmap_mode="r", allow_pickle=False)
        if arr.shape[1:] != (NUM_CLASSES, 480, 640):
            _die(f"Unexpected B logits shape fold{fold}: {arr.shape}")
        fold_logits[fold] = arr
        ids = val_ids_by_fold[fold]
        if len(ids) != arr.shape[0]:
            _die(f"Fold{fold}: val_ids {len(ids)} != logits N {arr.shape[0]}")
        for i, fid in enumerate(ids):
            if fid in id_to_pos:
                _die(f"Duplicate id across folds: {fid}")
            id_to_pos[fid] = (fold, i)
    return OOFIndexB(fold_logits=fold_logits, id_to_pos=id_to_pos)


def logits_b_for_id(b: OOFIndexB, file_id: str) -> np.ndarray:
    fold, idx = b.id_to_pos[file_id]
    return np.array(b.fold_logits[fold][idx], copy=False)  # (13,480,640) float16


def classwise_mix_logits(n: np.ndarray, b: np.ndarray, wB: np.ndarray) -> np.ndarray:
    """
    wB: (13,) in [0,1], applied per channel:
      mix[c] = (1-wB[c])*n[c] + wB[c]*b[c]
    """
    w = wB.astype(np.float32).reshape(NUM_CLASSES, 1, 1)
    return (n.astype(np.float32, copy=False) * (1.0 - w) + b.astype(np.float32, copy=False) * w).astype(np.float32, copy=False)

def quantize_wB_3level(wB: np.ndarray) -> np.ndarray:
    """
    Quantize weights into {0.00, 0.15, 0.30} after clipping to [0, 0.35].
    Boundaries chosen to reduce overfitting and keep B influence conservative:
      - <0.075 -> 0.00
      - [0.075, 0.225) -> 0.15
      - >=0.225 -> 0.30
    """
    w = np.clip(wB.astype(np.float32, copy=False), 0.0, 0.35)
    out = np.zeros_like(w)
    out[(w >= 0.075) & (w < 0.225)] = 0.15
    out[w >= 0.225] = 0.30
    return out


def top2_margin(logits_chw: np.ndarray) -> np.ndarray:
    """Return margin = top1 - top2 per pixel, using np.partition (C small)."""
    x = logits_chw.astype(np.float32, copy=False)
    part = np.partition(x, kth=[NUM_CLASSES - 2, NUM_CLASSES - 1], axis=0)
    top2 = part[NUM_CLASSES - 2]
    top1 = part[NUM_CLASSES - 1]
    return (top1 - top2).astype(np.float32, copy=False)

def pmax_from_logits_at(
    logits_chw: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
) -> np.ndarray:
    """
    Compute max softmax probability at selected pixels only.
    logits_chw: (C,H,W)
    ys,xs: (K,)
    returns: (K,) float32
    """
    x = logits_chw[:, ys, xs].astype(np.float32, copy=False)  # (C,K)
    m = np.max(x, axis=0)  # (K,)
    # pmax = exp(m) / sum exp(x) = 1 / sum exp(x-m)
    denom = np.exp(x - m[None, :]).sum(axis=0)
    return (1.0 / denom).astype(np.float32, copy=False)

def margin_from_logits_at(
    logits_chw: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
) -> np.ndarray:
    """
    Compute margin top1-top2 at selected pixels only.
    """
    x = logits_chw[:, ys, xs].astype(np.float32, copy=False)  # (C,K)
    part = np.partition(x, kth=[NUM_CLASSES - 2, NUM_CLASSES - 1], axis=0)
    top2 = part[NUM_CLASSES - 2]
    top1 = part[NUM_CLASSES - 1]
    return (top1 - top2).astype(np.float32, copy=False)

def boundary_mask_from_gt(gt: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Same policy as 01_nearest eval protocol:
      - replace 255->0 for morph ops
      - boundary = dilate != erode (k=3)
      - mask by gt!=255
    """
    _maybe_cv2()
    gt_u8 = gt.astype(np.uint8, copy=False)
    valid = (gt_u8 != IGNORE_INDEX)
    gt_clean = gt_u8.copy()
    gt_clean[~valid] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    d = cv2.dilate(gt_clean, kernel)
    e = cv2.erode(gt_clean, kernel)
    return (d != e) & valid

def depth_bin_at(depth_m: Optional[np.ndarray], ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """
    Bin edges fixed to E3: [0,1,2,3,5,10]
    Returns integer bin id in [0..4], or -1 if depth unavailable/invalid.
    """
    if depth_m is None:
        return -np.ones_like(ys, dtype=np.int64)
    d = depth_m[ys, xs]
    # invalid depth -> -1
    out = -np.ones_like(d, dtype=np.int64)
    m = d > 0
    # bins: 0-1,1-2,2-3,3-5,5-10
    edges = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 10.0], dtype=np.float32)
    # np.digitize returns 1..len(edges) for right=False; we want 0..4
    out[m] = np.clip(np.digitize(d[m], edges, right=False) - 1, 0, 4)
    return out.astype(np.int64, copy=False)

@dataclass
class LRModel:
    """
    Lightweight logistic regression with categorical offsets:
      score = b
            + w_pmaxN*pmaxN + w_pmaxB*pmaxB + w_mN*mN + w_mB*mB
            + w_boundary*boundary
            + w_depth[depth_bin] (depth_bin=-1 => 0)
            + w_predN[predN]
            + w_predB[predB]
    beta = sigmoid(score) = P(B correct)
    """
    b: float
    w_pmaxN: float
    w_pmaxB: float
    w_mN: float
    w_mB: float
    w_boundary: float
    w_depth: np.ndarray   # (5,)
    w_predN: np.ndarray   # (13,)
    w_predB: np.ndarray   # (13,)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

@dataclass
class PushLRModel:
    """
    Logistic regression for "push usefulness":
      y=1 if pushing L0 by eps*d fixes an error (and doesn't break correct).

    score = b
          + w_pmax0*pmax0 + w_m0*m0
          + w_pmaxN*pmaxN + w_pmaxB*pmaxB
          + w_mN*mN + w_mB*mB
          + w_dtop1*d_top1 + w_dgap*d_gap
          + w_boundary*boundary
          + w_depth[depth_bin]
          + w_pred0[pred0] + w_predB[predB]
    beta = sigmoid(score) = P(push improves).
    """
    b: float
    w_pmax0: float
    w_m0: float
    w_pmaxN: float
    w_pmaxB: float
    w_mN: float
    w_mB: float
    w_dtop1: float
    w_dgap: float
    w_boundary: float
    w_depth: np.ndarray   # (5,)
    w_pred0: np.ndarray   # (13,)
    w_predB: np.ndarray   # (13,)

def push_lr_train(
    samples: Dict[str, np.ndarray],
    *,
    l2: float = 1.0,
    lr: float = 0.05,
    epochs: int = 3,
    seed: int = 42,
    neg_keep_prob: float = 0.25,
) -> PushLRModel:
    """
    Train PushLRModel with Adam, with negative downsampling (to handle rare positives).
    samples keys:
      y (0/1), pmax0, m0, pmaxN, pmaxB, mN, mB, d_top1, d_gap, boundary(0/1),
      depth_bin(-1..4), pred0, predB
    """
    rng = np.random.default_rng(seed)
    y = samples["y"].astype(np.float32)

    # downsample negatives
    pos = (y > 0.5)
    neg = ~pos
    if np.any(neg):
        keep_neg = rng.random(neg.sum()) < float(neg_keep_prob)
        keep = pos.copy()
        keep[neg] = keep_neg
    else:
        keep = np.ones_like(y, dtype=bool)

    def sel(name: str, dtype=None):
        arr = samples[name][keep]
        return arr.astype(dtype, copy=False) if dtype is not None else arr

    y = sel("y", np.float32)
    pmax0 = sel("pmax0", np.float32)
    m0 = sel("m0", np.float32)
    pmaxN = sel("pmaxN", np.float32)
    pmaxB = sel("pmaxB", np.float32)
    mN = sel("mN", np.float32)
    mB = sel("mB", np.float32)
    d_top1 = sel("d_top1", np.float32)
    d_gap = sel("d_gap", np.float32)
    boundary = sel("boundary", np.float32)
    depth_bin = sel("depth_bin", np.int64)
    pred0 = sel("pred0", np.int64)
    predB = sel("predB", np.int64)

    n = y.shape[0]
    if n == 0:
        _die("No samples left after negative downsampling.")

    # params init
    b = 0.0
    w_pmax0 = w_m0 = 0.0
    w_pmaxN = w_pmaxB = 0.0
    w_mN = w_mB = 0.0
    w_dtop1 = w_dgap = 0.0
    w_boundary = 0.0
    w_depth = np.zeros((5,), dtype=np.float32)
    w_pred0 = np.zeros((NUM_CLASSES,), dtype=np.float32)
    w_predB = np.zeros((NUM_CLASSES,), dtype=np.float32)

    # Adam state
    def adam_init(shape=()):
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

    mb, vb = adam_init()
    mp0, vp0 = adam_init()
    mm0, vm0 = adam_init()
    mpn, vpn = adam_init()
    mpb, vpb = adam_init()
    mmn, vmn = adam_init()
    mmb, vmb = adam_init()
    md1, vd1 = adam_init()
    mdg, vdg = adam_init()
    mbr, vbr = adam_init()
    mdep, vdep = adam_init(w_depth.shape)
    mp0C, vp0C = adam_init(w_pred0.shape)
    mpbC, vpbC = adam_init(w_predB.shape)

    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    step = 0
    batch = min(200_000, n)

    for _ep in range(int(epochs)):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            step += 1
            idx = order[s : s + batch]

            yy = y[idx]
            sc = (
                b
                + w_pmax0 * pmax0[idx]
                + w_m0 * m0[idx]
                + w_pmaxN * pmaxN[idx]
                + w_pmaxB * pmaxB[idx]
                + w_mN * mN[idx]
                + w_mB * mB[idx]
                + w_dtop1 * d_top1[idx]
                + w_dgap * d_gap[idx]
                + w_boundary * boundary[idx]
                + w_pred0[pred0[idx]]
                + w_predB[predB[idx]]
            )
            db = depth_bin[idx]
            mvalid = db >= 0
            if np.any(mvalid):
                sc[mvalid] += w_depth[db[mvalid]]

            p = sigmoid(sc.astype(np.float32, copy=False)).astype(np.float32, copy=False)
            g = (p - yy).astype(np.float32, copy=False)

            gb = float(g.mean())
            gw_pmax0 = float((g * pmax0[idx]).mean())
            gw_m0 = float((g * m0[idx]).mean())
            gw_pmaxN = float((g * pmaxN[idx]).mean())
            gw_pmaxB = float((g * pmaxB[idx]).mean())
            gw_mN = float((g * mN[idx]).mean())
            gw_mB = float((g * mB[idx]).mean())
            gw_dtop1 = float((g * d_top1[idx]).mean())
            gw_dgap = float((g * d_gap[idx]).mean())
            gw_boundary = float((g * boundary[idx]).mean())

            gw_pred0 = np.zeros_like(w_pred0)
            gw_predB = np.zeros_like(w_predB)
            np.add.at(gw_pred0, pred0[idx], g)
            np.add.at(gw_predB, predB[idx], g)
            gw_pred0 /= float(len(idx))
            gw_predB /= float(len(idx))

            gw_depth = np.zeros_like(w_depth)
            if np.any(mvalid):
                np.add.at(gw_depth, db[mvalid], g[mvalid])
                gw_depth /= float(len(idx))

            # L2
            gb += l2 * b
            gw_pmax0 += l2 * w_pmax0
            gw_m0 += l2 * w_m0
            gw_pmaxN += l2 * w_pmaxN
            gw_pmaxB += l2 * w_pmaxB
            gw_mN += l2 * w_mN
            gw_mB += l2 * w_mB
            gw_dtop1 += l2 * w_dtop1
            gw_dgap += l2 * w_dgap
            gw_boundary += l2 * w_boundary
            gw_depth += l2 * w_depth
            gw_pred0 += l2 * w_pred0
            gw_predB += l2 * w_predB

            def adam_step(param, grad, m, v):
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad * grad)
                mhat = m / (1 - beta1**step)
                vhat = v / (1 - beta2**step)
                param = param - lr * mhat / (np.sqrt(vhat) + eps)
                return param, m, v

            b, mb, vb = adam_step(b, gb, mb, vb)
            w_pmax0, mp0, vp0 = adam_step(w_pmax0, gw_pmax0, mp0, vp0)
            w_m0, mm0, vm0 = adam_step(w_m0, gw_m0, mm0, vm0)
            w_pmaxN, mpn, vpn = adam_step(w_pmaxN, gw_pmaxN, mpn, vpn)
            w_pmaxB, mpb, vpb = adam_step(w_pmaxB, gw_pmaxB, mpb, vpb)
            w_mN, mmn, vmn = adam_step(w_mN, gw_mN, mmn, vmn)
            w_mB, mmb, vmb = adam_step(w_mB, gw_mB, mmb, vmb)
            w_dtop1, md1, vd1 = adam_step(w_dtop1, gw_dtop1, md1, vd1)
            w_dgap, mdg, vdg = adam_step(w_dgap, gw_dgap, mdg, vdg)
            w_boundary, mbr, vbr = adam_step(w_boundary, gw_boundary, mbr, vbr)
            w_depth, mdep, vdep = adam_step(w_depth, gw_depth, mdep, vdep)
            w_pred0, mp0C, vp0C = adam_step(w_pred0, gw_pred0, mp0C, vp0C)
            w_predB, mpbC, vpbC = adam_step(w_predB, gw_predB, mpbC, vpbC)

    return PushLRModel(
        b=float(b),
        w_pmax0=float(w_pmax0),
        w_m0=float(w_m0),
        w_pmaxN=float(w_pmaxN),
        w_pmaxB=float(w_pmaxB),
        w_mN=float(w_mN),
        w_mB=float(w_mB),
        w_dtop1=float(w_dtop1),
        w_dgap=float(w_dgap),
        w_boundary=float(w_boundary),
        w_depth=w_depth.astype(np.float32, copy=False),
        w_pred0=w_pred0.astype(np.float32, copy=False),
        w_predB=w_predB.astype(np.float32, copy=False),
    )

def push_lr_predict_beta(
    model: PushLRModel,
    *,
    pmax0: np.ndarray,
    m0: np.ndarray,
    pmaxN: np.ndarray,
    pmaxB: np.ndarray,
    mN: np.ndarray,
    mB: np.ndarray,
    d_top1: np.ndarray,
    d_gap: np.ndarray,
    boundary: np.ndarray,
    depth_bin: np.ndarray,
    pred0: np.ndarray,
    predB: np.ndarray,
) -> np.ndarray:
    sc = (
        model.b
        + model.w_pmax0 * pmax0
        + model.w_m0 * m0
        + model.w_pmaxN * pmaxN
        + model.w_pmaxB * pmaxB
        + model.w_mN * mN
        + model.w_mB * mB
        + model.w_dtop1 * d_top1
        + model.w_dgap * d_gap
        + model.w_boundary * boundary
        + model.w_pred0[pred0]
        + model.w_predB[predB]
    )
    mvalid = depth_bin >= 0
    if np.any(mvalid):
        sc[mvalid] += model.w_depth[depth_bin[mvalid]]
    return sigmoid(sc.astype(np.float32, copy=False)).astype(np.float32, copy=False)

def lr_train(
    samples: Dict[str, np.ndarray],
    *,
    l2: float = 1.0,
    lr: float = 0.05,
    epochs: int = 3,
    seed: int = 42,
) -> LRModel:
    """
    Train LR with Adam on sampled pixels.
    samples keys:
      y (0/1), pmaxN, pmaxB, mN, mB, boundary (0/1), depth_bin (-1..4), predN, predB
    """
    rng = np.random.default_rng(seed)
    y = samples["y"].astype(np.float32)
    pmaxN = samples["pmaxN"].astype(np.float32)
    pmaxB = samples["pmaxB"].astype(np.float32)
    mN = samples["mN"].astype(np.float32)
    mB = samples["mB"].astype(np.float32)
    boundary = samples["boundary"].astype(np.float32)
    depth_bin = samples["depth_bin"].astype(np.int64)
    predN = samples["predN"].astype(np.int64)
    predB = samples["predB"].astype(np.int64)

    n = y.shape[0]
    if n == 0:
        _die("No training samples after filtering; cannot train LR gating.")

    # params init (zeros is fine with strong L2)
    b = 0.0
    w_pmaxN = 0.0
    w_pmaxB = 0.0
    w_mN = 0.0
    w_mB = 0.0
    w_boundary = 0.0
    w_depth = np.zeros((5,), dtype=np.float32)
    w_predN = np.zeros((NUM_CLASSES,), dtype=np.float32)
    w_predB = np.zeros((NUM_CLASSES,), dtype=np.float32)

    # Adam state
    def adam_init(shape=()):
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

    mb, vb = adam_init()
    mpn, vpn = adam_init()
    mpb, vpb = adam_init()
    mmn, vmn = adam_init()
    mmb, vmb = adam_init()
    mbr, vbr = adam_init()
    mdep, vdep = adam_init(w_depth.shape)
    mpnC, vpnC = adam_init(w_predN.shape)
    mpbC, vpbC = adam_init(w_predB.shape)

    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    step = 0

    # mini-batch SGD for speed
    batch = min(200_000, n)

    for ep in range(int(epochs)):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            step += 1
            idx = order[s : s + batch]
            yy = y[idx]
            pn = pmaxN[idx]
            pb = pmaxB[idx]
            mn_ = mN[idx]
            mb_ = mB[idx]
            bd = boundary[idx]
            db = depth_bin[idx]
            cN = predN[idx]
            cB = predB[idx]

            # score
            sc = (
                b
                + w_pmaxN * pn
                + w_pmaxB * pb
                + w_mN * mn_
                + w_mB * mb_
                + w_boundary * bd
                + w_predN[cN]
                + w_predB[cB]
            )
            mvalid = db >= 0
            if np.any(mvalid):
                sc[mvalid] += w_depth[db[mvalid]]

            p = sigmoid(sc).astype(np.float32, copy=False)
            # logistic loss grad: (p - y)
            g = (p - yy).astype(np.float32, copy=False)

            # grads
            gb = float(g.mean())
            gw_pmaxN = float((g * pn).mean())
            gw_pmaxB = float((g * pb).mean())
            gw_mN = float((g * mn_).mean())
            gw_mB = float((g * mb_).mean())
            gw_boundary = float((g * bd).mean())

            # categorical grads (mean over samples of each class)
            gw_predN = np.zeros_like(w_predN)
            gw_predB = np.zeros_like(w_predB)
            np.add.at(gw_predN, cN, g)
            np.add.at(gw_predB, cB, g)
            gw_predN /= float(len(idx))
            gw_predB /= float(len(idx))

            gw_depth = np.zeros_like(w_depth)
            if np.any(mvalid):
                np.add.at(gw_depth, db[mvalid], g[mvalid])
                gw_depth /= float(len(idx))

            # L2
            gb += l2 * b
            gw_pmaxN += l2 * w_pmaxN
            gw_pmaxB += l2 * w_pmaxB
            gw_mN += l2 * w_mN
            gw_mB += l2 * w_mB
            gw_boundary += l2 * w_boundary
            gw_depth += l2 * w_depth
            gw_predN += l2 * w_predN
            gw_predB += l2 * w_predB

            # Adam update helper
            def adam_step(param, grad, m, v):
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad * grad)
                mhat = m / (1 - beta1**step)
                vhat = v / (1 - beta2**step)
                param = param - lr * mhat / (np.sqrt(vhat) + eps)
                return param, m, v

            b, mb, vb = adam_step(b, gb, mb, vb)
            w_pmaxN, mpn, vpn = adam_step(w_pmaxN, gw_pmaxN, mpn, vpn)
            w_pmaxB, mpb, vpb = adam_step(w_pmaxB, gw_pmaxB, mpb, vpb)
            w_mN, mmn, vmn = adam_step(w_mN, gw_mN, mmn, vmn)
            w_mB, mmb, vmb = adam_step(w_mB, gw_mB, mmb, vmb)
            w_boundary, mbr, vbr = adam_step(w_boundary, gw_boundary, mbr, vbr)
            w_depth, mdep, vdep = adam_step(w_depth, gw_depth, mdep, vdep)
            w_predN, mpnC, vpnC = adam_step(w_predN, gw_predN, mpnC, vpnC)
            w_predB, mpbC, vpbC = adam_step(w_predB, gw_predB, mpbC, vpbC)

    return LRModel(
        b=float(b),
        w_pmaxN=float(w_pmaxN),
        w_pmaxB=float(w_pmaxB),
        w_mN=float(w_mN),
        w_mB=float(w_mB),
        w_boundary=float(w_boundary),
        w_depth=w_depth.astype(np.float32, copy=False),
        w_predN=w_predN.astype(np.float32, copy=False),
        w_predB=w_predB.astype(np.float32, copy=False),
    )

def lr_predict_beta(
    model: LRModel,
    *,
    pmaxN: np.ndarray,
    pmaxB: np.ndarray,
    mN: np.ndarray,
    mB: np.ndarray,
    boundary: np.ndarray,
    depth_bin: np.ndarray,
    predN: np.ndarray,
    predB: np.ndarray,
) -> np.ndarray:
    sc = (
        model.b
        + model.w_pmaxN * pmaxN
        + model.w_pmaxB * pmaxB
        + model.w_mN * mN
        + model.w_mB * mB
        + model.w_boundary * boundary
        + model.w_predN[predN]
        + model.w_predB[predB]
    )
    mvalid = depth_bin >= 0
    if np.any(mvalid):
        sc[mvalid] += model.w_depth[depth_bin[mvalid]]
    return sigmoid(sc.astype(np.float32, copy=False)).astype(np.float32, copy=False)


def gating_disagree_v0(
    *,
    logits_n: np.ndarray,
    logits_b: np.ndarray,
    gt: np.ndarray,
    tau: float,
    depth_m: Optional[np.ndarray],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Rule-based gating (v0):
    - If agree: choose Nearest
    - Else:
      - Stop-loss: if Nearest predicts {books,table} -> choose Nearest
      - Else choose B if margin_B > margin_N + tau, else N
      - Optional depth rule (5-10m): if depth in [5,10) and B predicts one of {objects,wall,furniture},
        lower the threshold slightly (tau - 0.02) to allow more B in that zone.

    Returns:
      pred (H,W) and selection stats (fractions + good/bad on selected-B pixels).
    """
    pred_n = np.argmax(logits_n, axis=0).astype(np.int64, copy=False)
    pred_b = np.argmax(logits_b, axis=0).astype(np.int64, copy=False)
    agree = pred_n == pred_b

    mN = top2_margin(logits_n)
    mB = top2_margin(logits_b)

    valid = (gt != IGNORE_INDEX)
    disagree = (~agree) & valid

    choose_b = np.zeros_like(agree, dtype=np.bool_)
    # base: margin rule on disagree
    choose_b[disagree] = (mB[disagree] > (mN[disagree] + float(tau)))

    # stop-loss: if Nearest predicts books/table, force N
    stop = (pred_n == CID_BOOKS) | (pred_n == CID_TABLE)
    choose_b[stop] = False

    # depth rule (optional): in far range, allow B slightly more when it claims specific classes
    if depth_m is not None:
        d = depth_m
        far = (d >= 5.0) & (d < 10.0) & valid & (~agree)
        b_claim = (pred_b == CID_OBJECTS) | (pred_b == CID_WALL) | (pred_b == CID_FURNITURE)
        far2 = far & b_claim
        # looser threshold only on far2
        choose_b[far2] = (mB[far2] > (mN[far2] + float(tau) - 0.02))

    # final pred: N everywhere, overwrite with B where choose_b
    pred = pred_n.copy()
    pred[choose_b] = pred_b[choose_b]

    # selection stats (evaluate only where choose_b AND valid)
    sel = choose_b & valid
    n_correct = (pred_n == gt) & valid
    b_correct = (pred_b == gt) & valid
    good = sel & b_correct & (~n_correct)  # B saves
    bad = sel & (~b_correct) & n_correct   # B breaks
    both_wrong = sel & (~b_correct) & (~n_correct)
    both_right = sel & b_correct & n_correct

    denom_valid = float(valid.sum()) if valid.any() else 1.0
    denom_sel = float(sel.sum()) if sel.any() else 1.0
    stats = {
        "select_b_ratio": float(sel.sum() / denom_valid),
        "select_b_good_ratio_of_valid": float(good.sum() / denom_valid),
        "select_b_bad_ratio_of_valid": float(bad.sum() / denom_valid),
        "select_b_good_frac": float(good.sum() / denom_sel),
        "select_b_bad_frac": float(bad.sum() / denom_sel),
        "select_b_both_wrong_frac": float(both_wrong.sum() / denom_sel),
        "select_b_both_right_frac": float(both_right.sum() / denom_sel),
    }
    return pred, stats


def method_id(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace(",", "_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="00_data/output/e3_nearest_plus_b_opt")
    ap.add_argument("--label-dir", default="00_data/train/label")
    ap.add_argument("--depth-dir", default="00_data/train/depth")

    ap.add_argument("--splits-manifest", default="00_data/splits/folds_v1.json")
    ap.add_argument("--nearest-oof-logits", default="01_nearest/golden_artifacts/oof_logits.npy")
    ap.add_argument("--nearest-oof-ids", default="01_nearest/golden_artifacts/oof_file_ids.npy")
    ap.add_argument("--b-oof-dir", default="00_data/02_nonstruct_frozen/golden_artifacts/oof")

    ap.add_argument("--fixed-wb", type=float, default=0.2)

    # classwise candidates
    ap.add_argument("--classwise-default", type=float, default=0.2)
    ap.add_argument("--classwise-good", type=float, default=0.3)
    ap.add_argument("--classwise-stop", type=float, default=0.0)
    ap.add_argument("--classwise-good-classes", default="tv,ceiling", help="comma-separated names")

    # gating
    ap.add_argument("--tau", type=float, default=0.03)
    ap.add_argument("--tau-sweep", default="0.02,0.03,0.04,0.05")
    ap.add_argument("--lr-S", default="tv,ceiling,objects,picture", help="comma-separated class names for restricted LR gating")
    ap.add_argument("--lambda-sweep", default="0.25,0.5,0.75", help="comma-separated lambdas for local-correction")
    ap.add_argument("--push-eps", type=float, default=0.25, help="epsilon for defining push label: argmax(L0 + eps*(B-N))")
    ap.add_argument("--freeze-dir", default="", help="If set, write Ensemble-2 FROZEN bundle here")
    args = ap.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    _ensure(args.label_dir, "GT label dir")
    _ensure(args.splits_manifest, "splits manifest")
    _ensure(args.nearest_oof_logits, "Nearest oof_logits.npy")
    _ensure(args.nearest_oof_ids, "Nearest oof_file_ids.npy")
    _ensure(args.b_oof_dir, "Model B oof dir")

    # Load Nearest OOF
    n_ids = np.load(args.nearest_oof_ids, allow_pickle=False)
    file_ids = [str(x) for x in n_ids.tolist()]
    n_logits = np.load(args.nearest_oof_logits, mmap_mode="r", allow_pickle=False)
    if n_logits.shape[0] != len(file_ids) or n_logits.shape[1:] != (NUM_CLASSES, 480, 640):
        _die(f"Unexpected Nearest logits shape: {n_logits.shape} vs ids={len(file_ids)}")

    # Build Model B index from splits val_ids (canonical fold ordering)
    val_ids_by_fold = load_splits_val_ids_by_fold(args.splits_manifest)
    b_index = build_b_index(args.b_oof_dir, val_ids_by_fold)

    # Build class-wise weights (channel-wise mixing)
    good_names = [s.strip() for s in str(args.classwise_good_classes).split(",") if s.strip()]
    good_ids = set()
    for name in good_names:
        if name not in CLASS_NAMES:
            _die(f"Unknown class name in --classwise-good-classes: {name}")
        good_ids.add(CLASS_NAMES.index(name))

    wB = np.full((NUM_CLASSES,), float(args.classwise_default), dtype=np.float32)
    for c in good_ids:
        wB[c] = float(args.classwise_good)

    # Stop-loss init from existing E3 per_class_iou.csv if available
    per_class_path = "00_data/output/e3_nearest_plus_b/per_class_iou.csv"
    if os.path.exists(per_class_path):
        with open(per_class_path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                cid = int(row["class_id"])
                d = float(row["delta_iou"])
                if not math.isnan(d) and d < 0:
                    wB[cid] = float(args.classwise_stop)
    else:
        # Minimal hard stop if per-class table not present
        wB[CID_BOOKS] = float(args.classwise_stop)
        wB[CID_TABLE] = float(args.classwise_stop)

    # classwise_v1: clip + 3-level quantization (anti-overfit)
    wB_v1 = quantize_wB_3level(wB)

    # Define methods to evaluate
    methods: Dict[str, Dict[str, object]] = {}
    methods["nearest"] = {"type": "nearest"}
    methods[f"fixed_wB={args.fixed_wb:.3f}"] = {"type": "fixed", "wB": float(args.fixed_wb)}
    methods["classwise_v0"] = {"type": "classwise", "wB_vec": wB.copy()}
    methods["classwise_v1_q3"] = {"type": "classwise", "wB_vec": wB_v1.copy()}

    tau_list = [float(x) for x in str(args.tau_sweep).split(",") if x.strip()]
    if float(args.tau) not in tau_list:
        tau_list = [float(args.tau)] + tau_list
    tau_list = sorted(set(tau_list))
    for t in tau_list:
        methods[f"gating_disagree_v0_tau={t:.3f}"] = {"type": "gating", "tau": float(t)}

    # --- Restricted LR gating (train on OOF, evaluate on same OOF for now) ---
    # Target mask per user instruction: only allow gating when predB in S.
    # IMPORTANT: books/table are excluded by design (B is harmful there).
    S_names = [s.strip() for s in str(args.lr_S).split(",") if s.strip()]
    for nm in S_names:
        if nm not in CLASS_NAMES:
            _die(f"Unknown class name in --lr-S: {nm}")
    S = set(CLASS_NAMES.index(x) for x in S_names)
    if CID_BOOKS in S or CID_TABLE in S:
        _die("Do not include books/table in --lr-S. They are stop-loss classes.")

    # Collect training samples for PushLR ("push usefulness"):
    # mask: disagree & predB in S, and define y by whether L0+eps*d fixes without breaking.
    # Keep it small and class-balanced (per image, per class cap).
    sample_cap_per_img = 1500
    sample_cap_per_class = 350

    samples_push: Dict[str, List[np.ndarray]] = {
        "y": [],
        "pmax0": [],
        "m0": [],
        "pmaxN": [],
        "pmaxB": [],
        "mN": [],
        "mB": [],
        "d_top1": [],
        "d_gap": [],
        "boundary": [],
        "depth_bin": [],
        "pred0": [],
        "predB": [],
    }

    rng = np.random.default_rng(42)

    for i, fid in enumerate(file_ids):
        gt = load_label(args.label_dir, fid)
        ln = np.array(n_logits[i], copy=False)
        lb = logits_b_for_id(b_index, fid)
        predN = np.argmax(ln, axis=0).astype(np.int64, copy=False)
        predB = np.argmax(lb, axis=0).astype(np.int64, copy=False)
        # base logits (classwise_v1_q3) + direction
        L0 = classwise_mix_logits(ln, lb, wB_v1)
        d = (lb.astype(np.float32, copy=False) - ln.astype(np.float32, copy=False))
        pred0 = np.argmax(L0, axis=0).astype(np.int64, copy=False)
        valid = (gt != IGNORE_INDEX)
        disagree = (predN != predB) & valid
        inS = np.isin(predB, list(S))
        mask = disagree & inS
        if not np.any(mask):
            continue

        # define y by push effectiveness using eps
        eps_push = float(args.push_eps)
        pred_eps = np.argmax((L0 + eps_push * d), axis=0).astype(np.int64, copy=False)
        # y=1: pred0 wrong -> pred_eps correct
        y_pos = (pred0 != gt) & (pred_eps == gt) & mask
        # y=0: everything else in mask (including breaking correct or no change)
        mask2 = mask  # sample from full mask, label accordingly
        if not np.any(mask2):
            continue

        # boundary + depth (optional but available in this dataset)
        bnd = boundary_mask_from_gt(gt, k=3)
        depth = load_depth_m(args.depth_dir, fid)

        ys_all, xs_all = np.where(mask2)
        # class-balanced sampling by predB class
        picked_y = []
        picked_x = []
        for c in S:
            m_c = mask2 & (predB == c)
            if not np.any(m_c):
                continue
            ys, xs = np.where(m_c)
            k = min(sample_cap_per_class, len(ys))
            if k <= 0:
                continue
            sel = rng.choice(len(ys), size=k, replace=False)
            picked_y.append(ys[sel])
            picked_x.append(xs[sel])

        if picked_y:
            ys = np.concatenate(picked_y)
            xs = np.concatenate(picked_x)
        else:
            # fallback random
            k = min(sample_cap_per_img, len(ys_all))
            sel = rng.choice(len(ys_all), size=k, replace=False)
            ys = ys_all[sel]
            xs = xs_all[sel]

        # cap total per image
        if len(ys) > sample_cap_per_img:
            sel = rng.choice(len(ys), size=sample_cap_per_img, replace=False)
            ys = ys[sel]
            xs = xs[sel]

        # labels: y=1 if pushing helps, else 0
        y = y_pos[ys, xs].astype(np.int64)

        # features at sampled pixels
        pmax0 = pmax_from_logits_at(L0, ys, xs)
        m0 = margin_from_logits_at(L0, ys, xs)
        pmaxN = pmax_from_logits_at(ln, ys, xs)
        pmaxB = pmax_from_logits_at(lb, ys, xs)
        mN = margin_from_logits_at(ln, ys, xs)
        mB = margin_from_logits_at(lb, ys, xs)

        # delta summaries: d_top1 and d_gap based on base top1/top2 indices
        # get top1/top2 indices of L0 at sampled pixels
        x0 = L0[:, ys, xs].astype(np.float32, copy=False)  # (C,K)
        part = np.partition(x0, kth=[NUM_CLASSES - 2, NUM_CLASSES - 1], axis=0)
        top2v = part[NUM_CLASSES - 2]
        top1v = part[NUM_CLASSES - 1]
        # indices: compute via argmax and argmax of masked
        top1i = np.argmax(x0, axis=0).astype(np.int64, copy=False)
        x0_mask = x0.copy()
        x0_mask[top1i, np.arange(x0.shape[1])] = -1e9
        top2i = np.argmax(x0_mask, axis=0).astype(np.int64, copy=False)
        d_s = d[:, ys, xs]
        d_top1 = d_s[top1i, np.arange(d_s.shape[1])].astype(np.float32, copy=False)
        d_top2 = d_s[top2i, np.arange(d_s.shape[1])].astype(np.float32, copy=False)
        d_gap = (d_top1 - d_top2).astype(np.float32, copy=False)

        bd = bnd[ys, xs].astype(np.int64)
        db = depth_bin_at(depth, ys, xs)
        c0 = pred0[ys, xs].astype(np.int64)
        cB = predB[ys, xs].astype(np.int64)

        samples_push["y"].append(y)
        samples_push["pmax0"].append(pmax0)
        samples_push["m0"].append(m0)
        samples_push["pmaxN"].append(pmaxN)
        samples_push["pmaxB"].append(pmaxB)
        samples_push["mN"].append(mN)
        samples_push["mB"].append(mB)
        samples_push["d_top1"].append(d_top1)
        samples_push["d_gap"].append(d_gap)
        samples_push["boundary"].append(bd)
        samples_push["depth_bin"].append(db)
        samples_push["pred0"].append(c0)
        samples_push["predB"].append(cB)

        if (i + 1) % 200 == 0:
            print(f"[E3-OPT][LR] sampled up to image {i+1}/{len(file_ids)}", flush=True)

    # Train PushLR (push usefulness beta) (model only; not a prediction method by itself)
    if samples_push["y"]:
        packed = {k: np.concatenate(v) for k, v in samples_push.items()}
        push_model = push_lr_train(packed, l2=1.0, lr=0.05, epochs=3, seed=42, neg_keep_prob=0.25)
    else:
        push_model = None

    # Evaluate in a SINGLE pass over images (much faster than per-method passes).
    cms: Dict[str, np.ndarray] = {m: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64) for m in methods.keys()}
    gating_stats_sum: Dict[str, Dict[str, float]] = {
        m: {} for m, cfg in methods.items() if str(cfg["type"]) in ("gating", "lr_gating")
    }
    gating_stats_n: Dict[str, int] = {
        m: 0 for m, cfg in methods.items() if str(cfg["type"]) in ("gating", "lr_gating")
    }

    # Local correction methods: base=classwise_v1_q3, apply only on disagree & predB in S:
    #   logits = logits_base + λ * beta * (logits_B - logits_N)
    lambdas = [float(x) for x in str(args.lambda_sweep).split(",") if x.strip()]
    lambdas = sorted(set(lambdas))
    local_methods = [f"local_corr_lr_S_tauNA_lam={lam:.2f}" for lam in lambdas]
    for m in local_methods:
        methods[m] = {"type": "local_corr", "lambda": float(m.split("lam=")[-1])}
        cms[m] = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        gating_stats_sum[m] = {}
        gating_stats_n[m] = 0

    # Per-class (predB class) selection stats for LR-based methods
    # Accumulate counts over VALID pixels as denominator reference.
    sel_cls_sum: Dict[str, Dict[int, Dict[str, float]]] = {}  # method -> class_id -> stats sums
    sel_cls_n: Dict[str, int] = {}  # method -> number of images aggregated
    for m in (["lr_gating_restricted_v0"] if "lr_gating_restricted_v0" in methods else []) + local_methods:
        sel_cls_sum[m] = {c: {} for c in S}
        sel_cls_n[m] = 0

    # Transition stats for "do not break correct" and "fix wrong" relative to base pred0
    trans_sum: Dict[str, Dict[str, float]] = {}
    trans_n: Dict[str, int] = {}
    for m in local_methods:
        trans_sum[m] = {"break": 0.0, "improve": 0.0, "count_correct0": 0.0, "count_wrong0": 0.0}
        trans_n[m] = 0

    for i, fid in enumerate(file_ids):
        if fid not in b_index.id_to_pos:
            _die(f"ID missing in Model B folds: {fid}")

        gt = load_label(args.label_dir, fid)
        if gt.shape != (480, 640):
            _die(f"Unexpected GT shape {gt.shape} for {fid}")

        ln = np.array(n_logits[i], copy=False)  # (13,480,640) float32
        lb = logits_b_for_id(b_index, fid)      # (13,480,640) float16

        # Precompute common preds
        pred_n = np.argmax(ln, axis=0).astype(np.int64, copy=False)
        confusion_update(cms["nearest"], gt, pred_n)

        # Fixed blend
        fixed_name = f"fixed_wB={args.fixed_wb:.3f}"
        mix_fixed = (ln.astype(np.float32, copy=False) * (1.0 - float(args.fixed_wb)) + lb.astype(np.float32, copy=False) * float(args.fixed_wb))
        pred_fixed = np.argmax(mix_fixed, axis=0).astype(np.int64, copy=False)
        confusion_update(cms[fixed_name], gt, pred_fixed)

        # Classwise
        mix_cw = classwise_mix_logits(ln, lb, wB)
        pred_cw = np.argmax(mix_cw, axis=0).astype(np.int64, copy=False)
        confusion_update(cms["classwise_v0"], gt, pred_cw)

        mix_cw1 = classwise_mix_logits(ln, lb, wB_v1)
        pred_cw1 = np.argmax(mix_cw1, axis=0).astype(np.int64, copy=False)
        confusion_update(cms["classwise_v1_q3"], gt, pred_cw1)

        # Base for push-beta local correction
        L0 = mix_cw1
        pred0 = pred_cw1
        d = (lb.astype(np.float32, copy=False) - ln.astype(np.float32, copy=False))

        # Depth once if needed for any gating method
        depth = None
        if gating_stats_sum:
            depth = load_depth_m(args.depth_dir, fid)

        # Gating sweep
        for t in tau_list:
            name = f"gating_disagree_v0_tau={t:.3f}"
            pred_g, st = gating_disagree_v0(logits_n=ln, logits_b=lb, gt=gt, tau=t, depth_m=depth)
            confusion_update(cms[name], gt, pred_g)
            # aggregate stats
            ss = gating_stats_sum[name]
            for k, v in st.items():
                ss[k] = ss.get(k, 0.0) + float(v)
            gating_stats_n[name] += 1

        # NOTE: legacy "lr_gating_restricted_v0" path kept in earlier experiments.
        # The current winning direction is push-usefulness beta; we skip re-running old LR gating here.

        # Local correction: base=classwise_v1_q3, adjust only on restricted mask using LR beta
        # Push-beta local correction (Step B redefined):
        # beta = P(push improves) trained by pred0->pred_eps, and applied only when pred0 is wrong.
        if push_model is not None and local_methods:
            predB_full = np.argmax(lb, axis=0).astype(np.int64, copy=False)
            valid = (gt != IGNORE_INDEX)
            disagree = (pred0 != predB_full) & valid
            inS = np.isin(predB_full, list(S))
            mask = disagree & inS
            if np.any(mask):
                bnd = boundary_mask_from_gt(gt, k=3)
                ys, xs = np.where(mask)
                # features
                pmax0 = pmax_from_logits_at(L0, ys, xs)
                m0 = margin_from_logits_at(L0, ys, xs)
                pmaxN = pmax_from_logits_at(ln, ys, xs)
                pmaxB = pmax_from_logits_at(lb, ys, xs)
                mN_ = margin_from_logits_at(ln, ys, xs)
                mB_ = margin_from_logits_at(lb, ys, xs)
                bd = bnd[ys, xs].astype(np.float32)
                db = depth_bin_at(depth, ys, xs)
                c0 = pred0[ys, xs].astype(np.int64)
                cB = predB_full[ys, xs].astype(np.int64)
                # delta summaries
                d_s = d[:, ys, xs]
                top1i = c0
                # top2 index of base at those pixels
                x0 = L0[:, ys, xs].astype(np.float32, copy=False)
                x0_mask = x0.copy()
                x0_mask[top1i, np.arange(x0.shape[1])] = -1e9
                top2i = np.argmax(x0_mask, axis=0).astype(np.int64, copy=False)
                d_top1 = d_s[top1i, np.arange(d_s.shape[1])].astype(np.float32, copy=False)
                d_top2 = d_s[top2i, np.arange(d_s.shape[1])].astype(np.float32, copy=False)
                d_gap = (d_top1 - d_top2).astype(np.float32, copy=False)

                beta = push_lr_predict_beta(
                    push_model,
                    pmax0=pmax0,
                    m0=m0,
                    pmaxN=pmaxN,
                    pmaxB=pmaxB,
                    mN=mN_,
                    mB=mB_,
                    d_top1=d_top1,
                    d_gap=d_gap,
                    boundary=bd,
                    depth_bin=db,
                    pred0=c0,
                    predB=cB,
                )

                # Safety: beta only active where base pred0 is WRONG.
                gt_pix = gt[ys, xs]
                pred0_pix = c0
                beta_eff = beta.copy()
                beta_eff[pred0_pix == gt_pix] = 0.0

                pred_base = pred0.copy()
                # evaluate for each lambda
                for lam, mname in zip(lambdas, local_methods):
                    adj = (L0[:, ys, xs] + (float(lam) * beta_eff[None, :] * d[:, ys, xs]))
                    pred_adj = np.argmax(adj, axis=0).astype(np.int64, copy=False)
                    pred = pred_base.copy()
                    pred[ys, xs] = pred_adj
                    confusion_update(cms[mname], gt, pred)

                    # Transition stats relative to base pred0
                    correct0 = (pred0_pix == gt_pix)
                    wrong0 = ~correct0
                    break_ct = float(np.sum(correct0 & (pred_adj != gt_pix)))
                    improve_ct = float(np.sum(wrong0 & (pred_adj == gt_pix)))
                    trans_sum[mname]["break"] += break_ct
                    trans_sum[mname]["improve"] += improve_ct
                    trans_sum[mname]["count_correct0"] += float(np.sum(correct0))
                    trans_sum[mname]["count_wrong0"] += float(np.sum(wrong0))
                    trans_n[mname] += 1

                    # selection stats: define "select B" as beta_eff>0.5 (within mask)
                    sel = beta_eff > 0.5
                    predN_pix = pred_n[ys, xs]
                    predB_pix = predB_full[ys, xs]
                    n_ok = (predN_pix == gt_pix)
                    b_ok = (predB_pix == gt_pix)
                    good = np.sum(sel & b_ok & (~n_ok))
                    bad = np.sum(sel & (~b_ok) & n_ok)
                    both_wrong = np.sum(sel & (~b_ok) & (~n_ok))
                    both_right = np.sum(sel & b_ok & n_ok)
                    denom_valid = float(valid.sum()) if valid.any() else 1.0
                    denom_sel = float(np.sum(sel)) if np.any(sel) else 1.0
                    st = {
                        "select_b_ratio": float(np.sum(sel) / denom_valid),
                        "select_b_good_ratio_of_valid": float(good / denom_valid),
                        "select_b_bad_ratio_of_valid": float(bad / denom_valid),
                        "select_b_good_frac": float(good / denom_sel),
                        "select_b_bad_frac": float(bad / denom_sel),
                        "select_b_both_wrong_frac": float(both_wrong / denom_sel),
                        "select_b_both_right_frac": float(both_right / denom_sel),
                    }
                    ss = gating_stats_sum[mname]
                    for k, v in st.items():
                        ss[k] = ss.get(k, 0.0) + float(v)
                    gating_stats_n[mname] += 1

                # class-wise stats by predB class for the *beta itself* (use mname for lam=0.25 as representative)
                rep = local_methods[0]
                predB_cls = cB
                for c in S:
                    m_c = (predB_cls == c)
                    if not np.any(m_c):
                        continue
                    betac = beta_eff[m_c]
                    selc = betac > 0.5
                    denom_valid = float(valid.sum()) if valid.any() else 1.0
                    denom_selc = float(np.sum(selc)) if np.any(selc) else 1.0
                    # correctness for good/bad (relative N/B)
                    predN_c = pred_n[ys, xs][m_c]
                    predB_c = predB_full[ys, xs][m_c]
                    gt_c = gt_pix[m_c]
                    n_ok_c = (predN_c == gt_c)
                    b_ok_c = (predB_c == gt_c)
                    good_c = np.sum(selc & b_ok_c & (~n_ok_c))
                    bad_c = np.sum(selc & (~b_ok_c) & n_ok_c)
                    ss2 = sel_cls_sum[rep][c]
                    ss2["mask_ratio_of_valid"] = ss2.get("mask_ratio_of_valid", 0.0) + float(np.sum(m_c) / denom_valid)
                    ss2["select_b_ratio_of_valid"] = ss2.get("select_b_ratio_of_valid", 0.0) + float(np.sum(selc) / denom_valid)
                    ss2["good_of_valid"] = ss2.get("good_of_valid", 0.0) + float(good_c / denom_valid)
                    ss2["bad_of_valid"] = ss2.get("bad_of_valid", 0.0) + float(bad_c / denom_valid)
                    ss2["good_frac_selB"] = ss2.get("good_frac_selB", 0.0) + float(good_c / denom_selc)
                    ss2["bad_frac_selB"] = ss2.get("bad_frac_selB", 0.0) + float(bad_c / denom_selc)
                    ss2["mean_beta_mask"] = ss2.get("mean_beta_mask", 0.0) + float(np.mean(betac))
                    ss2["mean_beta_selB"] = ss2.get("mean_beta_selB", 0.0) + (float(np.mean(betac[selc])) if np.any(selc) else 0.0)
                sel_cls_n[rep] += 1
            else:
                for mname in local_methods:
                    confusion_update(cms[mname], gt, pred_cw1)

        if (i + 1) % 50 == 0:
            print(f"[E3-OPT] processed {i+1}/{len(file_ids)}", flush=True)

    # Build outputs
    metrics_rows: List[List[object]] = []
    selection_rows: List[List[object]] = []

    for mname in methods.keys():
        cm = cms[mname]
        miou = miou_from_cm(cm)
        metrics_rows.append([mname, miou])

        # per-class export
        ious = iou_from_cm(cm)
        pc_rows: List[List[object]] = [[c, CLASS_NAMES[c], ious[c]] for c in range(NUM_CLASSES)]
        write_csv(
            os.path.join(out_dir, f"per_class_iou_{method_id(mname)}.csv"),
            ["class_id", "class_name", "iou"],
            pc_rows,
        )

        if mname in gating_stats_sum and gating_stats_n.get(mname, 0) > 0:
            n = float(gating_stats_n[mname])
            avg = {k: v / n for k, v in gating_stats_sum[mname].items()}
            selection_rows.append(
                [mname]
                + [
                    avg.get(k, float("nan"))
                    for k in [
                        "select_b_ratio",
                        "select_b_good_ratio_of_valid",
                        "select_b_bad_ratio_of_valid",
                        "select_b_good_frac",
                        "select_b_bad_frac",
                        "select_b_both_wrong_frac",
                        "select_b_both_right_frac",
                    ]
                ]
            )

    # Export wB vectors for copy/paste decision-making
    w_rows = []
    for c in range(NUM_CLASSES):
        w_rows.append([c, CLASS_NAMES[c], float(wB[c]), float(wB_v1[c])])
    write_csv(os.path.join(out_dir, "wB_classwise.csv"), ["class_id", "class_name", "wB_v0", "wB_v1_q3"], w_rows)

    metrics_rows.sort(key=lambda r: float(r[1]), reverse=True)
    write_csv(os.path.join(out_dir, "metrics.csv"), ["method", "miou_global"], metrics_rows)
    if selection_rows:
        write_csv(
            os.path.join(out_dir, "selection_stats.csv"),
            [
                "method",
                "select_b_ratio",
                "select_b_good_ratio_of_valid",
                "select_b_bad_ratio_of_valid",
                "select_b_good_frac",
                "select_b_bad_frac",
                "select_b_both_wrong_frac",
                "select_b_both_right_frac",
            ],
            selection_rows,
        )

    # Write class-wise selection stats (predB class) for LR-based methods
    if sel_cls_sum:
        rows = []
        for mname, byc in sel_cls_sum.items():
            nimg = float(max(1, sel_cls_n.get(mname, 1)))
            for c in sorted(byc.keys()):
                ss = byc[c]
                if not ss:
                    continue
                rows.append([
                    mname,
                    int(c),
                    CLASS_NAMES[int(c)],
                    ss.get("mask_ratio_of_valid", 0.0) / nimg,
                    ss.get("select_b_ratio_of_valid", 0.0) / nimg,
                    ss.get("good_of_valid", 0.0) / nimg,
                    ss.get("bad_of_valid", 0.0) / nimg,
                    ss.get("good_frac_selB", 0.0) / nimg,
                    ss.get("bad_frac_selB", 0.0) / nimg,
                    ss.get("mean_beta_mask", 0.0) / nimg,
                    ss.get("mean_beta_selB", 0.0) / nimg,
                ])
        write_csv(
            os.path.join(out_dir, "selection_stats_by_predB_class.csv"),
            [
                "method",
                "predB_class_id",
                "predB_class_name",
                "mask_ratio_of_valid",
                "select_b_ratio_of_valid",
                "good_of_valid",
                "bad_of_valid",
                "good_frac_selB",
                "bad_frac_selB",
                "mean_beta_mask",
                "mean_beta_selB",
            ],
            rows,
        )

    # Write transition stats (break / improve relative to base pred0 for local correction)
    if trans_sum:
        rows = []
        for mname, ss in trans_sum.items():
            c0 = max(1.0, ss["count_correct0"])
            w0 = max(1.0, ss["count_wrong0"])
            rows.append(
                [
                    mname,
                    float(ss["break"] / c0),
                    float(ss["improve"] / w0),
                    int(ss["count_correct0"]),
                    int(ss["count_wrong0"]),
                ]
            )
        write_csv(
            os.path.join(out_dir, "transition_stats.csv"),
            ["method", "break_rate(pred0==GT->wrong)", "improve_rate(pred0!=GT->GT)", "count_correct0", "count_wrong0"],
            rows,
        )

    # --- FROZEN bundle (Ensemble-2) ---
    # Captures trained PushLR coefficients + exact feature order + sampling rules + manifest.
    if args.freeze_dir:
        if push_model is None:
            _die("Cannot freeze: push_model is None (no training samples).")

        freeze_dir = args.freeze_dir
        os.makedirs(freeze_dir, exist_ok=True)

        # choose best local correction from this run (typically run with single lambda for final)
        best_local = None
        best_miou = -1.0
        for m, v in metrics_rows:
            if str(m).startswith("local_corr_lr_S_tauNA_lam="):
                if float(v) > best_miou:
                    best_miou = float(v)
                    best_local = str(m)

        S_list = sorted(list(S))
        S_names_sorted = [CLASS_NAMES[i] for i in S_list]
        wB_map = {CLASS_NAMES[i]: float(wB_v1[i]) for i in range(NUM_CLASSES)}

        spec = {
            "name": "Ensemble-2 (FROZEN)",
            "oof_global_miou": float(best_miou) if best_local else None,
            "base": {
                "type": "classwise_v1_q3",
                "wB_levels": [0.0, 0.15, 0.30],
                "wB": wB_map,
            },
            "mask": {
                "definition": "agree==False & predB∈S’",
                "S_prime": S_names_sorted,
                "exclude_fixed": ["books", "table"],
            },
            "pushlr": {
                "target_eps_train": float(args.push_eps),
                "features_order": [
                    "pmax0",
                    "m0",
                    "pmaxN",
                    "pmaxB",
                    "mN",
                    "mB",
                    "d_top1",
                    "d_gap",
                    "boundary_flag",
                    "depth_bin",
                    "pred0_class",
                    "predB_class",
                ],
                "sampling": {
                    "mask": "disagree & predB in S’",
                    "per_image_cap": 1500,
                    "per_class_cap": 350,
                    "seed": 42,
                    "neg_keep_prob": 0.25,
                },
                "optimizer": {"type": "adam", "epochs": 3, "lr": 0.05, "l2": 1.0},
            },
            "inference": {
                "lambda": float(best_local.split("lam=")[-1]) if best_local else None,
                "formula": "L = L0 + lambda * beta_eff * (B-N)",
                "safety_device": "EVAL-ONLY: pred0==GT => beta_eff=0 (uses GT; not deployable on test).",
            },
        }

        coef = {
            "b": float(push_model.b),
            "w_pmax0": float(push_model.w_pmax0),
            "w_m0": float(push_model.w_m0),
            "w_pmaxN": float(push_model.w_pmaxN),
            "w_pmaxB": float(push_model.w_pmaxB),
            "w_mN": float(push_model.w_mN),
            "w_mB": float(push_model.w_mB),
            "w_dtop1": float(push_model.w_dtop1),
            "w_dgap": float(push_model.w_dgap),
            "w_boundary": float(push_model.w_boundary),
            "w_depth": [float(x) for x in push_model.w_depth.tolist()],
            "w_pred0": [float(x) for x in push_model.w_pred0.tolist()],
            "w_predB": [float(x) for x in push_model.w_predB.tolist()],
            "class_names": CLASS_NAMES,
            "depth_bins": ["0-1", "1-2", "2-3", "3-5", "5-10"],
        }

        write_json(os.path.join(freeze_dir, "spec.json"), spec)
        write_json(os.path.join(freeze_dir, "pushlr_coefficients.json"), coef)

        write_text(
            os.path.join(freeze_dir, "README.md"),
            "\n".join(
                [
                    "Ensemble-2 (FROZEN)",
                    f"- OOF global mIoU: {spec['oof_global_miou']}",
                    "",
                    "## Base",
                    f"- classwise_v1_q3 wB levels: {spec['base']['wB_levels']}",
                    "",
                    "## Mask",
                    f"- S': {', '.join(S_names_sorted)}",
                    "",
                    "## PushLR",
                    f"- eps_train: {args.push_eps}",
                    "",
                    "## Inference",
                    f"- lambda: {spec['inference']['lambda']}",
                    f"- formula: `{spec['inference']['formula']}`",
                    f"- safety: {spec['inference']['safety_device']}",
                    "",
                ]
            )
            + "\n",
        )

        rel_files = ["spec.json", "pushlr_coefficients.json", "README.md"]
        manifest_lines = [f"{sha256_file(os.path.join(freeze_dir, fn))}  {fn}" for fn in rel_files]
        write_text(os.path.join(freeze_dir, "sha256_manifest.txt"), "\n".join(manifest_lines) + "\n")
        write_text(os.path.join(freeze_dir, "FROZEN.lock"), "Ensemble-2 frozen. Do not edit.\n")

    lines: List[str] = []
    lines.append("E3-OPT 2-model ensemble results (global mIoU)")
    lines.append("")
    lines.append("| method | miou_global |")
    lines.append("|---|---:|")
    for m, v in metrics_rows:
        lines.append(f"| {m} | {float(v):.6f} |")
    lines.append("")
    if selection_rows:
        lines.append("Gating selection stats (averaged over images; ratios are of VALID pixels unless noted)")
        lines.append("")
        lines.append("| method | select_b_ratio | good_of_valid | bad_of_valid | good_frac(selB) | bad_frac(selB) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in selection_rows:
            m = row[0]
            sb, gV, bV, gF, bF = row[1], row[2], row[3], row[4], row[5]
            lines.append(f"| {m} | {sb:.4f} | {gV:.4f} | {bV:.4f} | {gF:.3f} | {bF:.3f} |")
        lines.append("")

    with open(os.path.join(out_dir, "notes.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[E3-OPT] Done. Outputs: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
