"""
E3: 2モデル補完・死角分析（01_nearest + Model B）
===================================================

Generates the E3 artifact set (CSV + optional visualizations) described by the user.

Contract (must hold)
--------------------
- OOF only.
- ID-based matching (no index alignment hacks).
- Ignore index=255 masked for ALL metrics.
- Logit resizing uses bilinear ONLY (never nearest).
- Compare Nearest / B / Ensemble(wN/wB) consistently.

Inputs required
---------------
Nearest (01_nearest):
  - 01_nearest/golden_artifacts/oof_logits.npy
  - 01_nearest/golden_artifacts/oof_file_ids.npy

Model B (02_nonstruct frozen bundle):
  - 00_data/02_nonstruct_frozen/golden_artifacts/oof/oof_fold{k}_logits.npy  (k=0..4)
    Note: we assume ordering matches val_ids derived from folds_v1.json + train_ids.txt.
          If it doesn't, you must provide the exact val_ids list used when saving those logits.

GT dataset (NYUv2 train):
  - --label-dir /path/to/NYUv2/train/label   (required)
  - --image-dir /path/to/NYUv2/train/image   (optional, for viz)
  - --depth-dir /path/to/NYUv2/train/depth   (optional, for depth bins/viz)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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


def _p(*xs: str) -> str:
    return os.path.join(*xs)


def _die(msg: str) -> None:
    raise SystemExit(f"[E3][FATAL] {msg}")


def _ensure_exists(path: str, desc: str) -> None:
    if not os.path.exists(path):
        _die(f"Missing {desc}: {path}")


def _nanmean(xs: Sequence[float]) -> float:
    ys = [x for x in xs if not (math.isnan(x) or math.isinf(x))]
    return float(sum(ys) / len(ys)) if ys else float("nan")


def _maybe_load_cv2() -> None:
    if cv2 is None:
        _die("OpenCV (cv2) is required (pip install opencv-python).")


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _load_splits_folds_map(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # Support two formats:
    # (A) Mapping format: {"folds": {"000006.png": 0, ...}, ...}
    # (B) Manifest format: {"folds": ["fold0.json", ...], ...} where each fold file
    #     contains {"train_ids":[...], "val_ids":[...]}.
    if "folds" not in obj:
        _die(f"Invalid splits file (missing key 'folds'): {path}")

    def norm_id(x: str) -> str:
        s = str(x).strip()
        # Repo conventions are mixed: some splits store "000002", others "000002.png".
        if s and (not s.endswith(".png")):
            s = s + ".png"
        return s

    if isinstance(obj["folds"], dict):
        out: Dict[str, int] = {}
        for k, v in obj["folds"].items():
            out[norm_id(k)] = int(v)
        return out

    if isinstance(obj["folds"], list):
        base = os.path.dirname(path)
        out: Dict[str, int] = {}
        for fold, fname in enumerate(obj["folds"]):
            fpath = os.path.join(base, str(fname))
            if not os.path.exists(fpath):
                _die(f"Split fold file not found: {fpath}")
            with open(fpath, "r", encoding="utf-8") as f:
                fd = json.load(f)
            val_ids = fd.get("val_ids")
            if not isinstance(val_ids, list):
                _die(f"Split fold file missing 'val_ids' list: {fpath}")
            for fid in val_ids:
                sfid = norm_id(fid)
                if sfid in out and out[sfid] != fold:
                    _die(f"ID appears in multiple folds: {sfid} (fold {out[sfid]} and {fold})")
                out[sfid] = fold
        return out

    _die(f"Invalid splits file format for 'folds': {path}")


def _load_val_ids_by_fold(path: str) -> Optional[List[List[str]]]:
    """
    If splits file is the manifest format (fold json filenames), return val_ids per fold
    preserving the exact ordering in each fold file. Otherwise return None.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    folds = obj.get("folds")
    if not isinstance(folds, list):
        return None

    def norm_id(x: str) -> str:
        s = str(x).strip()
        if s and (not s.endswith(".png")):
            s = s + ".png"
        return s
    base = os.path.dirname(path)
    out: List[List[str]] = []
    for fname in folds:
        fpath = os.path.join(base, str(fname))
        if not os.path.exists(fpath):
            _die(f"Split fold file not found: {fpath}")
        with open(fpath, "r", encoding="utf-8") as ff:
            fd = json.load(ff)
        val_ids = fd.get("val_ids")
        if not isinstance(val_ids, list):
            _die(f"Split fold file missing 'val_ids' list: {fpath}")
        out.append([norm_id(x) for x in val_ids])
    return out


def confusion_matrix(gt: np.ndarray, pred: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Confusion matrix over valid pixels only (gt!=255)."""
    if gt.shape != pred.shape:
        _die(f"Shape mismatch gt={gt.shape} pred={pred.shape}")
    gt_i = gt.astype(np.int64, copy=False)
    pr_i = pred.astype(np.int64, copy=False)
    m = (gt_i != IGNORE_INDEX) & (gt_i >= 0) & (gt_i < num_classes)
    if not np.any(m):
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    idx = gt_i[m] * num_classes + pr_i[m]
    cm = np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm.astype(np.int64, copy=False)


def iou_from_confusion(cm: np.ndarray) -> List[float]:
    """Per-class IoU; union==0 => NaN."""
    num_classes = cm.shape[0]
    ious: List[float] = []
    for c in range(num_classes):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - cm[c, c])
        fn = float(cm[c, :].sum() - cm[c, c])
        union = tp + fp + fn
        ious.append(float("nan") if union <= 0 else tp / union)
    return ious


def miou_from_confusion(cm: np.ndarray) -> float:
    return _nanmean(iou_from_confusion(cm))


def overall_miou_global(
    *,
    folds_map: Dict[str, int],
    train_ids: List[str],
    per_fold_cm: List[np.ndarray],
) -> float:
    """
    Compute OOF-global mIoU by summing fold confusion matrices then mIoU.
    This is the correct counterpart to Oracle-2 overall (both are global IoU).
    """
    if len(per_fold_cm) != 5:
        _die(f"Expected 5 folds confusion matrices, got {len(per_fold_cm)}")
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for f in range(5):
        cm += per_fold_cm[f]
    return miou_from_confusion(cm)


def load_label(label_dir: str, file_id: str) -> np.ndarray:
    _maybe_load_cv2()
    p = _p(label_dir, file_id)
    if not os.path.exists(p):
        _die(f"GT label not found: {p}")
    lbl = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if lbl is None:
        _die(f"Failed to read GT label: {p}")
    if lbl.ndim == 3:
        lbl = lbl[:, :, 0]
    return lbl.astype(np.int64, copy=False)


def load_depth(depth_dir: str, file_id: str) -> Optional[np.ndarray]:
    if not depth_dir:
        return None
    _maybe_load_cv2()
    p = _p(depth_dir, file_id)
    if not os.path.exists(p):
        return None
    d = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    return (d.astype(np.float32) / 1000.0).astype(np.float32, copy=False)


def load_rgb(image_dir: str, file_id: str) -> Optional[np.ndarray]:
    if not image_dir:
        return None
    _maybe_load_cv2()
    p = _p(image_dir, file_id)
    if not os.path.exists(p):
        return None
    img = cv2.imread(p)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def boundary_mask_from_gt(gt: np.ndarray, k: int = 3) -> np.ndarray:
    """01_nearest protocol: replace 255->0, boundary=dilate!=erode, then mask by valid."""
    _maybe_load_cv2()
    gt_u8 = gt.astype(np.uint8, copy=False)
    valid = (gt_u8 != IGNORE_INDEX)
    gt_clean = gt_u8.copy()
    gt_clean[~valid] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    d = cv2.dilate(gt_clean, kernel)
    e = cv2.erode(gt_clean, kernel)
    return (d != e) & valid


def maxprob_from_logits(logits_chw: np.ndarray) -> np.ndarray:
    """
    max softmax probability without full softmax:
    p_max = 1 / sum_j exp(l_j - max_l)
    """
    x = logits_chw.astype(np.float32, copy=False)
    m = np.max(x, axis=0)  # (H,W)
    denom = np.exp(x - m[None, :, :]).sum(axis=0)
    return (1.0 / denom).astype(np.float32, copy=False)


def resize_logits_bilinear(logits_chw: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    _maybe_load_cv2()
    c, h, w = logits_chw.shape
    out_h, out_w = out_hw
    if (h, w) == (out_h, out_w):
        return logits_chw.astype(np.float32, copy=False)
    out = np.empty((c, out_h, out_w), dtype=np.float32)
    for i in range(c):
        out[i] = cv2.resize(
            logits_chw[i].astype(np.float32, copy=False),
            (out_w, out_h),
            interpolation=cv2.INTER_LINEAR,
        )
    return out


@dataclass
class ModelAccum:
    cm: np.ndarray

    def __init__(self) -> None:
        self.cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    def update(self, gt: np.ndarray, pred: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        if mask is None:
            self.cm += confusion_matrix(gt, pred, NUM_CLASSES)
        else:
            gt2 = gt.copy()
            gt2[~mask] = IGNORE_INDEX
            self.cm += confusion_matrix(gt2, pred, NUM_CLASSES)


def _write_csv(path: str, header: List[str], rows: List[List[object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def _palette() -> np.ndarray:
    # fixed palette for 13 classes (RGB)
    return np.array(
        [
            [128, 0, 0],    # bed
            [0, 128, 0],    # books
            [128, 128, 0],  # ceiling
            [0, 0, 128],    # chair
            [128, 0, 128],  # floor
            [0, 128, 128],  # furniture
            [192, 192, 192],# objects
            [64, 0, 0],     # picture
            [0, 64, 0],     # sofa
            [64, 64, 0],    # table
            [0, 0, 64],     # tv
            [64, 0, 64],    # wall
            [0, 64, 64],    # window
        ],
        dtype=np.uint8,
    )


def colorize_label(lbl: np.ndarray) -> np.ndarray:
    pal = _palette()
    out = np.zeros((lbl.shape[0], lbl.shape[1], 3), dtype=np.uint8)
    m = (lbl != IGNORE_INDEX) & (lbl >= 0) & (lbl < NUM_CLASSES)
    out[m] = pal[lbl[m].astype(np.int64)]
    # ignore -> white
    out[~m] = np.array([255, 255, 255], dtype=np.uint8)
    return out


def gray_depth(depth_m: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    # normalize depth for display (0..255), using valid pixels only
    d = depth_m.astype(np.float32, copy=False)
    m = valid_mask & (d > 0)
    if not np.any(m):
        return np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
    v = d[m]
    lo, hi = float(np.percentile(v, 2)), float(np.percentile(v, 98))
    hi = max(hi, lo + 1e-6)
    x = (np.clip(d, lo, hi) - lo) / (hi - lo)
    g = (x * 255.0).astype(np.uint8)
    rgb = np.stack([g, g, g], axis=-1)
    rgb[~m] = 0
    return rgb


def overlay_mask(img_rgb: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.6) -> np.ndarray:
    out = img_rgb.copy()
    c = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    m = mask.astype(bool)
    out[m] = (alpha * out[m].astype(np.float32) + (1 - alpha) * c).astype(np.uint8)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Users/koizumihinata/NYUv2")

    ap.add_argument("--nearest-oof-logits", default="01_nearest/golden_artifacts/oof_logits.npy")
    ap.add_argument("--nearest-oof-ids", default="01_nearest/golden_artifacts/oof_file_ids.npy")

    ap.add_argument("--b-oof-dir", default="00_data/02_nonstruct_frozen/golden_artifacts/oof")
    ap.add_argument("--splits", default="00_data/splits/folds_v1.json")
    ap.add_argument("--train-ids", default="00_data/ids/train_ids.txt")

    ap.add_argument("--label-dir", default="", help="NYUv2 train label dir: /path/to/NYUv2/train/label (required)")
    ap.add_argument("--image-dir", default="", help="NYUv2 train image dir: /path/to/NYUv2/train/image (optional)")
    ap.add_argument("--depth-dir", default="", help="NYUv2 train depth dir: /path/to/NYUv2/train/depth (optional)")

    ap.add_argument("--out-dir", default="00_data/output/e3_nearest_plus_b")
    ap.add_argument("--w-nearest", type=float, default=0.8)
    ap.add_argument("--w-b", type=float, default=0.2)

    ap.add_argument("--target-h", type=int, default=480)
    ap.add_argument("--target-w", type=int, default=640)

    ap.add_argument("--highconf-thresholds", default="0.9")
    ap.add_argument("--depth-bins", default="0,1,2,3,5,10")

    ap.add_argument("--cc-small-max", type=int, default=200)
    ap.add_argument("--cc-medium-max", type=int, default=2000)

    ap.add_argument("--viz-topk", type=int, default=20)
    args = ap.parse_args()

    root = args.root
    out_dir = _p(root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # --- Existence checks (fail fast, actionable) ---
    splits_path = _p(root, args.splits)
    train_ids_path = _p(root, args.train_ids)
    _ensure_exists(splits_path, "splits file")
    _ensure_exists(train_ids_path, "train ids file")

    nearest_logits_path = _p(root, args.nearest_oof_logits)
    nearest_ids_path = _p(root, args.nearest_oof_ids)
    _ensure_exists(nearest_ids_path, "Nearest OOF ids (oof_file_ids.npy)")
    _ensure_exists(nearest_logits_path, "Nearest OOF logits (oof_logits.npy)")

    b_oof_dir = _p(root, args.b_oof_dir)
    for k in range(5):
        _ensure_exists(_p(b_oof_dir, f"oof_fold{k}_logits.npy"), f"Model B OOF logits fold{k}")

    if not args.label_dir:
        _die(
            "GT labels are required. Provide --label-dir /path/to/NYUv2/train/label.\n"
            "Optional: --image-dir /path/to/NYUv2/train/image, --depth-dir /path/to/NYUv2/train/depth"
        )
    label_dir = args.label_dir
    image_dir = args.image_dir
    depth_dir = args.depth_dir
    _ensure_exists(label_dir, "GT label directory")
    if image_dir and not os.path.exists(image_dir):
        _die(f"Image dir does not exist: {image_dir}")
    if depth_dir and not os.path.exists(depth_dir):
        _die(f"Depth dir does not exist: {depth_dir}")

    out_hw = (int(args.target_h), int(args.target_w))
    wN = float(args.w_nearest)
    wB = float(args.w_b)
    if wN < 0 or wB < 0 or abs((wN + wB) - 1.0) > 1e-6:
        _die(f"Ensemble weights must be non-negative and sum to 1.0. Got wN={wN}, wB={wB}")

    thresholds = [float(x) for x in args.highconf_thresholds.split(",") if x.strip()]
    depth_edges = [float(x) for x in args.depth_bins.split(",") if x.strip()]
    if sorted(depth_edges) != depth_edges or len(depth_edges) < 2:
        _die(f"Invalid --depth-bins: {args.depth_bins}")

    # --- Load ids/splits ---
    train_ids = _read_lines(train_ids_path)
    folds_map = _load_splits_folds_map(splits_path)
    val_ids_by_fold = _load_val_ids_by_fold(splits_path)
    missing_fold = [fid for fid in train_ids if fid not in folds_map]
    if missing_fold:
        _die(f"{len(missing_fold)} train ids missing in folds mapping. Example: {missing_fold[:5]}")

    # --- Load Nearest logits (memmap) + id mapping ---
    nearest_ids = np.load(nearest_ids_path, allow_pickle=False)
    nearest_ids_list = [str(x) for x in nearest_ids.tolist()]
    nearest_index: Dict[str, int] = {fid: i for i, fid in enumerate(nearest_ids_list)}

    n_logits = np.load(nearest_logits_path, mmap_mode="r", allow_pickle=False)
    if n_logits.ndim != 4 or n_logits.shape[1] != NUM_CLASSES:
        _die(f"Nearest oof_logits.npy expected shape (N,13,H,W), got {n_logits.shape}")

    # --- Load Model B fold logits (memmap) and per-fold ID index assumption ---
    b_logits_by_fold: Dict[int, np.ndarray] = {}
    b_index_by_fold: Dict[int, Dict[str, int]] = {}
    for fold in range(5):
        path = _p(b_oof_dir, f"oof_fold{fold}_logits.npy")
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        if arr.ndim != 4 or arr.shape[1] != NUM_CLASSES:
            _die(f"Model B fold{fold} logits expected (N,13,H,W), got {arr.shape}")
        b_logits_by_fold[fold] = arr
        if val_ids_by_fold is not None:
            # Use canonical ordering from the split fold json (matches how OOF is typically saved).
            val_ids = list(val_ids_by_fold[fold])
        else:
            # Fallback: derive val_ids from mapping, preserving train_ids order.
            val_ids = [fid for fid in train_ids if folds_map[fid] == fold]
        if len(val_ids) != arr.shape[0]:
            _die(
                f"Fold{fold}: derived val_ids length {len(val_ids)} != logits N {arr.shape[0]}.\n"
                "Ordering cannot be inferred. Provide the exact val_ids list used when saving the fold logits."
            )
        b_index_by_fold[fold] = {fid: i for i, fid in enumerate(val_ids)}

    # --- Global accumulators ---
    acc_n = ModelAccum()
    acc_b = ModelAccum()
    acc_e = ModelAccum()
    acc_oracle = ModelAccum()

    fold_acc_n = [ModelAccum() for _ in range(5)]
    fold_acc_b = [ModelAccum() for _ in range(5)]
    fold_acc_e = [ModelAccum() for _ in range(5)]

    # pixel partitions overall + per GT class
    part_A = part_B1 = part_B2 = part_C = 0
    gt_counts = np.zeros((NUM_CLASSES,), dtype=np.int64)
    b1_counts = np.zeros((NUM_CLASSES,), dtype=np.int64)
    b2_counts = np.zeros((NUM_CLASSES,), dtype=np.int64)
    c_counts = np.zeros((NUM_CLASSES,), dtype=np.int64)

    # blindspot by image
    blindspot_rows: List[Tuple[str, float, float, str]] = []

    # boundary/interior
    region_acc = {
        "boundary": {"n": ModelAccum(), "b": ModelAccum(), "e": ModelAccum(), "c": 0, "valid": 0},
        "interior": {"n": ModelAccum(), "b": ModelAccum(), "e": ModelAccum(), "c": 0, "valid": 0},
    }

    # depth bins (optional)
    depth_enabled = bool(depth_dir)
    depth_bin_labels = [f"{depth_edges[i]}-{depth_edges[i+1]}" for i in range(len(depth_edges) - 1)]
    depth_bin_acc = []
    if depth_enabled:
        for _ in depth_bin_labels:
            depth_bin_acc.append(
                {"n": ModelAccum(), "b": ModelAccum(), "e": ModelAccum(), "c": 0, "valid": 0, "c_gt_hist": np.zeros((NUM_CLASSES,), dtype=np.int64)}
            )

    # high-conf wrong
    hi = {}
    for t in thresholds:
        hi[t] = {
            "count": 0,
            "valid": 0,
            "gt_hist": np.zeros((NUM_CLASSES,), dtype=np.int64),
            "depthbin_counts": np.zeros((len(depth_bin_labels),), dtype=np.int64) if depth_enabled else None,
            "depthbin_valid": np.zeros((len(depth_bin_labels),), dtype=np.int64) if depth_enabled else None,
        }

    # error correlation (binary error within GT class)
    err_n11 = np.zeros((NUM_CLASSES,), dtype=np.int64)
    err_n10 = np.zeros((NUM_CLASSES,), dtype=np.int64)
    err_n01 = np.zeros((NUM_CLASSES,), dtype=np.int64)
    err_n00 = np.zeros((NUM_CLASSES,), dtype=np.int64)

    # connected component size buckets
    cc_bins = [
        ("small", 1, int(args.cc_small_max)),
        ("medium", int(args.cc_small_max) + 1, int(args.cc_medium_max)),
        ("large", int(args.cc_medium_max) + 1, 1_000_000_000),
    ]
    cc_acc = {name: {"n": ModelAccum(), "b": ModelAccum(), "e": ModelAccum(), "c": 0, "valid": 0} for (name, _mn, _mx) in cc_bins}

    # store per-image for later viz (topK only; store minimal)
    store_for_viz: Dict[str, Dict[str, np.ndarray]] = {}

    # --- Iterate samples (OOF only) ---
    for file_id in train_ids:
        if file_id not in nearest_index:
            _die(f"Nearest OOF ids missing file_id={file_id}")
        fold = int(folds_map[file_id])

        gt = load_label(label_dir, file_id)
        if gt.shape != out_hw:
            _die(f"GT label shape must be {out_hw} but got {gt.shape} for {file_id}")
        valid = (gt != IGNORE_INDEX)
        valid_pix = int(valid.sum())

        # logits (CHW), resize bilinear only
        n_chw = np.array(n_logits[nearest_index[file_id]], copy=False)
        b_arr = b_logits_by_fold[fold]
        b_pos = b_index_by_fold[fold][file_id]
        b_chw = np.array(b_arr[b_pos], copy=False)

        if n_chw.shape[1:] != out_hw:
            n_chw = resize_logits_bilinear(n_chw, out_hw)
        else:
            n_chw = n_chw.astype(np.float32, copy=False)
        if b_chw.shape[1:] != out_hw:
            b_chw = resize_logits_bilinear(b_chw, out_hw)
        else:
            b_chw = b_chw.astype(np.float32, copy=False)

        pred_n = np.argmax(n_chw, axis=0).astype(np.int64, copy=False)
        pred_b = np.argmax(b_chw, axis=0).astype(np.int64, copy=False)
        logits_e = (wN * n_chw + wB * b_chw).astype(np.float32, copy=False)
        pred_e = np.argmax(logits_e, axis=0).astype(np.int64, copy=False)

        acc_n.update(gt, pred_n)
        acc_b.update(gt, pred_b)
        acc_e.update(gt, pred_e)
        fold_acc_n[fold].update(gt, pred_n)
        fold_acc_b[fold].update(gt, pred_b)
        fold_acc_e[fold].update(gt, pred_e)

        # partitions
        correct_n = (pred_n == gt) & valid
        correct_b = (pred_b == gt) & valid
        A = correct_n & correct_b
        B1 = correct_n & (~correct_b) & valid
        B2 = (~correct_n) & correct_b & valid
        C = (~correct_n) & (~correct_b) & valid

        part_A += int(A.sum())
        part_B1 += int(B1.sum())
        part_B2 += int(B2.sum())
        part_C += int(C.sum())

        gt_counts += np.bincount(gt[valid], minlength=NUM_CLASSES).astype(np.int64)
        if np.any(B1):
            b1_counts += np.bincount(gt[B1], minlength=NUM_CLASSES).astype(np.int64)
        if np.any(B2):
            b2_counts += np.bincount(gt[B2], minlength=NUM_CLASSES).astype(np.int64)
        if np.any(C):
            c_counts += np.bincount(gt[C], minlength=NUM_CLASSES).astype(np.int64)

        # blindspot per image
        c_pix = int(C.sum())
        c_area_ratio = float(c_pix / valid_pix) if valid_pix > 0 else 0.0
        ens_miou_img = miou_from_confusion(confusion_matrix(gt, pred_e, NUM_CLASSES))
        top_classes = ""
        if c_pix > 0:
            hist_c = np.bincount(gt[C], minlength=NUM_CLASSES)
            top3 = np.argsort(hist_c)[::-1][:3]
            top_classes = ";".join([f"{CLASS_NAMES[int(k)]}:{int(hist_c[int(k)])}" for k in top3 if hist_c[int(k)] > 0])
        blindspot_rows.append((file_id, c_area_ratio, ens_miou_img, top_classes))

        # boundary/interior
        bmask = boundary_mask_from_gt(gt, k=3)
        imask = valid & (~bmask)
        for region, rmask in [("boundary", bmask), ("interior", imask)]:
            region_acc[region]["n"].update(gt, pred_n, rmask)
            region_acc[region]["b"].update(gt, pred_b, rmask)
            region_acc[region]["e"].update(gt, pred_e, rmask)
            region_acc[region]["c"] += int((C & rmask).sum())
            region_acc[region]["valid"] += int(rmask.sum())

        # depth bins (optional)
        depth = load_depth(depth_dir, file_id) if depth_enabled else None
        if depth_enabled:
            if depth is None:
                _die(f"Depth enabled but missing: {file_id}")
            if depth.shape != out_hw:
                _die(f"Depth shape must be {out_hw} but got {depth.shape} for {file_id}")
            d_valid = (depth > 0) & valid
            for bi in range(len(depth_edges) - 1):
                lo, hi_e = depth_edges[bi], depth_edges[bi + 1]
                m_bin = d_valid & (depth >= lo) & (depth < hi_e)
                if not np.any(m_bin):
                    continue
                depth_bin_acc[bi]["n"].update(gt, pred_n, m_bin)
                depth_bin_acc[bi]["b"].update(gt, pred_b, m_bin)
                depth_bin_acc[bi]["e"].update(gt, pred_e, m_bin)
                depth_bin_acc[bi]["c"] += int((C & m_bin).sum())
                depth_bin_acc[bi]["valid"] += int(m_bin.sum())
                if np.any(C & m_bin):
                    depth_bin_acc[bi]["c_gt_hist"] += np.bincount(gt[C & m_bin], minlength=NUM_CLASSES).astype(np.int64)

        # high-confidence wrong (ens)
        pmax = maxprob_from_logits(logits_e)
        wrong_e = (pred_e != gt) & valid
        for t in thresholds:
            hi[t]["valid"] += valid_pix
            m_hi = wrong_e & (pmax > t)
            if np.any(m_hi):
                hi[t]["count"] += int(m_hi.sum())
                hi[t]["gt_hist"] += np.bincount(gt[m_hi], minlength=NUM_CLASSES).astype(np.int64)
                if depth_enabled and depth is not None:
                    d_valid = (depth > 0) & valid
                    for bi in range(len(depth_edges) - 1):
                        lo, hi_e = depth_edges[bi], depth_edges[bi + 1]
                        m_bin = d_valid & (depth >= lo) & (depth < hi_e)
                        if not np.any(m_bin):
                            continue
                        hi[t]["depthbin_valid"][bi] += int(m_bin.sum())
                        hi[t]["depthbin_counts"][bi] += int((m_hi & m_bin).sum())

        # error correlation by GT class: accumulate binary table over GT==c pixels
        for c in range(NUM_CLASSES):
            m_c = (gt == c)
            if not np.any(m_c):
                continue
            en = (pred_n != gt) & m_c
            eb = (pred_b != gt) & m_c
            err_n11[c] += int(np.count_nonzero(en & eb))
            err_n10[c] += int(np.count_nonzero(en & (~eb)))
            err_n01[c] += int(np.count_nonzero((~en) & eb))
            err_n00[c] += int(np.count_nonzero((~en) & (~eb)))

        # oracle-2: pixelwise choose correct from (N,B) if exists
        oracle_pred = pred_e.copy()
        oracle_pred[correct_n] = gt[correct_n]
        oracle_pred[(~correct_n) & correct_b] = gt[(~correct_n) & correct_b]
        acc_oracle.update(gt, oracle_pred)

        # connected component size metrics (per GT class component)
        _maybe_load_cv2()
        for c in range(NUM_CLASSES):
            m = (gt == c)
            if not np.any(m):
                continue
            m_u8 = m.astype(np.uint8)
            ncc, cc_map, stats, _ = cv2.connectedComponentsWithStats(m_u8, connectivity=8)
            for cc_id in range(1, ncc):
                area = int(stats[cc_id, cv2.CC_STAT_AREA])
                comp_mask = (cc_map == cc_id)
                # bucket
                bucket = None
                for name, mn, mx in cc_bins:
                    if mn <= area <= mx:
                        bucket = name
                        break
                if bucket is None:
                    continue
                cc_acc[bucket]["n"].update(gt, pred_n, comp_mask)
                cc_acc[bucket]["b"].update(gt, pred_b, comp_mask)
                cc_acc[bucket]["e"].update(gt, pred_e, comp_mask)
                cc_acc[bucket]["c"] += int(np.count_nonzero(C & comp_mask))
                cc_acc[bucket]["valid"] += int(np.count_nonzero(comp_mask))

        # keep minimal tensors for viz selection later
        # store logits_e for pmax thresholding? not needed. store preds + C + gt.
        store_for_viz[file_id] = {"gt": gt, "pred_n": pred_n, "pred_b": pred_b, "pred_e": pred_e, "C": C.astype(np.uint8)}

    # --- 1) per_class_iou.csv ---
    iou_n = iou_from_confusion(acc_n.cm)
    iou_b = iou_from_confusion(acc_b.cm)
    iou_e = iou_from_confusion(acc_e.cm)
    rows = []
    for c in range(NUM_CLASSES):
        dn = iou_e[c] - iou_n[c] if not (math.isnan(iou_e[c]) or math.isnan(iou_n[c])) else float("nan")
        resid = 1.0 - iou_e[c] if not math.isnan(iou_e[c]) else float("nan")
        rows.append([c, CLASS_NAMES[c], iou_n[c], iou_b[c], iou_e[c], dn, resid])
    _write_csv(
        _p(out_dir, "per_class_iou.csv"),
        ["class_id", "class_name", "iou_nearest", "iou_B", "iou_ens", "delta_iou", "residual"],
        rows,
    )

    # --- 1-2) pixel_partition.csv (overall + by class) ---
    total_valid = part_A + part_B1 + part_B2 + part_C
    def _ratio(x: int) -> float:
        return float(x / total_valid) if total_valid > 0 else 0.0

    rows = []
    rows.append(["overall", "", _ratio(part_A), _ratio(part_B1), _ratio(part_B2), _ratio(part_C)])
    for c in range(NUM_CLASSES):
        denom = int(gt_counts[c])
        b2r = float(b2_counts[c] / denom) if denom > 0 else float("nan")
        cr = float(c_counts[c] / denom) if denom > 0 else float("nan")
        rows.append(["class", c, "", "", b2r, cr])
    _write_csv(
        _p(out_dir, "pixel_partition.csv"),
        ["level", "class_id", "area_ratio_A", "area_ratio_B1", "area_ratio_B2", "area_ratio_C"],
        rows,
    )

    # --- 1-3) fold_delta_miou.csv ---
    rows = []
    for fold in range(5):
        mi_n = miou_from_confusion(fold_acc_n[fold].cm)
        mi_b = miou_from_confusion(fold_acc_b[fold].cm)
        mi_e = miou_from_confusion(fold_acc_e[fold].cm)
        rows.append([fold, mi_n, mi_b, mi_e, mi_e - mi_n])

    # Add an explicit GLOBAL row to avoid common confusion:
    # mean(fold mIoU) != global mIoU in general (nonlinear IoU + per-fold class presence).
    mi_n_g = miou_from_confusion(acc_n.cm)
    mi_b_g = miou_from_confusion(acc_b.cm)
    mi_e_g = miou_from_confusion(acc_e.cm)
    rows.append(["global", mi_n_g, mi_b_g, mi_e_g, mi_e_g - mi_n_g])
    _write_csv(
        _p(out_dir, "fold_delta_miou.csv"),
        ["fold", "miou_nearest", "miou_B", "miou_ens", "delta_ens_vs_nearest"],
        rows,
    )

    # --- 2-1) blindspot_by_image_top50.csv ---
    blindspot_rows.sort(key=lambda x: x[1], reverse=True)
    top50 = blindspot_rows[:50]
    _write_csv(
        _p(out_dir, "blindspot_by_image_top50.csv"),
        ["image_id", "c_area_ratio", "ens_miou_image", "top_classes_in_C"],
        [[a, b, c, d] for (a, b, c, d) in top50],
    )

    # --- 3-1) boundary_interior.csv ---
    rows = []
    for region in ["boundary", "interior"]:
        mi_n = miou_from_confusion(region_acc[region]["n"].cm)
        mi_b = miou_from_confusion(region_acc[region]["b"].cm)
        mi_e = miou_from_confusion(region_acc[region]["e"].cm)
        v = int(region_acc[region]["valid"])
        cpx = int(region_acc[region]["c"])
        c_ratio = float(cpx / v) if v > 0 else 0.0
        rows.append([region, mi_n, mi_b, mi_e, c_ratio])
    _write_csv(
        _p(out_dir, "boundary_interior.csv"),
        ["region", "miou_nearest", "miou_B", "miou_ens", "c_area_ratio"],
        rows,
    )

    # --- 3-2) depthbin_metrics.csv (if depth enabled) ---
    if depth_enabled:
        rows = []
        for bi, label in enumerate(depth_bin_labels):
            mi_n = miou_from_confusion(depth_bin_acc[bi]["n"].cm)
            mi_b = miou_from_confusion(depth_bin_acc[bi]["b"].cm)
            mi_e = miou_from_confusion(depth_bin_acc[bi]["e"].cm)
            v = int(depth_bin_acc[bi]["valid"])
            cpx = int(depth_bin_acc[bi]["c"])
            c_ratio = float(cpx / v) if v > 0 else 0.0
            top_classes = ""
            hist = depth_bin_acc[bi]["c_gt_hist"]
            if int(hist.sum()) > 0:
                top3 = np.argsort(hist)[::-1][:3]
                top_classes = ";".join([f"{CLASS_NAMES[int(k)]}:{int(hist[int(k)])}" for k in top3 if hist[int(k)] > 0])
            rows.append([label, mi_n, mi_b, mi_e, c_ratio, top_classes])
        _write_csv(
            _p(out_dir, "depthbin_metrics.csv"),
            ["bin", "miou_nearest", "miou_B", "miou_ens", "c_area_ratio", "top_classes_in_C"],
            rows,
        )

    # --- 3-3) highconf_wrong.csv ---
    rows = []
    for t in thresholds:
        valid = int(hi[t]["valid"])
        cnt = int(hi[t]["count"])
        area_ratio = float(cnt / valid) if valid > 0 else 0.0
        # class breakdown: top5
        hist = hi[t]["gt_hist"]
        top5 = np.argsort(hist)[::-1][:5]
        cls_break = ";".join([f"{CLASS_NAMES[int(k)]}:{int(hist[int(k)])}" for k in top5 if hist[int(k)] > 0])
        rows.append([t, cnt, area_ratio, cls_break])
        if depth_enabled:
            # add one extra row per depth bin for this threshold
            for bi, label in enumerate(depth_bin_labels):
                vb = int(hi[t]["depthbin_valid"][bi])
                cb = int(hi[t]["depthbin_counts"][bi])
                rows.append([f"{t}@{label}", cb, float(cb / vb) if vb > 0 else 0.0, ""])
    _write_csv(
        _p(out_dir, "highconf_wrong.csv"),
        ["threshold", "count", "area_ratio", "class_breakdown_top5"],
        rows,
    )

    # --- 3-4) error_correlation_by_class.csv (phi coefficient) ---
    rows = []
    for c in range(NUM_CLASSES):
        n11, n10, n01, n00 = float(err_n11[c]), float(err_n10[c]), float(err_n01[c]), float(err_n00[c])
        denom = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
        corr = (n11 * n00 - n10 * n01) / math.sqrt(denom) if denom > 0 else float("nan")
        denom_gt = float(gt_counts[c])
        c_ratio = float(c_counts[c] / denom_gt) if denom_gt > 0 else float("nan")
        rows.append([c, CLASS_NAMES[c], corr, c_ratio])
    _write_csv(
        _p(out_dir, "error_correlation_by_class.csv"),
        ["class_id", "class_name", "corr_error_mask", "c_area_ratio"],
        rows,
    )

    # --- 3-5) cc_size_metrics.csv ---
    rows = []
    for name, _mn, _mx in cc_bins:
        mi_n = miou_from_confusion(cc_acc[name]["n"].cm)
        mi_b = miou_from_confusion(cc_acc[name]["b"].cm)
        mi_e = miou_from_confusion(cc_acc[name]["e"].cm)
        v = int(cc_acc[name]["valid"])
        cpx = int(cc_acc[name]["c"])
        c_ratio = float(cpx / v) if v > 0 else 0.0
        rows.append([name, mi_n, mi_b, mi_e, c_ratio, v])
    _write_csv(
        _p(out_dir, "cc_size_metrics.csv"),
        ["cc_size", "miou_nearest", "miou_B", "miou_ens", "c_area_ratio", "pixel_count"],
        rows,
    )

    # --- 4) oracle2_upperbound.csv (overall + per-class gap) ---
    mi_oracle = miou_from_confusion(acc_oracle.cm)
    mi_ens = miou_from_confusion(acc_e.cm)
    rows = [["overall", mi_oracle, mi_oracle - mi_ens]]
    iou_or = iou_from_confusion(acc_oracle.cm)
    for c in range(NUM_CLASSES):
        if math.isnan(iou_or[c]) or math.isnan(iou_e[c]):
            gap = float("nan")
        else:
            gap = iou_or[c] - iou_e[c]
        rows.append([CLASS_NAMES[c], iou_or[c], gap])
    _write_csv(
        _p(out_dir, "oracle2_upperbound.csv"),
        ["key", "score_oracle2", "gap_to_ens"],
        rows,
    )

    # --- 2-2) visualizations topK (optional) ---
    if image_dir:
        viz_dir = _p(out_dir, "viz", "e3_blindspot_top")
        os.makedirs(viz_dir, exist_ok=True)
        topk = min(int(args.viz_topk), len(top50))
        for file_id, _c_ratio, _mi, _tc in top50[:topk]:
            rgb = load_rgb(image_dir, file_id)
            if rgb is None:
                continue
            gt = store_for_viz[file_id]["gt"]
            pred_n = store_for_viz[file_id]["pred_n"]
            pred_b = store_for_viz[file_id]["pred_b"]
            pred_e = store_for_viz[file_id]["pred_e"]
            C = store_for_viz[file_id]["C"].astype(bool)
            valid = (gt != IGNORE_INDEX)
            depth = load_depth(depth_dir, file_id) if depth_dir else None

            # panels (all RGB uint8)
            pan_rgb = rgb
            pan_depth = gray_depth(depth, valid) if depth is not None else np.zeros_like(rgb)
            pan_gt = colorize_label(gt)
            pan_n = colorize_label(pred_n)
            pan_b = colorize_label(pred_b)
            pan_e = colorize_label(pred_e)
            err_n = overlay_mask(pan_n, (pred_n != gt) & valid, (255, 0, 0), alpha=0.4)
            err_b = overlay_mask(pan_b, (pred_b != gt) & valid, (255, 0, 0), alpha=0.4)
            err_e = overlay_mask(pan_e, (pred_e != gt) & valid, (255, 0, 0), alpha=0.4)
            blind = overlay_mask(pan_rgb, C, (255, 0, 255), alpha=0.5)

            # compose 3x3 grid: RGB, Depth, GT / PredN, PredB, PredE / ErrN, ErrB, ErrE, plus Blindspot as last (we'll do 3x4)
            row1 = np.concatenate([pan_rgb, pan_depth, pan_gt], axis=1)
            row2 = np.concatenate([pan_n, pan_b, pan_e], axis=1)
            row3 = np.concatenate([err_n, err_b, err_e], axis=1)
            grid = np.concatenate([row1, row2, row3], axis=0)

            # append blindspot mask under as full width
            blind_row = np.concatenate([blind, blind, blind], axis=1)
            grid = np.concatenate([grid, blind_row], axis=0)

            out_path = _p(viz_dir, file_id.replace(".png", "") + "_e3.png")
            _maybe_load_cv2()
            cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    print(f"[E3] Done. Outputs: {out_dir}")


if __name__ == "__main__":
    main()

