"""
optimize_ensemble_3model.py

Ensemble-3: Base=Ensemble-2(frozen), Delta=(Model C - Base), learn beta for push usefulness.
Focus on books/table only, with conservative gating.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# Reuse utilities from Ensemble-2 optimize script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OPT_PATH = os.path.join(REPO_ROOT, "02_nonstruct", "research")
sys.path.append(OPT_PATH)

import optimize_ensemble_2model as opt


NUM_CLASSES = 13
IGNORE_INDEX = 255
CLASS_NAMES = opt.CLASS_NAMES

CID_BOOKS = 1
CID_TABLE = 9
CID_FURNITURE = 5
CID_OBJECTS = 6


def _die(msg: str) -> None:
    raise SystemExit(f"[E3C][FATAL] {msg}")


def _ensure(path: str, desc: str) -> None:
    if not os.path.exists(path):
        _die(f"Missing {desc}: {path}")


def load_ids_txt(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            out.append(t)
    return out


def build_c_index(c_oof_dir: str) -> Dict[str, Tuple[int, int]]:
    """Map file_id -> (fold, idx) using val_ids_fold{k}.txt in Model C output."""
    out: Dict[str, Tuple[int, int]] = {}
    for fold in range(5):
        ids_path = os.path.join(c_oof_dir, f"val_ids_fold{fold}.txt")
        logits_path = os.path.join(c_oof_dir, f"oof_fold{fold}_logits.npy")
        _ensure(ids_path, f"Model C val_ids_fold{fold}.txt")
        _ensure(logits_path, f"Model C oof_fold{fold}_logits.npy")
        ids = np.loadtxt(ids_path, dtype=str)
        for idx, fid in enumerate(ids.tolist()):
            sfid = opt.norm_id(fid)
            out[sfid] = (fold, idx)
    return out


def load_c_logits(c_oof_dir: str, fold: int) -> np.ndarray:
    path = os.path.join(c_oof_dir, f"oof_fold{fold}_logits.npy")
    _ensure(path, f"Model C oof_fold{fold}_logits.npy")
    return np.load(path, mmap_mode="r")


def top3_from_row(row: np.ndarray) -> List[Tuple[int, int]]:
    pairs = [(i, int(row[i])) for i in range(len(row)) if int(row[i]) > 0]
    pairs.sort(key=lambda x: (-x[1], x[0]))
    return pairs[:3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-dir", default="00_data/ensemble2_frozen")
    ap.add_argument("--label-dir", default="00_data/train/label")
    ap.add_argument("--depth-dir", default="00_data/train/depth")
    ap.add_argument("--splits-manifest", default="00_data/splits/folds_v1.json")
    ap.add_argument("--nearest-oof-logits", default="01_nearest/golden_artifacts/oof/oof_logits.npy")
    ap.add_argument("--nearest-oof-ids", default="01_nearest/golden_artifacts/oof_file_ids.npy")
    ap.add_argument("--b-oof-dir", default="00_data/02_nonstruct_frozen/golden_artifacts/oof")
    ap.add_argument("--c-oof-dir", default="03_model_c/output")
    ap.add_argument("--out-dir", default="00_data/output/ensemble3_c_push")
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--push-eps", type=float, default=0.25)
    ap.add_argument("--per-image-cap", type=int, default=1500)
    ap.add_argument("--per-class-cap", type=int, default=350)
    ap.add_argument("--neg-keep-prob", type=float, default=0.25)
    ap.add_argument("--require-base-fo", action="store_true", help="Only gate when pred_base in {furniture,objects}")
    ap.add_argument("--save-logits", action="store_true")
    args = ap.parse_args()

    # Ensure paths
    _ensure(args.freeze_dir, "freeze dir")
    _ensure(args.label_dir, "label dir")
    _ensure(args.splits_manifest, "splits manifest")
    _ensure(args.nearest_oof_logits, "nearest oof logits")
    _ensure(args.nearest_oof_ids, "nearest oof ids")
    _ensure(args.b_oof_dir, "model B oof dir")
    _ensure(args.c_oof_dir, "model C oof dir")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load frozen spec + coefficients (Ensemble-2)
    spec_path = os.path.join(args.freeze_dir, "spec.json")
    coef_path = os.path.join(args.freeze_dir, "pushlr_coefficients.json")
    _ensure(spec_path, "spec.json")
    _ensure(coef_path, "pushlr_coefficients.json")

    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    with open(coef_path, "r", encoding="utf-8") as f:
        coef = json.load(f)

    wB_map: Dict[str, float] = spec["base"]["wB"]
    wB = np.array([float(wB_map[nm]) for nm in CLASS_NAMES], dtype=np.float32)
    S_names = list(spec["mask"]["S_prime"])
    S = set(CLASS_NAMES.index(x) for x in S_names)

    base_push = opt.PushLRModel(
        b=float(coef["b"]),
        w_pmax0=float(coef["w_pmax0"]),
        w_m0=float(coef["w_m0"]),
        w_pmaxN=float(coef["w_pmaxN"]),
        w_pmaxB=float(coef["w_pmaxB"]),
        w_mN=float(coef["w_mN"]),
        w_mB=float(coef["w_mB"]),
        w_dtop1=float(coef["w_dtop1"]),
        w_dgap=float(coef["w_dgap"]),
        w_boundary=float(coef["w_boundary"]),
        w_depth=np.array(coef["w_depth"], dtype=np.float32),
        w_pred0=np.array(coef["w_pred0"], dtype=np.float32),
        w_predB=np.array(coef["w_predB"], dtype=np.float32),
    )

    # Load Nearest OOF logits/ids
    n_ids = np.load(args.nearest_oof_ids, allow_pickle=False)
    file_ids = [opt.norm_id(str(x)) for x in n_ids.tolist()]
    n_logits = np.load(args.nearest_oof_logits, mmap_mode="r", allow_pickle=False)
    if n_logits.shape[0] != len(file_ids) or n_logits.shape[1:] != (NUM_CLASSES, 480, 640):
        _die(f"Unexpected nearest logits shape: {n_logits.shape} vs ids={len(file_ids)}")

    # Model B index
    val_ids_by_fold = opt.load_splits_val_ids_by_fold(args.splits_manifest)
    b_index = opt.build_b_index(args.b_oof_dir, val_ids_by_fold)

    # Model C index
    c_index = build_c_index(args.c_oof_dir)
    c_logits_by_fold = {f: load_c_logits(args.c_oof_dir, f) for f in range(5)}

    # Prepare samples for PushLR training
    rng = np.random.default_rng(42)
    samples_push: Dict[str, List[np.ndarray]] = {k: [] for k in [
        "y","pmax0","m0","pmaxN","pmaxB","mN","mB","d_top1","d_gap","boundary","depth_bin","pred0","predB"
    ]}

    sample_cap_per_img = int(args.per_image_cap)
    sample_cap_per_class = int(args.per_class_cap)

    for i, fid in enumerate(file_ids):
        if fid not in b_index.id_to_pos or fid not in c_index:
            _die(f"Missing ID in indices: {fid}")

        gt = opt.load_label(args.label_dir, fid)
        valid = (gt != IGNORE_INDEX)

        ln = np.array(n_logits[i], copy=False)  # Nearest
        lb = opt.logits_b_for_id(b_index, fid)  # Model B

        # Base (Ensemble-2 frozen): classwise mix + pushLR
        L0 = opt.classwise_mix_logits(ln, lb, wB)
        pred0 = np.argmax(L0, axis=0).astype(np.int64, copy=False)

        predB = np.argmax(lb, axis=0).astype(np.int64, copy=False)
        predN = np.argmax(ln, axis=0).astype(np.int64, copy=False)

        disagree = (predN != predB) & valid
        inS = np.isin(predB, list(S))
        mask = disagree & inS

        L_base = L0.copy()
        if np.any(mask):
            ys, xs = np.where(mask)
            bnd = opt.boundary_mask_from_gt(gt, k=3)
            depth = opt.load_depth_m(args.depth_dir, fid)
            db = opt.depth_bin_at(depth, ys, xs)

            pmax0 = opt.pmax_from_logits_at(L0, ys, xs)
            m0 = opt.margin_from_logits_at(L0, ys, xs)
            pmaxN = opt.pmax_from_logits_at(ln, ys, xs)
            pmaxB = opt.pmax_from_logits_at(lb, ys, xs)
            mN = opt.margin_from_logits_at(ln, ys, xs)
            mB = opt.margin_from_logits_at(lb, ys, xs)
            bd = bnd[ys, xs].astype(np.float32, copy=False)
            c0 = pred0[ys, xs].astype(np.int64, copy=False)
            cB = predB[ys, xs].astype(np.int64, copy=False)

            d = (lb - ln).astype(np.float32, copy=False)
            d_s = d[:, ys, xs]
            x0 = L0[:, ys, xs]
            x0_mask = x0.copy()
            x0_mask[c0, np.arange(x0_mask.shape[1])] = -1e9
            top2 = np.argmax(x0_mask, axis=0).astype(np.int64, copy=False)
            d_top1 = d_s[c0, np.arange(d_s.shape[1])].astype(np.float32, copy=False)
            d_top2 = d_s[top2, np.arange(d_s.shape[1])].astype(np.float32, copy=False)
            d_gap = (d_top1 - d_top2).astype(np.float32, copy=False)

            beta = opt.push_lr_predict_beta(
                base_push,
                pmax0=pmax0,
                m0=m0,
                pmaxN=pmaxN,
                pmaxB=pmaxB,
                mN=mN,
                mB=mB,
                d_top1=d_top1,
                d_gap=d_gap,
                boundary=bd,
                depth_bin=db,
                pred0=c0,
                predB=cB,
            )
            beta_eff = beta.copy()
            beta_eff[c0 == gt[ys, xs]] = 0.0
            L_base[:, ys, xs] = (L_base[:, ys, xs] + (float(spec["inference"]["lambda"]) * beta_eff[None, :] * d[:, ys, xs])).astype(np.float32, copy=False)

        # Model C logits
        fold_c, idx_c = c_index[fid]
        lc = np.array(c_logits_by_fold[fold_c][idx_c], copy=False)

        pred_c = np.argmax(lc, axis=0).astype(np.int64, copy=False)
        pred_base = np.argmax(L_base, axis=0).astype(np.int64, copy=False)

        # Gating mask for sampling
        mask_c = (pred_c == CID_BOOKS) | (pred_c == CID_TABLE)
        disagree_bc = (pred_base != pred_c)
        mask_gate = mask_c & disagree_bc & valid
        if args.require_base_fo:
            mask_gate &= (pred_base == CID_FURNITURE) | (pred_base == CID_OBJECTS)
        if not np.any(mask_gate):
            continue

        # y label from small push eps
        d_bc = (lc.astype(np.float32, copy=False) - L_base.astype(np.float32, copy=False))
        pred_eps = np.argmax((L_base + float(args.push_eps) * d_bc), axis=0).astype(np.int64, copy=False)
        y_pos = (pred_base != gt) & (pred_eps == gt) & mask_gate

        ys_all, xs_all = np.where(mask_gate)
        if len(ys_all) == 0:
            continue
        # sample capped per class (pred_c class)
        picked_y = []
        picked_x = []
        for cid in (CID_BOOKS, CID_TABLE):
            m_c = mask_gate & (pred_c == cid)
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
            k = min(sample_cap_per_img, len(ys_all))
            sel = rng.choice(len(ys_all), size=k, replace=False)
            ys = ys_all[sel]
            xs = xs_all[sel]

        if len(ys) > sample_cap_per_img:
            sel = rng.choice(len(ys), size=sample_cap_per_img, replace=False)
            ys = ys[sel]
            xs = xs[sel]

        y = y_pos[ys, xs].astype(np.int64, copy=False)

        # features
        pmax0 = opt.pmax_from_logits_at(L_base, ys, xs)
        m0 = opt.margin_from_logits_at(L_base, ys, xs)
        pmaxN = pmax0
        mN = m0
        pmaxB = opt.pmax_from_logits_at(lc, ys, xs)
        mB = opt.margin_from_logits_at(lc, ys, xs)

        x0 = L_base[:, ys, xs].astype(np.float32, copy=False)
        top1i = np.argmax(x0, axis=0).astype(np.int64, copy=False)
        x0_mask = x0.copy()
        x0_mask[top1i, np.arange(x0.shape[1])] = -1e9
        top2i = np.argmax(x0_mask, axis=0).astype(np.int64, copy=False)
        d_s = d_bc[:, ys, xs]
        d_top1 = d_s[top1i, np.arange(d_s.shape[1])].astype(np.float32, copy=False)
        d_top2 = d_s[top2i, np.arange(d_s.shape[1])].astype(np.float32, copy=False)
        d_gap = (d_top1 - d_top2).astype(np.float32, copy=False)

        bnd = opt.boundary_mask_from_gt(gt, k=3)
        depth = opt.load_depth_m(args.depth_dir, fid)
        db = opt.depth_bin_at(depth, ys, xs)
        c0 = pred_base[ys, xs].astype(np.int64, copy=False)
        cB = pred_c[ys, xs].astype(np.int64, copy=False)

        samples_push["y"].append(y)
        samples_push["pmax0"].append(pmax0)
        samples_push["m0"].append(m0)
        samples_push["pmaxN"].append(pmaxN)
        samples_push["pmaxB"].append(pmaxB)
        samples_push["mN"].append(mN)
        samples_push["mB"].append(mB)
        samples_push["d_top1"].append(d_top1)
        samples_push["d_gap"].append(d_gap)
        samples_push["boundary"].append(bnd[ys, xs].astype(np.int64))
        samples_push["depth_bin"].append(db)
        samples_push["pred0"].append(c0)
        samples_push["predB"].append(cB)

        if (i + 1) % 200 == 0:
            print(f"[E3C][LR] sampled up to image {i+1}/{len(file_ids)}", flush=True)

    # Train PushLR for E3C
    if not samples_push["y"]:
        _die("No samples collected for E3C PushLR.")
    packed = {k: np.concatenate(v) for k, v in samples_push.items()}
    push_model = opt.push_lr_train(
        packed,
        l2=1.0,
        lr=0.05,
        epochs=3,
        seed=42,
        neg_keep_prob=float(args.neg_keep_prob),
    )

    # Save coefficients
    coef_out = {
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
        "w_depth": push_model.w_depth.tolist(),
        "w_pred0": push_model.w_pred0.tolist(),
        "w_predB": push_model.w_predB.tolist(),
    }
    with open(os.path.join(args.out_dir, "pushlr_coefficients.json"), "w", encoding="utf-8") as f:
        json.dump(coef_out, f, indent=2)

    # Evaluate E3C (single pass)
    cm_base = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    cm_e3 = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    books_row = np.zeros((NUM_CLASSES,), dtype=np.int64)
    table_row = np.zeros((NUM_CLASSES,), dtype=np.int64)

    if args.save_logits:
        out_logits = np.lib.format.open_memmap(
            os.path.join(args.out_dir, "oof_logits_e3.npy"),
            mode="w+",
            dtype=np.float16,
            shape=(len(file_ids), NUM_CLASSES, 480, 640),
        )
    else:
        out_logits = None

    for i, fid in enumerate(file_ids):
        gt = opt.load_label(args.label_dir, fid)
        valid = (gt != IGNORE_INDEX)

        ln = np.array(n_logits[i], copy=False)
        lb = opt.logits_b_for_id(b_index, fid)

        L0 = opt.classwise_mix_logits(ln, lb, wB)
        pred0 = np.argmax(L0, axis=0).astype(np.int64, copy=False)

        predB = np.argmax(lb, axis=0).astype(np.int64, copy=False)
        predN = np.argmax(ln, axis=0).astype(np.int64, copy=False)

        disagree = (predN != predB) & valid
        inS = np.isin(predB, list(S))
        mask = disagree & inS

        L_base = L0.copy()
        if np.any(mask):
            ys, xs = np.where(mask)
            bnd = opt.boundary_mask_from_gt(gt, k=3)
            depth = opt.load_depth_m(args.depth_dir, fid)
            db = opt.depth_bin_at(depth, ys, xs)

            pmax0 = opt.pmax_from_logits_at(L0, ys, xs)
            m0 = opt.margin_from_logits_at(L0, ys, xs)
            pmaxN = opt.pmax_from_logits_at(ln, ys, xs)
            pmaxB = opt.pmax_from_logits_at(lb, ys, xs)
            mN = opt.margin_from_logits_at(ln, ys, xs)
            mB = opt.margin_from_logits_at(lb, ys, xs)
            bd = bnd[ys, xs].astype(np.float32, copy=False)
            c0 = pred0[ys, xs].astype(np.int64, copy=False)
            cB = predB[ys, xs].astype(np.int64, copy=False)

            d = (lb - ln).astype(np.float32, copy=False)
            d_s = d[:, ys, xs]
            x0 = L0[:, ys, xs]
            x0_mask = x0.copy()
            x0_mask[c0, np.arange(x0_mask.shape[1])] = -1e9
            top2 = np.argmax(x0_mask, axis=0).astype(np.int64, copy=False)
            d_top1 = d_s[c0, np.arange(d_s.shape[1])].astype(np.float32, copy=False)
            d_top2 = d_s[top2, np.arange(d_s.shape[1])].astype(np.float32, copy=False)
            d_gap = (d_top1 - d_top2).astype(np.float32, copy=False)

            beta = opt.push_lr_predict_beta(
                base_push,
                pmax0=pmax0,
                m0=m0,
                pmaxN=pmaxN,
                pmaxB=pmaxB,
                mN=mN,
                mB=mB,
                d_top1=d_top1,
                d_gap=d_gap,
                boundary=bd,
                depth_bin=db,
                pred0=c0,
                predB=cB,
            )
            beta_eff = beta.copy()
            beta_eff[c0 == gt[ys, xs]] = 0.0
            L_base[:, ys, xs] = (L_base[:, ys, xs] + (float(spec["inference"]["lambda"]) * beta_eff[None, :] * d[:, ys, xs])).astype(np.float32, copy=False)

        # Model C logits
        fold_c, idx_c = c_index[fid]
        lc = np.array(c_logits_by_fold[fold_c][idx_c], copy=False)
        pred_c = np.argmax(lc, axis=0).astype(np.int64, copy=False)
        pred_base = np.argmax(L_base, axis=0).astype(np.int64, copy=False)

        # Gating mask for inference
        mask_c = (pred_c == CID_BOOKS) | (pred_c == CID_TABLE)
        disagree_bc = (pred_base != pred_c)
        mask_gate = mask_c & disagree_bc & valid
        if args.require_base_fo:
            mask_gate &= (pred_base == CID_FURNITURE) | (pred_base == CID_OBJECTS)

        L_final = L_base.copy()
        if np.any(mask_gate):
            ys, xs = np.where(mask_gate)
            bnd = opt.boundary_mask_from_gt(gt, k=3)
            depth = opt.load_depth_m(args.depth_dir, fid)
            db = opt.depth_bin_at(depth, ys, xs)

            pmax0 = opt.pmax_from_logits_at(L_base, ys, xs)
            m0 = opt.margin_from_logits_at(L_base, ys, xs)
            pmaxN = pmax0
            mN = m0
            pmaxB = opt.pmax_from_logits_at(lc, ys, xs)
            mB = opt.margin_from_logits_at(lc, ys, xs)

            d_bc = (lc.astype(np.float32, copy=False) - L_base.astype(np.float32, copy=False))
            x0 = L_base[:, ys, xs]
            top1i = np.argmax(x0, axis=0).astype(np.int64, copy=False)
            x0_mask = x0.copy()
            x0_mask[top1i, np.arange(x0.shape[1])] = -1e9
            top2i = np.argmax(x0_mask, axis=0).astype(np.int64, copy=False)
            d_s = d_bc[:, ys, xs]
            d_top1 = d_s[top1i, np.arange(d_s.shape[1])].astype(np.float32, copy=False)
            d_top2 = d_s[top2i, np.arange(d_s.shape[1])].astype(np.float32, copy=False)
            d_gap = (d_top1 - d_top2).astype(np.float32, copy=False)

            c0 = pred_base[ys, xs].astype(np.int64, copy=False)
            cB = pred_c[ys, xs].astype(np.int64, copy=False)
            beta = opt.push_lr_predict_beta(
                push_model,
                pmax0=pmax0,
                m0=m0,
                pmaxN=pmaxN,
                pmaxB=pmaxB,
                mN=mN,
                mB=mB,
                d_top1=d_top1,
                d_gap=d_gap,
                boundary=bnd[ys, xs].astype(np.float32, copy=False),
                depth_bin=db,
                pred0=c0,
                predB=cB,
            )
            beta_eff = beta.copy()
            beta_eff[c0 == gt[ys, xs]] = 0.0
            L_final[:, ys, xs] = (L_final[:, ys, xs] + (float(args.lam) * beta_eff[None, :] * d_bc[:, ys, xs])).astype(np.float32, copy=False)

        pred_final = np.argmax(L_final, axis=0).astype(np.int64, copy=False)

        opt.confusion_update(cm_base, gt, pred_base)
        opt.confusion_update(cm_e3, gt, pred_final)

        # books/table top3
        valid_gt = valid
        bmask = (gt == CID_BOOKS) & valid_gt
        tmask = (gt == CID_TABLE) & valid_gt
        if np.any(bmask):
            vals, counts = np.unique(pred_final[bmask], return_counts=True)
            for v, c in zip(vals, counts):
                books_row[int(v)] += int(c)
        if np.any(tmask):
            vals, counts = np.unique(pred_final[tmask], return_counts=True)
            for v, c in zip(vals, counts):
                table_row[int(v)] += int(c)

        if out_logits is not None:
            out_logits[i] = L_final.astype(np.float16, copy=False)

        if (i + 1) % 200 == 0:
            print(f"[E3C] processed {i+1}/{len(file_ids)}", flush=True)

    # Summaries
    iou_base = opt.iou_from_cm(cm_base)
    iou_e3 = opt.iou_from_cm(cm_e3)
    miou_base = opt.miou_from_cm(cm_base)
    miou_e3 = opt.miou_from_cm(cm_e3)

    summary = {
        "miou_base": float(miou_base),
        "miou_e3": float(miou_e3),
        "delta_miou": float(miou_e3 - miou_base),
        "per_class_iou_base": {CLASS_NAMES[i]: float(iou_base[i]) for i in range(NUM_CLASSES)},
        "per_class_iou_e3": {CLASS_NAMES[i]: float(iou_e3[i]) for i in range(NUM_CLASSES)},
        "books_top3": top3_from_row(books_row),
        "table_top3": top3_from_row(table_row),
        "lambda": float(args.lam),
        "push_eps": float(args.push_eps),
        "require_base_fo": bool(args.require_base_fo),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # CSV top3
    with open(os.path.join(args.out_dir, "books_table_top3.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gt_class","rank","pred_class_id","pred_class_name","count"])
        for gt_name, row in [("books", books_row), ("table", table_row)]:
            top = top3_from_row(row)
            for rank, (cid, cnt) in enumerate(top, start=1):
                w.writerow([gt_name, rank, cid, CLASS_NAMES[cid], cnt])

    # Save spec
    with open(os.path.join(args.out_dir, "spec.json"), "w", encoding="utf-8") as f:
        json.dump({
            "name": "Ensemble-3 (Base=E2 frozen + C push)",
            "base": "ensemble2_frozen",
            "lambda": float(args.lam),
            "push_eps": float(args.push_eps),
            "gating": {
                "pred_c": ["books","table"],
                "disagree_base_c": True,
                "require_base_furniture_objects": bool(args.require_base_fo),
            },
        }, f, indent=2)

    print(f"[E3C] Done. summary: {os.path.join(args.out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
