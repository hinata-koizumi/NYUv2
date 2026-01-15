"""
analyze_ensemble2_books_table_confusion.py

Compute the GT->Pred confusion breakdown for books/table for the FINAL Ensemble-2 (FROZEN) spec,
in the same OOF setting used during ensemble optimization.

Outputs (under --out-dir):
- books_table_confusion_top3.csv
- books_table_confusion_full.csv
- books_table_error_by_depthbin.csv

Notes
-----
- Uses the frozen bundle at 00_data/ensemble2_frozen for S' + PushLR coefficients + lambda.
- Uses the SAME data contract as optimize_ensemble_2model.py:
  - Nearest OOF logits/ids
  - Model B OOF fold logits (split manifest val_ids order)
  - GT labels (ignore=255 masked)
  - Depth bins from 00_data/train/depth (meters)

- Safety device: pred0==GT => beta=0 (uses GT; evaluation-only). This matches the OOF score you froze.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from . import optimize_ensemble_2model as opt


def _topk_counts(counts: np.ndarray, k: int = 3) -> List[Tuple[int, int]]:
    pairs = [(i, int(counts[i])) for i in range(len(counts)) if int(counts[i]) > 0]
    pairs.sort(key=lambda x: (-x[1], x[0]))
    return pairs[:k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-dir", default="00_data/ensemble2_frozen")
    ap.add_argument("--out-dir", default="00_data/output/ensemble2_frozen_analysis")
    ap.add_argument("--label-dir", default="00_data/train/label")
    ap.add_argument("--depth-dir", default="00_data/train/depth")
    ap.add_argument("--splits-manifest", default="00_data/splits/folds_v1.json")
    ap.add_argument("--nearest-oof-logits", default="01_nearest/golden_artifacts/oof_logits.npy")
    ap.add_argument("--nearest-oof-ids", default="01_nearest/golden_artifacts/oof_file_ids.npy")
    ap.add_argument("--b-oof-dir", default="00_data/02_nonstruct_frozen/golden_artifacts/oof")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load frozen spec + coefficients
    spec_path = os.path.join(args.freeze_dir, "spec.json")
    coef_path = os.path.join(args.freeze_dir, "pushlr_coefficients.json")
    if not os.path.exists(spec_path):
        opt._die(f"Missing spec.json: {spec_path}")
    if not os.path.exists(coef_path):
        opt._die(f"Missing pushlr_coefficients.json: {coef_path}")

    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    with open(coef_path, "r", encoding="utf-8") as f:
        coef = json.load(f)

    # Reconstruct frozen components
    lam = float(spec["inference"]["lambda"])
    wB_map: Dict[str, float] = spec["base"]["wB"]
    wB = np.array([float(wB_map[nm]) for nm in opt.CLASS_NAMES], dtype=np.float32)  # (13,)

    S_names = list(spec["mask"]["S_prime"])
    S = set(opt.CLASS_NAMES.index(x) for x in S_names)

    push_model = opt.PushLRModel(
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
    opt._ensure(args.nearest_oof_ids, "Nearest oof_file_ids.npy")
    opt._ensure(args.nearest_oof_logits, "Nearest oof_logits.npy")
    n_ids = np.load(args.nearest_oof_ids, allow_pickle=False)
    file_ids = [str(x) for x in n_ids.tolist()]
    n_logits = np.load(args.nearest_oof_logits, mmap_mode="r", allow_pickle=False)
    if n_logits.shape[0] != len(file_ids) or n_logits.shape[1:] != (opt.NUM_CLASSES, 480, 640):
        opt._die(f"Unexpected Nearest logits shape: {n_logits.shape} vs ids={len(file_ids)}")

    # Model B index
    opt._ensure(args.splits_manifest, "splits manifest")
    opt._ensure(args.b_oof_dir, "Model B oof dir")
    val_ids_by_fold = opt.load_splits_val_ids_by_fold(args.splits_manifest)
    b_index = opt.build_b_index(args.b_oof_dir, val_ids_by_fold)

    # Accumulators: counts[gt_class, pred_class]
    counts = np.zeros((opt.NUM_CLASSES, opt.NUM_CLASSES), dtype=np.int64)
    counts_base = np.zeros((opt.NUM_CLASSES, opt.NUM_CLASSES), dtype=np.int64)  # pred0 confusion (before push)
    # Depth-bin error stats for GT in {books, table}
    # bins: 0-1,1-2,2-3,3-5,5-10 (same as optimize_ensemble_2model)
    n_bins = 5
    gt_bin_total_final = np.zeros((opt.NUM_CLASSES, n_bins), dtype=np.int64)
    gt_bin_err_final = np.zeros((opt.NUM_CLASSES, n_bins), dtype=np.int64)
    gt_bin_total_base = np.zeros((opt.NUM_CLASSES, n_bins), dtype=np.int64)
    gt_bin_err_base = np.zeros((opt.NUM_CLASSES, n_bins), dtype=np.int64)

    for i, fid in enumerate(file_ids):
        if fid not in b_index.id_to_pos:
            opt._die(f"ID missing in Model B index: {fid}")
        fold, j = b_index.id_to_pos[fid]
        ln = n_logits[i].astype(np.float32, copy=False)
        lb = b_index.fold_logits[fold][j].astype(np.float32, copy=False)

        # Base classwise_v1_q3 (frozen wB)
        L0 = ((1.0 - wB[:, None, None]) * ln + (wB[:, None, None] * lb)).astype(np.float32, copy=False)
        pred0 = np.argmax(L0, axis=0).astype(np.int64, copy=False)

        gt = opt.load_label(args.label_dir, fid)
        valid = (gt != opt.IGNORE_INDEX)
        depth = opt.load_depth_m(args.depth_dir, fid)

        # pred0 confusion (for context)
        gt_v = gt[valid].astype(np.int64, copy=False)
        p0_v = pred0[valid].astype(np.int64, copy=False)
        idx0 = gt_v * opt.NUM_CLASSES + p0_v
        counts_base += np.bincount(idx0, minlength=opt.NUM_CLASSES * opt.NUM_CLASSES).reshape(opt.NUM_CLASSES, opt.NUM_CLASSES)

        # Depth-bin breakdown for GT=books/table (base)
        for gt_cid in (opt.CID_BOOKS, opt.CID_TABLE):
            mgt = valid & (gt == gt_cid)
            if not np.any(mgt):
                continue
            ys, xs = np.where(mgt)
            db = opt.depth_bin_at(depth, ys, xs)  # (P,)
            # totals
            gt_bin_total_base[gt_cid] += np.bincount(db, minlength=n_bins).astype(np.int64, copy=False)
            # errors
            err = (pred0[ys, xs] != gt_cid)
            if np.any(err):
                gt_bin_err_base[gt_cid] += np.bincount(db[err], minlength=n_bins).astype(np.int64, copy=False)

        # Local correction (frozen)
        predB = np.argmax(lb, axis=0).astype(np.int64, copy=False)
        disagree = (pred0 != predB) & valid
        inS = np.isin(predB, list(S))
        mask = disagree & inS
        L = L0.copy()
        if np.any(mask):
            ys, xs = np.where(mask)
            bnd = opt.boundary_mask_from_gt(gt, k=3)
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
                push_model,
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
            # Safety: do not touch correct base pixels (eval-only, uses GT)
            beta_eff = beta.copy()
            beta_eff[c0 == gt[ys, xs]] = 0.0

            L[:, ys, xs] = (L[:, ys, xs] + (lam * beta_eff[None, :] * d[:, ys, xs])).astype(np.float32, copy=False)

        pred = np.argmax(L, axis=0).astype(np.int64, copy=False)
        pv = pred[valid].astype(np.int64, copy=False)
        idx = gt_v * opt.NUM_CLASSES + pv
        counts += np.bincount(idx, minlength=opt.NUM_CLASSES * opt.NUM_CLASSES).reshape(opt.NUM_CLASSES, opt.NUM_CLASSES)

        # Depth-bin breakdown for GT=books/table (final)
        for gt_cid in (opt.CID_BOOKS, opt.CID_TABLE):
            mgt = valid & (gt == gt_cid)
            if not np.any(mgt):
                continue
            ys, xs = np.where(mgt)
            db2 = opt.depth_bin_at(depth, ys, xs)
            gt_bin_total_final[gt_cid] += np.bincount(db2, minlength=n_bins).astype(np.int64, copy=False)
            err = (pred[ys, xs] != gt_cid)
            if np.any(err):
                gt_bin_err_final[gt_cid] += np.bincount(db2[err], minlength=n_bins).astype(np.int64, copy=False)

    # Extract books/table GT rows
    out_full = os.path.join(args.out_dir, "books_table_confusion_full.csv")
    out_top = os.path.join(args.out_dir, "books_table_confusion_top3.csv")
    out_depth = os.path.join(args.out_dir, "books_table_error_by_depthbin.csv")

    rows_full: List[List[object]] = []
    rows_top: List[List[object]] = []
    for gt_cid, gt_name in [(opt.CID_BOOKS, "books"), (opt.CID_TABLE, "table")]:
        row = counts[gt_cid].copy()
        row0 = counts_base[gt_cid].copy()
        total = int(row.sum())
        total0 = int(row0.sum())
        for pred_cid, pred_name in enumerate(opt.CLASS_NAMES):
            rows_full.append(
                [
                    gt_cid,
                    gt_name,
                    pred_cid,
                    pred_name,
                    int(row[pred_cid]),
                    float(row[pred_cid] / total) if total > 0 else float("nan"),
                    int(row0[pred_cid]),
                    float(row0[pred_cid] / total0) if total0 > 0 else float("nan"),
                ]
            )

        top = _topk_counts(row, k=3)
        top0 = _topk_counts(row0, k=3)
        for rank, (pred_cid, cnt) in enumerate(top, start=1):
            rows_top.append(
                [
                    gt_cid,
                    gt_name,
                    rank,
                    pred_cid,
                    opt.CLASS_NAMES[pred_cid],
                    int(cnt),
                    float(cnt / total) if total > 0 else float("nan"),
                    "ensemble2_final",
                ]
            )
        for rank, (pred_cid, cnt) in enumerate(top0, start=1):
            rows_top.append(
                [
                    gt_cid,
                    gt_name,
                    rank,
                    pred_cid,
                    opt.CLASS_NAMES[pred_cid],
                    int(cnt),
                    float(cnt / total0) if total0 > 0 else float("nan"),
                    "base_pred0(classwise_v1_q3)",
                ]
            )

    with open(out_full, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "gt_class_id",
                "gt_class_name",
                "pred_class_id",
                "pred_class_name",
                "count_final",
                "ratio_final",
                "count_base",
                "ratio_base",
            ]
        )
        w.writerows(rows_full)

    with open(out_top, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gt_class_id", "gt_class_name", "rank", "pred_class_id", "pred_class_name", "count", "ratio", "which"])
        w.writerows(rows_top)

    # Depth-bin error table
    bin_names = ["0-1m", "1-2m", "2-3m", "3-5m", "5-10m"]
    rows_depth: List[List[object]] = []
    for gt_cid, gt_name in [(opt.CID_BOOKS, "books"), (opt.CID_TABLE, "table")]:
        tot_f = gt_bin_total_final[gt_cid]
        err_f = gt_bin_err_final[gt_cid]
        tot_b = gt_bin_total_base[gt_cid]
        err_b = gt_bin_err_base[gt_cid]
        sum_err_f = int(err_f.sum())
        sum_err_b = int(err_b.sum())
        for bi, bn in enumerate(bin_names):
            tf, ef = int(tot_f[bi]), int(err_f[bi])
            tb, eb = int(tot_b[bi]), int(err_b[bi])
            rows_depth.append(
                [
                    gt_cid,
                    gt_name,
                    bn,
                    tf,
                    ef,
                    (float(ef) / float(tf)) if tf > 0 else float("nan"),
                    (float(ef) / float(sum_err_f)) if sum_err_f > 0 else float("nan"),
                    tb,
                    eb,
                    (float(eb) / float(tb)) if tb > 0 else float("nan"),
                    (float(eb) / float(sum_err_b)) if sum_err_b > 0 else float("nan"),
                ]
            )

    with open(out_depth, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "gt_class_id",
                "gt_class_name",
                "depth_bin",
                "gt_pixels_final",
                "err_pixels_final",
                "err_rate_final",
                "err_share_of_all_errors_final",
                "gt_pixels_base",
                "err_pixels_base",
                "err_rate_base",
                "err_share_of_all_errors_base",
            ]
        )
        w.writerows(rows_depth)

    print(f"[ENSEMBLE2-ANALYZE] wrote: {os.path.abspath(out_top)}")
    print(f"[ENSEMBLE2-ANALYZE] wrote: {os.path.abspath(out_full)}")
    print(f"[ENSEMBLE2-ANALYZE] wrote: {os.path.abspath(out_depth)}")


if __name__ == "__main__":
    main()

