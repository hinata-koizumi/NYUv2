"""
build_hard_masks.py

Generate hard masks for Model C-v1 training using frozen Ensemble-2 (OOF):
  hard_books: GT=books & pred_E2 in {furniture, objects}
  hard_table: GT=table & pred_E2 in {furniture, objects}

Outputs (per image, under --out-dir):
  {id}.npz with keys: hard_books, hard_table (uint8 0/1)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OPT_PATH = os.path.join(REPO_ROOT, "02_nonstruct", "research")
sys.path.append(OPT_PATH)

import optimize_ensemble_2model as opt


def _die(msg: str) -> None:
    raise SystemExit(f"[MODEL_C][HARD_MASK][FATAL] {msg}")


def load_ids_txt(path: str) -> list[str]:
    out: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            out.append(t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-dir", default="00_data/ensemble2_frozen")
    ap.add_argument("--train-ids", default="00_data/ids/train_ids.txt")
    ap.add_argument("--label-dir", default="00_data/train/label")
    ap.add_argument("--splits-manifest", default="00_data/splits/folds_v1.json")
    ap.add_argument("--nearest-oof-logits", default="01_nearest/golden_artifacts/oof_logits.npy")
    ap.add_argument("--nearest-oof-ids", default="01_nearest/golden_artifacts/oof_file_ids.npy")
    ap.add_argument("--b-oof-dir", default="00_data/02_nonstruct_frozen/golden_artifacts/oof")
    ap.add_argument("--out-dir", default="00_data/output/model_c_hard_masks")
    ap.add_argument("--save-empty", action="store_true", help="write empty masks too")
    ap.add_argument("--limit", type=int, default=0, help="debug: limit to N images")
    args = ap.parse_args()

    opt._ensure(args.freeze_dir, "freeze dir")
    opt._ensure(args.train_ids, "train ids")
    opt._ensure(args.label_dir, "label dir")
    opt._ensure(args.splits_manifest, "splits manifest")
    opt._ensure(args.nearest_oof_ids, "nearest oof ids")
    opt._ensure(args.nearest_oof_logits, "nearest oof logits")
    opt._ensure(args.b_oof_dir, "model B oof dir")

    # Load frozen spec + coefficients
    spec_path = os.path.join(args.freeze_dir, "spec.json")
    coef_path = os.path.join(args.freeze_dir, "pushlr_coefficients.json")
    if not os.path.exists(spec_path):
        _die(f"Missing spec.json: {spec_path}")
    if not os.path.exists(coef_path):
        _die(f"Missing pushlr_coefficients.json: {coef_path}")

    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    with open(coef_path, "r", encoding="utf-8") as f:
        coef = json.load(f)

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
    n_ids = np.load(args.nearest_oof_ids, allow_pickle=False)
    file_ids = [str(x) for x in n_ids.tolist()]
    n_logits = np.load(args.nearest_oof_logits, mmap_mode="r", allow_pickle=False)
    if n_logits.shape[0] != len(file_ids) or n_logits.shape[1:] != (opt.NUM_CLASSES, 480, 640):
        _die(f"Unexpected Nearest logits shape: {n_logits.shape} vs ids={len(file_ids)}")
    n_index = {opt.norm_id(x): i for i, x in enumerate(file_ids)}

    # Model B index (OOF)
    val_ids_by_fold = opt.load_splits_val_ids_by_fold(args.splits_manifest)
    b_index = opt.build_b_index(args.b_oof_dir, val_ids_by_fold)

    train_ids = load_ids_txt(args.train_ids)
    if args.limit and args.limit > 0:
        train_ids = train_ids[: int(args.limit)]

    os.makedirs(args.out_dir, exist_ok=True)

    total = 0
    wrote = 0
    hard_books_pixels = 0
    hard_table_pixels = 0

    for tid in train_ids:
        fid = opt.norm_id(tid)
        if fid not in n_index:
            _die(f"ID missing in nearest oof ids: {fid}")
        if fid not in b_index.id_to_pos:
            _die(f"ID missing in Model B OOF index: {fid}")

        i = n_index[fid]
        fold, j = b_index.id_to_pos[fid]

        ln = n_logits[i].astype(np.float32, copy=False)
        lb = b_index.fold_logits[fold][j].astype(np.float32, copy=False)

        # Base classwise_v1_q3 (frozen wB)
        L0 = ((1.0 - wB[:, None, None]) * ln + (wB[:, None, None] * lb)).astype(np.float32, copy=False)
        pred0 = np.argmax(L0, axis=0).astype(np.int64, copy=False)

        gt = opt.load_label(args.label_dir, fid)
        valid = (gt != opt.IGNORE_INDEX)
        depth = opt.load_depth_m(os.path.join(os.path.dirname(args.label_dir), "depth"), fid)
        if depth is None:
            _die(f"Missing depth for {fid}")

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

        hard_books = (gt == opt.CID_BOOKS) & valid & np.isin(pred, [opt.CID_FURNITURE, opt.CID_OBJECTS])
        hard_table = (gt == opt.CID_TABLE) & valid & np.isin(pred, [opt.CID_FURNITURE, opt.CID_OBJECTS])

        total += 1
        hard_books_pixels += int(hard_books.sum())
        hard_table_pixels += int(hard_table.sum())

        if args.save_empty or hard_books.any() or hard_table.any():
            out_path = os.path.join(args.out_dir, f"{fid[:-4]}.npz")
            np.savez_compressed(
                out_path,
                hard_books=hard_books.astype(np.uint8, copy=False),
                hard_table=hard_table.astype(np.uint8, copy=False),
            )
            wrote += 1

    print(f"[MODEL_C][HARD_MASK] total={total} wrote={wrote}")
    print(f"[MODEL_C][HARD_MASK] hard_books_pixels={hard_books_pixels} hard_table_pixels={hard_table_pixels}")
    print(f"[MODEL_C][HARD_MASK] out_dir={os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
