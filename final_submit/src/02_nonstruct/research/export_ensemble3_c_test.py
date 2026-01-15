"""
export_ensemble3_c_test.py

Build test logits and submission for Ensemble-3 (Base=E2 frozen + C push).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from typing import Dict, List

import numpy as np

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
    raise SystemExit(f"[E3C-TEST][FATAL] {msg}")


def load_ids_txt(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            out.append(t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-ids", default="00_data/ids/test_ids.txt")
    ap.add_argument("--base-logits", required=True, help="E2 frozen test logits (float16)")
    ap.add_argument("--base-ids", required=True, help="ids for base logits (.txt)")
    ap.add_argument("--c-test-logits", required=True, help="Model C test logits (float16)")
    ap.add_argument("--c-test-ids", required=True, help="ids for Model C logits (.txt)")
    ap.add_argument("--push-coef", required=True, help="E3C pushlr_coefficients.json")
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--out-dir", default="00_data/output/ensemble3_c_test")
    ap.add_argument("--require-base-fo", action="store_true")
    args = ap.parse_args()

    test_ids = load_ids_txt(args.test_ids)
    base_ids = load_ids_txt(args.base_ids)
    c_ids = load_ids_txt(args.c_test_ids)

    if len(test_ids) != len(base_ids) or len(test_ids) != len(c_ids):
        _die("ID length mismatch among test_ids/base_ids/c_ids")

    idx_base = {opt.norm_id(x): i for i, x in enumerate(base_ids)}
    idx_c = {opt.norm_id(x): i for i, x in enumerate(c_ids)}

    # reorder indices for base/c based on test_ids
    ord_base = np.array([idx_base[opt.norm_id(x)] for x in test_ids], dtype=np.int64)
    ord_c = np.array([idx_c[opt.norm_id(x)] for x in test_ids], dtype=np.int64)

    base_logits = np.load(args.base_logits, mmap_mode="r")
    c_logits = np.load(args.c_test_logits, mmap_mode="r")
    if base_logits.shape[1:] != (NUM_CLASSES, 480, 640):
        _die(f"Unexpected base logits shape: {base_logits.shape}")
    if c_logits.shape[1:] != (NUM_CLASSES, 480, 640):
        _die(f"Unexpected C logits shape: {c_logits.shape}")

    with open(args.push_coef, "r", encoding="utf-8") as f:
        coef = json.load(f)
    push = opt.PushLRModel(
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

    os.makedirs(args.out_dir, exist_ok=True)
    out_logits_path = os.path.join(args.out_dir, "submission.npy")
    out_zip_path = os.path.join(args.out_dir, "submission.zip")

    N = len(test_ids)
    out_logits = np.lib.format.open_memmap(out_logits_path, mode="w+", dtype=np.float16, shape=(N, NUM_CLASSES, 480, 640))

    for i in range(N):
        base = np.array(base_logits[ord_base[i]], copy=False).astype(np.float32, copy=False)
        c = np.array(c_logits[ord_c[i]], copy=False).astype(np.float32, copy=False)
        pred_base = np.argmax(base, axis=0).astype(np.int64, copy=False)
        pred_c = np.argmax(c, axis=0).astype(np.int64, copy=False)

        # gating mask
        mask = (pred_c == CID_BOOKS) | (pred_c == CID_TABLE)
        disagree = (pred_base != pred_c)
        mask_gate = mask & disagree
        if args.require_base_fo:
            mask_gate &= (pred_base == CID_FURNITURE) | (pred_base == CID_OBJECTS)

        L_final = base.copy()
        if np.any(mask_gate):
            ys, xs = np.where(mask_gate)
            # boundary/depth not available for test: set zeros
            boundary = np.zeros_like(pred_base, dtype=np.float32)
            depth_bin = np.zeros_like(pred_base, dtype=np.int64)

            pmax0 = opt.pmax_from_logits_at(base, ys, xs)
            m0 = opt.margin_from_logits_at(base, ys, xs)
            pmaxN = pmax0
            mN = m0
            pmaxB = opt.pmax_from_logits_at(c, ys, xs)
            mB = opt.margin_from_logits_at(c, ys, xs)

            d = c - base
            x0 = base[:, ys, xs]
            top1i = np.argmax(x0, axis=0).astype(np.int64, copy=False)
            x0_mask = x0.copy()
            x0_mask[top1i, np.arange(x0.shape[1])] = -1e9
            top2i = np.argmax(x0_mask, axis=0).astype(np.int64, copy=False)
            d_s = d[:, ys, xs]
            d_top1 = d_s[top1i, np.arange(d_s.shape[1])].astype(np.float32, copy=False)
            d_top2 = d_s[top2i, np.arange(d_s.shape[1])].astype(np.float32, copy=False)
            d_gap = (d_top1 - d_top2).astype(np.float32, copy=False)

            beta = opt.push_lr_predict_beta(
                push,
                pmax0=pmax0,
                m0=m0,
                pmaxN=pmaxN,
                pmaxB=pmaxB,
                mN=mN,
                mB=mB,
                d_top1=d_top1,
                d_gap=d_gap,
                boundary=boundary[ys, xs].astype(np.float32, copy=False),
                depth_bin=depth_bin[ys, xs],
                pred0=pred_base[ys, xs].astype(np.int64, copy=False),
                predB=pred_c[ys, xs].astype(np.int64, copy=False),
            )
            L_final[:, ys, xs] = (L_final[:, ys, xs] + (float(args.lam) * beta[None, :] * d[:, ys, xs])).astype(np.float32, copy=False)

        out_logits[i] = L_final.astype(np.float16, copy=False)

    # zip submission
    with zipfile.ZipFile(out_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_logits_path, arcname="submission.npy")

    print(f"[E3C-TEST] wrote: {os.path.abspath(out_zip_path)}")


if __name__ == "__main__":
    main()
