#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
except Exception:
    torch = None


NUM_CLASSES = 13
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


def _die(msg: str) -> None:
    raise SystemExit(f"[RUN-INFER][FATAL] {msg}")


def _ensure(path: str, desc: str) -> None:
    if not os.path.exists(path):
        _die(f"Missing {desc}: {path}")


def norm_id(x: str) -> str:
    s = str(x).strip()
    if s.endswith(".png"):
        return s
    return f"{s}.png"


def load_ids_txt(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            out.append(norm_id(t))
    return out


def load_ids_any(path: str) -> List[str]:
    if path.endswith(".npy"):
        arr = np.load(path, allow_pickle=True)
        return [norm_id(x) for x in arr.tolist()]
    if path.endswith(".txt"):
        return load_ids_txt(path)
    _die(f"Unknown ids format: {path} (expected .npy or .txt)")
    return []


def _ensure_logits_shape(x: np.ndarray, name: str) -> None:
    if x.ndim != 4:
        _die(f"{name}: expected logits ndim=4 (N,C,H,W), got {x.shape}")
    if x.shape[1] != NUM_CLASSES:
        _die(f"{name}: expected C=13, got {x.shape}")


def run_cmd(cmd: List[str], cwd: str | None = None) -> None:
    print("[RUN-INFER] exec:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


@dataclass
class PushLRModel:
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
    w_depth: np.ndarray  # (5,)
    w_pred0: np.ndarray  # (13,)
    w_predB: np.ndarray  # (13,)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float32, copy=False)
    z = np.clip(z, -50.0, 50.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32, copy=False)


def push_lr_predict_beta(
    m: PushLRModel,
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
    z = (
        float(m.b)
        + float(m.w_pmax0) * pmax0
        + float(m.w_m0) * m0
        + float(m.w_pmaxN) * pmaxN
        + float(m.w_pmaxB) * pmaxB
        + float(m.w_mN) * mN
        + float(m.w_mB) * mB
        + float(m.w_dtop1) * d_top1
        + float(m.w_dgap) * d_gap
        + float(m.w_boundary) * boundary
        + m.w_depth[depth_bin.astype(np.int64)]
        + m.w_pred0[pred0.astype(np.int64)]
        + m.w_predB[predB.astype(np.int64)]
    )
    return sigmoid(z)


def pmax_from_logits_at(logits_chw: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    x = logits_chw[:, ys, xs].astype(np.float32, copy=False)
    m = np.max(x, axis=0, keepdims=True)
    ex = np.exp(x - m)
    pr = ex / np.sum(ex, axis=0, keepdims=True)
    return np.max(pr, axis=0).astype(np.float32, copy=False)


def margin_from_logits_at(logits_chw: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    x = logits_chw[:, ys, xs].astype(np.float32, copy=False)
    m = np.max(x, axis=0, keepdims=True)
    ex = np.exp(x - m)
    pr = ex / np.sum(ex, axis=0, keepdims=True)
    top1 = np.max(pr, axis=0)
    pr2 = pr.copy()
    top1i = np.argmax(pr2, axis=0)
    pr2[top1i, np.arange(pr2.shape[1])] = -1.0
    top2 = np.max(pr2, axis=0)
    return (top1 - top2).astype(np.float32, copy=False)


def load_pushlr(obj: Dict[str, object]) -> PushLRModel:
    return PushLRModel(
        b=float(obj["b"]),
        w_pmax0=float(obj["w_pmax0"]),
        w_m0=float(obj["w_m0"]),
        w_pmaxN=float(obj["w_pmaxN"]),
        w_pmaxB=float(obj["w_pmaxB"]),
        w_mN=float(obj["w_mN"]),
        w_mB=float(obj["w_mB"]),
        w_dtop1=float(obj["w_dtop1"]),
        w_dgap=float(obj["w_dgap"]),
        w_boundary=float(obj["w_boundary"]),
        w_depth=np.array(obj["w_depth"], dtype=np.float32),
        w_pred0=np.array(obj["w_pred0"], dtype=np.float32),
        w_predB=np.array(obj["w_predB"], dtype=np.float32),
    )


def load_model_bundle(path: str) -> Dict[str, object]:
    if torch is None:
        _die("torch is required to load model.pth")
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        _die("model.pth must be a dict")
    return obj


def average_nearest_folds(
    *,
    test_ids: List[str],
    fold_dirs: List[str],
    out_dir: str,
    chunk: int = 4,
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    out_logits_path = os.path.join(out_dir, "test_logits.npy")
    out_ids_path = os.path.join(out_dir, "test_ids.txt")
    acc_path = os.path.join(out_dir, "_test_logits_acc_f32.npy")

    if not fold_dirs:
        _die("No nearest fold dirs provided")

    N = len(test_ids)
    acc = np.lib.format.open_memmap(acc_path, mode="w+", dtype=np.float32, shape=(N, NUM_CLASSES, 480, 640))
    acc[:] = 0.0

    for fdir in fold_dirs:
        logits_path = os.path.join(fdir, "test_logits.npy")
        ids_path = os.path.join(fdir, "test_file_ids.npy")
        _ensure(logits_path, "nearest test_logits.npy")
        _ensure(ids_path, "nearest test_file_ids.npy")
        ids = load_ids_any(ids_path)
        if len(ids) == 0:
            _die(f"Empty ids in {ids_path}")
        idx_map = {k: i for i, k in enumerate(ids)}
        missing = [k for k in test_ids if k not in idx_map]
        if missing:
            _die(f"Nearest fold ids missing {len(missing)} ids (first3={missing[:3]}) in {fdir}")
        order = np.array([idx_map[k] for k in test_ids], dtype=np.int64)

        logits = np.load(logits_path, mmap_mode="r")
        _ensure_logits_shape(logits, f"nearest logits {fdir}")

        for s0 in range(0, N, int(chunk)):
            s1 = min(N, s0 + int(chunk))
            sel = order[s0:s1]
            acc[s0:s1] += logits[sel].astype(np.float32, copy=False)

        print(f"[RUN-INFER] nearest fold averaged: {fdir}", flush=True)

    acc /= float(len(fold_dirs))
    out = np.lib.format.open_memmap(out_logits_path, mode="w+", dtype=np.float16, shape=(N, NUM_CLASSES, 480, 640))
    for s0 in range(0, N, int(chunk)):
        s1 = min(N, s0 + int(chunk))
        out[s0:s1] = acc[s0:s1].astype(np.float16, copy=False)

    with open(out_ids_path, "w", encoding="utf-8") as f:
        for k in test_ids:
            f.write(f"{k}\n")

    return out_logits_path, out_ids_path


def build_base_e2(
    *,
    test_ids: List[str],
    nearest_logits_path: str,
    nearest_ids_path: str,
    modelb_logits_path: str,
    modelb_ids_path: str,
    out_dir: str,
    wB: np.ndarray,
    push_b: PushLRModel,
    mask_s: List[int],
    lam: float,
    proxy_margin0_thr: float,
    disable_proxy_safety: bool,
    chunk: int = 4,
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    out_logits_path = os.path.join(out_dir, "base_logits.npy")
    out_ids_path = os.path.join(out_dir, "test_ids.txt")

    ids_n = load_ids_any(nearest_ids_path)
    ids_b = load_ids_any(modelb_ids_path)
    map_n = {k: i for i, k in enumerate(ids_n)}
    map_b = {k: i for i, k in enumerate(ids_b)}
    idx_n = np.array([map_n[k] for k in test_ids], dtype=np.int64)
    idx_b = np.array([map_b[k] for k in test_ids], dtype=np.int64)

    ln_all = np.load(nearest_logits_path, mmap_mode="r")
    lb_all = np.load(modelb_logits_path, mmap_mode="r")
    _ensure_logits_shape(ln_all, "nearest_test_logits")
    _ensure_logits_shape(lb_all, "modelb_test_logits")

    N = len(test_ids)
    out_logits = np.lib.format.open_memmap(out_logits_path, mode="w+", dtype=np.float16, shape=(N, NUM_CLASSES, 480, 640))

    w = wB.reshape(1, NUM_CLASSES, 1, 1).astype(np.float32)
    S = np.array(mask_s, dtype=np.int64)

    for s0 in range(0, N, int(chunk)):
        s1 = min(N, s0 + int(chunk))
        ln = np.asarray(ln_all[idx_n[s0:s1]], dtype=np.float32)
        lb = np.asarray(lb_all[idx_b[s0:s1]], dtype=np.float32)

        L0 = (1.0 - w) * ln + w * lb
        d = (lb - ln)

        for bi in range(s1 - s0):
            L0_i = L0[bi]
            d_i = d[bi]
            lb_i = lb[bi]
            ln_i = ln[bi]

            pred0 = np.argmax(L0_i, axis=0).astype(np.int64, copy=False)
            predB = np.argmax(lb_i, axis=0).astype(np.int64, copy=False)
            mask = (pred0 != predB) & np.isin(predB, S)

            L = L0_i.copy()
            if np.any(mask):
                ys, xs = np.where(mask)
                pmax0 = pmax_from_logits_at(L0_i, ys, xs)
                m0 = margin_from_logits_at(L0_i, ys, xs)
                pmaxN = pmax_from_logits_at(ln_i, ys, xs)
                pmaxB = pmax_from_logits_at(lb_i, ys, xs)
                mN = margin_from_logits_at(ln_i, ys, xs)
                mB = margin_from_logits_at(lb_i, ys, xs)
                boundary = np.zeros_like(pmax0, dtype=np.float32)
                depth_bin = np.zeros_like(pred0[ys, xs], dtype=np.int64)
                c0 = pred0[ys, xs].astype(np.int64, copy=False)
                cB = predB[ys, xs].astype(np.int64, copy=False)

                d_s = d_i[:, ys, xs]
                x0 = L0_i[:, ys, xs]
                x0_mask = x0.copy()
                x0_mask[c0, np.arange(x0_mask.shape[1])] = -1e9
                top2 = np.argmax(x0_mask, axis=0).astype(np.int64, copy=False)
                d_top1 = d_s[c0, np.arange(d_s.shape[1])]
                d_top2 = d_s[top2, np.arange(d_s.shape[1])]
                d_gap = (d_top1 - d_top2).astype(np.float32, copy=False)

                beta = push_lr_predict_beta(
                    push_b,
                    pmax0=pmax0,
                    m0=m0,
                    pmaxN=pmaxN,
                    pmaxB=pmaxB,
                    mN=mN,
                    mB=mB,
                    d_top1=d_top1.astype(np.float32, copy=False),
                    d_gap=d_gap,
                    boundary=boundary,
                    depth_bin=depth_bin,
                    pred0=c0,
                    predB=cB,
                )

                beta_eff = beta
                if not disable_proxy_safety:
                    beta_eff = beta.copy()
                    beta_eff[m0 > float(proxy_margin0_thr)] = 0.0

                L[:, ys, xs] = (L[:, ys, xs] + (float(lam) * beta_eff[None, :] * d_i[:, ys, xs])).astype(
                    np.float32, copy=False
                )

            out_logits[s0 + bi] = L.astype(np.float16, copy=False)

        print(f"[RUN-INFER] base E2 chunk {s0}:{s1}", flush=True)

    with open(out_ids_path, "w", encoding="utf-8") as f:
        for k in test_ids:
            f.write(f"{k}\n")

    return out_logits_path, out_ids_path


def apply_e3_c_push(
    *,
    test_ids: List[str],
    base_logits_path: str,
    base_ids_path: str,
    c_logits_path: str,
    c_ids_path: str,
    push_c: PushLRModel,
    lambda_c: float,
    require_base_fo: bool,
    out_path: str,
    chunk: int = 1,
) -> str:
    ids_base = load_ids_any(base_ids_path)
    ids_c = load_ids_any(c_ids_path)
    idx_base = {k: i for i, k in enumerate(ids_base)}
    idx_c = {k: i for i, k in enumerate(ids_c)}
    ord_base = np.array([idx_base[k] for k in test_ids], dtype=np.int64)
    ord_c = np.array([idx_c[k] for k in test_ids], dtype=np.int64)

    base_logits = np.load(base_logits_path, mmap_mode="r")
    c_logits = np.load(c_logits_path, mmap_mode="r")
    _ensure_logits_shape(base_logits, "base_logits")
    _ensure_logits_shape(c_logits, "modelc_logits")

    CID_BOOKS = CLASS_NAMES.index("books")
    CID_TABLE = CLASS_NAMES.index("table")
    CID_FURNITURE = CLASS_NAMES.index("furniture")
    CID_OBJECTS = CLASS_NAMES.index("objects")

    N = len(test_ids)
    out_logits = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(N, NUM_CLASSES, 480, 640))

    for i in range(0, N, int(chunk)):
        j = min(N, i + int(chunk))
        for k in range(i, j):
            base = np.array(base_logits[ord_base[k]], copy=False).astype(np.float32, copy=False)
            c = np.array(c_logits[ord_c[k]], copy=False).astype(np.float32, copy=False)
            pred_base = np.argmax(base, axis=0).astype(np.int64, copy=False)
            pred_c = np.argmax(c, axis=0).astype(np.int64, copy=False)

            mask = (pred_c == CID_BOOKS) | (pred_c == CID_TABLE)
            mask &= (pred_base != pred_c)
            if require_base_fo:
                mask &= (pred_base == CID_FURNITURE) | (pred_base == CID_OBJECTS)

            L_final = base.copy()
            if np.any(mask):
                ys, xs = np.where(mask)
                boundary = np.zeros_like(pred_base, dtype=np.float32)
                depth_bin = np.zeros_like(pred_base, dtype=np.int64)

                pmax0 = pmax_from_logits_at(base, ys, xs)
                m0 = margin_from_logits_at(base, ys, xs)
                pmaxN = pmax0
                mN = m0
                pmaxB = pmax_from_logits_at(c, ys, xs)
                mB = margin_from_logits_at(c, ys, xs)

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

                beta = push_lr_predict_beta(
                    push_c,
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

                L_final[:, ys, xs] = (
                    L_final[:, ys, xs] + (float(lambda_c) * beta[None, :] * d[:, ys, xs])
                ).astype(np.float32, copy=False)

            out_logits[k] = L_final.astype(np.float16, copy=False)

        print(f"[RUN-INFER] E3 push chunk {i}:{j}", flush=True)

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-ids", required=True)
    ap.add_argument("--test-image-dir", required=True)
    ap.add_argument("--test-depth-dir", required=True)
    ap.add_argument("--model-pth", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lambda-c", type=float, default=1.0)
    ap.add_argument("--nearest-fold-dirs", required=True, help="comma-separated fold dirs")
    ap.add_argument("--modelb-ckpt-dir", default="")
    ap.add_argument("--modelc-ckpt-dir", default="")
    ap.add_argument("--modelb-folds", default="0,1,2,3,4")
    ap.add_argument("--modelc-folds", default="0,1,2,3,4")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-proxy-safety", action="store_true")
    ap.add_argument("--proxy-margin0-thr", type=float, default=0.25)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src_root = Path(__file__).resolve().parents[1]
    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    test_ids = load_ids_txt(args.test_ids)
    print(f"[RUN-INFER] test_ids={len(test_ids)} first5={test_ids[:5]}")

    bundle = load_model_bundle(args.model_pth)
    e2 = bundle.get("e2", {})
    e3 = bundle.get("e3", {})
    paths = bundle.get("paths", {})

    modelb_ckpt_dir = args.modelb_ckpt_dir or str(paths.get("modelb_ckpt_dir", ""))
    modelc_ckpt_dir = args.modelc_ckpt_dir or str(paths.get("modelc_ckpt_dir", ""))
    if not modelb_ckpt_dir:
        _die("modelb_ckpt_dir not provided (use --modelb-ckpt-dir or model.pth)")
    if not modelc_ckpt_dir:
        _die("modelc_ckpt_dir not provided (use --modelc-ckpt-dir or model.pth)")

    # Step A: Model B test logits
    modelb_out_dir = os.path.join(work_dir, "modelb")
    modelb_logits_path = os.path.join(modelb_out_dir, "test_logits.npy")
    modelb_ids_path = os.path.join(modelb_out_dir, "test_ids.txt")
    if args.force or not os.path.exists(modelb_logits_path):
        cmd = [
            sys.executable,
            "-m",
            "02_nonstruct.research.infer_modelb_test_logits",
            "--test-ids",
            os.path.abspath(args.test_ids),
            "--test-image-dir",
            os.path.abspath(args.test_image_dir),
            "--ckpt-dir",
            os.path.abspath(modelb_ckpt_dir),
            "--folds",
            str(args.modelb_folds),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--device",
            str(args.device),
            "--out-dir",
            os.path.abspath(modelb_out_dir),
        ]
        run_cmd(cmd, cwd=str(src_root))
    _ensure(modelb_logits_path, "modelb test_logits.npy")
    _ensure(modelb_ids_path, "modelb test_ids.txt")

    # Step B: Model C test logits
    modelc_out_dir = os.path.join(work_dir, "modelc")
    modelc_logits_path = os.path.join(modelc_out_dir, "test_logits.npy")
    modelc_ids_path = os.path.join(modelc_out_dir, "test_ids.txt")
    if args.force or not os.path.exists(modelc_logits_path):
        cmd = [
            sys.executable,
            "tools/infer_modelc_test_logits.py",
            "--test-ids",
            os.path.abspath(args.test_ids),
            "--test-image-dir",
            os.path.abspath(args.test_image_dir),
            "--test-depth-dir",
            os.path.abspath(args.test_depth_dir),
            "--ckpt-dir",
            os.path.abspath(modelc_ckpt_dir),
            "--folds",
            str(args.modelc_folds),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--device",
            str(args.device),
            "--out-dir",
            os.path.abspath(modelc_out_dir),
        ]
        run_cmd(cmd, cwd=str(src_root / "03_model_c"))
    _ensure(modelc_logits_path, "modelc test_logits.npy")
    _ensure(modelc_ids_path, "modelc test_ids.txt")

    # Step C: Nearest fold-average
    nearest_dirs = [x.strip() for x in str(args.nearest_fold_dirs).split(",") if x.strip()]
    nearest_out_dir = os.path.join(work_dir, "nearest")
    nearest_logits_path, nearest_ids_path = average_nearest_folds(
        test_ids=test_ids,
        fold_dirs=nearest_dirs,
        out_dir=nearest_out_dir,
        chunk=4,
    )

    # Step D: Base E2 (classwise + PushLR(B))
    if not isinstance(e2, dict):
        _die("model.pth missing e2 config dict")
    e2_spec = e2.get("spec", {})
    e2_wB = np.array(e2.get("wB_classwise_q3", []), dtype=np.float32)
    if e2_wB.size != NUM_CLASSES:
        _die("model.pth missing e2 wB_classwise_q3 (len=13)")
    push_b = load_pushlr(e2.get("pushlr_coefficients", {}))
    mask_s_names = e2_spec.get("mask", {}).get("S_prime", [])
    mask_s = [CLASS_NAMES.index(n) for n in mask_s_names] if mask_s_names else [2, 6, 7, 10]
    lam_e2 = float(e2_spec.get("inference", {}).get("lambda", 1.9))

    base_out_dir = os.path.join(work_dir, "base_e2")
    base_logits_path, base_ids_path = build_base_e2(
        test_ids=test_ids,
        nearest_logits_path=nearest_logits_path,
        nearest_ids_path=nearest_ids_path,
        modelb_logits_path=modelb_logits_path,
        modelb_ids_path=modelb_ids_path,
        out_dir=base_out_dir,
        wB=e2_wB,
        push_b=push_b,
        mask_s=mask_s,
        lam=lam_e2,
        proxy_margin0_thr=float(args.proxy_margin0_thr),
        disable_proxy_safety=bool(args.no_proxy_safety),
        chunk=4,
    )

    # Step E: E3 C-push
    if not isinstance(e3, dict):
        _die("model.pth missing e3 config dict")
    e3_spec = e3.get("spec", {})
    require_base_fo = bool(e3_spec.get("gating", {}).get("require_base_furniture_objects", True))
    push_c = load_pushlr(e3.get("pushlr_coefficients", {}))

    out_path = os.path.abspath(args.out)
    apply_e3_c_push(
        test_ids=test_ids,
        base_logits_path=base_logits_path,
        base_ids_path=base_ids_path,
        c_logits_path=modelc_logits_path,
        c_ids_path=modelc_ids_path,
        push_c=push_c,
        lambda_c=float(args.lambda_c),
        require_base_fo=require_base_fo,
        out_path=out_path,
        chunk=1,
    )

    # Final checks
    arr = np.load(out_path, mmap_mode="r")
    print(f"[RUN-INFER] submission shape={arr.shape} dtype={arr.dtype}")
    finite = np.isfinite(arr[:1]).all()
    print(f"[RUN-INFER] submission finite(sample): {finite}")


if __name__ == "__main__":
    main()
