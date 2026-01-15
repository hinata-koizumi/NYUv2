"""
export_ensemble2_test.py

LB確認（最短・事故らない）向け：
- 2モデルの test logits を test_ids.txt 順に突合して並べ直す（事故ポイント潰し）
- Ensemble-2 (FROZEN) を適用して
  - float16 logits (N,13,480,640) を保存
  - 予測ラベル (N,480,640) uint8 を保存
  - PNG（任意）と提出zip（submission.npy をzip）を生成

NOTE:
- OOFで使った「安全装置 pred0==GT => beta=0」は test ではGTが無いので使えません。
  代わりに、デフォルトでは「pred0 が高自信（margin0が大）なら押さない」proxy safety を入れています。
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import cv2


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
NUM_CLASSES = 13


def _die(msg: str) -> None:
    raise SystemExit(f"[ENSEMBLE2-TEST][FATAL] {msg}")


def norm_id(x: str) -> str:
    s = str(x).strip()
    if s.endswith(".png"):
        s = s[:-4]
    return s


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


def _resize_chw_bilinear(logits_chw: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """logits_chw: (C,H,W) float32 -> (C,hw[0],hw[1]) float32"""
    C, H, W = logits_chw.shape
    Ht, Wt = hw
    if (H, W) == (Ht, Wt):
        return logits_chw
    out = np.empty((C, Ht, Wt), dtype=np.float32)
    for c in range(C):
        out[c] = cv2.resize(logits_chw[c], (Wt, Ht), interpolation=cv2.INTER_LINEAR)
    return out


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
    # top1, top2
    top1 = np.max(pr, axis=0)
    pr2 = pr.copy()
    top1i = np.argmax(pr2, axis=0)
    pr2[top1i, np.arange(pr2.shape[1])] = -1.0
    top2 = np.max(pr2, axis=0)
    return (top1 - top2).astype(np.float32, copy=False)


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


def load_pushlr_coeff(path: str) -> PushLRModel:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-dir", default="00_data/ensemble2_frozen")
    ap.add_argument("--test-ids", default="00_data/ids/test_ids.txt", help="order source (MUST be respected)")
    ap.add_argument("--nearest-test-logits", required=True)
    ap.add_argument("--nearest-test-ids", required=True, help=".npy or .txt ids for nearest logits")
    ap.add_argument("--b-test-logits", required=True)
    ap.add_argument("--b-test-ids", required=True, help=".npy or .txt ids for model B logits")
    ap.add_argument("--out-dir", default="00_data/output/ensemble2_frozen_test")
    ap.add_argument("--write-png", action="store_true")
    ap.add_argument("--chunk", type=int, default=8)
    # deployable proxy safety (事故らない)
    ap.add_argument("--no-proxy-safety", action="store_true", help="disable proxy safety (NOT recommended for LB check)")
    ap.add_argument("--proxy-margin0-thr", type=float, default=0.25)
    args = ap.parse_args()

    # Load freeze spec + coeff
    spec_path = os.path.join(args.freeze_dir, "spec.json")
    coef_path = os.path.join(args.freeze_dir, "pushlr_coefficients.json")
    if not os.path.exists(spec_path):
        _die(f"Missing spec.json: {spec_path}")
    if not os.path.exists(coef_path):
        _die(f"Missing pushlr_coefficients.json: {coef_path}")
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    push = load_pushlr_coeff(coef_path)

    lam = float(spec.get("inference", {}).get("lambda", 1.9))
    wB_map: Dict[str, float] = spec["base"]["wB"]
    wB = np.array([float(wB_map[n]) for n in CLASS_NAMES], dtype=np.float32)  # (C,)
    S_names = spec["mask"]["S_prime"]
    S = sorted([CLASS_NAMES.index(n) for n in S_names])

    # Load ids, build alignment
    test_ids = load_ids_txt(args.test_ids)
    ids_n = load_ids_any(args.nearest_test_ids)
    ids_b = load_ids_any(args.b_test_ids)
    if len(set(test_ids)) != len(test_ids):
        _die("test_ids contains duplicates")

    map_n = {k: i for i, k in enumerate(ids_n)}
    map_b = {k: i for i, k in enumerate(ids_b)}

    missing_n = [k for k in test_ids if k not in map_n]
    missing_b = [k for k in test_ids if k not in map_b]
    if missing_n:
        _die(f"Nearest logits missing {len(missing_n)} ids (first3={missing_n[:3]})")
    if missing_b:
        _die(f"Model B logits missing {len(missing_b)} ids (first3={missing_b[:3]})")

    idx_n = np.array([map_n[k] for k in test_ids], dtype=np.int64)
    idx_b = np.array([map_b[k] for k in test_ids], dtype=np.int64)

    # Load logits (mmap)
    ln_all = np.load(args.nearest_test_logits, mmap_mode="r")
    lb_all = np.load(args.b_test_logits, mmap_mode="r")
    _ensure_logits_shape(ln_all, "nearest_test_logits")
    _ensure_logits_shape(lb_all, "b_test_logits")

    N = len(test_ids)
    Ht, Wt = 480, 640
    os.makedirs(args.out_dir, exist_ok=True)
    out_logits_path = os.path.join(args.out_dir, "test_logits_ensemble2_f16.npy")
    out_pred_path = os.path.join(args.out_dir, "submission.npy")
    out_zip_path = os.path.join(args.out_dir, "submission.zip")
    out_ids_path = os.path.join(args.out_dir, "test_ids_used.txt")
    png_dir = os.path.join(args.out_dir, "pred_png")
    if args.write_png:
        os.makedirs(png_dir, exist_ok=True)

    # open_memmap outputs
    out_logits = np.lib.format.open_memmap(
        out_logits_path, mode="w+", dtype=np.float16, shape=(N, NUM_CLASSES, Ht, Wt)
    )
    out_pred = np.lib.format.open_memmap(out_pred_path, mode="w+", dtype=np.uint8, shape=(N, Ht, Wt))

    # write ids used (exact order)
    with open(out_ids_path, "w", encoding="utf-8") as f:
        for k in test_ids:
            f.write(f"{k}.png\n")

    # process
    for s0 in range(0, N, int(args.chunk)):
        s1 = min(N, s0 + int(args.chunk))
        bn = idx_n[s0:s1]
        bb = idx_b[s0:s1]

        ln = np.asarray(ln_all[bn], dtype=np.float32)  # (B,C,H,W)
        lb = np.asarray(lb_all[bb], dtype=np.float32)

        # resize if needed
        if ln.shape[2:] != (Ht, Wt):
            ln = np.stack([_resize_chw_bilinear(x, (Ht, Wt)) for x in ln], axis=0)
        if lb.shape[2:] != (Ht, Wt):
            lb = np.stack([_resize_chw_bilinear(x, (Ht, Wt)) for x in lb], axis=0)

        # classwise base
        w = wB[None, :, None, None]
        L0 = (1.0 - w) * ln + w * lb  # (B,C,H,W)

        # delta
        d = (lb - ln)  # (B,C,H,W)

        for bi in range(s1 - s0):
            L0_i = L0[bi]
            d_i = d[bi]
            lb_i = lb[bi]

            pred0 = np.argmax(L0_i, axis=0).astype(np.int64, copy=False)
            predB = np.argmax(lb_i, axis=0).astype(np.int64, copy=False)

            mask = (pred0 != predB) & np.isin(predB, S)
            L = L0_i.copy()
            if np.any(mask):
                ys, xs = np.where(mask)
                # features (boundary/depth are unavailable here -> zeros; weights are small)
                pmax0 = pmax_from_logits_at(L0_i, ys, xs)
                m0 = margin_from_logits_at(L0_i, ys, xs)
                pmaxN = pmax_from_logits_at(ln[bi], ys, xs)
                pmaxB = pmax_from_logits_at(lb_i, ys, xs)
                mN = margin_from_logits_at(ln[bi], ys, xs)
                mB = margin_from_logits_at(lb_i, ys, xs)
                boundary = np.zeros_like(pmax0, dtype=np.float32)
                depth_bin = np.zeros_like(pred0[ys, xs], dtype=np.int64)
                c0 = pred0[ys, xs].astype(np.int64, copy=False)
                cB = predB[ys, xs].astype(np.int64, copy=False)

                # delta summaries
                d_s = d_i[:, ys, xs]  # (C,P)
                x0 = L0_i[:, ys, xs]
                x0_mask = x0.copy()
                x0_mask[c0, np.arange(x0_mask.shape[1])] = -1e9
                top2 = np.argmax(x0_mask, axis=0).astype(np.int64, copy=False)
                d_top1 = d_s[c0, np.arange(d_s.shape[1])]
                d_top2 = d_s[top2, np.arange(d_s.shape[1])]
                d_gap = (d_top1 - d_top2).astype(np.float32, copy=False)

                beta = push_lr_predict_beta(
                    push,
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
                if not args.no_proxy_safety:
                    # conservative: do not push where base is very confident
                    beta_eff = beta.copy()
                    beta_eff[m0 > float(args.proxy_margin0_thr)] = 0.0

                # apply
                L[:, ys, xs] = (L[:, ys, xs] + (lam * beta_eff[None, :] * d_i[:, ys, xs])).astype(
                    np.float32, copy=False
                )

            # save
            out_logits[s0 + bi] = L.astype(np.float16, copy=False)
            pred = np.argmax(L, axis=0).astype(np.uint8, copy=False)
            if pred.max() > 12:
                _die(f"pred has invalid max={int(pred.max())} at idx={s0+bi}")
            out_pred[s0 + bi] = pred

            if args.write_png:
                fid = test_ids[s0 + bi]
                cv2.imwrite(os.path.join(png_dir, f"{fid}.png"), pred)

    # zip submission
    with zipfile.ZipFile(out_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_pred_path, arcname="submission.npy")

    print(f"[ENSEMBLE2-TEST] wrote logits: {os.path.abspath(out_logits_path)}")
    print(f"[ENSEMBLE2-TEST] wrote submission: {os.path.abspath(out_zip_path)}")
    print(f"[ENSEMBLE2-TEST] ids order file: {os.path.abspath(out_ids_path)}")


if __name__ == "__main__":
    main()

