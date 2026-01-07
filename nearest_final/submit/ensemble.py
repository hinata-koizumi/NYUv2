"""
Fold ensemble inference (Exp100 Final) -> submission.npy (+ submission.zip).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zipfile

from .utils import (
    best_ckpt_path,
    discover_folds,
    fixed_tta_combs,
    load_cfg_from_fold_dir,
    safe_torch_load,
)


def _load_global_temp(*, exp_dir: str, override_temp: float | None, oof_summary: str | None) -> float:
    if override_temp is not None:
        return float(override_temp)

    p = str(oof_summary) if oof_summary is not None else os.path.join(exp_dir, "oof_summary.json")
    if not os.path.exists(p):
        raise FileNotFoundError(
            "Global temperature is required but OOF summary was not found.\n"
            f"- looked for: {p}\n"
            "Provide one of:\n"
            "  - --temp <T>\n"
            "  - --oof_summary <path/to/oof_summary.json>\n"
            "Or generate it via: python -m nearest_final.submit.oof_temp --exp_dir <EXP_DIR>"
        )
    with open(p, "r") as f:
        s = json.load(f)
    if "best_oof_temp" not in s:
        raise KeyError(f"Missing key 'best_oof_temp' in {p}")
    return float(s["best_oof_temp"])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m nearest_final.submit.ensemble",
        description="Fold ensemble inference for an existing experiment directory (Exp100).",
    )
    p.add_argument("--exp_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--folds", type=str, default=None, help="Comma-separated fold indices.")
    p.add_argument("--temp", type=float, default=None, help="Override global temperature.")
    p.add_argument("--oof_summary", type=str, default=None, help="Path to oof_summary.json.")
    p.add_argument("--data_root", type=str, default=None, help="Override cfg.DATA_ROOT.")
    p.add_argument("--batch_mul", type=int, default=2, help="Batch size multiplier.")
    p.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars.")
    return p


def main(argv: list[str] | None = None) -> None:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from ..data.dataset import NYUDataset
    from ..data.transforms import get_valid_transforms
    from ..engine.inference import Predictor
    from ..model.meta_arch import SegFPN
    from ..utils.misc import configure_runtime, seed_everything, worker_init_fn

    args = build_parser().parse_args(argv)
    exp_dir = os.path.abspath(os.path.expanduser(str(args.exp_dir)))
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"--exp_dir not found: {exp_dir}")

    out_dir = str(args.out_dir) if args.out_dir is not None else os.path.join(exp_dir, "ensemble_folds")
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    all_folds = discover_folds(exp_dir)
    if args.folds is None:
        folds = all_folds
    else:
        folds = [int(x.strip()) for x in str(args.folds).split(",") if x.strip() != ""]
    for f in folds:
        if f not in all_folds:
            raise ValueError(f"Requested fold {f} not found under {exp_dir}. Available: {all_folds}")

    cfg = load_cfg_from_fold_dir(os.path.join(exp_dir, f"fold{folds[0]}"))
    if args.data_root is not None:
        cfg = cfg.with_overrides(DATA_ROOT=str(args.data_root))
        cfg.validate()

    seed_everything(cfg.SEED)
    configure_runtime(cfg)

    global_temp = _load_global_temp(exp_dir=exp_dir, override_temp=args.temp, oof_summary=args.oof_summary)
    # Use config TTA settings (Unified with training config)
    tta_combs = list(cfg.TTA_COMBS)

    test_image_dir = os.path.join(cfg.TEST_DIR, "image")
    test_depth_dir = os.path.join(cfg.TEST_DIR, "depth")
    image_files = sorted([os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir) if f.endswith(".png")])
    if len(image_files) == 0:
        raise FileNotFoundError(f"No test images found under: {test_image_dir}")

    depth_files = []
    for img_p in image_files:
        base = os.path.basename(img_p)
        d_p = os.path.join(test_depth_dir, base)
        depth_files.append(d_p if os.path.exists(d_p) else None)

    test_ds = NYUDataset(
        image_paths=np.array(image_files),
        label_paths=None,
        depth_paths=np.array(depth_files),
        cfg=cfg,
        transform=get_valid_transforms(cfg),
        color_transform=None,
        enable_smart_crop=False,
        is_train=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=max(1, int(cfg.BATCH_SIZE) * max(1, int(args.batch_mul))),
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # Use first sample to determine shape
    sample0 = test_ds[0]
    if isinstance(sample0, (tuple, list)) and len(sample0) == 3:
        _x0, _y0, meta0 = sample0
    elif isinstance(sample0, (tuple, list)) and len(sample0) == 5:
        _x0, _y0, meta0, _d0, _v0 = sample0
    else:
        raise ValueError("Unexpected test_ds[0] structure from NYUDataset")
    h_orig, w_orig = int(meta0["orig_h"]), int(meta0["orig_w"])
    n = len(image_files)

    mm_path = os.path.join(out_dir, f"acc_probs_{int(time.time())}.dat")
    acc = None
    try:
        acc = np.memmap(mm_path, dtype="float32", mode="w+", shape=(n, h_orig, w_orig, int(cfg.NUM_CLASSES)))
        acc[:] = 0.0
        acc.flush()

        for f in folds:
            fold_dir = os.path.join(exp_dir, f"fold{f}")
            wp = best_ckpt_path(fold_dir)
            w_fold = 1.0 / float(len(folds))
            print(f"[Fold {f}] T={global_temp:.3f} tta={len(tta_combs)} ckpt={os.path.basename(wp)}")

            model = SegFPN(
                num_classes=cfg.NUM_CLASSES, 
                in_channels=cfg.IN_CHANNELS, 
                pretrained=False,
                use_depth_aux=bool(getattr(cfg, "USE_DEPTH_AUX", False))
            )
            # state = safe_torch_load(wp, map_location="cpu")
            # model.load_state_dict(state, strict=True)
            # model.to(cfg.DEVICE)

            state = safe_torch_load(wp, map_location="cpu")
            model.load_state_dict(state, strict=True)
            model.to(cfg.DEVICE)
            model.eval()

            predictor = Predictor(model, test_loader, cfg.DEVICE, cfg)
            it = predictor.predict_logits(tta_combs=tta_combs, temperature=float(global_temp), return_details=True)
            it = tqdm(it, total=n, desc=f"Infer fold{f}", disable=bool(getattr(args, "no_progress", False)))

            # Folder for detailed logits
            logits_out_dir = os.path.join(out_dir, "logits_detailed", f"fold{f}")
            os.makedirs(logits_out_dir, exist_ok=True)

            idx = 0
            for item in it:
                # item keys: merged_probs (H,W,C), branches (list), meta (dict)
                probs_item = item["merged_probs"]
                meta = item["meta"]
                
                # Update Ensemble Accumulator
                acc[idx] += probs_item * float(w_fold)
                
                # Save Per-Fold Merged Logits
                # User requested "val_oof_logits.npy" and "test_logits.npy".
                # Here we are in test ensemble. We save per-fold logits.
                # "merged_logits" from predictor is (C, H, W) numpy float32 (or float16 if we optimized return).
                # Implementation in Predictor returns float32 for consistency, we cast here.
                
                fid = meta.get("file_id", f"img_{idx:05d}")
                fname = f"{fid}.npy"
                
                if "merged_logits" in item:
                    # Save Merged Logits (C, H, W)
                    l_np = item["merged_logits"].astype(np.float16)
                    np.save(os.path.join(logits_out_dir, fname), l_np)
                
                # We skip saving individual branches to save space, unless debug needed.
                # If needed, uncomment:
                # for b_idx, b_data in enumerate(item["branches"]):
                #     ...

                idx += 1
            acc.flush()

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("Generating submission...")
        preds = []
        for i in tqdm(range(n), disable=bool(getattr(args, "no_progress", False))):
            preds.append(np.argmax(acc[i], axis=2).astype(np.uint8))
        preds = np.array(preds)

        submission_npy = os.path.join(out_dir, "submission.npy")
        submission_zip = os.path.join(out_dir, "submission.zip")
        np.save(submission_npy, preds)
        with zipfile.ZipFile(submission_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(submission_npy, arcname="submission.npy")

        manifest = {
            "exp_dir": exp_dir,
            "folds": folds,
            "global_temp": float(global_temp),
            "override_temp": args.temp,
            "oof_summary": (str(args.oof_summary) if args.oof_summary is not None else os.path.join(exp_dir, "oof_summary.json")),
            "tta_combs": [{"scale": float(s), "hflip": bool(f)} for s, f in tta_combs],
            "data_root": str(getattr(cfg, "DATA_ROOT", "")),
            "test_dir": str(getattr(cfg, "TEST_DIR", "")),
            "num_classes": int(cfg.NUM_CLASSES),
            "num_test_images": int(n),
            "output": {"submission_npy": submission_npy, "submission_zip": submission_zip},
        }
        with open(os.path.join(out_dir, "ensemble_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Saved: {submission_npy}")
        print(f"Saved: {submission_zip}")
    finally:
        try:
            if acc is not None:
                try:
                    acc.flush()
                except Exception:
                    pass
            if os.path.exists(mm_path):
                os.remove(mm_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()