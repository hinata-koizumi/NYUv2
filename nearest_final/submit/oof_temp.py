"""
Compute a *global* temperature T* via OOF (out-of-fold) evaluation.
Exp100 Fix: Correctly handles Depth Nearest Interpolation during TTA.
"""

from __future__ import annotations

import argparse
import json
import os

from main.submit.utils import discover_folds, fixed_tta_combs, load_cfg_from_fold_dir, safe_torch_load


def _collect_train_paths(cfg):
    image_dir = os.path.join(cfg.TRAIN_DIR, "image")
    label_dir = os.path.join(cfg.TRAIN_DIR, "label")
    depth_dir = os.path.join(cfg.TRAIN_DIR, "depth")

    images = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    labels = sorted([f for f in os.listdir(label_dir) if f.endswith(".png")])
    depths = sorted([f for f in os.listdir(depth_dir) if f.endswith(".png")])

    img_map = {f: os.path.join(image_dir, f) for f in images}
    lbl_map = {f: os.path.join(label_dir, f) for f in labels}
    dep_map = {f: os.path.join(depth_dir, f) for f in depths}

    keys = sorted(list(set(img_map.keys()) & set(lbl_map.keys()) & set(dep_map.keys())))
    if len(keys) == 0:
        raise ValueError("No common files found in train/image,label,depth")

    import numpy as np

    x_img = np.array([img_map[k] for k in keys])
    x_lbl = np.array([lbl_map[k] for k in keys])
    x_dep = np.array([dep_map[k] for k in keys])
    return keys, x_img, x_lbl, x_dep


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m main.submit.oof_temp",
        description="Compute global best temperature T* from OOF (all folds aggregated).",
    )
    p.add_argument("--exp_dir", type=str, required=True)
    p.add_argument("--out_json", type=str, default=None, help="Default: <exp_dir>/oof_summary.json")
    p.add_argument("--temps", type=str, default="0.7,0.8,0.9,1.0")
    p.add_argument("--data_root", type=str, default=None, help="Override cfg.DATA_ROOT.")
    p.add_argument("--batch_mul", type=int, default=1, help="Batch size multiplier for OOF relative to cfg.BATCH_SIZE.")
    return p


def main(argv: list[str] | None = None) -> None:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.model_selection import KFold
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from main.data.dataset import NYUDataset
    from main.data.transforms import get_valid_transforms
    from main.model.meta_arch import SegFPN
    from main.utils.metrics import compute_metrics, update_confusion_matrix
    from main.utils.misc import configure_runtime, seed_everything, worker_init_fn

    args = build_parser().parse_args(argv)
    exp_dir = os.path.abspath(os.path.expanduser(str(args.exp_dir)))
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"--exp_dir not found: {exp_dir}")

    folds = discover_folds(exp_dir)
    cfg = load_cfg_from_fold_dir(os.path.join(exp_dir, f"fold{folds[0]}"))
    if args.data_root is not None:
        cfg = cfg.with_overrides(DATA_ROOT=str(args.data_root))
        cfg.validate()

    seed_everything(cfg.SEED)
    configure_runtime(cfg)

    temps = [float(x.strip()) for x in str(args.temps).split(",") if x.strip() != ""]
    if len(temps) == 0:
        raise ValueError("--temps must contain at least one value")

    tta_combs = fixed_tta_combs()

    keys, x_img, x_lbl, x_dep = _collect_train_paths(cfg)
    kf = KFold(n_splits=int(cfg.N_FOLDS), shuffle=True, random_state=int(cfg.SEED))
    splits = list(kf.split(x_img))

    cms = {t: np.zeros((int(cfg.NUM_CLASSES), int(cfg.NUM_CLASSES)), dtype=np.int64) for t in temps}

    device_type = "cuda" if str(cfg.DEVICE) == "cuda" else "cpu"
    use_cuda = device_type == "cuda"
    use_amp = bool(getattr(cfg, "USE_AMP", False)) and use_cuda
    amp_dtype_name = str(getattr(cfg, "AMP_DTYPE", "bf16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16

    def _tta_probs(model, x: torch.Tensor, temperature: float) -> torch.Tensor:
        base_h, base_w = int(x.shape[2]), int(x.shape[3])
        acc = None
        for scale, hflip in tta_combs:
            x_aug = x
            if float(scale) != 1.0:
                # 1. Enforce 32px Alignment
                nh = max(32, int(round(base_h * float(scale) / 32.0)) * 32)
                nw = max(32, int(round(base_w * float(scale) / 32.0)) * 32)
                
                # 2. Split: RGB(Bilinear) vs Depth(Nearest)
                rgb = x_aug[:, :3, :, :]
                dep = x_aug[:, 3:, :, :]
                
                rgb = F.interpolate(rgb, size=(nh, nw), mode="bilinear", align_corners=False)
                dep = F.interpolate(dep, size=(nh, nw), mode="nearest") # <--- NEAREST ENFORCED
                
                x_aug = torch.cat([rgb, dep], dim=1)

            if bool(hflip):
                x_aug = torch.flip(x_aug, dims=[3])

            with torch.amp.autocast(device_type=device_type, enabled=use_amp, dtype=amp_dtype):
                out = model(x_aug)
            logits = out[0] if isinstance(out, (tuple, list)) else out

            if bool(hflip):
                logits = torch.flip(logits, dims=[3])
            
            if float(scale) != 1.0:
                logits = F.interpolate(logits, size=(base_h, base_w), mode="bilinear", align_corners=False)

            probs = torch.softmax(logits / float(temperature), dim=1)
            acc = probs if acc is None else (acc + probs)
        return acc / float(len(tta_combs))

    for fold_idx in folds:
        if fold_idx >= len(splits):
            raise ValueError(f"Found fold{fold_idx} under exp_dir but cfg.N_FOLDS={cfg.N_FOLDS}")
        _tr_idx, va_idx = splits[fold_idx]

        fold_dir = os.path.join(exp_dir, f"fold{fold_idx}")
        ckpt = os.path.join(fold_dir, "model_best.pth")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Missing weights: {ckpt}")

        ds = NYUDataset(
            image_paths=x_img[va_idx],
            label_paths=x_lbl[va_idx],
            depth_paths=x_dep[va_idx],
            cfg=cfg,
            transform=get_valid_transforms(cfg),
            color_transform=None,
            enable_smart_crop=False,
            is_train=False,
        )
        loader = DataLoader(
            ds,
            batch_size=max(1, int(cfg.BATCH_SIZE) * max(1, int(args.batch_mul))),
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
            prefetch_factor=2,
        )

        model = SegFPN(num_classes=cfg.NUM_CLASSES, in_channels=cfg.IN_CHANNELS, pretrained=False)
        state = safe_torch_load(ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        model.to(cfg.DEVICE)
        if cfg.DEVICE == "cuda" and bool(getattr(cfg, "USE_CHANNELS_LAST", False)):
            model = model.to(memory_format=torch.channels_last)
        model.eval()

        print(f"[Fold {fold_idx}] OOF samples={len(ds)} ckpt=model_best.pth tta={len(tta_combs)}")

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"OOF fold{fold_idx}", leave=False):
                x, y = batch[0], batch[1]
                x = x.to(cfg.DEVICE, non_blocking=True)
                y = y.to(cfg.DEVICE, non_blocking=True)

                for t in temps:
                    probs = _tta_probs(model, x, temperature=float(t))
                    pred = torch.argmax(probs, dim=1).detach().cpu().numpy().astype(np.int32)
                    gt = y.detach().cpu().numpy().astype(np.int32)
                    for i in range(int(pred.shape[0])):
                        cms[t] = update_confusion_matrix(pred[i], gt[i], int(cfg.NUM_CLASSES), int(cfg.IGNORE_INDEX), cms[t])

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    per_temp = {}
    best_t = None
    best_miou = -1.0
    for t in temps:
        _pix, miou, _iou = compute_metrics(cms[t])
        per_temp[str(t)] = float(miou)
        if float(miou) > best_miou:
            best_miou = float(miou)
            best_t = float(t)

    out_json = str(args.out_json) if args.out_json is not None else os.path.join(exp_dir, "oof_summary.json")
    out_json = os.path.abspath(os.path.expanduser(out_json))

    summary = {
        "exp_dir": exp_dir,
        "folds": folds,
        "temps": temps,
        "tta_combs": [{"scale": float(s), "hflip": bool(f)} for s, f in tta_combs],
        "best_oof_temp": float(best_t),
        "best_oof_miou": float(best_miou),
        "per_temp_oof_miou": per_temp,
        "num_train_keys": int(len(keys)),
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()