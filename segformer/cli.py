import argparse
import ast
import os
import re
import warnings  # 追加
from typing import Any, get_args, get_origin

# --- 警告抑制を追加 ---
warnings.filterwarnings("ignore", category=UserWarning) 
# ----------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m main", description="NYUv2 Exp100 Pipeline (Train Only)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Train command
    tr = sub.add_parser("train", help="Train Exp100 model")
    tr.add_argument("--preset", type=str, default="exp100", help="Config preset (default: exp100)")
    tr.add_argument("--exp_name", type=str, default="exp100_final_nearest", help="Experiment name")
    tr.add_argument("--fold", type=int, default=None, help="Run specific fold (0-4)")
    tr.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override Config fields (repeatable). Example: --set LEARNING_RATE=2e-4",
    )

    return p


def _coerce_scalar(val: str) -> Any:
    """
    Best-effort parsing for CLI overrides.
    """
    s = str(val).strip()
    low = s.lower()
    if low in ("true", "t", "yes", "y", "1", "on"):
        return True
    if low in ("false", "f", "no", "n", "0", "off"):
        return False

    # int / float
    try:
        if re.fullmatch(r"[+-]?\d+", s):
            return int(s)
        if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?", s) or re.fullmatch(
            r"[+-]?\d+(?:[eE][+-]?\d+)", s
        ):
            return float(s)
    except Exception:
        pass

    # Python literal
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def _coerce_to_field_type(field_type: Any, raw: Any) -> Any:
    """
    Coerce override value to Config field type.
    """
    origin = get_origin(field_type)
    args = get_args(field_type)

    if origin is None and str(field_type).startswith("typing.Optional"):
        pass
    if origin is type(None):
        return None
    
    # Tuples
    if origin is tuple:
        elem_t = args[0] if args else Any
        if isinstance(raw, str):
            if "," in raw and not (raw.strip().startswith(("(", "[", "{"))):
                parts = [p.strip() for p in raw.split(",") if p.strip()]
                if elem_t is float:
                    return tuple(float(p) for p in parts)
                return tuple(int(p) for p in parts)
            raw2 = _coerce_scalar(raw)
        else:
            raw2 = raw
        
        if isinstance(raw2, (list, tuple)):
            return tuple(_coerce_to_field_type(elem_t, x) for x in raw2)
        return (_coerce_to_field_type(elem_t, raw2),)

    # Scalars
    if field_type is bool:
        if isinstance(raw, bool): return raw
        return bool(_coerce_scalar(str(raw)))
    if field_type is int:
        if isinstance(raw, int) and not isinstance(raw, bool): return raw
        return int(_coerce_scalar(str(raw)))
    if field_type is float:
        if isinstance(raw, (int, float)) and not isinstance(raw, bool): return float(raw)
        return float(_coerce_scalar(str(raw)))
    if field_type is str:
        return str(raw)

    if isinstance(raw, str):
        return _coerce_scalar(raw)
    return raw


def _parse_set_overrides(pairs: list[str], cfg) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    fields = getattr(cfg, "__dataclass_fields__", {})
    for item in pairs:
        if "=" not in str(item):
            raise ValueError(f"Invalid --set {item!r}. Expected KEY=VALUE.")
        k, v = str(item).split("=", 1)
        key = k.strip()
        if key not in fields:
            raise ValueError(f"Unknown config field in --set: {key!r}")
        raw = v.strip()
        field_t = fields[key].type
        overrides[key] = _coerce_to_field_type(field_t, raw)
    return overrides


def _build_cfg(*, preset: str | None = None, exp_name: str | None = None, set_pairs: list[str] | None = None):
    from main.configs.base_config import Config

    cfg = Config()
    if preset:
        cfg = cfg.apply_preset(preset)
    if exp_name:
        cfg = cfg.with_overrides(EXP_NAME=str(exp_name))
    if set_pairs:
        overrides = _parse_set_overrides(list(set_pairs), cfg)
        if overrides:
            cfg = cfg.with_overrides(**overrides)
    cfg.validate()
    return cfg


def _train(*, preset: str | None, exp_name: str | None, fold: int | None, set_pairs: list[str] | None) -> None:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.model_selection import KFold
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader
    import time

    from main.data.dataset import NYUDataset
    from main.data.transforms import get_color_transforms, get_train_transforms, get_valid_transforms
    from main.engine.trainer import train_one_epoch, validate, validate_tta_sweep
    from main.model.meta_arch import build_model
    from main.utils.misc import (
        CheckpointManager,
        Logger,
        ModelEMA,
        configure_runtime,
        save_config,
        seed_everything,
        worker_init_fn,
    )
    from main.utils.sam import SAM

    cfg = _build_cfg(preset=preset, exp_name=exp_name, set_pairs=set_pairs)
    seed_everything(cfg.SEED)
    configure_runtime(cfg)

    output_root = str(getattr(cfg, "OUTPUT_ROOT", os.path.join("data", "output")))
    output_dir = os.path.join(output_root, cfg.EXP_NAME)
    os.makedirs(output_dir, exist_ok=True)

    # --- Data Loading (Shared across folds) ---
    image_dir = os.path.join(cfg.TRAIN_DIR, "image")
    label_dir = os.path.join(cfg.TRAIN_DIR, "label")
    depth_dir = os.path.join(cfg.TRAIN_DIR, "depth")

    images = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    img_map = {f: os.path.join(image_dir, f) for f in images}
    lbl_map = {f: os.path.join(label_dir, f) for f in images}
    dep_map = {f: os.path.join(depth_dir, f) for f in images}

    # Intersection check
    keys = sorted(list(set(img_map.keys()) & set(lbl_map.keys()) & set(dep_map.keys())))
    if len(keys) == 0:
        raise ValueError("No common files found in train/image,label,depth")

    X_img = np.array([img_map[k] for k in keys])
    X_lbl = np.array([lbl_map[k] for k in keys])
    X_dep = np.array([dep_map[k] for k in keys])

    with open(os.path.join(output_dir, "train_keys.txt"), "w") as f:
        f.write("\n".join(keys))

    kf = KFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    splits = list(kf.split(X_img))

    folds_to_run = range(cfg.N_FOLDS) if fold is None else [int(fold)]

    # --- Fold Loop ---
    for fold_idx in folds_to_run:
        print(f"\n{'='*20} Fold {fold_idx} {'='*20}")
        fold_dir = os.path.join(output_dir, f"fold{fold_idx}")
        save_config(cfg, fold_dir)
        logger = Logger(fold_dir)

        tr_idx, va_idx = splits[fold_idx]

        train_ds = NYUDataset(
            image_paths=X_img[tr_idx],
            label_paths=X_lbl[tr_idx],
            depth_paths=X_dep[tr_idx],
            cfg=cfg,
            transform=get_train_transforms(cfg),
            color_transform=get_color_transforms(cfg),
            enable_smart_crop=True,
            is_train=True,
        )
        valid_ds = NYUDataset(
            image_paths=X_img[va_idx],
            label_paths=X_lbl[va_idx],
            depth_paths=X_dep[va_idx],
            cfg=cfg,
            transform=get_valid_transforms(cfg),
            color_transform=None,
            enable_smart_crop=False,
            is_train=False,
        )

        dl_common = dict(
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            persistent_workers=(cfg.NUM_WORKERS > 0),
            prefetch_factor=2 if cfg.NUM_WORKERS > 0 else None,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            **dl_common,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            **dl_common,
        )

        model = build_model(cfg).to(cfg.DEVICE)
        if cfg.DEVICE == "cuda" and bool(getattr(cfg, "USE_CHANNELS_LAST", False)):
            model = model.to(memory_format=torch.channels_last)

        # EMA
        ema = None
        if bool(getattr(cfg, "USE_EMA", True)):
            ema = ModelEMA(model, cfg.EMA_DECAY)

        criterion = nn.CrossEntropyLoss(ignore_index=cfg.IGNORE_INDEX).to(cfg.DEVICE)

        # Optimizer (SAM Enforced)
        optimizer = SAM(
            model.parameters(),
            optim.AdamW,
            rho=float(getattr(cfg, "SAM_RHO", 0.02)),
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY
        )

        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=cfg.ETA_MIN)
        ckpt = CheckpointManager(fold_dir, top_k=cfg.SAVE_TOP_K)

        best_miou = -1.0
        best_epoch = -1
        
        # Mixed Precision
        use_amp = bool(getattr(cfg, "USE_AMP", False)) and (cfg.DEVICE == "cuda")
        amp_dtype = str(getattr(cfg, "AMP_DTYPE", "bf16")).lower()
        use_scaler = use_amp and (amp_dtype == "fp16") # No scaler for BF16
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

        # --- Epoch Loop ---
        for epoch_idx in range(cfg.EPOCHS):
            epoch = epoch_idx + 1
            train_ds.set_epoch(epoch)

            t0 = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            tr_loss = train_one_epoch(
                model,
                ema,
                train_loader,
                criterion,
                optimizer,
                cfg.DEVICE,
                scaler=scaler,
                use_amp=use_amp,
                grad_accum_steps=getattr(cfg, "GRAD_ACCUM_STEPS", 1),
                cfg=cfg,
            )

            dt = time.time() - t0
            throughput = len(train_loader.dataset) / dt if dt > 0 else 0.0
            mem_max = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

            scheduler.step()

            eval_model = ema.ema if ema is not None else model
            va_loss, miou, acc, class_iou = validate(eval_model, valid_loader, criterion, cfg.DEVICE, cfg)

            print(
                f"Epoch {epoch}: TrLoss={tr_loss:.4f}, VaLoss={va_loss:.4f}, "
                f"mIoU={miou:.4f}, Speed={throughput:.1f} img/s, Mem={mem_max:.1f}GB"
            )

            lr = optimizer.param_groups[0]["lr"]
            logger.log_epoch(epoch, {
                "lr": lr,
                "train_loss": tr_loss,
                "valid_loss": va_loss,
                "valid_miou": miou,
                "valid_pixel_acc": acc,
                "class_iou": class_iou,
                "throughput": throughput,
                "gpu_mem": mem_max,
            })

            ckpt.save(eval_model, epoch, miou)
            # Save "last.pth" for resuming
            ckpt.save_last(model, optimizer, epoch, miou)

            if miou > best_miou:
                best_miou = float(miou)
                best_epoch = int(epoch)
                # 【修正箇所】_atomic_torch_save -> save_artifact に変更
                ckpt.save_artifact(eval_model.state_dict(), os.path.join(fold_dir, "model_best.pth"))
                logger.save_summary({"best_miou": best_miou, "best_epoch": best_epoch, "config": cfg.to_dict()})

        # --- End of Fold: TTA Sweep ---
        try:
            best_temp, best_tta_miou, temp_results = validate_tta_sweep(eval_model, valid_loader, cfg.DEVICE, cfg)
            logger.save_summary(
                {
                    "best_miou": best_miou,
                    "best_epoch": best_epoch,
                    "best_tta_temp": float(best_temp),
                    "best_tta_miou": float(best_tta_miou),
                    "tta_temp_results": {str(k): float(v) for k, v in temp_results.items()},
                    "config": cfg.to_dict(),
                }
            )
        except Exception as e:
            print(f"[WARN] TTA sweep failed: {e}")

        logger.close()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "train":
        _train(preset=args.preset, exp_name=args.exp_name, fold=args.fold, set_pairs=getattr(args, "set", None))
        return
    # 'submit' command removed. Use 'python -m main.submit' instead.
    raise ValueError(f"Unknown command: {args.cmd}")