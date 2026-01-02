import argparse
import ast
import json
import os
import re
import time
import zipfile
from typing import Any, get_args, get_origin


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m main", description="NYUv2 minimal pipeline (train + submit)")
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Train model")
    tr.add_argument("--preset", type=str, default=None, help="Doc reproduction preset (e.g. exp093_5)")
    tr.add_argument("--exp_name", type=str, default=None, help="Override experiment name")
    tr.add_argument("--fold", type=int, default=None, help="Run a specific fold (0-4)")
    tr.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override Config fields (repeatable). Example: --set COPY_PASTE_ENABLE=true --set COPY_PASTE_PROB=0.5",
    )

    sb = sub.add_parser("submit", help="Run inference + generate submission.zip (logits+TTA)")
    sb.add_argument("--exp_name", type=str, default=None, help="Override experiment name")
    sb.add_argument("--folds", type=int, default=None, help="Number of folds to use (default: cfg.N_FOLDS)")
    sb.add_argument(
        "--temp",
        type=float,
        default=None,
        help="Override temperature. If omitted, uses fold0 summary.json best_tta_temp (or cfg.TEMPERATURES[0]).",
    )
    sb.add_argument(
        "--ckpt_k",
        type=int,
        default=None,
        help="Checkpoint ensemble per fold: use top-K epoch checkpoints (+ model_best) at submit-time. "
        "If omitted, uses cfg.SUBMIT_CKPT_ENSEMBLE_K. 1 disables.",
    )
    sb.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override Config fields (repeatable). Example: --set N_FOLDS=5",
    )
    return p


def _coerce_scalar(val: str) -> Any:
    """
    Best-effort parsing for CLI overrides.
    Supports:
      - bool: true/false/1/0
      - int/float
      - tuples/lists/dicts via Python literal (ast.literal_eval)
      - fallback: raw string
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

    # Python literal (tuple/list/dict/str/...)
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def _parse_seq_of_numbers(val: str, elem_type: type) -> tuple:
    """
    Parse "1,2,3" into tuple[int,...] or tuple[float,...].
    """
    parts = [p.strip() for p in str(val).split(",") if p.strip() != ""]
    if elem_type is float:
        return tuple(float(p) for p in parts)
    return tuple(int(p) for p in parts)


def _coerce_to_field_type(field_type: Any, raw: Any) -> Any:
    """
    Coerce an override value to the annotated Config field type.
    Keeps behavior permissive but predictable.
    """
    origin = get_origin(field_type)
    args = get_args(field_type)

    # Optional[T] => Union[T, NoneType]
    if origin is None and str(field_type).startswith("typing.Optional"):
        # typing.Optional doesn't always roundtrip nicely in runtime typing;
        # let literal parsing handle None or user can pass "None".
        pass
    if origin is type(None):
        return None
    if origin is not None and origin is list:
        # not used in current Config, but keep it generic
        inner = args[0] if args else Any
        if isinstance(raw, str):
            raw = _coerce_scalar(raw)
        if isinstance(raw, (tuple, list)):
            return [(_coerce_to_field_type(inner, x)) for x in raw]
        return [(_coerce_to_field_type(inner, raw))]

    # Tuples: Tuple[int, ...] etc
    if origin is tuple:
        elem_t = args[0] if args else Any
        if isinstance(raw, str):
            # allow "1,2,3" as shorthand
            if "," in raw and not (raw.strip().startswith(("(", "[", "{"))):
                if elem_t in (int, float):
                    return _parse_seq_of_numbers(raw, elem_t)
            raw2 = _coerce_scalar(raw)
        else:
            raw2 = raw
        if isinstance(raw2, tuple):
            return tuple(_coerce_to_field_type(elem_t, x) for x in raw2)
        if isinstance(raw2, list):
            return tuple(_coerce_to_field_type(elem_t, x) for x in raw2)
        # scalar -> 1-tuple
        return ( _coerce_to_field_type(elem_t, raw2), )

    # Scalars
    if field_type is bool:
        if isinstance(raw, bool):
            return raw
        return bool(_coerce_scalar(str(raw)))
    if field_type is int:
        if isinstance(raw, int) and not isinstance(raw, bool):
            return raw
        return int(_coerce_scalar(str(raw)))
    if field_type is float:
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return float(raw)
        return float(_coerce_scalar(str(raw)))
    if field_type is str:
        return str(raw)

    # Fallback: best-effort scalar parse
    if isinstance(raw, str):
        return _coerce_scalar(raw)
    return raw


def _parse_set_overrides(pairs: list[str], cfg) -> dict[str, Any]:
    """
    Parse repeated --set KEY=VALUE pairs into a dict compatible with cfg.with_overrides().
    """
    overrides: dict[str, Any] = {}
    fields = getattr(cfg, "__dataclass_fields__", {})
    for item in pairs:
        if "=" not in str(item):
            raise ValueError(f"Invalid --set {item!r}. Expected KEY=VALUE.")
        k, v = str(item).split("=", 1)
        key = k.strip()
        if key == "":
            raise ValueError(f"Invalid --set {item!r}. Key is empty.")
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

    cfg = _build_cfg(preset=preset, exp_name=exp_name, set_pairs=set_pairs)
    seed_everything(cfg.SEED)
    configure_runtime(cfg)

    output_root = str(getattr(cfg, "OUTPUT_ROOT", os.path.join("data", "output")))
    output_dir = os.path.join(output_root, cfg.EXP_NAME)
    os.makedirs(output_dir, exist_ok=True)

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

    X_img = np.array([img_map[k] for k in keys])
    X_lbl = np.array([lbl_map[k] for k in keys])
    X_dep = np.array([dep_map[k] for k in keys])

    with open(os.path.join(output_dir, "train_keys.txt"), "w") as f:
        f.write("\n".join(keys))

    kf = KFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    splits = list(kf.split(X_img))

    folds_to_run = range(cfg.N_FOLDS) if fold is None else [int(fold)]

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

        # DataLoader perf knobs (CUDA):
        # - pin_memory=True enables async H2D copies with non_blocking=True
        # - prefetch_factor helps keep GPU fed when num_workers>0
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
        # Optional memory format optimization (best on CUDA)
        if cfg.DEVICE == "cuda" and bool(getattr(cfg, "USE_CHANNELS_LAST", False)):
            model = model.to(memory_format=torch.channels_last)
        # Optional torch.compile (PyTorch 2.x)
        if cfg.DEVICE == "cuda" and bool(getattr(cfg, "USE_TORCH_COMPILE", False)):
            try:
                model = torch.compile(model, mode=str(getattr(cfg, "TORCH_COMPILE_MODE", "default")))
            except Exception as e:
                print(f"[WARN] torch.compile failed, continuing without compile: {e}")
        # EMA (Optional per config)
        ema = None
        if bool(getattr(cfg, "USE_EMA", True)):
            ema = ModelEMA(model, cfg.EMA_DECAY)

        # Criterion
        criterion = nn.CrossEntropyLoss(ignore_index=cfg.IGNORE_INDEX).to(cfg.DEVICE)

        # Optimizer
        opt_name = str(getattr(cfg, "OPTIMIZER", "adamw")).lower()
        if opt_name == "sam_adamw":
            from main.utils.sam import SAM
            optimizer = SAM(
                model.parameters(),
                optim.AdamW,
                rho=float(getattr(cfg, "SAM_RHO", 0.02)),
                lr=cfg.LEARNING_RATE,
                weight_decay=cfg.WEIGHT_DECAY
            )
        else:
            optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)

        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=cfg.ETA_MIN)

        ckpt = CheckpointManager(fold_dir, top_k=cfg.SAVE_TOP_K)

        best_miou = -1.0
        best_epoch = -1
        no_improve = 0
        min_epochs = getattr(cfg, "MIN_EPOCHS", 20)
        patience = getattr(cfg, "EARLY_STOPPING_PATIENCE", 10)
        min_delta = getattr(cfg, "EARLY_STOPPING_MIN_DELTA", 1e-4)

        # GradScaler is only needed for fp16. For bf16, keep scaler disabled.
        use_amp = bool(getattr(cfg, "USE_AMP", False)) and (cfg.DEVICE == "cuda")
        amp_dtype = str(getattr(cfg, "AMP_DTYPE", "bf16")).lower()
        use_scaler = use_amp and (amp_dtype == "fp16")
        # `torch.cuda.amp.GradScaler` is deprecated; use `torch.amp.GradScaler`.
        # Scaler is only used for fp16 on CUDA, so device_type is effectively "cuda" here.
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

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

            logger.log_epoch(
                epoch,
                {
                    "lr": lr,
                    "train_loss": tr_loss,
                    "valid_loss": va_loss,
                    "valid_miou": miou,
                    "valid_pixel_acc": acc,
                    "class_iou": class_iou,
                    "throughput": throughput,
                    "gpu_mem": mem_max,
                },
            )

            ckpt.save(eval_model, epoch, miou)

            improved = miou > best_miou + min_delta
            if improved:
                best_miou = float(miou)
                best_epoch = int(epoch)
                no_improve = 0
                ckpt._atomic_torch_save(eval_model.state_dict(), os.path.join(fold_dir, "model_best.pth"))
                logger.save_summary({"best_miou": best_miou, "best_epoch": best_epoch, "config": cfg.to_dict()})
            else:
                no_improve += 1

            if bool(getattr(cfg, "USE_EARLY_STOPPING", True)) and epoch >= min_epochs and no_improve >= patience:
                print(
                    f"[EARLY STOP] epoch={epoch} best={best_miou:.4f} "
                    f"best_epoch={best_epoch} no_improve={no_improve}/{patience}"
                )
                break

        # Optional: doc reproduction temperature sweep
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


def _submit(
    *,
    exp_name: str | None,
    folds: int | None,
    temp: float | None,
    ckpt_k: int | None,
    set_pairs: list[str] | None,
) -> None:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from main.data.dataset import NYUDataset
    from main.data.transforms import get_valid_transforms
    from main.engine.inference import Predictor
    from main.model.meta_arch import SegFPN
    from main.utils.misc import configure_runtime, seed_everything, worker_init_fn

    cfg = _build_cfg(exp_name=exp_name, set_pairs=set_pairs)
    if folds is not None:
        cfg = cfg.with_overrides(N_FOLDS=int(folds))

    seed_everything(cfg.SEED)
    configure_runtime(cfg)

    output_root = str(getattr(cfg, "OUTPUT_ROOT", os.path.join("data", "output")))
    output_dir = os.path.join(output_root, cfg.EXP_NAME)

    # Submit-time options
    if ckpt_k is None:
        ckpt_k = int(getattr(cfg, "SUBMIT_CKPT_ENSEMBLE_K", 1))
    ckpt_k = max(1, int(ckpt_k))

    def _fold_best_temp(fold_dir: str) -> float:
        """
        Per-fold temperature:
          - CLI --temp overrides everything
          - else fold summary.json best_tta_temp
          - else cfg.TEMPERATURES[0]
          - else 1.0
        """
        if temp is not None:
            return float(temp)
        try:
            p = os.path.join(fold_dir, "summary.json")
            if os.path.exists(p):
                with open(p, "r") as f:
                    s = json.load(f)
                if "best_tta_temp" in s:
                    return float(s["best_tta_temp"])
        except Exception:
            pass
        ts = list(getattr(cfg, "TEMPERATURES", []))
        return float(ts[0]) if len(ts) else 1.0

    def _fold_ckpts(fold_dir: str, k: int) -> list[str]:
        """
        model_best.pth + top-K epoch checkpoints by mIoU parsed from filename.
        """
        best = os.path.join(fold_dir, "model_best.pth")
        if not os.path.exists(best):
            raise FileNotFoundError(f"Missing weights: {best}")

        # pattern: model_epoch{epoch}_miou{miou:.4f}.pth
        pat = re.compile(r"^model_epoch(\d+)_miou([0-9.]+)\.pth$")
        scored: list[tuple[float, str]] = []
        for fn in os.listdir(fold_dir):
            m = pat.match(fn)
            if not m:
                continue
            miou = float(m.group(2))
            scored.append((miou, os.path.join(fold_dir, fn)))
        scored.sort(key=lambda x: x[0], reverse=True)

        picked = [best] + [p for _m, p in scored[: int(k)]]
        # dedupe while preserving order
        out: list[str] = []
        seen = set()
        for p in picked:
            if p not in seen:
                out.append(p)
                seen.add(p)
        return out

    test_image_dir = os.path.join(cfg.TEST_DIR, "image")
    test_depth_dir = os.path.join(cfg.TEST_DIR, "depth")

    image_files = sorted([os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir) if f.endswith(".png")])
    if len(image_files) == 0:
        raise FileNotFoundError("No test images found.")

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
        batch_size=cfg.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # Allocate memmap at original resolution
    sample0 = test_ds[0]
    # NYUDataset may return either (x,y,meta) or (x,y,meta,depth_target,depth_valid)
    if isinstance(sample0, (tuple, list)) and len(sample0) == 3:
        _x0, _y0, meta0 = sample0
    elif isinstance(sample0, (tuple, list)) and len(sample0) == 5:
        _x0, _y0, meta0, _d0, _v0 = sample0
    else:
        raise ValueError(f"Unexpected test_ds[0] structure: type={type(sample0)} len={len(sample0) if hasattr(sample0,'__len__') else 'n/a'}")
    H_orig, W_orig = int(meta0["orig_h"]), int(meta0["orig_w"])
    N = len(image_files)

    os.makedirs("tmp", exist_ok=True)
    mm_path = f"tmp/acc_logits_{int(time.time())}.dat"
    acc = np.memmap(mm_path, dtype="float32", mode="w+", shape=(N, H_orig, W_orig, cfg.NUM_CLASSES))
    acc[:] = 0.0
    acc.flush()

    # Per-fold normalized ensembling:
    # Each fold contributes weight 1.0 total, regardless of how many checkpoints are used.
    for f in range(int(cfg.N_FOLDS)):
        fold_dir = os.path.join(output_dir, f"fold{f}")
        fold_temp = _fold_best_temp(fold_dir)
        ckpts = _fold_ckpts(fold_dir, ckpt_k)
        w = 1.0 / float(len(ckpts))
        print(f"[Fold {f}] temp={fold_temp:.3f} ckpts={len(ckpts)} (ckpt_k={ckpt_k})")

        for wp in ckpts:
            print(f"  Processing {wp}...")
            model = SegFPN(num_classes=cfg.NUM_CLASSES, in_channels=cfg.IN_CHANNELS)
            state = torch.load(wp, map_location="cpu")
            incompatible = model.load_state_dict(state, strict=False)
            if (len(getattr(incompatible, "missing_keys", [])) > 0) or (len(getattr(incompatible, "unexpected_keys", [])) > 0):
                print(
                    f"[INFO] Loaded with strict=False "
                    f"(missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)})"
                )
            model.to(cfg.DEVICE)
            if cfg.DEVICE == "cuda" and bool(getattr(cfg, "USE_CHANNELS_LAST", False)):
                model = model.to(memory_format=torch.channels_last)
            if cfg.DEVICE == "cuda" and bool(getattr(cfg, "USE_TORCH_COMPILE", False)):
                try:
                    model = torch.compile(model, mode=str(getattr(cfg, "TORCH_COMPILE_MODE", "default")))
                except Exception as e:
                    print(f"[WARN] torch.compile failed, continuing without compile: {e}")
            model.eval()

            predictor = Predictor(model, test_loader, cfg.DEVICE, cfg)
            idx = 0
            for probs_item in predictor.predict_logits(temperature=float(fold_temp)):
                acc[idx] += probs_item * w
                idx += 1
            acc.flush()

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("Generating submission...")
    preds = []
    for i in tqdm(range(N)):
        preds.append(np.argmax(acc[i], axis=2).astype(np.uint8))
    preds = np.array(preds)

    np.save("tmp/submission.npy", preds)
    with zipfile.ZipFile("submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write("tmp/submission.npy", arcname="tmp/submission.npy")
    print("Done! submission.zip created.")

    if os.path.exists(mm_path):
        os.remove(mm_path)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "train":
        _train(preset=args.preset, exp_name=args.exp_name, fold=args.fold, set_pairs=getattr(args, "set", None))
        return
    if args.cmd == "submit":
        _submit(
            exp_name=args.exp_name,
            folds=args.folds,
            temp=args.temp,
            ckpt_k=getattr(args, "ckpt_k", None),
            set_pairs=getattr(args, "set", None),
        )
        return
    raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
