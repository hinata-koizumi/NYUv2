
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import sys

from ..data.dataset import NYUDataset
from ..data.transforms import get_color_transforms, get_train_transforms, get_valid_transforms
from ..engine.trainer import train_one_epoch, validate, validate_tta_sweep
from ..model.meta_arch import build_model
from ..utils.misc import (
    CheckpointManager,
    Logger,
    ModelEMA,
    configure_runtime,
    save_config,
    seed_everything,
    worker_init_fn,
    get_git_hash,
)
from ..utils.sam import SAM
from ..engine.inference import Predictor

def _save_validation_results(cfg, fold_dir: str, loader, ckpt_path: str):
    """"
    Run inference on validation set and save per-sample outputs.
    """
    
    val_out_dir = os.path.join(fold_dir, "valid_preds")
    os.makedirs(val_out_dir, exist_ok=True)
    
    if not os.path.exists(ckpt_path):
        print(f"[WARN] No checkpoint found at {ckpt_path}, skipping validation saving.")
        return

    model = build_model(cfg)
    print(f"Loading best checkpoint from {ckpt_path} for validation output...")
    try:
        ckpt = torch.load(ckpt_path, map_location=getattr(cfg, "DEVICE", "cpu"))
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
    except Exception as e:
        print(f"Failed to load checkpoint for validation output: {e}")
        return
    model.to(cfg.DEVICE)
    if cfg.DEVICE == "cuda" and bool(getattr(cfg, "USE_CHANNELS_LAST", False)):
        model = model.to(memory_format=torch.channels_last)
    
    predictor = Predictor(model, loader, cfg.DEVICE, cfg)
    
    # Just 1.0 scale, no flip for "standard" validation output
    it = predictor.predict_logits(tta_combs=[(1.0, False)], temperature=1.0, return_details=True)
    
    meta_list = []
    
    for i, item in enumerate(it):
        meta = item["meta"]
        fid = meta.get("file_id", f"val_{i:04d}")
        
        probs = item["merged_probs"]
        pred_mask = np.argmax(probs, axis=0).astype(np.uint8) # H,W
        
        logits_chw = item["branches"][0]["logits"] # fp16 numpy
        
        np.save(os.path.join(val_out_dir, f"{fid}_mask.npy"), pred_mask)
        np.save(os.path.join(val_out_dir, f"{fid}_logits.npy"), logits_chw)
        
        meta_entry = {k: float(v) if isinstance(v, (np.float32, float)) else str(v) for k,v in meta.items()}
        meta_entry["file_id"] = fid
        meta_list.append(meta_entry)
        
    with open(os.path.join(fold_dir, "valid_meta.json"), "w") as f:
        json.dump(meta_list, f, indent=2)
        
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_train(cfg, fold: int | None, limit: int = 0):
    # Save Run Meta (Git Hash etc)
    git_hash = get_git_hash()

    seed_everything(cfg.SEED)
    configure_runtime(cfg)

    output_root = str(getattr(cfg, "OUTPUT_ROOT", os.path.join("data", "output")))
    output_dir = os.path.join(output_root, cfg.EXP_NAME)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save global meta
    with open(os.path.join(output_dir, "run_meta.json"), "w") as f:
         json.dump({
             "commit_hash": git_hash,
             "config_preset": cfg.to_dict(),
             "command": " ".join(sys.argv)
         }, f, indent=2)

    # --- Data Loading (Shared across folds) ---
    image_dir = os.path.join(cfg.TRAIN_DIR, "image")
    label_dir = os.path.join(cfg.TRAIN_DIR, "label")
    depth_dir = os.path.join(cfg.TRAIN_DIR, "depth")

    images = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    if limit > 0:
        images = images[:limit]
        print(f"!!! LIMITING TRAIN DATA TO {limit} IMAGES !!!")
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
        os.makedirs(fold_dir, exist_ok=True)
        # Metadata for this fold
        with open(os.path.join(fold_dir, "fold_meta.json"), "w") as f:
            json.dump({
                "fold": fold_idx,
                "commit_hash": git_hash,
                "seed": int(cfg.SEED),
                "is_ema": bool(getattr(cfg, "USE_EMA", True))
            }, f, indent=2)

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

            tr_loss, tr_aux, tr_grad = train_one_epoch(
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
                f"Epoch {epoch}: TrLoss={tr_loss:.4f}, Aux={tr_aux:.4f}, Grad={tr_grad:.4f}, VaLoss={va_loss:.4f}, "
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
                "grad_norm": tr_grad,
                "aux_loss": tr_aux,
            })

            ckpt.save(eval_model, epoch, miou)
            # Save "last.pth" for resuming
            ckpt.save_last(model, optimizer, epoch, miou)

            if miou > best_miou:
                best_miou = float(miou)
                best_epoch = int(epoch)
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
            
        # --- Final Validation Saving ---
        print("Saving validation results (mask, logits, meta)...")
        _save_validation_results(cfg, fold_dir, valid_loader, os.path.join(fold_dir, "model_best.pth"))

        logger.close()

if __name__ == "__main__":
    import argparse
    from ..configs.base_config import BaseConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=str, default=None, help="Comma separated fold indices to run (e.g. '0,1'). If None, run all.")
    parser.add_argument("--epochs", type=int, default=None, help="Override EPOCHS in config")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (fast run)")
    parser.add_argument("--limit-batches", type=int, default=0, help="Limit number of batches per epoch (debug)")
    
    args = parser.parse_args()
    
    cfg = BaseConfig()
    
    if args.epochs is not None:
        cfg = cfg.with_overrides(EPOCHS=args.epochs)
        
    if args.debug:
        # Debug mode: 2 epochs, enable debug flag
        cfg = cfg.with_overrides(DEBUG=True, EPOCHS=2)
        # cfg.limit_batches = 5 # Handle inside trainer if needed, or just pass to run_train logic?
        # Current run_train uses 'limit' arg for images, not batches.
        
    folds = None
    if args.folds is not None:
        folds = [int(x) for x in args.folds.split(",")]
        
    # If debug, maybe limit images?
    limit = 0
    if args.debug:
        limit = 100

    if folds is None:
        # Run all
        print("Running All Folds...")
        run_train(cfg, None, limit=limit)
    else:
        for f in folds:
            print(f"Running Fold {f}...")
            run_train(cfg, f, limit=limit)
