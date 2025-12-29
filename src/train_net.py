import os
import sys
sys.path.append(os.getcwd())
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import numpy as np
import time

from configs.base_config import Config
from src.data.dataset import NYUDataset
from src.data.transforms import get_train_transforms, get_valid_transforms, get_color_transforms
from src.data.adapters import get_adapter
from src.model.meta_arch import SegFPN
from src.utils.metrics import CombinedSegLoss
from src.engine.trainer import train_one_epoch, validate
from src.utils.misc import seed_everything, worker_init_fn, ModelEMA, CheckpointManager, EarlyStopping, Logger, save_config

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--fold", type=int, default=None, help="Run specific fold (0-4)")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config()
    
    if args.exp_name:
        cfg.EXP_NAME = args.exp_name
        
    seed_everything(cfg.SEED)
    
    output_dir = os.path.join("data", "outputs", cfg.EXP_NAME)
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Data Preparation ---
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
    
    # Save keys for reproducibility
    with open(os.path.join(output_dir, "train_keys.txt"), "w") as f:
        f.write("\n".join(keys))

    kf = KFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    splits = list(kf.split(X_img))

    adapter = get_adapter(cfg)

    # --- Training Loop ---
    folds_to_run = range(cfg.N_FOLDS)
    if args.fold is not None:
        folds_to_run = [args.fold]
        
    for fold_idx in folds_to_run:
        print(f"\n{'='*20} Fold {fold_idx} {'='*20}")
        fold_dir = os.path.join(output_dir, f"fold{fold_idx}")
        save_config(cfg, fold_dir)
        # Logger
        logger = Logger(fold_dir)
        
        tr_idx, va_idx = splits[fold_idx]
        
        # Datasets
        train_ds = NYUDataset(
            image_paths=X_img[tr_idx],
            label_paths=X_lbl[tr_idx],
            depth_paths=X_dep[tr_idx],
            cfg=cfg,
            transform=get_train_transforms(cfg),
            color_transform=get_color_transforms(cfg),
            enable_smart_crop=True,
            adapter=adapter,
        )
        valid_ds = NYUDataset(
            image_paths=X_img[va_idx],
            label_paths=X_lbl[va_idx],
            depth_paths=X_dep[va_idx],
            cfg=cfg,
            transform=get_valid_transforms(cfg),
            color_transform=None,
            enable_smart_crop=False,
            adapter=adapter,
        )
        
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            persistent_workers=(cfg.NUM_WORKERS > 0),
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            persistent_workers=(cfg.NUM_WORKERS > 0),
        )
        
        # Model
        model = SegFPN(num_classes=cfg.NUM_CLASSES, in_channels=cfg.IN_CHANNELS).to(cfg.DEVICE)
        ema = ModelEMA(model, cfg.EMA_DECAY)
        
        criterion = CombinedSegLoss(cfg).to(cfg.DEVICE)
        
        # Optimizer with optional LR Boost
        if hasattr(model, 'get_lr_params') and getattr(cfg, "STEM_LR_MULT", 1.0) != 1.0:
            print(f"Using Stem LR Boost: x{cfg.STEM_LR_MULT}")
            groups = model.get_lr_params(cfg.LEARNING_RATE, multiplier=cfg.STEM_LR_MULT)
            optimizer = optim.AdamW(groups, weight_decay=cfg.WEIGHT_DECAY)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=cfg.ETA_MIN)
        
        ckpt = CheckpointManager(fold_dir, top_k=cfg.SAVE_TOP_K)
        ckpt = CheckpointManager(fold_dir, top_k=cfg.SAVE_TOP_K)
        
        # Early Stopping State
        best_miou = -1.0
        best_epoch = -1
        no_improve = 0
        min_epochs = getattr(cfg, "MIN_EPOCHS", 20)
        patience = getattr(cfg, "EARLY_STOPPING_PATIENCE", 10)
        min_delta = getattr(cfg, "EARLY_STOPPING_MIN_DELTA", 1e-4)        
        # Scaler for AMP
        scaler = torch.cuda.amp.GradScaler(enabled=getattr(cfg, "USE_AMP", False))

        # Epoch Loop
        for epoch_idx in range(cfg.EPOCHS):
             epoch = epoch_idx + 1
             # Time and Memory Tracking
             t0 = time.time()
             torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
             
             tr_loss = train_one_epoch(
                model, ema, train_loader, criterion, optimizer, cfg.DEVICE,
                scaler=scaler,
                use_amp=getattr(cfg, "USE_AMP", False),
                grad_accum_steps=getattr(cfg, "GRAD_ACCUM_STEPS", 1)
             )
             
             t1 = time.time()
             dt = t1 - t0
             num_samples = len(train_loader.dataset)
             throughput = num_samples / dt if dt > 0 else 0
             
             mem_max = 0
             if torch.cuda.is_available():
                 mem_max = torch.cuda.max_memory_allocated() / 1e9 # GB
             
             scheduler.step()
             
             # Validation
             va_loss, miou, acc, class_iou = validate(ema.ema, valid_loader, criterion, cfg.DEVICE, cfg)
             
             # Visualization Artifacts (Epoch 1, Best, Last)
             if epoch == 1 or miou > best_miou or epoch == cfg.EPOCHS:
                 # Visualize first sample of validation set
                 # Create a simple visualizer or just get item 0
                 # We need inputs and preds.
                 # Re-instantiate dataset or just grab from loader?
                 # Loader shuffles=False, so first batch is stable.
                 try:
                     vis_batch = next(iter(valid_loader))
                     vx, vy, vmeta = vis_batch
                     vx = vx.to(cfg.DEVICE)
                     # Predict
                     with torch.no_grad():
                         vlogits = ema.ema(vx)
                         vpred = vlogits.argmax(dim=1).cpu().numpy()[0] # First item
                     
                     # Get Data
                     vimg = vx[0, :3].cpu().numpy().transpose(1, 2, 0)
                     # Denorm
                     mean = np.array(cfg.MEAN)
                     std = np.array(cfg.STD)
                     vimg = (vimg * std + mean) * 255.0
                     vimg = np.clip(vimg, 0, 255).astype(np.uint8)
                     
                     vgt = vy[0].numpy().astype(np.uint8)
                     
                     # Depth (Inv)
                     vinv = vx[0, 3].cpu().numpy()
                     vinv_vis = (vinv * 255).astype(np.uint8)
                     import cv2
                     vinv_vis = cv2.applyColorMap(vinv_vis, cv2.COLORMAP_JET)
                     
                     # Pred/GT Color
                     # Simple grayscale or colormap? Labels are 0-12
                     # Scale to 0-255 roughly (x20)
                     vpred_vis = (vpred * 20).astype(np.uint8)
                     vgt_vis = (vgt * 20).astype(np.uint8)
                     vpred_vis = cv2.applyColorMap(vpred_vis, cv2.COLORMAP_TURBO)
                     vgt_vis = cv2.applyColorMap(vgt_vis, cv2.COLORMAP_TURBO)
                     
                     # Concat: RGB | Depth | GT | Pred
                     # All should be same H,W
                     # Depth is H,W,3 (after colormap)
                     
                     row = np.hstack([vimg, vinv_vis, vgt_vis, vpred_vis])
                     
                     suffix = "epoch1" if epoch == 1 else ("last" if epoch == cfg.EPOCHS else "best")
                     if miou > best_miou: suffix = "best" # Priority
                     
                     logger.save_image(f"vis_{suffix}.png", row)
                 except Exception as e:
                     print(f"Vis failed: {e}")
             
             print(f"Epoch {epoch}: TrLoss={tr_loss:.4f}, VaLoss={va_loss:.4f}, mIoU={miou:.4f}, Speed={throughput:.1f} img/s, Mem={mem_max:.1f}GB")
             
             # Log Epoch
             # Get LRs
             lrs = [g['lr'] for g in optimizer.param_groups]
             lr_stem = lrs[0] if len(lrs) > 1 else lrs[0]
             lr_base = lrs[1] if len(lrs) > 1 else lrs[0]
             
             logger.log_epoch(epoch, {
                 "lr": lr_base, # backward compat
                 "lr_base": lr_base,
                 "lr_stem": lr_stem,
                 "train_loss": tr_loss,
                 "valid_loss": va_loss,
                 "valid_miou": miou,
                 "valid_pixel_acc": acc,
                 "class_iou": class_iou,
                 "throughput": throughput,
                 "gpu_mem": mem_max
             })
             
             ckpt.save(ema.ema, epoch, miou)

             improved = (miou > best_miou + min_delta)
             if improved:
                 best_miou = miou
                 best_epoch = epoch
                 no_improve = 0
                 ckpt._atomic_torch_save(ema.ema.state_dict(), os.path.join(fold_dir, "model_best.pth"))
                 # Update Summary with Best
                 logger.save_summary({
                     "best_miou": best_miou,
                     "best_epoch": best_epoch,
                     "config": cfg.to_dict()
                 })
             else:
                 no_improve += 1
                 
             # ---- Early stopping gate ----
             if epoch >= min_epochs and no_improve >= patience:
                 print(f"[EARLY STOP] epoch={epoch} best={best_miou:.4f} best_epoch={best_epoch} no_improve={no_improve}/{patience}")
                 break
        
        logger.close()

if __name__ == "__main__":
    main()
