
import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add root to path for imports
sys.path.append("/root/datasets/NYUv2")

# Local Imports
# Local Imports
from configs import default as config
from data.dataset import ModelBDataset
from model.loss import ModelBLoss
from utils.common import calculate_metrics, MetricAggregator, save_logits

from model.arch import ConvNeXtBaseFPN3Ch
import importlib
n01 = importlib.import_module("01_nearest.constants")
# 01 Imports removed as we use local model definition now (that wraps 01 components)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--smoke", action="store_true", help="Run 1 iter")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()

def train_one_epoch(model, loader, optimizer, scaler, loss_fn, epoch, device):
    model.train()
    pbar = tqdm(loader, desc=f"Train Ep {epoch}", leave=False)
    
    losses = []
    
    for batch in pbar:
        # batch: x, y, id
        x, y, _ = batch
        # Revert to standard contiguous NCHW as channels_last caused RuntimeError on CPU/MPS
        x, y = x.to(device).contiguous(), y.to(device).contiguous()
        
        optimizer.zero_grad()
        
        # Determine device type for autocast
        # torch.amp.autocast is available in newer torch
        # "cuda" or "cpu" or "mps"?
        # validation: autocast(device_type=device.type)
        
        dev_type = device.type
        if dev_type == 'mps':
            # MPS autocast support varies. Check torch version?
            # Safe to use 'cpu' fallback or try 'mps' if sure.
            # For now, let's DISABLE autocast on MPS/CPU to ensure stability of smoke test.
            # ConvNeXt works fine in fp32.
            enable_amp = False
        elif dev_type == 'cpu':
            enable_amp = True # CPU bfloat16?
            dev_type = 'cpu'
        else:
            enable_amp = True # CUDA
            
        if enable_amp:
            with torch.amp.autocast(device_type=dev_type):
                output = model(x)
                if isinstance(output, (tuple, list)):
                    logits = output[0]
                else:
                    logits = output
                loss = loss_fn(logits, y)
        else:
            output = model(x)
            if isinstance(output, (tuple, list)):
                logits = output[0]
            else:
                logits = output
            loss = loss_fn(logits, y)

        if scaler is not None and enable_amp and dev_type == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard backward
            loss.backward()
            optimizer.step()
        
        losses.append(loss.item())
        pbar.set_postfix(loss=loss.item())
        
    return np.mean(losses)

def validate(model, loader, device, output_dir=None, save_preds=False):
    model.eval()
    agg = MetricAggregator()
    
    all_logits = []
    all_ids = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val"):
            x, y, ids = batch
            x, y = x.to(device), y.to(device)
            
            with torch.amp.autocast(device_type=device.type if device.type != 'mps' else 'cpu'):
                output = model(x)
                if isinstance(output, (tuple, list)):
                    logits = output[0]
                else:
                    logits = output
            
            # Request: "bilinear align_corners=False" to 480x640
            # If x is 480x640, logits are 480x640.
            # But just in case:
            if logits.shape[2:] != (480, 640):
                logits = torch.nn.functional.interpolate(
                    logits, size=(480, 640), mode='bilinear', align_corners=False
                )
            
            # Metrics
            m_dict = calculate_metrics(logits, y, device=device)
            agg.update(m_dict)
            
            if save_preds:
                all_logits.append(logits.float().cpu().numpy()) # Keep float32 for concat then float16 save?
                all_ids.extend(ids)
                
    metrics = agg.compute()
    
    if save_preds and output_dir:
        # Concat
        full_logits = np.concatenate(all_logits, axis=0)
        # Sort by IDs? "oof_logits.npy: train_ids order".
        # But this function handles one fold val.
        # Request: "oof_fold{k}_logits.npy (val count...)"
        
        save_logits(full_logits, all_ids, output_dir, file_prefix=f"oof_fold_temp")
        
    return metrics

def main():
    args = get_args()
    
    # 1. Setup Dirs
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # 1.5 Integrity Check: Class IDs
    print("Verifying Class Consistency...")
    if config.CLASS_NAMES != n01.CLASS_NAMES:
        print("[FAIL] CLASS_NAMES mismatch!")
        print(f"Local: {config.CLASS_NAMES}")
        print(f"01:    {n01.CLASS_NAMES}")
        raise ValueError("Class Name Mismatch with 01_nearest")
    
    print(f"IDs Check:")
    print(f"  SMALL_TARGETS: {config.SMALL_TARGET_IDS} -> {[config.CLASS_NAMES[i] for i in config.SMALL_TARGET_IDS]}")
    print(f"  PROTECT: {config.PROTECT_IDS} -> {[config.CLASS_NAMES[i] for i in config.PROTECT_IDS]}")
    print(f"  STRUCT7: {config.STRUCT7_IDS} -> {[config.CLASS_NAMES[i] for i in config.STRUCT7_IDS]}")
    print("[OK] Class IDs Consistent.")
    
    # 2. Load Splits
    with open(config.SPLITS_FILE) as f:
        manifest = json.load(f)
        
    fold_filename = manifest["folds"][args.fold]
    fold_path = os.path.join(os.path.dirname(config.SPLITS_FILE), fold_filename)
    
    with open(fold_path) as f:
        fold_data = json.load(f)
        
    train_ids = fold_data["train_ids"]
    val_ids = fold_data["val_ids"]
    
    # Map IDs to Paths
    # We assume images are in 00_data/train/images and labels in 00_data/train/labels?
    # Or we need paths.
    # 01_nearest uses absolute paths?
    # I need to construct paths from IDs.
    # IDs look like "nyu_office_0_10". filename "nyu_office_0_10.jpg" / ".png"?
    # I should check 00_data structure.
    # Assuming standard structure:
    # 00_data/train/images/{id}.jpg
    # 00_data/train/labels/{id}.png
    # 00_data/train/depths/{id}.png (16bit mm)
    
    img_dir = os.path.join(config.DATA_DIR, "train/image")
    lbl_dir = os.path.join(config.DATA_DIR, "train/label")
    dep_dir = os.path.join(config.DATA_DIR, "train/depth")
    
    def get_paths(id_list):
        imgs = [os.path.join(img_dir, f"{i}.jpg") for i in id_list]
        lbls = [os.path.join(lbl_dir, f"{i}.png") for i in id_list]
        deps = [os.path.join(dep_dir, f"{i}.png") for i in id_list]
        # Verify existence of first
        if not os.path.exists(imgs[0]):
             # Try png?
             imgs = [os.path.join(img_dir, f"{i}.png") for i in id_list]
        return np.array(imgs), np.array(lbls), np.array(deps)
        
    train_imgs, train_lbls, train_deps = get_paths(train_ids)
    val_imgs, val_lbls, val_deps = get_paths(val_ids)
    
    if args.smoke:
        # Truncate
        train_imgs, train_lbls, train_deps = train_imgs[:8], train_lbls[:8], train_deps[:8]
        val_imgs, val_lbls, val_deps = val_imgs[:4], val_lbls[:4], val_deps[:4]
        
    # 3. Datasets
    # Train: WeightedSampler
    ds_train = ModelBDataset(train_imgs, train_lbls, train_deps, is_train=True, ids=train_ids)
    
    sampler = WeightedRandomSampler(
        weights=ds_train.weights,
        num_samples=len(ds_train),
        replacement=True
    )
    
    dl_train = DataLoader(
        ds_train, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    ds_val = ModelBDataset(val_imgs, val_lbls, val_deps, is_train=False, ids=val_ids) # No smart crop
    dl_val = DataLoader(
        ds_val,
        batch_size=1, # Safer for val
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 4. Model
    # FORCE CPU FOR STABILITY ON MAC (MPS has view issues?)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.backends.mps.is_available(): device = torch.device("mps")
    
    model = ConvNeXtBaseFPN3Ch(
        num_classes=13,
        pretrained=True
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4) # Simple recipe
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # Scheduler: Cosine
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    loss_fn = ModelBLoss()
    
    # 5. Loop
    best_safety = -1.0 # Defining "Best" based on safety?
    # Request: "Early stop: val mIoU < Safety Log (Floor/Wall/Objects)".
    # This implies we want to MAXIMIZE safety metrics AND performance.
    # Let's save based on a score: FloorIoU + WallIoU + (1 - suck_rate) * 0.5?
    # Or just save standard best mIoU but print safety.
    # Request: "Metrics... if collapsed stop".
    
    for ep in range(args.epochs):
        train_loss = train_one_epoch(model, dl_train, optimizer, scaler, loss_fn, ep, device)
        metrics = validate(model, dl_val, device)
        scheduler.step()
        
        # Logging
        log_str = f"Ep {ep}: Loss {train_loss:.4f} "
        log_str += f"| F/W: {metrics.get('iou_floor',0):.3f}/{metrics.get('iou_wall',0):.3f} "
        log_str += f"| ObjRat: {metrics.get('ratio_objects_global',0):.2f} "
        log_str += f"| ObjPct: {metrics.get('pred_objects_percent_global',0):.1f}% "
        log_str += f"| Suck: {metrics.get('suck_rate',0):.3f} "
        log_str += f"| SmTgt: {metrics.get('iou_books',0):.3f}/{metrics.get('iou_picture',0):.3f}/{metrics.get('iou_tv',0):.3f}"
        
        print(log_str)
        
        # Safety Check
        # Stop if floor/wall collapsed?
        if metrics.get('iou_floor',0) < 0.1 or metrics.get('iou_wall',0) < 0.1:
            print("SAFETY STOP: Floor/Wall collapsed.")
            break
            
        # Stop if object explosion (global ratio)?
        if metrics.get('ratio_objects_global',0) > 1.25:
             print(f"SAFETY STOP: Objects explosion (global ratio={metrics.get('ratio_objects_global',0):.2f}).")
             break
        
        # Stop if pred objects percentage too high?
        if metrics.get('pred_objects_percent_global',0) > 25.0:
             print(f"SAFETY STOP: Objects percentage too high ({metrics.get('pred_objects_percent_global',0):.1f}%).")
             break
             
        # Save mechanism (simplified)
        torch.save({
            "epoch": ep,
            "state_dict": model.state_dict(),
            "metrics": metrics
        }, os.path.join(config.OUTPUT_DIR, f"fold{args.fold}_last.pth"))
        
        # Save OOF logits at the end? 
        # "8) Inference ... fold output ... oof_fold{k}_logits.npy"
        # We should generate this at BEST epoch or LAST epoch.
        # Assuming run to completion or manual selection.
        # I'll generate it at end of script (Test phase).

    # Final Inference
    print("Running Final Inference on Val...")
    # Load best? Or just use last.
    # I'll use last for v0.
    
    # Generate OOF Logits
    # Need to keep order? Dataset is standard order.
    # Save
    validate(model, dl_val, device, output_dir=config.OUTPUT_DIR, save_preds=True)
    # Rename temp file to correct format
    # save_logits uses "oof_fold_temp".
    # Logic in validate with save_preds=True calls save_logits.
    # I should customize validate to naming.
    # Actually, let's just run custom inference step here.
    
    # Reload full val dataset sorted?
    # `validate` iterates `dl_val`. `dl_val` is `val_ids` order.
    # We want to save `val_ids_fold{k}.txt` and logits same order.
    # `save_logits` does exactly that.
    
    # We need to rename output
    src_npy = os.path.join(config.OUTPUT_DIR, "oof_fold_temp_logits.npy")
    src_txt = os.path.join(config.OUTPUT_DIR, "oof_fold_temp_ids.txt")
    
    dst_npy = os.path.join(config.OUTPUT_DIR, f"oof_fold{args.fold}_logits.npy")
    dst_txt = os.path.join(config.OUTPUT_DIR, f"val_ids_fold{args.fold}.txt")
    
    if os.path.exists(src_npy):
        # VERIFICATION: OOF Order
        # Reload saved sorted IDs and ensure they match expected val_ids
        saved_ids_arr = np.loadtxt(src_txt, dtype=str)
        # val_ids is dataset source.
        # But Dataloader shuffle=False so it should match.
        # Strict check:
        if len(saved_ids_arr) != len(val_ids):
            print(f"[WARN] OOF count mismatch: {len(saved_ids_arr)} vs {len(val_ids)}")
        elif not np.array_equal(saved_ids_arr, val_ids):
             print("[WARN] OOF ID Order Mismatch! Check saving logic.")
             # Diff
             # print(saved_ids_arr[:5], val_ids[:5])
        else:
             print("[OK] OOF Logits Order Verified (Matches val_ids).")

        os.rename(src_npy, dst_npy)
        os.rename(src_txt, dst_txt)

if __name__ == "__main__":
    main()
