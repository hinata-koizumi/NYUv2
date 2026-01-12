import os
import json
import numpy as np
import glob
import cv2
import sys

sys.path.append(os.getcwd())
from ..utils.metrics import compute_metrics, update_confusion_matrix

def run_merge_golden():
    golden_root = "01_nearest/golden_artifacts"
    folds_dir = os.path.join(golden_root, "folds")
    oof_dir = os.path.join(golden_root, "oof")
    os.makedirs(oof_dir, exist_ok=True)
    
    # 1. Collect all fold parts
    fold_dirs = sorted(glob.glob(os.path.join(folds_dir, "fold*")))
    if not fold_dirs:
        print("No folds found!")
        return
        
    all_logits = []
    all_ids = []
    
    print(f"Found {len(fold_dirs)} folds: {[os.path.basename(f) for f in fold_dirs]}")
    
    for d in fold_dirs:
        lpath = os.path.join(d, "val_logits.npy")
        ipath = os.path.join(d, "val_file_ids.npy")
        
        if not os.path.exists(lpath):
            print(f"Skipping {d} (no logits)")
            continue
            
        print(f"Loading {d}...")
        logits = np.load(lpath)
        ids = np.load(ipath)
        all_logits.append(logits)
        all_ids.append(ids)
        
    if not all_logits:
        print("No data collected.")
        return

    # Concatenate
    oof_logits = np.concatenate(all_logits, axis=0) # (N, C, H, W)
    oof_ids = np.concatenate(all_ids, axis=0)
    
    # Sort by ID to ensure deterministic order
    sort_idx = np.argsort(oof_ids)
    oof_ids = oof_ids[sort_idx]
    oof_logits = oof_logits[sort_idx]
    
    # Save OOF
    print(f"Saving merged OOF ({oof_logits.shape}) to {oof_dir}")
    
    # Use memmap to write large file (>2GB) to avoid macOS 2GB write limit
    out_path = os.path.join(oof_dir, "oof_logits.npy")
    mout = np.lib.format.open_memmap(out_path, mode='w+', dtype=oof_logits.dtype, shape=oof_logits.shape)
    mout[:] = oof_logits[:]
    mout.flush()
    del mout
    # np.save(os.path.join(oof_dir, "oof_logits.npy"), oof_logits)
    np.save(os.path.join(oof_dir, "oof_file_ids.npy"), oof_ids)
    
    # 2. Compute Metrics (Validation against legacy)
    # label_root = "/root/datasets/NYUv2/data/train/label"
    label_root = "data/train/label"
    C = 13
    IGNORE_INDEX = 255
    cm = np.zeros((C, C), dtype=np.int64)
    
    print("Computing metrics...")
    for i in range(len(oof_ids)):
        fid = oof_ids[i]
        gt_path = os.path.join(label_root, f"{fid}.png")
        if not os.path.exists(gt_path):
             print(f"Warning: GT not found for {fid}")
             continue
             
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if gt.ndim == 3: gt = gt[:,:,0]
        
        # Pred
        logits = oof_logits[i] # (C, H, W)
        pred = np.argmax(logits, axis=0).astype(np.uint8)
        
        if gt.shape != pred.shape:
             gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
             
        update_confusion_matrix(pred, gt, C, IGNORE_INDEX, cm)
        
    pixel_acc, miou, ious = compute_metrics(cm)
    print(f"Global OOF mIoU: {miou:.4f}")
    
    metrics = {
        "mIoU": float(miou),
        "pixel_acc": float(pixel_acc),
        "per_class_iou": {str(i): float(ious[i]) for i in range(C)},
        "confusion_matrix": cm.tolist()
    }
    
    with open(os.path.join(oof_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Expected Check
    expected = 0.7077
    if abs(miou - expected) < 0.001:
        print("✅ Metric matches legacy 0.7077")
    else:
        print(f"⚠️ Metric divergence from {expected} (diff: {miou - expected})")

def main():
    run_merge_golden()

if __name__ == "__main__":
    main()
