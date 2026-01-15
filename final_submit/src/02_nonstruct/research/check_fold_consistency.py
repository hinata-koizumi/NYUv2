"""
check_fold_consistency.py
- Fold 0~4 それぞれで Delta mIoU を計算し、改善の安定性を確認する。
- 特定のクラス (TV, Objects, Picture) が全 Fold で伸びているかチェック。
"""

import os
import numpy as np
import torch
import cv2
from tqdm import tqdm
import sys
sys.path.append("/root/datasets/NYUv2/02_nonstruct"); sys.path.append("/root/datasets/NYUv2")
from configs import default as config

# Paths
MODEL_A_DIR = "/root/datasets/NYUv2/00_data/output/01_nearest_v1.0_frozen/golden_artifacts"
MODEL_A_LOGITS = os.path.join(MODEL_A_DIR, "oof_logits.npy")
MODEL_A_IDS = os.path.join(MODEL_A_DIR, "oof_file_ids.npy")
MODEL_B_DIR = "/root/datasets/NYUv2/02_nonstruct/output"
GT_LBL_DIR = "/root/datasets/NYUv2/00_data/train/label"

CLASS_NAMES = config.CLASS_NAMES
TARGET_CLASSES = ["tv", "objects", "picture", "bed", "window"] # Check targets + some others

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def calculate_iou_per_class(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    return iou

def main():
    # Load Model A Map
    ids_a = np.load(MODEL_A_IDS)
    if ids_a.dtype.kind in ['U', 'S']:
        ids_a = [str(x).split('/')[-1].replace('.png', '').replace('.jpg', '') for x in ids_a]
    id_to_a_idx = {img_id: i for i, img_id in enumerate(ids_a)}
    logits_a_full = np.load(MODEL_A_LOGITS, mmap_mode='r')
    
    print(f"{'Fold':<5} | {'Base mIoU':<10} | {'Ens mIoU':<10} | {'Delta':<10} | {'TV Δ':<8} | {'Obj Δ':<8} | {'Pic Δ':<8}")
    print("-" * 80)
    
    folds_stats = []
    
    for fold in range(5):
        # Load Fold IDs
        id_list_path = os.path.join(MODEL_B_DIR, f"val_ids_fold{fold}.txt")
        if not os.path.exists(id_list_path):
            print(f"Fold {fold}: ID file not found.")
            continue
            
        ids_b = np.loadtxt(id_list_path, dtype=str)
        
        # Load Fold Logits
        b_path = os.path.join(MODEL_B_DIR, f"oof_fold{fold}_logits.npy")
        logits_b = np.load(b_path, mmap_mode='r')
        
        hist_a = np.zeros((13, 13))
        hist_ens = np.zeros((13, 13))
        
        valid_count = 0
        
        for idx_b, img_id in enumerate(ids_b):
            if img_id not in id_to_a_idx: continue
            
            gt_path = os.path.join(GT_LBL_DIR, f"{img_id}.png")
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt is None: continue
            
            idx_a = id_to_a_idx[img_id]
            la = logits_a_full[idx_a].copy()
            lb = logits_b[idx_b].copy()
            
            sa = torch.softmax(torch.from_numpy(la).float(), dim=0).numpy()
            sb = torch.softmax(torch.from_numpy(lb).float(), dim=0).numpy()
            
            # 0.8 / 0.2 Ensemble
            p_ens = 0.8 * sa + 0.2 * sb
            
            pred_a = np.argmax(sa, axis=0)
            pred_ens = np.argmax(p_ens, axis=0)
            
            hist_a += fast_hist(gt.flatten(), pred_a.flatten(), 13)
            hist_ens += fast_hist(gt.flatten(), pred_ens.flatten(), 13)
            valid_count += 1
            
        iou_a = calculate_iou_per_class(hist_a)
        iou_ens = calculate_iou_per_class(hist_ens)
        miou_a = np.nanmean(iou_a)
        miou_ens = np.nanmean(iou_ens)
        delta = miou_ens - miou_a
        
        # Class Deltas
        d_tv = iou_ens[10] - iou_a[10]
        d_obj = iou_ens[6] - iou_a[6]
        d_pic = iou_ens[7] - iou_a[7]
        
        print(f"{fold:<5} | {miou_a:<10.4f} | {miou_ens:<10.4f} | {delta:<+10.4f} | {d_tv:<+8.4f} | {d_obj:<+8.4f} | {d_pic:<+8.4f}")
        
        folds_stats.append({
            "fold": fold,
            "delta": delta,
            "d_tv": d_tv,
            "d_obj": d_obj,
            "d_pic": d_pic
        })
        
    print("-" * 80)
    
    # Check Consistency
    all_positive = all(f["delta"] > 0 for f in folds_stats)
    tv_positive = all(f["d_tv"] > -0.005 for f in folds_stats) # Allow slight noise, but generally positive
    obj_positive = all(f["d_obj"] > 0 for f in folds_stats)
    
    if all_positive:
        print("✅ Consistency Check Passed: All folds show positive Delta mIoU.")
    else:
        print("⚠️ Consistency Check Warning: Some folds generally negative.")

if __name__ == "__main__":
    main()
