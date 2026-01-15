"""
final_analysis.py
1. アンサンブル評価 (0.8 Nearest + 0.2 ModelB)
   - Baseline (Nearest) vs Ensemble のクラス別 IoU とその差分を表示。
2. Fold 3 Floor IoU 健全性チェック
   - 現在の "Mean IoU" に対し、GT Floor=0 の画像を除外した場合の数値を計算。
   - "Mean IoU (All)" vs "Mean IoU (Exclude Empty GT)" vs "Global IoU"
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

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def calculate_iou_from_hist(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    return iou

def main():
    # --- Load IDs ---
    ids_a = np.load(MODEL_A_IDS)
    if ids_a.dtype.kind in ['U', 'S']:
        ids_a = [str(x).split('/')[-1].replace('.png', '').replace('.jpg', '') for x in ids_a]
    id_to_a_idx = {img_id: i for i, img_id in enumerate(ids_a)}
    
    # Load Model B IDs map
    id_to_b_info = {}
    fold_to_ids = {f: [] for f in range(5)}
    
    for f in range(5):
        id_list_path = os.path.join(MODEL_B_DIR, f"val_ids_fold{f}.txt")
        if os.path.exists(id_list_path):
            ids = np.loadtxt(id_list_path, dtype=str)
            fold_to_ids[f] = list(ids)
            for idx, img_id in enumerate(ids):
                id_to_b_info[img_id] = (f, idx)
    
    # Common IDs for Ensemble
    common_ids = sorted(list(set(ids_a) & set(id_to_b_info.keys())))
    
    # --- Load Logits ---
    logits_a_full = np.load(MODEL_A_LOGITS, mmap_mode='r')
    
    # We need to cache B logits
    logits_b_cache = {}
    
    # Storage for Ensemble evaluation (Global Hist accumulators)
    hist_a = np.zeros((13, 13))
    hist_ens = np.zeros((13, 13))
    
    # Storage for Fold 3 Floor Analysis
    # We need image-wise IoUs for Floor
    f3_floor_ious = []
    f3_has_gt_floor = []
    
    print(f"Processing {len(common_ids)} images...")
    
    logit_cache_miss = 0
    
    for img_id in tqdm(common_ids):
        # Load GT
        gt_path = os.path.join(GT_LBL_DIR, f"{img_id}.png")
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None: continue
        gt_f = gt.flatten()
        
        # Get Model A
        idx_a = id_to_a_idx[img_id]
        la = logits_a_full[idx_a].copy() # (13, H, W)
        
        # Get Model B
        fold_b, idx_b = id_to_b_info[img_id]
        if fold_b not in logits_b_cache:
            path = os.path.join(MODEL_B_DIR, f"oof_fold{fold_b}_logits.npy")
            logits_b_cache[fold_b] = np.load(path, mmap_mode='r')
        
        lb = logits_b_cache[fold_b][idx_b].copy()
        
        # Softmax mixing
        sa = torch.softmax(torch.from_numpy(la).float(), dim=0).numpy()
        sb = torch.softmax(torch.from_numpy(lb).float(), dim=0).numpy()
        
        # --- 1. Ensemble Evaluation (0.8 A + 0.2 B) ---
        p_ens = 0.8 * sa + 0.2 * sb
        
        pred_a = np.argmax(sa, axis=0)
        pred_ens = np.argmax(p_ens, axis=0)
        
        hist_a += fast_hist(gt_f, pred_a.flatten(), 13)
        hist_ens += fast_hist(gt_f, pred_ens.flatten(), 13)
        
        # --- 2. Fold 3 Floor Check ---
        if fold_b == 3:
            # Calculate Floor IoU for this image (using Model B raw output, as that was the concern)
            # Or should we check the ensemble? The user asked about "Fold 3 Floor evaluation", implying the Model B degradation.
            # I will check Model B's pure performance for this diagnosis.
            pred_b = np.argmax(lb, axis=0)
            
            # Floor = class 4
            inter = np.logical_and(pred_b == 4, gt == 4).sum()
            union = np.logical_or(pred_b == 4, gt == 4).sum()
            gt_floor_px = (gt == 4).sum()
            
            if union == 0:
                iou = 1.0 # Perfect match (both empty)
            else:
                iou = inter / union
                
            f3_floor_ious.append(iou)
            f3_has_gt_floor.append(gt_floor_px > 0)

    # --- Print Ensemble Results ---
    iou_a = calculate_iou_from_hist(hist_a)
    iou_ens = calculate_iou_from_hist(hist_ens)
    miou_a = np.nanmean(iou_a)
    miou_ens = np.nanmean(iou_ens)
    
    print("\n" + "="*60)
    print("【Ensemble Evaluation (0.8 Nearest + 0.2 ModelB)】")
    print(f"{'Class':<12} | {'Nearest':<10} | {'Ensemble':<10} | {'Delta':<10}")
    print("-" * 60)
    for i, name in enumerate(CLASS_NAMES):
        diff = iou_ens[i] - iou_a[i]
        mark = " (+)" if diff > 0 else " (-)"
        print(f"{name:<12} | {iou_a[i]:<10.4f} | {iou_ens[i]:<10.4f} | {diff:<+.4f}{mark}")
    print("-" * 60)
    print(f"{'mIoU':<12} | {miou_a:<10.4f} | {miou_ens:<10.4f} | {miou_ens - miou_a:<+.4f}")
    print("="*60)

    # --- Print Fold 3 Floor Analysis ---
    if len(f3_floor_ious) > 0:
        mean_floor_all = np.mean(f3_floor_ious)
        
        # Exclude empty GT
        valid_indices = [i for i, has_gt in enumerate(f3_has_gt_floor) if has_gt]
        mean_floor_excl = np.mean([f3_floor_ious[i] for i in valid_indices])
        
        print("\n" + "="*60)
        print("【Fold 3 Floor IoU Analysis (Model B)】")
        print(f"Total Images in Fold 3: {len(f3_floor_ious)}")
        print(f"Images with GT Floor > 0: {len(valid_indices)}")
        print(f"Images with GT Floor = 0: {len(f3_floor_ious) - len(valid_indices)}")
        print("-" * 60)
        print(f"Mean Floor IoU (All):              {mean_floor_all:.4f}")
        print(f"Mean Floor IoU (Exclude Empty GT): {mean_floor_excl:.4f}")
        print("="*60)
        
        if mean_floor_excl > mean_floor_all + 0.05:
            print("✅ Metric Check Passed: Low Floor IoU is largely due to empty GT penalty.")
        else:
            print("⚠️ Metric Check Warning: Degradation persists even after excluding empty GT.")

if __name__ == "__main__":
    main()
