"""
evaluate_ensemble.py
- Model A (nearest) と Model B (nonstruct) の OOF Logits を 0.5:0.5 で混合して評価。
- mIoU(nearest) と mIoU(mix) を算出し、ΔmIoU を報告。
"""

import os
import numpy as np
import torch
import cv2
from tqdm import tqdm
import json

import sys
sys.path.append("/root/datasets/NYUv2")
import config

# Paths for Model A (nearest - FROZEN)
MODEL_A_DIR = "/root/datasets/NYUv2/00_data/output/01_nearest_v1.0_frozen/golden_artifacts"
MODEL_A_LOGITS = os.path.join(MODEL_A_DIR, "oof_logits.npy")
MODEL_A_IDS = os.path.join(MODEL_A_DIR, "oof_file_ids.npy")

# Paths for Model B (02_nonstruct)
MODEL_B_DIR = "/root/datasets/NYUv2/02_nonstruct/output"

# GT Label dir
GT_LBL_DIR = "/root/datasets/NYUv2/00_data/train/label"

CLASS_NAMES = [
    "bed", "books", "ceiling", "chair", "floor", "furniture", 
    "objects", "picture", "sofa", "table", "tv", "wall", "window"
]

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def calculate_iou(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    return iou

def main():
    print("Loading Model A (nearest) IDs...")
    ids_a = np.load(MODEL_A_IDS) # Expected to be image names like '000002'
    if ids_a.dtype.kind in ['U', 'S']:
        ids_a = [str(x).split('/')[-1].replace('.png', '').replace('.jpg', '') for x in ids_a]
    
    print(f"Model A has {len(ids_a)} images in OOF.")
    
    # Load Model B IDs and map them to indices in Model B OOF files
    id_to_b_info = {}
    for f in range(5):
        id_list_path = os.path.join(MODEL_B_DIR, f"val_ids_fold{f}.txt")
        if not os.path.exists(id_list_path):
            print(f"Warning: Fold {f} ID list not found.")
            continue
        
        ids_b = np.loadtxt(id_list_path, dtype=str)
        for idx, img_id in enumerate(ids_b):
            id_to_b_info[img_id] = (f, idx)
    
    print(f"Model B has {len(id_to_b_info)} images across 5 folds.")
    
    # Intersection of IDs
    common_ids = sorted(list(set(ids_a) & set(id_to_b_info.keys())))
    print(f"Common images to evaluate: {len(common_ids)}")
    
    if len(common_ids) == 0:
        print("Error: No common IDs found between Model A and Model B OOF.")
        return

    # Load Model A Logits (Memory map to save RAM)
    logits_a_full = np.load(MODEL_A_LOGITS, mmap_mode='r')
    id_to_a_idx = {img_id: i for i, img_id in enumerate(ids_a)}
    
    # Pre-cache Model B Logits (Open one by one to save RAM)
    logits_b_cache = {}
    
    hist_a = np.zeros((13, 13))
    hist_mix = np.zeros((13, 13))
    
    print("Evaluating...")
    for i, img_id in enumerate(tqdm(common_ids)):
        # 1. Load GT
        gt_path = os.path.join(GT_LBL_DIR, f"{img_id}.png")
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        
        # 2. Get Model A Logits
        idx_a = id_to_a_idx[img_id]
        la = logits_a_full[idx_a].copy() # (13, 480, 640)
        
        # 3. Get Model B Logits
        fold_b, idx_b = id_to_b_info[img_id]
        if fold_b not in logits_b_cache:
            # Clear cache if too many folds loaded
            if len(logits_b_cache) > 1: logits_b_cache.clear()
            b_path = os.path.join(MODEL_B_DIR, f"oof_fold{fold_b}_logits.npy")
            logits_b_cache[fold_b] = np.load(b_path, mmap_mode='r')
            
        lb = logits_b_cache[fold_b][idx_b].copy() # (13, 480, 640)
        
        # Normalize/Softmax mixing
        # We'll use torch for easier softmax
        ta = torch.from_numpy(la).float()
        tb = torch.from_numpy(lb).float()
        
        sa = torch.softmax(ta, dim=0).numpy()
        sb = torch.softmax(tb, dim=0).numpy()
        
        # Nearest (Model A) Predictions
        pred_a = np.argmax(sa, axis=0)
        hist_a += fast_hist(gt.flatten(), pred_a.flatten(), 13)
        
        # Mixed Predictions (Average of probabilities)
        p_mix = 0.5 * sa + 0.5 * sb
        pred_mix = np.argmax(p_mix, axis=0)
        hist_mix += fast_hist(gt.flatten(), pred_mix.flatten(), 13)
        
        # Optional: Check distribution for the first few images
        if i == 0:
            print(f"\nImage {img_id} Pred Distribution:")
            for c in range(13):
                cnt_a = (pred_a == c).sum()
                cnt_b = (np.argmax(sb, axis=0) == c).sum()
                cnt_gt = (gt == c).sum()
                if cnt_a > 0 or cnt_b > 0 or cnt_gt > 0:
                    print(f"  {CLASS_NAMES[c]:<10}: A={cnt_a:>6}, B={cnt_b:>6}, GT={cnt_gt:>6}")
        
    # Results
    iou_a = calculate_iou(hist_a)
    iou_mix = calculate_iou(hist_mix)
    
    miou_a = np.nanmean(iou_a)
    miou_mix = np.nanmean(iou_mix)
    delta = miou_mix - miou_a
    
    print("\n" + "="*60)
    print(f"{'Class':<12} | {'Model A':<10} | {'Mix (0.5)':<10} | {'Delta':<10}")
    print("-" * 60)
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<12} | {iou_a[i]:<10.4f} | {iou_mix[i]:<10.4f} | {iou_mix[i]-iou_a[i]:<+10.4f}")
    print("-" * 60)
    print(f"{'mIoU':<12} | {miou_a:<10.4f} | {miou_mix:<10.4f} | {delta:<+10.4f}")
    print("="*60)
    
    if delta >= 0.003:
        print(f"\n✅ Delta mIoU = {delta:+.4f} (>= +0.003)")
        print("Model B is a confirmed partner for Model A.")
    else:
        print(f"\n⚠️ Delta mIoU = {delta:+.4f} (< +0.003)")
        print("Improvement is below target. Check for high correlation or weak B performance.")

if __name__ == "__main__":
    main()
