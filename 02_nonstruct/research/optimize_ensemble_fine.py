"""
optimize_ensemble_fine.py
- Model A と Model B の混合重みを細かくスキャン (0.10 ~ 0.30)。
- 最適な比率を特定する。
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

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def calculate_miou(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    return np.nanmean(iou)

def main():
    ids_a = np.load(MODEL_A_IDS)
    if ids_a.dtype.kind in ['U', 'S']:
        ids_a = [str(x).split('/')[-1].replace('.png', '').replace('.jpg', '') for x in ids_a]
    id_to_a_idx = {img_id: i for i, img_id in enumerate(ids_a)}
    
    id_to_b_info = {}
    for f in range(5):
        id_list_path = os.path.join(MODEL_B_DIR, f"val_ids_fold{f}.txt")
        if os.path.exists(id_list_path):
            ids = np.loadtxt(id_list_path, dtype=str)
            for idx, img_id in enumerate(ids):
                id_to_b_info[img_id] = (f, idx)
    
    common_ids = sorted(list(set(ids_a) & set(id_to_b_info.keys())))
    logits_a_full = np.load(MODEL_A_LOGITS, mmap_mode='r')
    
    weights = [0.10, 0.15, 0.20, 0.25, 0.30]
    hists = {w: np.zeros((13, 13)) for w in weights}
    
    logits_b_cache = {}
    
    print(f"Scanning weights {weights} on {len(common_ids)} images...")
    for img_id in tqdm(common_ids):
        gt_path = os.path.join(GT_LBL_DIR, f"{img_id}.png")
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None: continue
        gt_f = gt.flatten()
        
        idx_a = id_to_a_idx[img_id]
        la = logits_a_full[idx_a].copy()
        
        fold_b, idx_b = id_to_b_info[img_id]
        if fold_b not in logits_b_cache:
            if len(logits_b_cache) > 1: logits_b_cache.clear()
            b_path = os.path.join(MODEL_B_DIR, f"oof_fold{fold_b}_logits.npy")
            logits_b_cache[fold_b] = np.load(b_path, mmap_mode='r')
        lb = logits_b_cache[fold_b][idx_b].copy()
        
        sa = torch.softmax(torch.from_numpy(la).float(), dim=0).numpy()
        sb = torch.softmax(torch.from_numpy(lb).float(), dim=0).numpy()
        
        for w in weights:
            p_mix = (1.0 - w) * sa + w * sb
            pred = np.argmax(p_mix, axis=0)
            hists[w] += fast_hist(gt_f, pred.flatten(), 13)
            
    print("\nFine-grained Weight Scan:")
    print(f"{'Weight B':<10} | {'mIoU':<10} | {'Delta':<10}")
    print("-" * 30)
    
    # Need baseline for delta
    # Since we didn't calculate w=0 here, use known baseline 0.7124 (from prev runs)
    baseline_miou = 0.7124
    
    best_w = -1
    best_miou = -1
    
    for w in weights:
        miou = calculate_miou(hists[w])
        print(f"{w:<10.2f} | {miou:<10.4f} | {miou - baseline_miou:<+10.4f}")
        if miou > best_miou:
            best_miou = miou
            best_w = w
            
    print("-" * 30)
    print(f"Best Weight: {best_w} (mIoU: {best_miou:.4f})")

if __name__ == "__main__":
    main()
