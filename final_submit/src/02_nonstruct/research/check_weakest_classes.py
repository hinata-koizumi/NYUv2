"""
check_weakest_classes.py
Check 0.8/0.2 Ensemble class-wise IoU and list the weakest classes.
"""
import os
import numpy as np
import torch
import cv2
from tqdm import tqdm
import sys
sys.path.append("/root/datasets/NYUv2/02_nonstruct"); sys.path.append("/root/datasets/NYUv2")
from configs import default as config

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
    logits_b_cache = {}
    hist_ens = np.zeros((13, 13))
    
    print(f"Checking 0.8/0.2 Ensemble Weakest Classes ({len(common_ids)} images)...")
    
    for img_id in tqdm(common_ids):
        gt_path = os.path.join(GT_LBL_DIR, f"{img_id}.png")
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None: continue
        
        idx_a = id_to_a_idx[img_id]
        la = logits_a_full[idx_a].copy()
        fold_b, idx_b = id_to_b_info[img_id]
        if fold_b not in logits_b_cache:
            path = os.path.join(MODEL_B_DIR, f"oof_fold{fold_b}_logits.npy")
            logits_b_cache[fold_b] = np.load(path, mmap_mode='r')
        lb = logits_b_cache[fold_b][idx_b].copy()
        
        sa = torch.softmax(torch.from_numpy(la).float(), dim=0).numpy()
        sb = torch.softmax(torch.from_numpy(lb).float(), dim=0).numpy()
        
        p_ens = 0.8 * sa + 0.2 * sb
        pred_ens = np.argmax(p_ens, axis=0)
        hist_ens += fast_hist(gt.flatten(), pred_ens.flatten(), 13)

    iou_ens = calculate_iou_from_hist(hist_ens)
    
    # Store as (Class Name, IoU)
    class_ious = []
    for i, name in enumerate(CLASS_NAMES):
        class_ious.append((name, iou_ens[i]))
    
    # Sort by IoU ascending
    class_ious.sort(key=lambda x: x[1])
    
    print("\n" + "="*40)
    print("Ensemble (0.8/0.2) Weakest Classes:")
    print("-" * 40)
    for name, score in class_ious:
        print(f"{name:<15} : {score:.4f}")
    print("="*40)
    
    print(f"\nTop 3 Weakest: {[x[0] for x in class_ious[:3]]}")

if __name__ == "__main__":
    main()
