"""
visualize_ensemble_gains.py
- Model A で間違い、Ensemble (0.2) で正解したピクセルを可視化する。
- 改善領域が「小物 (Books/TV/Picture)」なのか「境界 (Boundary)」なのかを定性評価する。
"""

import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append("/root/datasets/NYUv2/02_nonstruct"); sys.path.append("/root/datasets/NYUv2")
from configs import default as config

# Paths
MODEL_A_DIR = "/root/datasets/NYUv2/00_data/output/01_nearest_v1.0_frozen/golden_artifacts"
MODEL_A_LOGITS = os.path.join(MODEL_A_DIR, "oof_logits.npy")
MODEL_A_IDS = os.path.join(MODEL_A_DIR, "oof_file_ids.npy")
MODEL_B_DIR = "/root/datasets/NYUv2/02_nonstruct/output"
GT_LBL_DIR = "/root/datasets/NYUv2/00_data/train/label"
RGB_DIR = "/root/datasets/NYUv2/00_data/train/image"
OUTPUT_DIR = "/root/datasets/NYUv2/02_nonstruct/output/ensemble_visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Weight
WEIGHT_B = 0.2

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
    
    # We want to find images with high improvement count
    best_improvement_count = -1
    best_img_id = None
    
    print("Scanning for best improvement examples...")
    
    # Check first 200 images to save time, or scan all? 
    # Let's scan a subset or until we find good examples.
    visualized_count = 0
    
    for img_id in tqdm(common_ids):
        if visualized_count >= 5: break # Save top 5
        
        gt_path = os.path.join(GT_LBL_DIR, f"{img_id}.png")
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None: continue
        
        idx_a = id_to_a_idx[img_id]
        la = logits_a_full[idx_a].copy()
        
        fold_b, idx_b = id_to_b_info[img_id]
        if fold_b not in logits_b_cache:
            if len(logits_b_cache) > 1: logits_b_cache.clear()
            b_path = os.path.join(MODEL_B_DIR, f"oof_fold{fold_b}_logits.npy")
            logits_b_cache[fold_b] = np.load(b_path, mmap_mode='r')
        lb = logits_b_cache[fold_b][idx_b].copy()
        
        # Softmax
        sa = torch.softmax(torch.from_numpy(la).float(), dim=0).numpy()
        sb = torch.softmax(torch.from_numpy(lb).float(), dim=0).numpy()
        
        pred_a = np.argmax(sa, axis=0)
        
        p_ens = (1.0 - WEIGHT_B) * sa + WEIGHT_B * sb
        pred_ens = np.argmax(p_ens, axis=0)
        
        # Mask: A was WRONG and Ens was CORRECT
        improved_mask = (pred_a != gt) & (pred_ens == gt) & (gt != 255)
        degraded_mask = (pred_a == gt) & (pred_ens != gt) & (gt != 255)
        
        imp_count = np.count_nonzero(improved_mask)
        deg_count = np.count_nonzero(degraded_mask)
        
        if imp_count > 1000 and imp_count > deg_count * 1.5: # Significant improvement
            # Visualize
            rgb_path = os.path.join(RGB_DIR, f"{img_id}.png") # Try png
            if not os.path.exists(rgb_path): rgb_path = os.path.join(RGB_DIR, f"{img_id}.jpg")
            
            rgb = cv2.imread(rgb_path)
            if rgb is not None:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (640, 480))
            else:
                rgb = np.zeros((480, 640, 3), dtype=np.uint8)

            # Create overlay
            overlay = rgb.copy()
            # Green for Improved
            overlay[improved_mask] = [0, 255, 0]
            # Red for Degraded
            overlay[degraded_mask] = [255, 0, 0]
            
            # Blend
            vis = cv2.addWeighted(rgb, 0.7, overlay, 0.3, 0)
            
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(rgb)
            plt.title(f"RGB {img_id}")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(vis)
            plt.title(f"Imp (Green) vs Deg (Red)\nGain: +{imp_count} px")
            plt.axis('off')
            
            save_path = os.path.join(OUTPUT_DIR, f"gain_{img_id}.png")
            plt.savefig(save_path)
            plt.close()
            
            print(f"Saved gain visualization: {save_path} (Imp: {imp_count}, Deg: {deg_count})")
            visualized_count += 1

if __name__ == "__main__":
    main()
