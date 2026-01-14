"""
visualize_fold3_bad.py
Fold 3 の Floor IoU = 0 画像を可視化して、何が起きているか確認する。
"""

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# Paths
GT_DIR = "/root/datasets/NYUv2/00_data/train/label"
RGB_DIR = "/root/datasets/NYUv2/00_data/train/image"
OUTPUT_DIR = "/root/datasets/NYUv2/02_nonstruct/output"
DIAG_DIR = os.path.join(OUTPUT_DIR, "diagnosis_fold3")
os.makedirs(DIAG_DIR, exist_ok=True)

# IDs to visualize
BAD_IDS = ["000101", "000102", "000119"]

def main():
    # Load Fold 3 Logits
    logits_path = os.path.join(OUTPUT_DIR, "oof_fold3_logits.npy")
    ids_path = os.path.join(OUTPUT_DIR, "val_ids_fold3.txt")
    
    val_ids = np.loadtxt(ids_path, dtype=str)
    logits = np.load(logits_path, mmap_mode='r')
    
    id_to_idx = {img_id: i for i, img_id in enumerate(val_ids)}
    
    for img_id in BAD_IDS:
        if img_id not in id_to_idx: continue
        
        idx = id_to_idx[img_id]
        l = logits[idx]
        pred = np.argmax(l, axis=0)
        
        # Load RGB
        rgb_path_j = os.path.join(RGB_DIR, f"{img_id}.jpg")
        rgb_path_p = os.path.join(RGB_DIR, f"{img_id}.png")
        rgb_path = rgb_path_j if os.path.exists(rgb_path_j) else rgb_path_p
        
        rgb = cv2.imread(rgb_path)
        if rgb is not None:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (640, 480))
        
        # Load GT
        gt_path = os.path.join(GT_DIR, f"{img_id}.png")
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        # Visualize
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        if rgb is not None: plt.imshow(rgb)
        plt.title(f"RGB {img_id}")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        # Highlight Floor (4), Wall (11), Objects (6)
        vis_gt = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
        vis_gt[gt == 4] = [255, 0, 0] # Floor: Red
        vis_gt[gt == 11] = [0, 255, 0] # Wall: Green
        plt.imshow(vis_gt)
        plt.title("GT (R:Floor, G:Wall)")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        vis_pred = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        vis_pred[pred == 4] = [255, 0, 0]
        vis_pred[pred == 11] = [0, 255, 0]
        plt.imshow(vis_pred)
        plt.title("Pred (R:Floor, G:Wall)")
        plt.axis('off')
        
        save_path = os.path.join(DIAG_DIR, f"fail_{img_id}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    main()
