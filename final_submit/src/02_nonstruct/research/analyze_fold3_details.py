"""
analyze_fold3_details.py
- Fold 3 の詳細分析。
- 各画像の Floor IoU, SmallTargets IoU を計算し、特に悪い画像の原因を探る。
"""

import os
import numpy as np
import torch
import cv2
from tqdm import tqdm
import json

import sys
sys.path.append("/root/datasets/NYUv2/02_nonstruct"); sys.path.append("/root/datasets/NYUv2")
from configs import default as config
from utils.common import calculate_metrics

# Paths
CHECKPOINT = "/root/datasets/NYUv2/02_nonstruct/output/fold3_last.pth"
VAL_IDS_FILE = "/root/datasets/NYUv2/02_nonstruct/output/val_ids_fold3.txt"
GT_LBL_DIR = "/root/datasets/NYUv2/00_data/train/label"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load IDs
    val_ids = np.loadtxt(VAL_IDS_FILE, dtype=str)
    print(f"Analyzing Fold 3: {len(val_ids)} images")
    
    # Load Logits (already saved as OOF)
    logits_b_path = "/root/datasets/NYUv2/02_nonstruct/output/oof_fold3_logits.npy"
    logits_b = np.load(logits_b_path, mmap_mode='r')
    
    results = []
    
    print("Evaluating image by image...")
    for i, img_id in enumerate(tqdm(val_ids)):
        gt_path = os.path.join(GT_LBL_DIR, f"{img_id}.png")
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None: continue
        
        l = logits_b[i]
        pred = np.argmax(l, axis=0)
        
        # Simple IoU calculation
        def get_iou(c_id):
            inter = np.logical_and(pred == c_id, gt == c_id).sum()
            union = np.logical_or(pred == c_id, gt == c_id).sum()
            if union == 0: return 1.0 if (gt == c_id).sum() == 0 else 0.0
            return inter / union
        
        floor_iou = get_iou(4)
        wall_iou = get_iou(11)
        books_iou = get_iou(1)
        pic_iou = get_iou(7)
        tv_iou = get_iou(10)
        sm_avg = (books_iou + pic_iou + tv_iou) / 3
        
        results.append({
            "id": img_id,
            "floor": floor_iou,
            "wall": wall_iou,
            "sm_avg": sm_avg
        })
        
    # Sort by Floor IoU
    results.sort(key=lambda x: x["floor"])
    
    print("\nWorst 10 images for Floor IoU in Fold 3:")
    print(f"{'ID':<10} | {'Floor':<10} | {'Wall':<10} | {'SmAvg':<10}")
    print("-" * 50)
    for r in results[:10]:
        print(f"{r['id']:<10} | {r['floor']:<10.4f} | {r['wall']:<10.4f} | {r['sm_avg']:<10.4f}")
        
    # Correlation between Floor and SmAvg
    floors = [r["floor"] for r in results]
    sm_avgs = [r["sm_avg"] for r in results]
    corr = np.corrcoef(floors, sm_avgs)[0, 1]
    print(f"\nCorrelation (Floor vs SmallTargets Avg): {corr:.4f}")

if __name__ == "__main__":
    main()
