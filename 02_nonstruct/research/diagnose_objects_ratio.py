"""
Objects Ratio診断スクリプト

このスクリプトは以下を実行します：
1. Objects Ratioの計算式を確認
2. Fold 0のvalidation画像10枚について、pred/gt objects比率を計算
3. 比率が最大の3枚について、RGB/GT/Predを可視化
"""

import os
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
import json

import sys
sys.path.append("/root/datasets/NYUv2/02_nonstruct"); sys.path.append("/root/datasets/NYUv2")

from configs import default as config
from data.dataset import ModelBDataset
from model.arch import ConvNeXtBaseFPN3Ch

def analyze_objects_ratio():
    """Objects Ratioの詳細分析"""
    
    # Load fold data
    with open(config.SPLITS_FILE) as f:
        manifest = json.load(f)
    
    fold_filename = manifest["folds"][0]  # fold 0
    fold_path = os.path.join(os.path.dirname(config.SPLITS_FILE), fold_filename)
    
    with open(fold_path) as f:
        fold_data = json.load(f)
    
    val_ids = fold_data["val_ids"]
    
    # Paths
    img_dir = os.path.join(config.DATA_DIR, "train/image")
    lbl_dir = os.path.join(config.DATA_DIR, "train/label")
    dep_dir = os.path.join(config.DATA_DIR, "train/depth")
    
    def get_paths(id_list):
        imgs = [os.path.join(img_dir, f"{i}.jpg") for i in id_list]
        lbls = [os.path.join(lbl_dir, f"{i}.png") for i in id_list]
        deps = [os.path.join(dep_dir, f"{i}.png") for i in id_list]
        if not os.path.exists(imgs[0]):
            imgs = [os.path.join(img_dir, f"{i}.png") for i in id_list]
        return np.array(imgs), np.array(lbls), np.array(deps)
    
    val_imgs, val_lbls, val_deps = get_paths(val_ids)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNeXtBaseFPN3Ch(num_classes=13, pretrained=False).to(device)
    
    checkpoint = torch.load("/root/datasets/NYUv2/02_nonstruct/output/fold0_last.pth", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    # Dataset (first 10 images)
    sample_size = min(10, len(val_ids))
    ds_val = ModelBDataset(
        val_imgs[:sample_size], 
        val_lbls[:sample_size], 
        val_deps[:sample_size], 
        is_train=False, 
        ids=val_ids[:sample_size]
    )
    
    results = []
    
    print("\n" + "="*80)
    print("Objects Ratio 計算式の確認:")
    print("="*80)
    print("現在の実装 (utils.py:57-73):")
    print("  p_obj = ((preds == 6) & valid_mask).sum().float()")
    print("  t_obj = ((targets == 6) & valid_mask).sum().float()")
    print("  ratio_objects = (p_obj / t_obj).item()  # バッチ全体で1つの値")
    print("\n⚠️  問題: GT objects が少ない画像で pred が少し増えるだけで比が爆発")
    print("="*80)
    
    print("\n" + "="*80)
    print(f"Validation画像 {sample_size}枚の詳細分析:")
    print("="*80)
    print(f"{'ID':<25} {'Pred%':>8} {'GT%':>8} {'Pred/GT':>10}")
    print("-"*80)
    
    with torch.no_grad():
        for idx in range(sample_size):
            x, y, img_id = ds_val[idx]
            x = x.unsqueeze(0).to(device)
            y = y.to(device)
            
            # Predict
            logits = model(x)
            if logits.shape[2:] != (480, 640):
                logits = torch.nn.functional.interpolate(
                    logits, size=(480, 640), mode='bilinear', align_corners=False
                )
            
            preds = torch.argmax(logits, dim=1).squeeze(0)
            
            # Calculate objects ratio for this image
            valid_mask = (y != 255)
            valid_count = valid_mask.sum().float().item()
            
            p_obj = ((preds == 6) & valid_mask).sum().float().item()
            t_obj = ((y == 6) & valid_mask).sum().float().item()
            
            pred_pct = (p_obj / valid_count * 100) if valid_count > 0 else 0
            gt_pct = (t_obj / valid_count * 100) if valid_count > 0 else 0
            ratio = (p_obj / t_obj) if t_obj > 0 else (999.0 if p_obj > 0 else 1.0)
            
            results.append({
                'id': img_id,
                'pred_pct': pred_pct,
                'gt_pct': gt_pct,
                'ratio': ratio,
                'pred': preds.cpu().numpy(),
                'gt': y.cpu().numpy(),
                'img_path': val_imgs[idx]
            })
            
            print(f"{img_id:<25} {pred_pct:>7.2f}% {gt_pct:>7.2f}% {ratio:>10.2f}")
    
    print("-"*80)
    
    # Global ratio (as currently calculated)
    total_pred = sum(r['pred_pct'] * r['pred'].size for r in results)
    total_gt = sum(r['gt_pct'] * r['gt'].size for r in results)
    total_pixels = sum(r['pred'].size for r in results)
    
    global_pred_pct = total_pred / total_pixels
    global_gt_pct = total_gt / total_pixels
    
    print(f"\nGlobal統計 ({sample_size}枚):")
    print(f"  Pred objects: {global_pred_pct:.2f}%")
    print(f"  GT objects:   {global_gt_pct:.2f}%")
    print(f"  Global ratio: {global_pred_pct/global_gt_pct if global_gt_pct > 0 else 999:.2f}")
    
    # Sort by ratio and visualize top 3
    results.sort(key=lambda x: x['ratio'], reverse=True)
    
    print("\n" + "="*80)
    print("上位3枚の可視化を保存中...")
    print("="*80)
    
    os.makedirs("/root/datasets/NYUv2/02_nonstruct/output/diagnosis", exist_ok=True)
    
    for i, r in enumerate(results[:3]):
        print(f"\n{i+1}. {r['id']} (ratio={r['ratio']:.2f}, pred={r['pred_pct']:.2f}%, gt={r['gt_pct']:.2f}%)")
        
        # Load RGB
        img = cv2.imread(r['img_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        fig_h, fig_w = 480, 640
        canvas = np.zeros((fig_h, fig_w * 3, 3), dtype=np.uint8)
        
        # RGB
        canvas[:, :fig_w] = cv2.resize(img, (fig_w, fig_h))
        
        # GT (colorize objects=6 as red)
        gt_vis = np.zeros((fig_h, fig_w, 3), dtype=np.uint8)
        gt_mask = (r['gt'] == 6)
        gt_vis[gt_mask] = [255, 0, 0]  # Red for objects
        canvas[:, fig_w:fig_w*2] = gt_vis
        
        # Pred (colorize objects=6 as red)
        pred_vis = np.zeros((fig_h, fig_w, 3), dtype=np.uint8)
        pred_mask = (r['pred'] == 6)
        pred_vis[pred_mask] = [255, 0, 0]  # Red for objects
        canvas[:, fig_w*2:] = pred_vis
        
        # Save
        out_path = f"/root/datasets/NYUv2/02_nonstruct/output/diagnosis/top{i+1}_{r['id']}.png"
        cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print(f"   保存: {out_path}")
    
    print("\n" + "="*80)
    print("診断完了")
    print("="*80)

if __name__ == "__main__":
    analyze_objects_ratio()
