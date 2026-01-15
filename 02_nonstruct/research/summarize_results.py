"""
Fold 0-4の結果集計スクリプト

全てのチェックポイントから最終メトリクスを抽出し、サマリーを作成する。
"""

import os
import torch
import numpy as np

def summarize_all_folds():
    output_dir = "/root/datasets/NYUv2/02_nonstruct/output"
    folds = range(5)
    
    summary = []
    
    print(f"{'Fold':<5} {'Floor':<7} {'Wall':<7} {'ObjRat':<7} {'Suck':<7} {'Books':<7} {'Pic':<7} {'TV':<7} {'SmTgtAvg':<8}")
    print("-" * 80)
    
    metrics_list = []
    
    for f in folds:
        path = os.path.join(output_dir, f"fold{f}_last.pth")
        if not os.path.exists(path):
            print(f"Fold {f}: Not found")
            continue
            
        data = torch.load(path, map_location="cpu")
        m = data.get("metrics", {})
        metrics_list.append(m)
        
        sm_avg = (m.get('iou_books',0) + m.get('iou_picture',0) + m.get('iou_tv',0)) / 3
        
        print(f"{f:<5} {m.get('iou_floor',0):<7.3f} {m.get('iou_wall',0):<7.3f} "
              f"{m.get('ratio_objects_global',0):<7.3f} {m.get('suck_rate',0):<7.3f} "
              f"{m.get('iou_books',0):<7.3f} {m.get('iou_picture',0):<7.3f} {m.get('iou_tv',0):<7.3f} {sm_avg:<8.3f}")

    # Calculate mean
    if metrics_list:
        keys = metrics_list[0].keys()
        means = {k: np.mean([m.get(k, 0) for m in metrics_list]) for k in keys}
        mean_sm_avg = (means.get('iou_books',0) + means.get('iou_picture',0) + means.get('iou_tv',0)) / 3
        
        print("-" * 80)
        print(f"{'MEAN':<5} {means.get('iou_floor',0):<7.3f} {means.get('iou_wall',0):<7.3f} "
              f"{means.get('ratio_objects_global',0):<7.3f} {means.get('suck_rate',0):<7.3f} "
              f"{means.get('iou_books',0):<7.3f} {means.get('iou_picture',0):<7.3f} {means.get('iou_tv',0):<7.3f} {mean_sm_avg:<8.3f}")

if __name__ == "__main__":
    summarize_all_folds()
