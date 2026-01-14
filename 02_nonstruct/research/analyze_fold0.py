"""
Fold0 Epoch推移分析スクリプト

既存のfold0_last.pthから全epochの推移を再計算し、best epochを決定する。
ただし、学習時のログが保存されていない場合は、最終epochのみ分析する。
"""

import os
import numpy as np
import torch
import json
from torch.utils.data import DataLoader

import sys
sys.path.append("/root/datasets/NYUv2/02_nonstruct"); sys.path.append("/root/datasets/NYUv2")

from configs import default as config
from data.dataset import ModelBDataset
from model.arch import ConvNeXtBaseFPN3Ch
from utils.common import calculate_metrics, MetricAggregator

def analyze_fold0_checkpoint():
    """Fold0の最終チェックポイントを分析"""
    
    print("\n" + "="*80)
    print("Fold 0 最終チェックポイント分析")
    print("="*80)
    
    # Load fold data
    with open(config.SPLITS_FILE) as f:
        manifest = json.load(f)
    
    fold_filename = manifest["folds"][0]
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
    
    # Dataset
    ds_val = ModelBDataset(val_imgs, val_lbls, val_deps, is_train=False, ids=val_ids)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=0)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNeXtBaseFPN3Ch(num_classes=13, pretrained=False).to(device)
    
    checkpoint_path = "/root/datasets/NYUv2/02_nonstruct/output/fold0_last.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print(f"\nチェックポイント情報:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  保存されたメトリクス: {list(checkpoint.get('metrics', {}).keys())}")
    
    # Load weights and evaluate
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    print(f"\nValidation全体を再評価中... ({len(val_ids)}枚)")
    
    agg = MetricAggregator()
    
    with torch.no_grad():
        for batch in dl_val:
            x, y, ids = batch
            x, y = x.to(device), y.to(device)
            
            with torch.amp.autocast(device_type=device.type if device.type != 'mps' else 'cpu'):
                output = model(x)
                if isinstance(output, (tuple, list)):
                    logits = output[0]
                else:
                    logits = output
            
            if logits.shape[2:] != (480, 640):
                logits = torch.nn.functional.interpolate(
                    logits, size=(480, 640), mode='bilinear', align_corners=False
                )
            
            m_dict = calculate_metrics(logits, y, device=device)
            agg.update(m_dict)
    
    metrics = agg.compute()
    
    print("\n" + "="*80)
    print("修正後のメトリクス (Global ratio使用)")
    print("="*80)
    
    print(f"\n【構造物の安定性】")
    print(f"  Floor IoU:           {metrics.get('iou_floor', 0):.3f}")
    print(f"  Wall IoU:            {metrics.get('iou_wall', 0):.3f}")
    
    print(f"\n【Objects関連 (修正後)】")
    print(f"  Ratio (Global):      {metrics.get('ratio_objects_global', 0):.3f}")
    print(f"  Pred % (Global):     {metrics.get('pred_objects_percent_global', 0):.2f}%")
    print(f"  Suck-in Rate:        {metrics.get('suck_rate', 0):.3f}")
    
    print(f"\n【Small Targets】")
    print(f"  Books IoU:           {metrics.get('iou_books', 0):.3f}")
    print(f"  Picture IoU:         {metrics.get('iou_picture', 0):.3f}")
    print(f"  TV IoU:              {metrics.get('iou_tv', 0):.3f}")
    
    print("\n" + "="*80)
    print("判定")
    print("="*80)
    
    # Safety checks
    issues = []
    
    if metrics.get('iou_floor', 0) < 0.7:
        issues.append(f"⚠️  Floor IoU低い ({metrics.get('iou_floor', 0):.3f})")
    
    if metrics.get('iou_wall', 0) < 0.6:
        issues.append(f"⚠️  Wall IoU低い ({metrics.get('iou_wall', 0):.3f})")
    
    if metrics.get('ratio_objects_global', 0) > 1.25:
        issues.append(f"⚠️  Objects ratio高い ({metrics.get('ratio_objects_global', 0):.3f})")
    
    if metrics.get('pred_objects_percent_global', 0) > 25.0:
        issues.append(f"⚠️  Objects percentage高い ({metrics.get('pred_objects_percent_global', 0):.2f}%)")
    
    if metrics.get('suck_rate', 0) > 0.05:
        issues.append(f"⚠️  Suck-in rate高い ({metrics.get('suck_rate', 0):.3f})")
    
    if issues:
        print("\n問題点:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ 全ての安全性チェックをパス")
    
    # Small target performance
    avg_small = (metrics.get('iou_books', 0) + metrics.get('iou_picture', 0) + metrics.get('iou_tv', 0)) / 3
    print(f"\nSmall Targets平均IoU: {avg_small:.3f}")
    
    if avg_small < 0.15:
        print("  ⚠️  Small targetsが弱い（ただしBは01_nearestと組む前提なので許容範囲内）")
    
    print("\n" + "="*80)
    print("結論")
    print("="*80)
    
    if not issues:
        print("✅ このモデルは5fold fullに進行可能")
        print("   - Global ratio = {:.3f} (正常範囲)".format(metrics.get('ratio_objects_global', 0)))
        print("   - Floor/Wall安定")
        print("   - Suck-in rate低い")
    else:
        print("⚠️  いくつかの問題が検出されました。上記の問題点を確認してください。")
    
    print("="*80)
    
    return metrics

if __name__ == "__main__":
    analyze_fold0_checkpoint()
