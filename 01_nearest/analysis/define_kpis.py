import os
import json
import numpy as np
import cv2
import sys
from tqdm import tqdm

sys.path.append(os.getcwd())
from nearest_final.utils.metrics import update_confusion_matrix, compute_metrics_from_confulstion_matrix

def compute_iou(cm):
    # cm: (C, C)
    true_pos = np.diag(cm)
    false_pos = np.sum(cm, axis=0) - true_pos
    false_neg = np.sum(cm, axis=1) - true_pos
    denom = true_pos + false_pos + false_neg
    iou = np.divide(true_pos, denom, out=np.zeros_like(true_pos, dtype=float), where=denom!=0)
    return iou

def run_define_kpis():
    golden_root = "nearest_final/golden_artifacts"
    oof_dir = os.path.join(golden_root, "oof")
    
    logits_path = os.path.join(oof_dir, "oof_logits.npy")
    ids_path = os.path.join(oof_dir, "oof_file_ids.npy")
    
    if not os.path.exists(logits_path):
        print("OOF Logits not found. Run merge_golden.py first.")
        return

    print("Loading OOF assets...")
    logits = np.load(logits_path) # (N, C, H, W)
    ids = np.load(ids_path)
    
    # Constants
    C = 13
    IGNORE_INDEX = 255
    CLASS_NAMES = [
        "bed", "books", "ceiling", "chair", "floor", "furniture", 
        "objects", "picture", "sofa", "table", "tv", "wall", "window"
    ]
    CID_BOOKS = 1
    CID_FURN = 5
    CID_OBJ = 6
    CID_TABLE = 9
    
    data_root = "/root/datasets/NYUv2/data/train"
    
    # Bins
    dist_bins = [(0,1), (1,2), (2,3), (3,5), (5,10)]
    
    # Storage for bin-wise confusion matrices
    # key: bin_idx -> (C, C)
    bin_cms = {i: np.zeros((C, C), dtype=np.int64) for i in range(len(dist_bins))}
    global_cm = np.zeros((C, C), dtype=np.int64)
    
    print("Computing KPIs...")
    for i in tqdm(range(len(ids))):
        fid = ids[i]
        
        # Load GT
        gt = cv2.imread(os.path.join(data_root, "label", f"{fid}.png"), cv2.IMREAD_UNCHANGED)
        if gt.ndim == 3: gt = gt[:,:,0]
        
        # Load Depth
        depth = cv2.imread(os.path.join(data_root, "depth", f"{fid}.png"), cv2.IMREAD_UNCHANGED)
        # Depth is usually uint16 in mm? Or customized?
        # NYUv2 originally is units... let's assume mm or consistent with training.
        # But we need meters for bins.
        # If standard 16bit PNG from NYU dataset, max is usually ~10000 (10m).
        # Let's assume unit is millimeters => / 1000.0
        if depth is None:
             print(f"Warning: No depth for {fid}")
             continue
        depth_m = depth.astype(float) / 1000.0
        
        # Pred
        # logits (C, H, W)
        l = logits[i]
        pred = np.argmax(l, axis=0).astype(np.uint8)
        
        # ResizingGT
        if gt.shape != pred.shape:
             gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
             
        # Resize Depth
        if depth_m.shape != pred.shape:
             depth_m = cv2.resize(depth_m, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
             
        # Global CM
        update_confusion_matrix(pred, gt, C, IGNORE_INDEX, global_cm)
        
        # Bin-wise
        mask_valid = (gt != IGNORE_INDEX)
        
        for b_idx, (d_min, d_max) in enumerate(dist_bins):
            mask_d = (depth_m >= d_min) & (depth_m < d_max) & mask_valid
            if not np.any(mask_d): continue
            
            p_bin = pred[mask_d]
            g_bin = gt[mask_d]
            
            # Fast update
            # bincount trick or explicit loop? bincount is faster for flat arrays.
            # But update_confusion_matrix might be optimized or we can use sklearn.
            # Let's just use the func from utils if efficient, or minimal manual update 
            # p_bin * C + g_bin ...
            
            # Manual flat update
            idx = p_bin * C + g_bin # flattened index
            count = np.bincount(idx, minlength=C*C)
            bin_cms[b_idx] += count.reshape(C, C)

    # Calculate KPIs
    kpis = {
        "global_miou": -1,
        "books_iou_per_bin": {},
        "books_confusion": {}, # {bin: {to_furn: %, to_obj: %}}
    }
    
    # Global
    iou_global = compute_iou(global_cm)
    kpis["global_miou"] = float(np.mean(iou_global))
    
    # Bins
    for b_idx, (d_min, d_max) in enumerate(dist_bins):
        bin_label = f"{d_min}-{d_max}m"
        cm_b = bin_cms[b_idx]
        iou_b = compute_iou(cm_b)
        
        # Books IoU
        books_iou = float(iou_b[CID_BOOKS])
        kpis["books_iou_per_bin"][bin_label] = books_iou
        
        # Confusion Analysis for Books (True Class = Books)
        # Confusion Row for Books: cm_b[CID_BOOKS, :] -> Predictions given GT=Books is false?
        # NO. Confusion Matrix usually (Pred, GT) or (GT, Pred)?
        # utils.metrics.update_confusion_matrix: index = pred * num_classes + label
        # rows = pred, cols = label? 
        # let's check code logic: cm[pred][label] += 1
        # So Row=Pred, Col=GT.
        # We want: Given GT=Books, what did we predict?
        # Sum of Column `CID_BOOKS` is Total GT Books
        # `cm_b[CID_FURN, CID_BOOKS]` is "Predicted Furn, GT Books"
        
        gt_books_total = np.sum(cm_b[:, CID_BOOKS])
        if gt_books_total > 0:
            rate_furn = float(cm_b[CID_FURN, CID_BOOKS]) / gt_books_total
            rate_obj = float(cm_b[CID_OBJ, CID_BOOKS]) / gt_books_total
        else:
            rate_furn = 0.0
            rate_obj = 0.0
            
        kpis["books_confusion"][bin_label] = {
            "to_furniture": rate_furn,
            "to_objects": rate_obj
        }

    out_path = os.path.join(golden_root, "next_model_kpis.json")
    print(f"Saving KPIs to {out_path}...")
    with open(out_path, "w") as f:
        json.dump(kpis, f, indent=2)
        
    print("Done.")

def main():
    run_define_kpis()

if __name__ == "__main__":
    main()
