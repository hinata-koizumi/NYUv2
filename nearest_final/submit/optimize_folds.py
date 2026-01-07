import numpy as np
import os
import json
import cv2
import sys
from scipy.optimize import minimize

sys.path.append(os.getcwd())
# metrics module might be slow for full loop, we'll implement fast cm
from nearest_final.utils.metrics import update_confusion_matrix, compute_metrics

def run_optimize_folds():
    golden_root = "golden_artifacts"
    label_root = "/root/datasets/NYUv2/data/train/label"

    # Load Folds
    print("Loading data for optimization...")
    fold_logits = []
    fold_gts = []
    
    for f in range(5):
        d = os.path.join(golden_root, "folds", f"fold{f}")
        if not os.path.exists(d): 
            print(f"Fold {f} missing, skipping.")
            continue
            
        logits = np.load(os.path.join(d, "val_logits.npy")) # (Ni, C, H, W)
        ids = np.load(os.path.join(d, "val_file_ids.npy"))
        
        fold_logits.append(logits)
        
        # Load GTs
        gts = []
        for fid in ids:
            p = os.path.join(label_root, f"{fid}.png")
            gt = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if gt.ndim == 3: gt = gt[:,:,0]
            # Resize GT to match logits (480x640)
            if gt.shape[0] != logits.shape[2] or gt.shape[1] != logits.shape[3]:
                 gt = cv2.resize(gt, (logits.shape[3], logits.shape[2]), interpolation=cv2.INTER_NEAREST)
            gts.append(gt)
        fold_gts.append(np.array(gts))

    if not fold_logits:
        print("No data.")
        return

    C = 13
    IGNORE_INDEX = 255

    # Optimization Function
    # We want to maximize mIoU <=> minimize -mIoU
    
    def objective(weights):
        # weights: (5,)
        # Since folds are disjoint, we calculate CM for each fold with its weight, then sum CMs.
        # But wait, weights affect argmax inside the fold.
        # If w > 0, argmax(w * logits) == argmax(logits).
        # So CM is constant per fold regardless of w (as long as w > 0).
        
        # We can verify this hypothesis.
        # So objective SHOULD be constant.
        
        return 0.0

    print("Checking score variance...")
    
    # Calculate baseline (all w=1)
    cm_total = np.zeros((C, C), dtype=np.int64)
    
    for i in range(len(fold_logits)):
        # w=1
        preds = np.argmax(fold_logits[i], axis=1)
        gts = fold_gts[i]
        for j in range(len(preds)):
            update_confusion_matrix(preds[j], gts[j], C, IGNORE_INDEX, cm_total)
            
    _, base_miou, _ = compute_metrics(cm_total)
    print(f"Baseline mIoU (w=[1...]): {base_miou:.6f}")
    
    # Calculate with skewed weights (w=[0.1, 10, ...])
    # Note: Argmax is scale invariant.
    print("Baseline is scale-invariant for disjoint OOF. Optimization is technically strictly flat.")
    print("Defaulting to Equal Weights (0.2).")
    
    final_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    
    out_path = os.path.join(golden_root, "fold_weights.json")
    with open(out_path, "w") as f:
        json.dump(final_weights, f)
        
    with open(os.path.join(golden_root, "oof_weighted_score.txt"), "w") as f:
        f.write(f"mIoU: {base_miou:.6f}\nWeights: {final_weights}\nNote: Optimization skipped due to scale-invariance on disjoint OOF.")

    with open(os.path.join(golden_root, "oof_weighted_score.txt"), "w") as f:
        f.write(f"mIoU: {base_miou:.6f}\nWeights: {final_weights}\nNote: Optimization skipped due to scale-invariance on disjoint OOF.")

def main():
    run_optimize_folds()

if __name__ == "__main__":
    main()
