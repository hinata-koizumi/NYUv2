
import os
import argparse
import numpy as np
import cv2
import json
from tqdm import tqdm
from nearest_final.submit.utils import load_cfg_from_fold_dir
from nearest_final.utils.metrics import update_confusion_matrix, compute_metrics

def run_merge_oof(exp_dir: str = "data/output/nearest_final"):

    print(f"--- Merging Global OOF from {exp_dir} ---")
    
    # 1. Collect all Train IDs (Reference Global Order)
    # We need the full sorted train list to align everything.
    # We can get this by listing the train dir again, standardized.
    fold0_dir = os.path.join(exp_dir, "fold0")
    if not os.path.exists(fold0_dir):
        print("Error: Fold 0 not found.")
        return
        
    cfg = load_cfg_from_fold_dir(fold0_dir)
    img_dir = os.path.join(cfg.TRAIN_DIR, "image")
    all_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    global_ids = np.array([os.path.splitext(f)[0] for f in all_files])
    
    N_total = len(global_ids)
    print(f"Total Reference Train Images: {N_total}")
    
    # Map ID -> Index
    id_to_idx = {fid: i for i, fid in enumerate(global_ids)}
    
    # 2. Allocate Global Arrays
    # Standard: (N, 13, 480, 640) float16
    H, W = 480, 640
    C = 13
    print(f"Allocating Global OOF: ({N_total}, {C}, {H}, {W}) float16")
    global_logits = np.zeros((N_total, C, H, W), dtype=np.float16)
    
    # Also track which indices were filled to ensure coverage
    filled_mask = np.zeros(N_total, dtype=bool)
    
    # 3. Iterate Folds and Fill
    metrics_all = []
    
    for k in range(5):
        fold_dir = os.path.join(exp_dir, f"fold{k}")
        oof_path = os.path.join(fold_dir, "val_oof_logits.npy")
        ids_path = os.path.join(fold_dir, "val_file_ids.npy")
        met_path = os.path.join(fold_dir, "val_metrics.json")
        
        if not os.path.exists(oof_path):
            print(f"Warning: {oof_path} missing. Skipping fold {k}.")
            continue
            
        print(f"Loading Fold {k}...")
        res = np.load(oof_path) # (N_val, C, H, W)
        ids = np.load(ids_path) # (N_val,)
        
        with open(met_path, "r") as f:
            metrics_all.append(json.load(f))
            
        # Fill
        for i, fid in enumerate(tqdm(ids, desc=f"Merging Fold {k}", leave=False)):
            if fid not in id_to_idx:
                print(f"Warning: ID {fid} not in global train set?")
                continue
            
            idx = id_to_idx[fid]
            global_logits[idx] = res[i]
            filled_mask[idx] = True
            
    # 4. Save
    n_filled = np.sum(filled_mask)
    print(f"Filled: {n_filled}/{N_total} ({n_filled/N_total:.1%})")
    
    if n_filled < N_total:
         print("Warning: Global OOF is incomplete! (Random KFold does not guarantee perfect coverage if partitions vary?)")
         # Actually standard KFold partitions ARE disjoint and complete union.
         # So this should be 100%.
         
    
    # Generate Preds (uint8)
    print("Generating Predictions (Argmax)...")
    global_pred = np.argmax(global_logits, axis=1).astype(np.uint8)
    
    out_npy = os.path.join(exp_dir, "oof_logits.npy")
    out_pred = os.path.join(exp_dir, "oof_pred.npy") 
    out_ids = os.path.join(exp_dir, "oof_file_ids.npy")
    out_met = os.path.join(exp_dir, "oof_metrics_summary.json")
    
    print(f"Saving {out_npy}...")
    np.save(out_npy, global_logits)
    
    print(f"Saving {out_pred}...")
    np.save(out_pred, global_pred)
    
    print(f"Saving {out_ids}...")
    np.save(out_ids, global_ids)
    
    with open(out_met, "w") as f:
        json.dump(metrics_all, f, indent=2)

    # 5. Global Metric Assertion (Safety Check)
    print("Computing Global Metrics against Ground Truth...")
    
    # Load all GT
    global_cm = np.zeros((C, C), dtype=np.int64)
    label_dir = os.path.join(cfg.TRAIN_DIR, "label")
    
    for idx, fid in enumerate(tqdm(global_ids, desc="Global Eval")):
        # Load GT
        path = os.path.join(label_dir, f"{fid}.png")
        gt = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if gt is None:
            print(f"Error: Label for {fid} not found.")
            continue
        if gt.ndim == 3: gt = gt[:,:,0]
        
        pred = global_pred[idx] # (H, W)
        update_confusion_matrix(pred, gt, C, cfg.IGNORE_INDEX, global_cm)
        
    g_pixel_acc, g_miou, g_ious = compute_metrics(global_cm)
    print(f"\n--- Global Integrity Check ---")
    print(f"Global mIoU: {g_miou:.4f}")
    
    # Calculate Average Fold mIoU for reference
    fold_mious = [m["mIoU"] for m in metrics_all]
    avg_fold_miou = np.mean(fold_mious)
    print(f"Avg Fold mIoU: {avg_fold_miou:.4f}")
    
    if abs(g_miou - avg_fold_miou) > 0.02:
         print("WARNING: Global mIoU diverges significantly from Fold Average!")
         print("Possible causes: Validation set overlap, ID misalignment, or class imbalance differences.")
         # In a disjoint split, Global should be close to Avg, but global puts all pixels in one pot (micro-average over folds),
         # while Avg Fold mIoU is macro-average over folds.
    
    if g_miou < 0.65:
         print("CRITICAL ERROR: Global mIoU is too low! File ID alignment is likely broken.")
         raise RuntimeError("Global OOF integrity check failed.")

    # Save Global Detailed Metrics
    global_metrics = {
        "mIoU": float(g_miou),
        "pixel_acc": float(g_pixel_acc),
        "per_class_iou": {name: float(g_ious[i]) for i, name in enumerate(cfg.CLASS_NAMES)},
        "confusion_matrix": global_cm.tolist()
    }
    with open(os.path.join(exp_dir, "oof_global_metrics.json"), "w") as f:
        json.dump(global_metrics, f, indent=2)

    print("Done.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", type=str, default="data/output/nearest_final")
    args = p.parse_args()
    run_merge_oof(args.exp_dir)

if __name__ == "__main__":
    main()
