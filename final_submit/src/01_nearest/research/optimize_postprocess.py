
import os
import json
import numpy as np
import cv2
import argparse
import itertools
from tqdm import tqdm
import sys

# Re-use logic from report_kpis? 
# To be fast, we should load data once and loop over params.

def compute_kpis_fast(logits, gt_list, valid_masks, ids):
    # logits: (N, C, H, W)
    # gt_list: list of (H, W) or array? Array is faster but memory heavy.
    # We valid masks: (N, H, W) bool
    
    # Constants
    C = 13
    STRUCT_IDS = [0, 3, 4, 8, 9, 11, 12] # bed, chair, floor, sofa, table, wall, window
    NONSTRUCT_IDS = [1, 2, 5, 6, 7, 10]
    CID_TABLE = 9
    IGNORE_INDEX = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    preds = np.argmax(logits, axis=1).astype(np.uint8)
    
    # Accumulators
    # We can use bincount for CM
    # flat_preds = preds.ravel()
    # flat_gts = gt_stack.ravel()
    # valid = (flat_gts != 255)
    # cm = bincount...
    
    # Iterate for robustness (and boundary generation)
    
    cm = np.zeros((C, C), dtype=np.int64)
    boundary_inter = 0
    boundary_union = 0
    
    struct_tp = 0
    struct_fp = 0
    
    entropy_sum = 0.0
    entropy_count = 0
    
    for i in range(len(preds)):
        p = preds[i]
        g = gt_list[i]
        valid = valid_masks[i]
        
        # 1. CM
        g_valid = g[valid]
        p_valid = p[valid]
        if len(g_valid) > 0:
            idx = p_valid * C + g_valid
            count = np.bincount(idx, minlength=C*C)
            cm += count.reshape(C, C)
            
        # 2. Boundary
        # Pre-computed GT boundary?
        # We should pre-compute GT derived data to speed up search!
        pass 
        
    return cm
    
# Better Approach:
# Pre-compute GT stats (Boundary Maps, Struct Masks, Table Masks).
# Loop params:
#   Apply params to Logits -> Preds
#   Compare Preds vs Pre-computed GTs
#   Calculate Scores

def optimize(exp_dir):
    oof_dir = os.path.join(exp_dir, "01_nearest/golden_artifacts/oof")
    if not os.path.exists(oof_dir):
        # Fallback
        if os.path.exists(os.path.join(exp_dir, "golden_artifacts/oof")):
            oof_dir = os.path.join(exp_dir, "golden_artifacts/oof")
            
    logits_path = os.path.join(oof_dir, "oof_logits.npy")
    ids_path = os.path.join(oof_dir, "oof_file_ids.npy")
    
    print("Loading OOF...")
    logits_raw = np.load(logits_path, mmap_mode='r') # (N, 13, H, W)
    ids = np.load(ids_path)
    
    # Load GTs
    print("Loading GTs...")
    data_root = "/root/datasets/NYUv2/00_data/train"
    gts = []
    
    # Pre-computed Boundary GTs
    gt_boundaries = []
    gt_struct_masks = []
    gt_nonstruct_masks = []
    gt_table_masks = []
    valid_masks = []
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    STRUCT_IDS = [0, 3, 4, 8, 9, 11, 12]
    NONSTRUCT_IDS = [1, 2, 5, 6, 7, 10]
    CID_TABLE = 9
    
    for fid in tqdm(ids):
        if fid.endswith(".png"):
            p = os.path.join(data_root, "label", fid)
        else:
            p = os.path.join(data_root, "label", f"{fid}.png")
            
        if not os.path.exists(p): # Check train/label vs data/train/label?
             # fallback
             p_alt = os.path.join("/root/datasets/NYUv2/00_data/train/label", fid if fid.endswith(".png") else f"{fid}.png")
             if os.path.exists(p_alt): p = p_alt
             
        g = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if g.ndim==3: g=g[:,:,0]
        
        # Resize if needed to match logits (assuming logits are correct size usually)
        # Check against logits[0]
        h, w = logits_raw.shape[2:]
        if g.shape != (h, w):
            g = cv2.resize(g, (w, h), interpolation=cv2.INTER_NEAREST)
            
        gts.append(g)
        
        v_mask = (g != 255)
        valid_masks.append(v_mask)
        
        # Boundary GT
        v_u8 = v_mask.astype(np.uint8)
        safe = cv2.erode(v_u8, kernel)
        g_u8 = g.astype(np.uint8)
        d = cv2.dilate(g_u8, kernel)
        e = cv2.erode(g_u8, kernel)
        grad = (d != e)
        b_gt = (grad & (safe > 0))
        gt_boundaries.append(b_gt)
        
        # Masks
        is_struct = np.isin(g, STRUCT_IDS)
        gt_struct_masks.append(is_struct & v_mask)
        
        is_nonstruct = np.isin(g, NONSTRUCT_IDS)
        gt_nonstruct_masks.append(is_nonstruct & v_mask) # For entropy
        
        gt_table_masks.append((g == CID_TABLE) & v_mask)
        
    print("Data Loaded. Starting Search...")
    # Revised Grid (User Request)
    t_struct_grid = [0.85, 0.90, 0.95, 1.00]
    t_nonstruct_grid = [1.00, 1.05, 1.10, 1.20]
    t_table_grid = [0.80, 0.90, 1.00] 
    
    tau_grid = [0.55, 0.58, 0.60]
    # Reduce delta range to avoid over-damping
    delta_grid = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    # Init stability vars
    fold_scores = []
    fold_struct = []
    fold_prec = []

    # Helper to calculate metrics for a single image
    def calculate_metrics(pred, g, vm, b_gt, gt_struct_mask):
        conf_mat = np.zeros((13, 13), dtype=np.int64)
        
        cm_idx = pred[vm] * 13 + g[vm]
        if cm_idx.size > 0:
            c = np.bincount(cm_idx, minlength=169)
            conf_mat += c.reshape(13, 13)
            
        tp = np.diag(conf_mat)
        fp = np.sum(conf_mat, axis=0) - tp
        fn = np.sum(conf_mat, axis=1) - tp
        iou = tp / (tp + fp + fn + 1e-6)
        
        miou_all = np.mean(iou)
        miou_struct = np.mean(iou[STRUCT_IDS])
        miou_table = iou[CID_TABLE]

        # Boundary
        d_p = cv2.dilate(pred, kernel)
        e_p = cv2.erode(pred, kernel)
        b_p = (d_p != e_p) & vm
        
        b_inter = np.sum(b_gt & b_p)
        b_union = np.sum(b_gt | b_p)
        miou_bound = b_inter / (b_union + 1e-6)
        
        # Struct Precision
        p_struct = np.isin(pred, STRUCT_IDS)
        mask_sp = p_struct & vm
        
        tp_struct = np.sum(mask_sp & gt_struct_mask)
        fp_struct = np.sum(mask_sp & (~gt_struct_mask))
        struct_prec = tp_struct / (tp_struct + fp_struct + 1e-6)
        
        # Consolidate Objective (Specialist Goal: Structure + Edge + Table)
        score = miou_struct + 0.2 * miou_bound + 0.3 * miou_table
        
        return {
            "mIoU_all": miou_all,
            "mIoU_struct": miou_struct,
            "mIoU_table": miou_table,
            "mIoU_boundary": miou_bound,
            "struct_precision": struct_prec,
            "score": score
        }

    # Re-defining eval_params for CV support
    def eval_cv(t_struct_grid, t_nonstruct_grid, t_table_grid, tau_grid, delta_grid, stride=50):
        all_params = list(itertools.product(t_struct_grid, t_nonstruct_grid, t_table_grid, tau_grid, delta_grid))
        
        # results will store: ((ts, tn, tt, tau, delta), [metrics_f0, metrics_f1, ..., metrics_f4])
        # where metrics_fk is a dict of scores for that fold, or None if no images in fold for stride
        results = {p: [None]*5 for p in all_params}

        print(f"Running CV evaluation for {len(all_params)} parameter combinations with stride {stride}...")
        for i in tqdm(range(0, len(ids), stride), desc="Processing images for CV"):
            fid = ids[i]
            k = folds_map.get(fid, -1)
            if k < 0: continue # Skip if fold not found
            
            g = gts[i]
            vm = valid_masks[i]
            b_gt = gt_boundaries[i]
            gt_struct_mask = gt_struct_masks[i]

            l_base = logits_raw[i].copy() # (13, H, W)
            
            for ts, tn, tt, tau, delta in all_params:
                l = l_base.copy()
                
                # 1. Temp
                l[STRUCT_IDS] /= ts
                l[NONSTRUCT_IDS] /= tn
                l[CID_TABLE] /= tt # Apply table temp
                
                # 2. Softmax / Damp
                if tau > 0 and delta > 0:
                    ma = np.max(l, axis=0, keepdims=True)
                    ex = np.exp(l - ma)
                    pr = ex / np.sum(ex, axis=0, keepdims=True)
                    pm = np.max(pr, axis=0)
                    
                    mask = (pm < tau)
                    for cid in NONSTRUCT_IDS:
                        l[cid][mask] -= delta

                # 3. Argmax
                pred = np.argmax(l, axis=0).astype(np.uint8)
                
                # Calculate metrics for this image and param set
                metrics = calculate_metrics(pred, g, vm, b_gt, gt_struct_mask)
                
                # Accumulate metrics for the current fold
                if results[(ts, tn, tt, tau, delta)][k] is None:
                    results[(ts, tn, tt, tau, delta)][k] = {key: [] for key in metrics.keys()}
                
                for key, value in metrics.items():
                    results[(ts, tn, tt, tau, delta)][k][key].append(value)
        
        # Average accumulated metrics per fold
        final_results = []
        for params, fold_data in results.items():
            avg_fold_data = [None]*5
            for k in range(5):
                if fold_data[k] is not None:
                    avg_metrics = {key: np.mean(values) for key, values in fold_data[k].items()}
                    avg_fold_data[k] = avg_metrics
            final_results.append((params, avg_fold_data))
        
        return final_results

    # Original eval_params function (renamed to avoid conflict and used for baseline)
    def eval_params(ts, tn, tau, delta, stride=1):
        
        conf_mat = np.zeros((13, 13), dtype=np.int64)
        b_inter = 0
        b_union = 0
        tp_struct = 0
        fp_struct = 0
        
        # Loop with stride
        for i in range(0, len(ids), stride):
            l = logits_raw[i].copy() # (13, H, W)
            
            # 1. Temp
            l[STRUCT_IDS] /= ts
            l[NONSTRUCT_IDS] /= tn
            
            # 2. Softmax / Damp
            if tau > 0 and delta > 0:
                ma = np.max(l, axis=0, keepdims=True)
                ex = np.exp(l - ma)
                pr = ex / np.sum(ex, axis=0, keepdims=True)
                pm = np.max(pr, axis=0)
                
                mask = (pm < tau)
                for cid in NONSTRUCT_IDS:
                    l[cid][mask] -= delta

            # 3. Argmax
            pred = np.argmax(l, axis=0).astype(np.uint8)
            
            # Metrics
            g = gts[i]
            cm_idx = pred[valid_masks[i]] * 13 + g[valid_masks[i]]
            if cm_idx.size > 0:
                c = np.bincount(cm_idx, minlength=169)
                conf_mat += c.reshape(13, 13)
                
            # Boundary
            d_p = cv2.dilate(pred, kernel)
            e_p = cv2.erode(pred, kernel)
            b_p = (d_p != e_p) & valid_masks[i]
            
            b_gt = gt_boundaries[i]
            b_inter += np.sum(b_gt & b_p)
            b_union += np.sum(b_gt | b_p)
            
            # Struct Precision
            p_struct = np.isin(pred, STRUCT_IDS)
            mask_sp = p_struct & valid_masks[i]
            
            tp_struct += np.sum(mask_sp & gt_struct_masks[i])
            fp_struct += np.sum(mask_sp & (~gt_struct_masks[i]))
            
        # Calc Scores
        tp = np.diag(conf_mat)
        fp = np.sum(conf_mat, axis=0) - tp
        fn = np.sum(conf_mat, axis=1) - tp
        iou = tp / (tp + fp + fn + 1e-6)
        
        miou_all = np.mean(iou)
        miou_struct = np.mean(iou[STRUCT_IDS])
        miou_table = iou[CID_TABLE]
        miou_bound = b_inter / (b_union + 1e-6)
        struct_prec = tp_struct / (tp_struct + fp_struct + 1e-6)
        
        return {
            "mIoU_all": miou_all,
            "mIoU_struct": miou_struct,
            "mIoU_table": miou_table,
            "mIoU_boundary": miou_bound,
            "struct_precision": struct_prec
        }

    print("Evaluating Baseline (Full)...")
    base_res = eval_params(1.0, 1.0, 0.0, 0.0, stride=1)
    print(f"Baseline: {base_res}")
    
    # Load Splits for CV
    split_path = os.path.join(exp_dir.replace("golden_artifacts", "03_ensemble/splits"), "splits.json")
    if not os.path.exists(split_path):
        # Fallback to local default if not found (e.g. if exp_dir is somewhere else)
        split_path = "/root/datasets/NYUv2/03_ensemble/splits/splits.json"
    
    folds_map = {}
    if os.path.exists(split_path):
        with open(split_path, "r") as f:
            splits = json.load(f)
            # splits: [{"train": [...], "val": [...]}, ...] for 5 folds
            # We want map: fid -> fold_idx (where it was VAL)
            for k in range(len(splits)):
                for fid in splits[k]["val"]:
                    folds_map[fid] = k
    else:
        print("WARN: splits.json not found. Using Random 5-Fold Mock.")
        for i, fid in enumerate(ids):
            folds_map[fid] = i % 5

    gt_folds = [folds_map.get(fid, -1) for fid in ids] # (N,) array
    gt_folds = np.array(gt_folds)

    results = eval_cv(t_struct_grid, t_nonstruct_grid, t_table_grid, tau_grid, delta_grid, stride=50)
    
    print("\nLeave-One-Fold-Out Results:")
    cv_scores = []
    best_params_votes = []
    
    # Baseline Score for Constraints
    b_all = base_res["mIoU_all"]
    b_tbl = base_res["mIoU_table"]
    b_prec = base_res["struct_precision"]
    
    for k in range(5):
        best_p = None
        best_train_score = -999
        
        for params, mets in results:
            valid_train = [mets[j] for j in range(5) if j != k and mets[j] is not None]
            if not valid_train: continue
            
            avg_score = np.mean([m["score"] for m in valid_train])
            
            # Constraints (Strict per User Request)
            # 1. mIoU_all >= baseline - 0.002
            avg_all = np.mean([m["mIoU_all"] for m in valid_train])
            if avg_all < (b_all - 0.002): continue
            
            # 2. mIoU_table >= baseline - 0.001
            avg_tbl = np.mean([m["mIoU_table"] for m in valid_train])
            if avg_tbl < (b_tbl - 0.001): continue
            
            # 3. struct_precision >= baseline - 0.003
            avg_prec = np.mean([m["struct_precision"] for m in valid_train])
            if avg_prec < (b_prec - 0.003): continue

            if avg_score > best_train_score:
                best_train_score = avg_score
                best_p = params
        
        if best_p:
            for params, mets in results:
                if params == best_p:
                    val_m = mets[k]
                    if val_m: # Ensure val_m is not None
                        cv_scores.append(val_m["score"])
                        best_params_votes.append(best_p)
                        print(f"Fold {k}: Best Params {best_p}, Val Score {val_m['score']:.4f}")
                    else:
                        print(f"Fold {k}: Best Params {best_p}, but no validation data for this fold.")
                    break
        else:
             print(f"Fold {k}: No valid params found.")
             
    if best_params_votes:
        from collections import Counter
        final_best = Counter(best_params_votes).most_common(1)[0][0]
        print(f"\nFinal Selected Params (CV Vote): {final_best}")
        
        # Stability Stats
        for params, mets in results:
            if params == final_best:
                for k in range(5):
                    if mets[k]:
                        fold_scores.append(mets[k]["score"])
                        fold_struct.append(mets[k]["mIoU_struct"])
                        fold_prec.append(mets[k]["struct_precision"])
                break
        
        print(f"Score: Mean={np.mean(fold_scores):.4f}, Std={np.std(fold_scores):.4f}")
        print(f"struct_mIoU: Mean={np.mean(fold_struct):.4f}, Std={np.std(fold_struct):.4f}")
        
        best_res = base_res # Placeholder, we trust CV
        best_params = final_best
    else:
        print("CV Failed. Using Baseline.")
        best_res = base_res
        best_params = (1.0, 1.0, 1.0, 0.0, 0.0) # Corrected tuple size

    print("-" * 50)
    
    if best_params:
        final_recipe = {
            "postprocess": {
                "T_struct": best_params[0],
                "T_nonstruct": best_params[1],
                "T_table": best_params[2],
                "damping_tau": best_params[3],
                "damping_delta": best_params[4],
                "stability": {
                     "score_std": float(np.std(fold_scores)) if fold_scores else 0.0,
                     "struct_std": float(np.std(fold_struct)) if fold_struct else 0.0
                }
            }
        }
        os.makedirs(os.path.join(exp_dir, "01_nearest/golden_artifacts"), exist_ok=True)
        with open(os.path.join(exp_dir, "01_nearest/golden_artifacts/final_recipe.json"), "w") as f:
            json.dump(final_recipe, f, indent=2)
            
        # Report
        report = {
            "baseline": base_res,
            "best_params": list(best_params),
            "cv_vote_count": len(best_params_votes)
        }
        with open(os.path.join(exp_dir, "01_nearest/golden_artifacts/postprocess_report.json"), "w") as f:
            json.dump(report, f, indent=2)
            
        print("Saved final_recipe.json")
        
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", type=str, required=True)
    args = p.parse_args()
    optimize(args.exp_dir)
