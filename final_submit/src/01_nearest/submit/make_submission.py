
import numpy as np
import os
import zipfile
import argparse
import json
from tqdm import tqdm

def run_make_submission(exp_dir: str, output_name: str = "submission"):
    print("--- Generating Robust Submission (Sequential RAM Optimized) ---")
    
    # 1. Config
    recipe_path = os.path.join(exp_dir, "final_recipe.json")
    if not os.path.exists(recipe_path):
        recipe_path = os.path.join(os.path.dirname(exp_dir), "final_recipe.json")
        
    T_struct = 1.0
    T_nonstruct = 1.0
    T_table = 1.0
    damping_tau = 0.0
    damping_delta = 0.0
    
    if os.path.exists(recipe_path):
        with open(recipe_path, "r") as f:
            recipe = json.load(f)
            pp = recipe.get("postprocess", {})
            T_struct = pp.get("T_struct", 1.0)
            T_nonstruct = pp.get("T_nonstruct", 1.0)
            T_table = pp.get("T_table", 1.0)
            damping_tau = pp.get("damping_tau", 0.0)
            damping_delta = pp.get("damping_delta", 0.0)
        print(f"Loaded Recipe: {pp}")
    else:
        print("Recipe not found, using defaults.")

    # 2. Identify Test Files
    # Test logits are usually in exp_dir/fold{k}/test_logits.npy
    # We need shape.
    f0 = os.path.join(exp_dir, "fold0", "test_logits.npy")
    if not os.path.exists(f0):
        print(f"Error: {f0} not found.")
        return
        
    m0 = np.load(f0, mmap_mode='r')
    print(f"Reference Shape: {m0.shape}, Dtype: {m0.dtype}")
    
    N, C, H, W = m0.shape
    
    # 3. Accumulate in RAM (Full Load)
    # 125GB RAM available. 11GB per fold is trivial.
    print("Allocating Accumulator (11GB)...")
    acc = np.zeros((N, C, H, W), dtype=np.float32)

    for k in range(5):
        path = os.path.join(exp_dir, f"fold{k}", "test_logits.npy")
        print(f"Loading Fold {k} (Full RAM): {path} ...")
        
        # Load fully
        m = np.load(path) # No mmap
        
        if m.ndim==3 and m.shape[-1]==13:
            m = m.transpose(0, 3, 1, 2)
            
        acc += m
        del m # Free memory
        print(f"Fold {k} Merged.")
            
    # Average
    acc /= 5.0
    
    # 4. Post-Process
    print("Post-processing...")
    
    CLASS_NAMES = ["bed", "books", "ceiling", "chair", "floor", "furniture", "objects", "picture", "sofa", "table", "tv", "wall", "window"]
    STRUCT_CLASSES = ["bed", "chair", "floor", "sofa", "table", "wall", "window"]
    STRUCT_IDS = [i for i, n in enumerate(CLASS_NAMES) if n in STRUCT_CLASSES]
    NONSTRUCT_IDS = [i for i in range(13) if i not in STRUCT_IDS]
    TABLE_ID = 9
    
    # In-place ops
    if abs(T_struct - 1.0) > 1e-3:
        for cid in STRUCT_IDS:
             acc[:, cid, :, :] /= T_struct
             
    if abs(T_nonstruct - 1.0) > 1e-3:
        for cid in NONSTRUCT_IDS:
             acc[:, cid, :, :] /= T_nonstruct
             
    if abs(T_table - T_struct) > 1e-3:
         acc[:, TABLE_ID, :, :] *= (T_struct / T_table)
         
    # Damping / Softmax
    # Need to do per-pixel? 
    # Or batched.
    # Softmax on (N, C, H, W) is huge.
    # But possible if 11GB.
    
    # We need argmax for submission.
    # We need probs for damping.
    
    final_pred = np.zeros((N, H, W), dtype=np.uint8)
    
    # To save memory, process final steps in chunks and write to output
    calc_logits_path = os.path.join(exp_dir, "test_logits_calibrated.npy")
    print(f"Saving logits {calc_logits_path}...")
    np.save(calc_logits_path, acc) # Save before damping if damping changes things? 
    # Wait, usually we save "averaged logits". Damping is post-proc for argmax.
    # But if user wants "calibrated logits", they usually mean applied T.
    
    # Apply Damping dynamically
    print("Generating predictions...")
    for i in tqdm(range(N)):
        l = acc[i] # (C, H, W)
        
        if damping_tau > 0 and damping_delta > 0:
             ma = np.max(l, axis=0, keepdims=True)
             ex = np.exp(l - ma)
             pr = ex / np.sum(ex, axis=0, keepdims=True)
             pm = np.max(pr, axis=0) # (H, W)
             
             mask = (pm < damping_tau)
             for cid in NONSTRUCT_IDS:
                 l[cid][mask] -= damping_delta
                 
        start_pred = np.argmax(l, axis=0).astype(np.uint8)
        final_pred[i] = start_pred
        
    # Save SUBMISSION
    npy_path = os.path.join(exp_dir, f"{output_name}.npy")
    zip_path = os.path.join(exp_dir, f"{output_name}.zip")
    
    print(f"Saving {npy_path}...")
    np.save(npy_path, final_pred)
    
    print(f"Zipping {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(npy_path, "submission.npy")
        
    print("Done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", type=str, required=True)
    p.add_argument("--output_name", type=str, default="submission")
    args = p.parse_args()
    run_make_submission(args.exp_dir, args.output_name)
