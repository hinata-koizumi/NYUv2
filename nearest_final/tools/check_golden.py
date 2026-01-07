import numpy as np
import os
import hashlib
import json
import argparse

def get_file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def get_array_fingerprint(arr):
    # Hash the raw bytes of a numpy array
    return hashlib.md5(arr.tobytes()).hexdigest()

def run_check_golden(output_path="data/golden_baseline.json", check_mode=False):
    # Default path relative to project root if running from root
    # But usually we want to be flexible.
    
    base_dir = "data/output/nearest_final"
    baseline_path = output_path
    
    action = "Verifying" if check_mode else "Creating"
    print(f"--- {action} Golden Baseline ---")
    
    baseline = {}
    
    # 1. Fold 0 Val Logits (First 10)
    # Path: data/output/nearest_final/fold0/val_oof_logits.npy
    val_path = os.path.join(base_dir, "fold0/val_oof_logits.npy")
    
    # Try alternate location if not found (maybe in fold subdir directly?)
    if not os.path.exists(val_path):
         # Sometimes it is val_logits.npy in golden_artifacts?
         # Or maybe the user meant the one from training?
         # Original script looked at data/output/nearest_final/fold0/val_oof_logits.npy
         pass

    if os.path.exists(val_path):
        print(f"Loading {val_path}...")
        val_logits = np.load(val_path, mmap_mode='r')
        val_first10 = val_logits[:10].copy() # Force read
        
        baseline["val_fold0_first10_shape"] = list(val_first10.shape)
        baseline["val_fold0_first10_mean"] = float(np.mean(val_first10))
        baseline["val_fold0_first10_hash"] = get_array_fingerprint(val_first10)
        
        print(f"  Shape: {val_first10.shape}")
        print(f"  Mean: {baseline['val_fold0_first10_mean']}")
        print(f"  Hash: {baseline['val_fold0_first10_hash']}")
    else:
        print(f"Error: {val_path} not found!")
        if check_mode: return False
        # If creating, maybe we just want to create what we can?
        # But original script exit(1) here.
        exit(1)

    # 2. Submission File
    sub_path = "submission.npy" 
    if not os.path.exists(sub_path):
        sub_path = "data/output/nearest_final/submission.npy" 
        
    sub_array = None
    if os.path.exists(sub_path):
        print(f"Loading {sub_path}...")
        sub_array = np.load(sub_path)
        print(f"  Submission Hash: {get_array_fingerprint(sub_array)}")
    else:
        print("Warning: submission.npy not found.")
        if not check_mode:
            print("Error: submission.npy not found, cannot create full baseline.")
            exit(1)
        else:
            print("Warning: submission.npy not found, skipping submission hash check.")

    # Populate current data
    current_data = {
        "val_fold0_first10_shape": baseline["val_fold0_first10_shape"],
        "val_fold0_first10_mean": baseline["val_fold0_first10_mean"],
        "val_fold0_first10_hash": baseline["val_fold0_first10_hash"],
    }
    
    if sub_array is not None:
        current_data["submission_hash"] = get_array_fingerprint(sub_array)

    if check_mode:
        if not os.path.exists(baseline_path):
            print(f"Error: No baseline found at {baseline_path} to check against.")
            return False
            
        with open(baseline_path, "r") as f:
            saved_baseline = json.load(f)
            
        print("\n--- Verifying against Baseline ---")
        param_match = True
        for k, v in saved_baseline.items():
            curr = current_data.get(k)
            if curr != v:
                print(f"[FAIL] {k}: Expected {v}, Got {curr}")
                param_match = False
            else:
                print(f"[PASS] {k}: {curr}")
                
        if param_match:
            print("\nSUCCESS: Current state matches Golden Baseline.")
        else:
            print("\nFAILURE: Regressions detected!")
        return param_match
    else:
        os.makedirs(os.path.dirname(os.path.abspath(baseline_path)), exist_ok=True)
        with open(baseline_path, "w") as f:
            json.dump(current_data, f, indent=2)
        print(f"Baseline saved to {os.path.abspath(baseline_path)}")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Check against existing baseline")
    parser.add_argument("--output", type=str, default="data/golden_baseline.json", help="Path to golden baseline json")
    args = parser.parse_args()
    
    run_check_golden(output_path=args.output, check_mode=args.check)
