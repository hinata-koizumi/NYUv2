
import os
import numpy as np
import argparse
import sys

def check_integrity(exp_dir):
    print("--- Submission Integrity Check ---")
    
    # 1. Collect Expected IDs from Folds
    folds_dir = os.path.join(exp_dir, "01_nearest/golden_artifacts/folds")
    
    expected_ids = []
    
    # We need to know the Test Set order.
    # The submission script iterates fold 0..4 test_logits.npy.
    # Does it assume they are aligned? Yes, make_submission does acc += fold_k.
    # This implies ALL folds must have SAME ordering of test files.
    # Let's verify that fold0 test_ids == fold1 test_ids == ... == fold4 test_ids
    
    ref_ids = None
    
    for k in range(5):
        f_dir = os.path.join(folds_dir, f"fold{k}")
        ids_path = os.path.join(f_dir, "test_file_ids.npy")
        
        if not os.path.exists(ids_path):
            print(f"Error: {ids_path} missing.")
            return False
            
        ids = np.load(ids_path)
        
        if ref_ids is None:
            ref_ids = ids
            print(f"Fold 0 Test Size: {len(ids)}")
        else:
            if not np.array_equal(ref_ids, ids):
                print(f"Error: Fold {k} IDs do not match Fold 0!")
                return False
                
    print("PASS: Fold 0-4 Test IDs Consistency")
    
    # 2. Check Submission Alignment
    sub_path = os.path.join(folds_dir, "submission.npy")
    if not os.path.exists(sub_path):
        print(f"Error: Submission {sub_path} missing.")
        return False
        
    sub = np.load(sub_path)
    print(f"Submission Shape: {sub.shape}")
    
    if len(ref_ids) != sub.shape[0]:
        print(f"Error: Mismatch! IDs={len(ref_ids)}, Sub={sub.shape[0]}")
        return False
        
    print("PASS: Submission Count Alignment")
    
    # 3. Check Uniqueness
    if len(np.unique(ref_ids)) != len(ref_ids):
        print("Error: Duplicate IDs found in test set!")
        return False
        
    print(f"PASS: Uniqueness ({len(ref_ids)} unique files)")
    
    # 4. Check Count
    if len(ref_ids) != 654:
        print(f"Warning: Expected 654 files, found {len(ref_ids)}.")
        if len(ref_ids) == 0: return False
    else:
        print("PASS: Exact Count (654)")

    print("--- Integrity Check Passed ---")
    return True

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", type=str, default="/root/datasets/NYUv2")
    args = p.parse_args()
    success = check_integrity(args.exp_dir)
    sys.exit(0 if success else 1)
