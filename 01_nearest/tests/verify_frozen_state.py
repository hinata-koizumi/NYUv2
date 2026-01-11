
import os
import sys
import json
import numpy as np
import hashlib

def verify_freeze(root_dir):
    print("--- Verifying Frozen State ---")
    
    base = os.path.join(root_dir, "01_nearest")
    ga = os.path.join(base, "golden_artifacts")
    
    # 1. Lock Check
    l1 = os.path.join(ga, "FROZEN.lock")
    l2 = os.path.join(base, "FROZEN.lock")
    
    if not os.path.exists(l1):
        print("FAIL: golden_artifacts/FROZEN.lock missing!")
        return False
    if not os.path.exists(l2):
        print("WARN: Root FROZEN.lock alias missing (minor).")
    
    print("PASS: Locks exist.")
    
    # 2. Manifest Check
    man_path = os.path.join(ga, "sha256_manifest.txt")
    if not os.path.exists(man_path):
        print("FAIL: Manifest missing.")
        return False
        
    with open(man_path, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        if not line.strip(): continue
        expected_hash, fname = line.strip().split("  ")
        fpath = os.path.join(ga, fname)
        
        if not os.path.exists(fpath):
            print(f"FAIL: Artifact {fname} missing!")
            return False
            
        # Verify Hash (Skip huge files if slow? No, Verify means Verify)
        # But 10GB hash takes time. Let's do it.
        print(f"Verifying {fname}...")
        sha256_hash = hashlib.sha256()
        with open(fpath, "rb") as f:
             for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        current_hash = sha256_hash.hexdigest()
        
        if current_hash != expected_hash:
            print(f"FAIL: Hash mismatch for {fname}!")
            return False
            
    print("PASS: Manifest Integrity.")
    
    # 3. Logical Integrity
    print("Checking Logical Integrity...")
    try:
        oof_ids = np.load(os.path.join(ga, "oof_file_ids.npy"))
        oof_logits = np.load(os.path.join(ga, "oof_logits.npy"), mmap_mode='r')
        test_ids = np.load(os.path.join(ga, "test_file_ids.npy"))
        test_logits = np.load(os.path.join(ga, "test_logits.npy"), mmap_mode='r')
        
        if len(oof_ids) != 795:
            print(f"FAIL: OOF count {len(oof_ids)} != 795")
            return False
        if len(test_ids) != 654:
            print(f"FAIL: Test count {len(test_ids)} != 654")
            return False
            
        if oof_logits.shape[0] != 795:
            print(f"FAIL: OOF Logits shape {oof_logits.shape}")
            return False
        if test_logits.shape[0] != 654:
            print(f"FAIL: Test Logits shape {test_logits.shape}")
            return False
            
        # Unique check
        if len(np.unique(oof_ids)) != 795:
             print("FAIL: Duplicate OOF IDs")
             return False
        if len(np.unique(test_ids)) != 654:
             print("FAIL: Duplicate Test IDs")
             return False
             
    except Exception as e:
        print(f"FAIL: Logical check crashed: {e}")
        return False
        
    print("PASS: Logical Integrity.")
    print("--- FROZEN STATE VERIFIED ---")
    return True

if __name__ == "__main__":
    success = verify_freeze("/root/datasets/NYUv2")
    sys.exit(0 if success else 1)
