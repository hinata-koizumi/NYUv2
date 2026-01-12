
import os
import hashlib
import numpy as np
import json
import glob

def check_ids(name, ids_path, img_dir):
    print(f"Checking {name}...")
    if not os.path.exists(ids_path):
        print(f"FAILED: {ids_path} does not exist.")
        return False
    
    with open(ids_path, 'r') as f:
        ids = [line.strip() for line in f if line.strip()]
    
    # Check duplicates
    if len(ids) != len(set(ids)):
        print(f"FAILED: Duplicate IDs found in {ids_path}")
        return False
    
    # Check existence
    # Note: IDs in file include extensions based on previous step
    missing = []
    case_issues = []
    
    # Get actual files map (lowercase -> realname) for case check
    if not os.path.exists(img_dir):
        # Fallback relative if full path not found (user env specific)
        if img_dir.startswith("/Users/koizumihinata/NYUv2/"):
             rel = img_dir.replace("/Users/koizumihinata/NYUv2/", "")
             if os.path.exists(rel):
                 img_dir = rel
             else:
                  print(f"FAILED: Image dir {img_dir} not found")
                  return False

    real_files = set(os.listdir(img_dir))
    
    for i in ids:
        if i not in real_files:
            missing.append(i)
    
    if missing:
        print(f"FAILED: {len(missing)} IDs not found in {img_dir}. First 3: {missing[:3]}")
        return False
        
    print(f"PASSED: {name} ({len(ids)} items) matches {img_dir}")
    return True

def check_golden():
    print("Checking Golden Artifacts...")
    path = "01_nearest/golden_artifacts/test_logits.npy"
    if not os.path.exists(path):
        print(f"WARNING: {path} not found. Skipping shape check.")
        return
    
    try:
        data = np.load(path)
        print(f"Golden Shape: {data.shape}")
        print(f"Golden Dtype: {data.dtype}")
        
        expected_shape = (654, 13, 480, 640)
        if data.shape != expected_shape:
            print(f"WARNING: Shape mismatch! Expected {expected_shape}, got {data.shape}")
        
        if data.dtype != np.float16:
            print(f"WARNING: Dtype mismatch! Expected float16, got {data.dtype}")
            
    except Exception as e:
        print(f"FAILED to load golden artifact: {e}")

def hash_file(path):
    sha = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(65536)
            if not chunk: break
            sha.update(chunk)
    return sha.hexdigest()

def hardening_splits():
    print("Hardening Splits...")
    split_dir = "00_data/splits"
    manifest_path = os.path.join(split_dir, "manifest.json")
    
    if os.path.exists(manifest_path):
        # Rename to v1
        v1_path = os.path.join(split_dir, "folds_v1.json")
        if not os.path.exists(v1_path):
            os.rename(manifest_path, v1_path)
            print(f"Renamed manifest.json to {v1_path}")
        else:
            print(f"{v1_path} already exists.")
            
        # Calc hash
        h = hash_file(v1_path)
        print(f"Split Manifest SHA256: {h}")
        
        # Save hash
        with open(os.path.join(split_dir, "sha256_manifest.txt"), "w") as f:
            f.write(f"{h}  folds_v1.json\n")
    else:
        print("manifest.json not found in splits dir.")

def main():
    root = os.getcwd()
    
    # 1. IDs
    check_ids("Train IDs", "00_data/ids/train_ids.txt", "00_data/train/image")
    check_ids("Test IDs", "00_data/ids/test_ids.txt", "00_data/test/image")
    
    # 2. Golden
    check_golden()
    
    # 3. Splits
    hardening_splits()

if __name__ == "__main__":
    main()
