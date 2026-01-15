"""
verify_frozen_state.py
Verifies the integrity of the 02_nonstruct_v0_frozen bundle.
Checks:
- Manifest exists
- All files in manifest match SHA256
"""

import os
import hashlib
import sys

# Assume script run from root or bundle root?
# Let's assume we point it to the bundle dir.
BUNDLE_ROOT = "/root/datasets/NYUv2/02_nonstruct/02_nonstruct_v0_frozen"
MANIFEST_FILE = os.path.join(BUNDLE_ROOT, "golden_artifacts/sha256_manifest.txt")

def calculate_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    if not os.path.exists(MANIFEST_FILE):
        print(f"[ERROR] Manifest not found: {MANIFEST_FILE}")
        sys.exit(1)
        
    print(f"Verifying Frozen Bundle: {BUNDLE_ROOT}")
    print(f"Manifest: {MANIFEST_FILE}")
    
    expected_hashes = {}
    with open(MANIFEST_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2: continue
            expected_hashes[parts[1]] = parts[0]
            
    print("-" * 80)
    print(f"{'File':<50} | {'Status':<10}")
    print("-" * 80)
    
    all_passed = True
    
    for rel_path, expected_hash in expected_hashes.items():
        # Manifest paths are relative to golden_artifacts?
        # My manifest generation used "oof/oof_fold0.npy"
        # So full path is BUNDLE_ROOT/golden_artifacts/oof/oof_fold0.npy
        
        full_path = os.path.join(BUNDLE_ROOT, "golden_artifacts", rel_path)
        
        if not os.path.exists(full_path):
            print(f"{rel_path:<50} | [MISSING]")
            all_passed = False
            continue
            
        current_hash = calculate_sha256(full_path)
        
        if current_hash == expected_hash:
            print(f"{rel_path:<50} | [OK]")
        else:
            print(f"{rel_path:<50} | [FAIL]")
            print(f"  Expected: {expected_hash}")
            print(f"  Got:      {current_hash}")
            all_passed = False
            
    print("-" * 80)
    
    # Extra Check: Splits & IDs existence
    # We might not have hashes for them in the logic yet, but check existence.
    extras = [
        "ids/train_ids.txt",
        "ids/test_ids.txt",
        "splits/folds_v1.json",
        "recipe/RECIPE.lock.md",
        "ckpts/best_fold0.pth"
    ]
    
    for e in extras:
        if os.path.exists(os.path.join(BUNDLE_ROOT, e)):
            print(f"{e:<50} | [OK] (Exists)")
        else:
            print(f"{e:<50} | [MISSING]")
            # soft fail?
    
    if all_passed:
        print("\n✅ Bundle Integrity Verified.")
        sys.exit(0)
    else:
        print("\n❌ Bundle Verification FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
