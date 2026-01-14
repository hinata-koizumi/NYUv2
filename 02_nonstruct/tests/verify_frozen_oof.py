"""
verify_frozen_oof.py
Checks the integrity of frozen artifacts against the SHA256 manifest.
"""

import os
import hashlib
import sys

# Paths
ROOT_DIR = "/root/datasets/NYUv2/02_nonstruct"
GOLDEN_DIR = os.path.join(ROOT_DIR, "golden_artifacts")
MANIFEST_FILE = os.path.join(GOLDEN_DIR, "manifests/sha256_manifest.txt")

def calculate_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    if not os.path.exists(MANIFEST_FILE):
        print(f"[ERROR] Manifest file not found: {MANIFEST_FILE}")
        sys.exit(1)
        
    print(f"Reading manifest: {MANIFEST_FILE}")
    
    expected_hashes = {}
    with open(MANIFEST_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2: continue
            expected_hashes[parts[1]] = parts[0]
            
    all_passed = True
    
    print("-" * 60)
    print(f"{'File':<40} | {'Status':<10}")
    print("-" * 60)
    
    for rel_path, expected_hash in expected_hashes.items():
        full_path = os.path.join(GOLDEN_DIR, rel_path)
        
        if not os.path.exists(full_path):
            print(f"{rel_path:<40} | [MISSING]")
            all_passed = False
            continue
            
        current_hash = calculate_sha256(full_path)
        
        if current_hash == expected_hash:
            print(f"{rel_path:<40} | [OK]")
        else:
            print(f"{rel_path:<40} | [FAIL]")
            print(f"  Expected: {expected_hash}")
            print(f"  Got:      {current_hash}")
            all_passed = False
            
    print("-" * 60)
    
    if all_passed:
        print("--- FROZEN OOF VERIFIED ---")
        sys.exit(0)
    else:
        print("--- VERIFICATION FAILED ---")
        sys.exit(1)

if __name__ == "__main__":
    main()
