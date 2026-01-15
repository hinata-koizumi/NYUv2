
import os
import hashlib
import glob

def calculate_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def generate_manifest(root_dir):
    artifacts_dir = os.path.join(root_dir, "01_nearest/golden_artifacts")
    manifest_path = os.path.join(artifacts_dir, "sha256_manifest.txt")
    
    print(f"Generating manifest for {artifacts_dir}...")
    
    # Files to hash (Explicit list from freeze requirement)
    # oof_logits.npy
    # oof_file_ids.npy
    # test_logits.npy
    # test_file_ids.npy
    # meta.json
    # metrics.json
    # final_recipe.json
    
    target_files = [
        "oof_logits.npy",
        "oof_file_ids.npy",
        "test_logits.npy",
        "test_file_ids.npy",
        "meta.json",
        "metrics.json",
        "final_recipe.json"
    ]
    
    lines = []
    for fname in target_files:
        fpath = os.path.join(artifacts_dir, fname)
        if os.path.exists(fpath):
            print(f"Hashing {fname}...")
            h = calculate_sha256(fpath)
            lines.append(f"{h}  {fname}")
        else:
            print(f"WARN: {fname} missing!")
            
    with open(manifest_path, "w") as f:
        f.write("\n".join(lines) + "\n")
        
    print(f"Manifest written to {manifest_path}")

if __name__ == "__main__":
    generate_manifest("/root/datasets/NYUv2")
