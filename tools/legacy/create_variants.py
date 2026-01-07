
import numpy as np
import os
import zipfile
import shutil

def load_and_fix(path):
    print(f"Loading {os.path.basename(path)}...")
    l = np.load(path)
    if l.ndim == 4:
        # Fix HWC -> CHW if needed
        # (N, H, W, C) -> (N, C, H, W)
        if l.shape[-1] == 13 and l.shape[1] != 13:
            print(f"  Fixing shape {l.shape} -> CHW")
            l = l.transpose(0, 3, 1, 2)
    return l.astype(np.float32)

def save_submission(logits, out_path):
    print(f"Generating {out_path}...")
    preds = np.argmax(logits, axis=1).astype(np.uint8)
    np.save(out_path, preds)

def create_zip(submission_path, zip_path, weights_dir):
    print(f"Creating ZIP: {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(submission_path, "submission.npy")
        # Weights excluded per user request

def generate_variant(name, files, weights=None, out_npy="submission.npy", out_zip="submission.zip"):
    print(f"\n--- Processing Variant {name} ---")
    if not files:
        print("No files provided.")
        return

    # Check existence
    for f in files:
        if not os.path.exists(f):
            print(f"Skipping {name}: {f} not found.")
            return

    # Initialize with first fold
    print(f"  Loading {os.path.basename(files[0])}...")
    acc = load_and_fix(files[0])
    
    w0 = weights[0] if weights else 1.0
    if weights:
        acc = acc * w0
    
    # Accumulate others
    for i in range(1, len(files)):
        print(f"  Adding {os.path.basename(files[i])}...")
        curr = load_and_fix(files[i])
        w = weights[i] if weights else 1.0
        acc += (curr * w)
        del curr # Free memory

    # Average
    if not weights:
         acc /= float(len(files))
    
    save_submission(acc, out_npy)
    # create_zip now excludes weights per user request
    create_zip(out_npy, out_zip, "data/output/nearest_final") # weights_dir unused mostly
    del acc # Free memory

def main():
    base_dir = "data/output/nearest_final"
    
    # Files
    files_prot = [os.path.join(base_dir, f"test_logits_fold{k}.npy") for k in range(5)]
    files_noprot = [os.path.join(base_dir, f"test_logits_fold{k}_noprot.npy") for k in range(5)]
    
    # Variant A: Standard Mean
    generate_variant("A (Standard)", files_prot, weights=None, out_npy="submission_A.npy", out_zip="submission_A.zip")
    
    # Variant B: Weighted
    # Weights: Fold0=0.30, Others=0.175
    w_B = [0.30, 0.175, 0.175, 0.175, 0.175]
    generate_variant("B (Weighted)", files_prot, weights=w_B, out_npy="submission_B.npy", out_zip="submission_B.zip")

    # Variant C: No Protection
    generate_variant("C (NoProt)", files_noprot, weights=None, out_npy="submission_C.npy", out_zip="submission_C.zip")

if __name__ == "__main__":
    main()
