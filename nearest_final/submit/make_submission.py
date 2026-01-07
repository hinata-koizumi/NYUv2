
import numpy as np
import os
import zipfile
import argparse

def search_files(base_dir):
    files = []
    for k in range(5):
        # Target: fold{k}/test_logits.npy
        path = os.path.join(base_dir, f"fold{k}", "test_logits.npy")
        if os.path.exists(path):
            files.append(path)
        else:
            print(f"Warning: {path} not found.")
    return files

def load_and_fix(path):
    print(f"Loading {path}...")
    l = np.load(path)
    # Ensure CHW float32
    if l.ndim == 4 and l.shape[-1] == 13:
         l = l.transpose(0, 3, 1, 2)
    return l.astype(np.float32)

def run_make_submission(exp_dir: str, output_name: str = "submission"):
    print("--- Generating Robust Submission ---")
    files = search_files(exp_dir)
    if len(files) != 5:
        print(f"Error: Expected 5 files, found {len(files)}.")
        return

    # Accumulate
    print(f"Accumulating {files[0]}...")
    acc = load_and_fix(files[0])
    
    for i in range(1, 5):
        print(f"Accumulating {files[i]}...")
        curr = load_and_fix(files[i])
        acc += curr
        del curr
        
    acc /= 5.0
    
    print("Argmaxing...")
    pred = np.argmax(acc, axis=1).astype(np.uint8)
    
    npy_path = os.path.join(exp_dir, f"{output_name}.npy")
    zip_path = os.path.join(exp_dir, f"{output_name}.zip")
    
    print(f"Saving {npy_path}...")
    np.save(npy_path, pred)
    
    print(f"Zipping {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(npy_path, os.path.basename(npy_path))
        
    print("Done. Ready for LB submission.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", type=str, default="data/output/nearest_final")
    p.add_argument("--output_name", type=str, default="submission")
    args = p.parse_args()
    
    run_make_submission(args.exp_dir, args.output_name)

if __name__ == "__main__":
    main()
