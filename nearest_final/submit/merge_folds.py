
import numpy as np
import os
import glob
from tqdm import tqdm

def main():
    exp_dir = "data/output/nearest_final"
    out_file = "submission.npy"
    
    # We expect 5 files named test_logits_fold0.npy, test_logits_fold1.npy, ...
    # Each shape: (N, 13, H, W) float16
    
    logits_files = [os.path.join(exp_dir, f"test_logits_fold{k}.npy") for k in range(5)]
    
    # Check all files exist
    for f in logits_files:
        if not os.path.exists(f):
            print(f"Error: Logit file for fold {f} not found!")
            return

    print("Loading logits...")
    print("Loading logits...")
    
    def _load_and_fix(path):
        l = np.load(path)
        if l.ndim == 4:
            # Check for HWC (channels last) vs CHW (channels first)
            # If shape[-1] is 13, it's likely (N, H, W, C)
            if l.shape[-1] == 13 and l.shape[1] != 13:
                print(f"  Fixing shape {os.path.basename(path)} {l.shape} -> (N, C, H, W)")
                l = l.transpose(0, 3, 1, 2)
        return l.astype(np.float32)

    logits_0 = _load_and_fix(logits_files[0])
    acc_logits = logits_0
    
    for i in range(1, 5):
        print(f"Adding fold {i}...")
        l = _load_and_fix(logits_files[i])
        acc_logits += l
        
    # Mean
    acc_logits /= 5.0
    
    print("Computing Argmax...")
    preds = np.argmax(acc_logits, axis=1).astype(np.uint8)
    
    print(f"Saving {out_file} shape={preds.shape} dtype={preds.dtype}...")
    np.save(out_file, preds)
    print("Done.")

if __name__ == "__main__":
    main()
