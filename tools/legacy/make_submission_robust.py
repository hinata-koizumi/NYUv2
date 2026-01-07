
import numpy as np
import os
import zipfile

def search_files():
    base_dir = "data/output/nearest_final"
    files = []
    for k in range(5):
        # Target: data/output/nearest_final/fold{k}/test_logits.npy
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

def main():
    print("--- Generating Robust Submission ---")
    files = search_files()
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
    
    print("Saving submission.npy...")
    np.save("submission.npy", pred)
    
    print("Zipping submission.zip...")
    with zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write("submission.npy")
        
    print("Done. Ready for LB submission.")

if __name__ == "__main__":
    main()
