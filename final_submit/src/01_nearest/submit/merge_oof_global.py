
import os
import argparse
import numpy as np
import json
from tqdm import tqdm

def merge_oof(exp_dir):
    folds_dir = os.path.join(exp_dir, "00_data/output/exp100_final_01_nearest")
    out_dir = os.path.join(exp_dir, "01_nearest/golden_artifacts/oof")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Collect all file info first
    all_tasks = []
    print("Scanning files...")
    
    total_imgs = 0
    ref_shape = None
    
    for k in range(5):
        f_dir = os.path.join(folds_dir, f"fold{k}")
        vp_dir = os.path.join(f_dir, "valid_preds")
        meta_path = os.path.join(f_dir, "valid_meta.json")
        
        if not os.path.exists(meta_path):
             print(f"Meta not found for fold {k}")
             continue
             
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
        for item in meta:
            fid = item["file_id"]
            # File ID is "xxxxx.png"
            l_path = os.path.join(vp_dir, f"{fid}_logits.npy")
            if not os.path.exists(l_path):
                # Try mismatch name?
                l_path = os.path.join(vp_dir, f"{fid.replace('.png','')}_logits.npy")
                if not os.path.exists(l_path):
                     print(f"Missing: {l_path}")
                     continue
            
            # Check shape of first one
            if ref_shape is None:
                l = np.load(l_path, mmap_mode='r')
                ref_shape = l.shape # (C, H, W)
                
            all_tasks.append((fid, l_path))
            
    total_imgs = len(all_tasks)
    if total_imgs == 0:
        print("No files found!")
        return

    print(f"Found {total_imgs} files. Shape: {ref_shape}")
    
    # 2. Create Memmap Output
    # Shape: (N, C, H, W)
    out_shape = (total_imgs,) + ref_shape
    out_path = os.path.join(out_dir, "oof_logits.npy")
    ids_path = os.path.join(out_dir, "oof_file_ids.npy")
    
    print(f"Allocating memmap at {out_path} {out_shape}...")
    # 'w+' creates or overwrites
    fp = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float32, shape=out_shape)
    
    all_ids = []
    
    # 3. Fill
    print("Filling data...")
    for i, (fid, l_path) in enumerate(tqdm(all_tasks)):
        # Load and write directly
        l = np.load(l_path) # Load to RAM (one image is fine)
        if l.shape != ref_shape:
            # Resize? Assuming consistency for now.
            pass
        fp[i] = l
        all_ids.append(fid)
        
    fp.flush()
    # Explicitly delete to close?
    del fp
    
    print("Saving IDs...")
    np.save(ids_path, np.array(all_ids))
    
    # Save meta
    meta = {
        "num_samples": total_imgs,
        "source": folds_dir
    }
    with open(os.path.join(exp_dir, "01_nearest/golden_artifacts/meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
        
    print("Done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str, default="/root/datasets/NYUv2")
    args = p.parse_args()
    merge_oof(args.root_dir)
