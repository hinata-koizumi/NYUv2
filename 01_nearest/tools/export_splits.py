"""
Export authoritative splits from nearest_final logic to ensemble_lab/splits.
Ensures compatible splits across all ensemble models.
"""

import os
import json
import numpy as np
import argparse
from datetime import datetime

# Import nearest_final modules
# Assumes running from project root (NYUv2)
from 01_nearest.configs.base_config import Config
from 01_nearest.data.fold_utils import get_split_files

def main():
    parser = argparse.ArgumentParser(description="Export Shared Splits")
    parser.add_argument("--out_dir", type=str, default="03_ensemble/splits", help="Output directory")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Load Default Config (Frozen)
    cfg = Config()
    
    # Metadata
    split_id = f"group_block{cfg.SPLIT_BLOCK_SIZE}_v1"
    created_at = datetime.now().strftime("%Y-%m-%d")
    
    manifest = {
        "split_id": split_id,
        "split_mode": cfg.SPLIT_MODE,
        "block_size": cfg.SPLIT_BLOCK_SIZE,
        "n_folds": cfg.N_FOLDS,
        "seed": cfg.SEED,
        "created_at": created_at,
        "created_by": "_01_nearest_final/tools/export_splits.py",
        "folds": []
    }

    print(f"Exporting splits to {out_dir}...")
    print(f"Config: Mode={cfg.SPLIT_MODE}, Block={cfg.SPLIT_BLOCK_SIZE}, Seed={cfg.SEED}")

    for fold_idx in range(cfg.N_FOLDS):
        # Use existing logic from fold_utils
        train_ids, val_ids = get_split_files(cfg, fold_idx)
        
        # Enforce basename (just in case)
        train_ids = [os.path.splitext(os.path.basename(f))[0] for f in train_ids]
        val_ids = [os.path.splitext(os.path.basename(f))[0] for f in val_ids]
        
        # Verify no overlap
        assert set(train_ids).isdisjoint(set(val_ids)), f"Fold {fold_idx} has overlapping IDs!"
        
        # Sort for determinism
        # train_ids.sort() # fold_utils might sort, but we respect the logic's order if possible. 
        # Actually strict sorting is better for JSON diffs.
        train_ids = sorted(train_ids)
        val_ids = sorted(val_ids)

        split_data = {
            "split_id": split_id,
            "fold": fold_idx,
            "num_train": len(train_ids),
            "num_val": len(val_ids),
            "train_ids": train_ids,
            "val_ids": val_ids,
            "created_by": "01_nearest/tools/export_splits.py",
            "created_at": created_at,
            "notes": f"Generated from nearest_final {cfg.SPLIT_MODE} logic"
        }

        chk_name = f"fold{fold_idx}.json"
        chk_path = os.path.join(out_dir, chk_name)
        
        with open(chk_path, "w") as f:
            json.dump(split_data, f, indent=2)
            
        print(f"Saved {chk_name}: Train={len(train_ids)}, Val={len(val_ids)}")
        manifest["folds"].append(chk_name)

    # Save Manifest
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {manifest_path}")

if __name__ == "__main__":
    main()
