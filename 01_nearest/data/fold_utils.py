
import os
import numpy as np
from sklearn.model_selection import KFold

def get_split_files(cfg, fold_idx: int):
    """
    Returns (train_file_ids, val_file_ids) for a given fold.
    Currently implements Random KFold (Stage 0 Baseline).
    Future: Will support GroupKFold (Blocking/Clustering).
    """
    img_dir = os.path.join(cfg.TRAIN_DIR, "image")
    # Sort to ensure potential temporal order if ids are numeric
    all_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    x_img = np.array(all_files)
    
    # Determine Split Strategy from Config
    split_mode = getattr(cfg, "SPLIT_MODE", "kfold")
    
    if split_mode == "group":
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=cfg.N_FOLDS)
        
        # Strategy: Temporal Blocking (Stage 1)
        # Assuming sorted file_ids correspond to temporal order.
        # Group every K frames into a block.
        block_size = int(getattr(cfg, "SPLIT_BLOCK_SIZE", 50))
        num_files = len(x_img)
        groups = np.arange(num_files) // block_size
        
        print(f"Using GroupKFold (Block={block_size}). Total Groups: {len(np.unique(groups))}")
        splits = list(gkf.split(x_img, groups=groups))
        
    elif split_mode == "hard":
        # S2 Hard Split (Clustering-based)
        manifest_path = "/root/datasets/NYUv2/00_data/hard_split_manifest.json"
        
        # Check if manifest exists
        if not os.path.exists(manifest_path):
             # Fallback to local path if running from root
             manifest_path = "00_data/hard_split_manifest.json"
             
        if not os.path.exists(manifest_path):
             raise FileNotFoundError(f"Hard split manifest not found at {manifest_path}")
             
        import json
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            
        # Manifest format: {"fold0": [ids...], "fold1": ...}
        fold_key = f"fold{fold_idx}"
        if fold_key not in manifest:
            raise ValueError(f"Fold {fold_key} not found in hard split manifest")
            
        val_ids = manifest[fold_key]
        val_set = set(val_ids)
        
        # Train IDs are all IDs NOT in Val IDs
        # We need the full list of files to derive train_ids
        train_ids = [fid for f in x_img if (fid := os.path.splitext(f)[0]) not in val_set]
        
        # Sort for determinism
        train_ids.sort()
        val_ids.sort()
        
        # Update splits info (hacky for return structure compatibility if needed, but we return directly)
        return train_ids, val_ids

    else:
        # Random KFold (Stage 0 Baseline)
        # Note: Seed must match training config to reproduce the same split!
        kf = KFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
        splits = list(kf.split(x_img))
    
    if fold_idx < 0 or fold_idx >= len(splits):
        raise ValueError(f"Fold {fold_idx} out of range (0-{len(splits)-1})")
        
    tr_idx, va_idx = splits[fold_idx]
    
    train_files = x_img[tr_idx]
    val_files = x_img[va_idx]
    
    # Remove extensions to return IDs
    train_ids = [os.path.splitext(f)[0] for f in train_files]
    val_ids = [os.path.splitext(f)[0] for f in val_files]
    
    return train_ids, val_ids
