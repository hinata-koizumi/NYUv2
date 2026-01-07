
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
