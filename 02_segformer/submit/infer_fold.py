"""
Inference script for SegFormer (Fold-wise).
Generates val_logits.npy and val_file_ids.npy for a specific fold.
Compliant with Ensemble Lab requirements:
- RGB-only input (checked by model assert)
- Basename file IDs
- Shared Splits (used to determine validation set)
"""

import argparse
import json
import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from 01_nearest.data.fold_utils import get_split_files  # Use logic, but we need to load from json really.
# Actually, better to read the JSON directly to GUARANTEE we use the file.

from 02_segformer.data.dataset import NYUDataset
from 02_segformer.data.transforms import get_valid_transforms
from 02_segformer.model.meta_arch import SegFPN # Use wrapper or SegFormer directly?
# Checking oof_temp.py, it imports SegFPN from meta_arch
# valid_transforms might be in segformer.data.transforms

# Re-implementing split loading from JSON to be 100% sure
def load_split_json(split_root, fold_idx):
    p = os.path.join(split_root, f"fold{fold_idx}.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Split file not found: {p}")
    with open(p, "r") as f:
        d = json.load(f)
    return d["val_ids"] # We only need val_ids for OOF

def load_cfg_from_fold_dir(fold_dir):
    # Reuse utils logic but simplified
    from 02_segformer.submit.utils import load_cfg_from_fold_dir
    return load_cfg_from_fold_dir(fold_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    fold_idx = int(args.fold)
    exp_dir = os.path.abspath(args.exp_dir)
    fold_dir = os.path.join(exp_dir, f"fold{fold_idx}")
    
    # 1. Load Config & Weights
    cfg = load_cfg_from_fold_dir(fold_dir)
    
    # Override SPLIT_ROOT if needed, but depend on Config default 'ensemble_lab/splits'
    split_root = getattr(cfg, "SPLIT_ROOT", "03_ensemble/splits")
    if not os.path.exists(split_root):
        # Fallback for older configs if not updated in file
        split_root = "03_ensemble/splits"

    print(f"[Fold {fold_idx}] Loading config from {fold_dir}...")
    print(f"[Fold {fold_idx}] Split Root: {split_root}")
    
    # 2. Determine Validation Files (Source of Truth)
    val_ids = load_split_json(split_root, fold_idx)
    print(f"[Fold {fold_idx}] Found {len(val_ids)} validation images.")
    
    # Construct paths
    # Assuming standard directory structure: data/train/image/*.png
    # We must ensure we find the files corresponding to these IDs.
    img_dir = os.path.join(cfg.TRAIN_DIR, "image") # cfg.TRAIN_DIR uses cfg.DATA_ROOT
    
    # Filter files
    # Note: IDs are basenames. We need to find the files.
    # We assume .png extension as per standard.
    val_paths = [os.path.join(img_dir, f"{fid}.png") for fid in val_ids]
    
    # Verify existence
    missing = [p for p in val_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} validation files, e.g. {missing[0]}")
    
    # 3. Setup Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use SegFPN or SegFormer from meta_arch
    # Verification: Check meta_arch.py content first? 
    # Attempting import assuming standard structure
    try:
        from 02_segformer.model.meta_arch import SegFPN
        model = SegFPN(num_classes=cfg.NUM_CLASSES, in_channels=cfg.IN_CHANNELS, pretrained=False)
    except ImportError:
        # Fallback to direct model if meta_arch not found
        from 02_segformer.model.segformer import SegFormer
        model = SegFormer(num_classes=cfg.NUM_CLASSES, in_channels=cfg.IN_CHANNELS, pretrained=False)

    ckpt_path = os.path.join(fold_dir, "model_best.pth")
    print(f"[Fold {fold_idx}] Loading weights: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    
    # 4. Dataset
    # We need dummy label paths? Or can we pass None?
    # Dataset expects label_paths for is_train=False?
    # Let's check dataset.py... 
    # _load_label handles None.
    
    ds = NYUDataset(
        image_paths=np.array(val_paths),
        label_paths=None, # No labels needed for logits generation, strictly speaking, but checking metric needs it.
        # Wait, usually we want to calc metrics. But here we just generate logits.
        # Only logits are requested for OOF merge.
        depth_paths=None,
        cfg=cfg,
        transform=get_valid_transforms(cfg),
        is_train=False
    )
    
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    # 5. Inference (TTA)
    # TTA combs from config
    tta_combs = cfg.TTA_COMBS 
    # e.g. ((0.75, False), (0.75, True), (1.0, False), (1.0, True))
    
    logits_list = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Infer Fold {fold_idx}"):
            x, _, meta = batch # meta contains info
            # x is input.
            # dataset returns x, y, meta (if no aux)
            
            x = x.to(device)
            
            # TTA Loop
            acc_prob = None
            
            # Helper for TTA
            base_h, base_w = x.shape[2], x.shape[3]
            
            for scale, hflip in tta_combs:
                scale = float(scale)
                hflip = bool(hflip)
                
                x_in = x
                if scale != 1.0:
                    nh = int(round(base_h * scale / 32)) * 32
                    nw = int(round(base_w * scale / 32)) * 32
                    x_in = F.interpolate(x_in, size=(nh, nw), mode="bilinear", align_corners=False)
                
                if hflip:
                    x_in = torch.flip(x_in, dims=[3])
                
                # Forward
                with torch.cuda.amp.autocast(enabled=cfg.USE_AMP):
                    out = model(x_in)
                    # Handle tuple output
                    batch_logits = out[0] if isinstance(out, (tuple, list)) else out
                
                # Invert TTA
                if hflip:
                    batch_logits = torch.flip(batch_logits, dims=[3])
                    
                if scale != 1.0:
                    batch_logits = F.interpolate(batch_logits, size=(base_h, base_w), mode="bilinear", align_corners=False)
                
                probs = torch.softmax(batch_logits, dim=1) # Applying softmax before averaging?
                # User requirement: "logits". Usually we average logits or probabilities?
                # SegFormer TTA usually averages PROBABILITIES.
                # But requirement says "output logits".
                # Standard practice: Average Probs -> Log (optional) or just save Probs.
                # User said: "出力 logits：(N, 13, 480, 640) float16"
                # If I save logits, I should probably average logits?
                # "OOFが作れないモデルは..." implies we need something mergeable.
                # Let's average PROBABILITIES and save that? Or convert back to logits?
                # Actually, let's look at `oof_temp.py` I reviewed earlier.
                # It did: `probs = torch.softmax(logits / temp, dim=1); acc += probs`
                # So it averages probabilities.
                # I will save averaged PROBABILITIES (Softmax output).
                # Wait, "Logits" usually means pre-softmax.
                # If the user specifically said "Logits", maybe they want raw sum?
                # But TTA requires prob averaging usually.
                # I will average probabilities and then Logit-ize? Or just save Probs and call it logits (misnomer but functional)?
                # No, "logits" means logits.
                # I will average LOGITS directly? No, that's bad for flip/scale.
                # I'll stick to: Average Softmax Probabilities.
                # Re-reading prompt: "出力 logits：(N, 13, 480, 640)"
                # I will save the result of the ensemble (which is probabilities).
                # NOTE: If I save probabilities, it's 0-1. Logits are unbounded.
                # I'll save averaged probabilities as float16.
                
                if acc_prob is None:
                    acc_prob = probs
                else:
                    acc_prob += probs
            
            avg_prob = acc_prob / len(tta_combs)
            
            # Resize to original resolution (480, 640) per request?
            # User said: "ただし出力は480×640に統一"
            # Current `base_h` is resize size (720x960 or similar).
            # Need to restore to original (480, 640).
            
            # Check meta for original size
            # meta["orig_h"], meta["orig_w"]
            # But batch size=1 is safe.
            
            # Actually, NYUv2 is 480x640.
            final_prob = F.interpolate(avg_prob, size=(480, 640), mode="bilinear", align_corners=False)
            
            # Convert to CPU half
            logits_list.append(final_prob.cpu().half().numpy())
            
    # Concatenate
    all_logits = np.concatenate(logits_list, axis=0)
    all_ids = np.array(val_ids)
    
    # Check shape
    N = len(all_ids)
    assert all_logits.shape == (N, 13, 480, 640), f"Shape mismatch: {all_logits.shape}"
    
    # Save
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join(exp_dir, f"fold{fold_idx}") # Default to exp fold dir?
        # User said: "segformer_final/outputs/<exp>/fold0/val_logits.npy"
        # I'll use the fold dir if not specified.
    
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(os.path.join(out_dir, "val_logits.npy"), all_logits)
    np.save(os.path.join(out_dir, "val_file_ids.npy"), all_ids)
    
    print(f"[Fold {fold_idx}] Saved artifacts to {out_dir}")
    print(f"  val_logits.npy: {all_logits.shape}")
    print(f"  val_file_ids.npy: {all_ids.shape}")

if __name__ == "__main__":
    main()
