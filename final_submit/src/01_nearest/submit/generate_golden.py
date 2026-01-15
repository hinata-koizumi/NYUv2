import argparse
import json
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Adjust path to allow imports from project root
sys.path.append(os.getcwd())

from ..data.dataset import NYUDataset
from ..data.transforms import get_valid_transforms
from ..data.fold_utils import get_split_files
from ..engine.inference import Predictor
from ..model.meta_arch import SegFPN
from ..utils.misc import configure_runtime, seed_everything
from ..utils.submit_utils import load_cfg_from_fold_dir, safe_torch_load

def run_inference(cfg, model, dataset, device, tta_combs, save_path_logits, save_path_ids):
    loader = DataLoader(
        dataset, 
        batch_size=max(1, int(cfg.BATCH_SIZE)),
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )

    predictor = Predictor(model, loader, device, cfg)
    
    # Pre-allocate memory (N, C, 480, 640) float16
    N = len(dataset)
    C = cfg.NUM_CLASSES
    H_std, W_std = 480, 640
    
    print(f"Allocating ({N}, {C}, {H_std}, {W_std}) float16...")
    results = np.zeros((N, C, H_std, W_std), dtype=np.float16)
    
    iterator = predictor.predict_logits(tta_combs=tta_combs, return_details=False, return_logits=True)
    
    for i, logits in enumerate(tqdm(iterator, total=N)):
        # Shape check (C, H, W)
        if logits.shape[0] != C:
            if logits.shape[2] == C: logits = logits.transpose(2, 0, 1)
            
        results[i] = logits.astype(np.float16)

    # Save
    print(f"Saving logits to {save_path_logits}...")
    np.save(save_path_logits, results)
    
    # Save IDs if dataset has them
    if hasattr(dataset, "image_paths"):
        # Extract file IDs from paths
        # dataset.image_paths is (N,) numpy array
        ids = [os.path.splitext(os.path.basename(p))[0] for p in dataset.image_paths]
        print(f"Saving IDs to {save_path_ids}...")
        np.save(save_path_ids, np.array(ids))

def run_generate_golden(sanity=False, all_folds=False, fold=0, limit=0, batch_size=None):
    # Load Recipe & Manifest
    recipe_path = "nearest_final/configs/final_recipe.json"
    manifest_path = "nearest_final/configs/ckpt_manifest.json"
    
    with open(recipe_path) as f: recipe = json.load(f)
    with open(manifest_path) as f: manifest = json.load(f)
    
    folds_to_run = [0, 1, 2, 3, 4] if all_folds else [fold]
    if sanity:
        folds_to_run = [0]
        limit = 8
        print("!!! SANITY CHECK MODE: Fold 0, Limit 8 !!!")

    # Output Root
    golden_root = "nearest_final/golden_artifacts"
    os.makedirs(golden_root, exist_ok=True)

    for fold in folds_to_run:
        print(f"\n=== Processing Fold {fold} ===")
        fold_str = str(fold)
        if fold_str not in manifest["folds"]:
            print(f"Fold {fold} not in manifest!")
            continue

        ckpt_path = manifest["folds"][fold_str]["best"]
        print(f"Checkpoint: {ckpt_path}")
        
        # Determine specific Save Dir
        save_dir = os.path.join(golden_root, "folds", f"fold{fold}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Load Config from training directory mechanism (to get model params correct)
        # We assume the ckpt path follows standard structure: .../foldX/model_best.pth
        # So we can infer fold_dir from ckpt_path
        fold_dir = os.path.dirname(ckpt_path)
        cfg = load_cfg_from_fold_dir(fold_dir)
        
        # OVERRIDE Config with Recipe
        # 1. Split Mode
        cfg = cfg.with_overrides(SPLIT_MODE=recipe["SPLIT_MODE"])
        
        # 2. Books Protection
        # Engine checks INFER_TTA_BOOKS_PROTECT, CLASS_ID_BOOKS
        cfg = cfg.with_overrides(
            INFER_TTA_BOOKS_PROTECT=recipe["BOOKS_PROTECT"]["enabled"],
            CLASS_ID_BOOKS=recipe["BOOKS_PROTECT"]["target_class_idx"]
        )
        
        # 3. TTA Combs
        # We pass this explicitly to predict_logits, but good to have in cfg too
        # Config object expects tuples, JSON gives lists. Convert.
        tta_combs = [tuple(x) for x in recipe["TTA_COMBS"]]
        
        # Override Batch Size for GPU optimization
        if batch_size is not None and batch_size > 0:
            print(f"Overriding Batch Size: {cfg.BATCH_SIZE} -> {batch_size}")
            cfg = cfg.with_overrides(BATCH_SIZE=int(batch_size))
        
        # Setup Model
        seed_everything(cfg.SEED)
        configure_runtime(cfg)
        
        model = SegFPN(num_classes=cfg.NUM_CLASSES, in_channels=cfg.IN_CHANNELS, use_depth_aux=bool(getattr(cfg, "USE_DEPTH_AUX", False)))
        state = safe_torch_load(ckpt_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.to(cfg.DEVICE)
        model.eval()
        
        # --- Validation Inference ---
        print("\n--- Validation Inference ---")
        _, val_ids = get_split_files(cfg, fold)
        
        if limit > 0:
            val_ids = val_ids[:limit]
            print(f"Limiting validation to {len(val_ids)} images")

        img_dir = os.path.join(cfg.TRAIN_DIR, "image")
        label_dir = os.path.join(cfg.TRAIN_DIR, "label")
        depth_dir = os.path.join(cfg.TRAIN_DIR, "depth")
        
        val_ds = NYUDataset(
            image_paths=np.array([os.path.join(img_dir, f"{fid}.png") for fid in val_ids]),
            label_paths=np.array([os.path.join(label_dir, f"{fid}.png") for fid in val_ids]),
            depth_paths=np.array([os.path.join(depth_dir, f"{fid}.png") for fid in val_ids]),
            cfg=cfg,
            transform=get_valid_transforms(cfg),
            is_train=False
        )
        
        run_inference(
            cfg, model, val_ds, cfg.DEVICE, tta_combs,
            save_path_logits=os.path.join(save_dir, "val_logits.npy"),
            save_path_ids=os.path.join(save_dir, "val_file_ids.npy")
        )
        
        # --- Test Inference ---
        print("\n--- Test Inference ---")
        test_img_dir = os.path.join(cfg.TEST_DIR, "image")
        test_files = sorted([f for f in os.listdir(test_img_dir) if f.endswith(".png")])
        test_ids = [os.path.splitext(f)[0] for f in test_files]
        
        if limit > 0:
            test_ids = test_ids[:limit]
            test_files = [f"{fid}.png" for fid in test_ids]
            print(f"Limiting test to {len(test_ids)} images")
            
        test_ds = NYUDataset(
            image_paths=np.array([os.path.join(test_img_dir, f) for f in test_files]),
            label_paths=None,
            depth_paths=np.array([os.path.join(cfg.TEST_DIR, "depth", f) for f in test_files]),
            cfg=cfg,
            transform=get_valid_transforms(cfg),
            is_train=False
        )
        
        run_inference(
            cfg, model, test_ds, cfg.DEVICE, tta_combs,
            save_path_logits=os.path.join(save_dir, "test_logits.npy"),
            save_path_ids=os.path.join(save_dir, "test_file_ids.npy")
        )

    print("\nGolden Generation Complete.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sanity", action="store_true", help="Run strictly on first 8 images of fold 0 only")
    p.add_argument("--all_folds", action="store_true", help="Run all 5 folds")
    p.add_argument("--fold", type=int, default=0, help="Run specific fold if not --all_folds")
    p.add_argument("--limit", type=int, default=0, help="Limit number of images per fold (debug)")
    args = p.parse_args()

    run_generate_golden(
        sanity=args.sanity,
        all_folds=args.all_folds,
        fold=args.fold,
        limit=args.limit
    )

if __name__ == "__main__":
    main()
