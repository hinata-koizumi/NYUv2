
import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2

from ..data.dataset import NYUDataset
from ..data.transforms import get_valid_transforms
from ..data.fold_utils import get_split_files
from ..engine.inference import Predictor
from ..model.meta_arch import SegFPN
from ..utils.misc import configure_runtime, seed_everything
from .utils import load_cfg_from_fold_dir, best_ckpt_path, safe_torch_load
from ..utils.metrics import update_confusion_matrix, compute_metrics

def run_oof_inference(exp_dir: str, fold: int, save_dir: str = None, batch_mul: int = 1, 
                      no_books_protect: bool = False, split_mode: str = None):
    
    fold_dir = os.path.join(exp_dir, f"fold{fold}")
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = fold_dir

    print(f"--- Generating OOF Assets for Fold {fold} ---")
    
    cfg = load_cfg_from_fold_dir(fold_dir)
    
    # Force Split Mode if provided
    if split_mode:
        print(f"Overriding SPLIT_MODE to {split_mode}")
        cfg = cfg.with_overrides(SPLIT_MODE=split_mode)
        
    if no_books_protect:
        cfg = cfg.with_overrides(INFER_TTA_BOOKS_PROTECT=False)

    seed_everything(cfg.SEED)
    configure_runtime(cfg)

    # 1. Load Model
    ckpt_path = best_ckpt_path(fold_dir)
    print(f"Loading {ckpt_path}...")
    model = SegFPN(num_classes=cfg.NUM_CLASSES, in_channels=cfg.IN_CHANNELS, use_depth_aux=bool(getattr(cfg, "USE_DEPTH_AUX", False)))
    state = safe_torch_load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(cfg.DEVICE)
    model.eval()

    # 2. Get Validation Split
    print("Generating validation split...")
    _, val_ids = get_split_files(cfg, fold)
    print(f"Validation set: {len(val_ids)} images")
    
    # 3. Create Dataset
    img_dir = os.path.join(cfg.TRAIN_DIR, "image")
    label_dir = os.path.join(cfg.TRAIN_DIR, "label")
    depth_dir = os.path.join(cfg.TRAIN_DIR, "depth")
    
    img_paths = [os.path.join(img_dir, f"{fid}.png") for fid in val_ids]
    lbl_paths = [os.path.join(label_dir, f"{fid}.png") for fid in val_ids]
    dep_paths = [os.path.join(depth_dir, f"{fid}.png") for fid in val_ids]
    
    ds = NYUDataset(
        image_paths=np.array(img_paths),
        label_paths=np.array(lbl_paths),
        depth_paths=np.array(dep_paths),
        cfg=cfg,
        transform=get_valid_transforms(cfg),
        is_train=False
    )
    
    # 4. Loader
    loader = DataLoader(
        ds, 
        batch_size=max(1, int(cfg.BATCH_SIZE) * max(1, args.batch_mul)),
        shuffle=False, 
        num_workers=0, # Safe default
        pin_memory=True
    )
    
    # 5. Inference
    predictor = Predictor(model, loader, cfg.DEVICE, cfg)
    tta_combs = list(cfg.TTA_COMBS)
    
    # Prepare artifacts
    # val_oof_logits.npy: (N, C, H, W) float16
    # We can pre-allocate since we know N.
    # We need H, W. Assuming 480x640 per 'Standardizing' constraint.
    # To be safe, we check existing shape or force config.
    # Standard: 480x640.
    
    H_std, W_std = 480, 640 # Target standardized resolution
    N = len(val_ids)
    C = cfg.NUM_CLASSES
    
    print(f"Allocating Memory: ({N}, {C}, {H_std}, {W_std}) float16")
    results = np.zeros((N, C, H_std, W_std), dtype=np.float16)
    
    # Confusion Matrix for Metrics
    cm = np.zeros((C, C), dtype=np.int64)
    
    print("Running Inference...")
    iterator = predictor.predict_logits(tta_combs=tta_combs, return_details=False, return_logits=True)
    
    for i, logits in enumerate(tqdm(iterator, total=N)):
        # logits: (C, H, W) float32 (from Predictor default return)
        
        # Check Shape
        if logits.shape[0] != C:
             # Unexpected format handling (HWC?)
             if logits.shape[2] == C:
                 logits = logits.transpose(2, 0, 1) # HWC -> CHW
        
        # Ensure it matches standardized HW
        _, h, w = logits.shape
        if h != H_std or w != W_std:
            # Resize if necessary (shouldn't happen with correct pipeline, but safety)
            # Use torch for easy interpolation or cv2 per channel
             pass # Assume correct for now as confirmed by check_val_consistency
        
        # Save to array
        results[i] = logits.astype(np.float16)

        # Update Metrics
        # pred = argmax
        pred = np.argmax(logits, axis=0).astype(np.uint8)
        
        # Load GT (Raw)
        gt_path = lbl_paths[i]
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if gt.ndim == 3: gt = gt[:,:,0]
        
        update_confusion_matrix(pred, gt, C, cfg.IGNORE_INDEX, cm)

    # 6. Save Artifacts
    oof_path = os.path.join(save_dir, "val_oof_logits.npy")
    ids_path = os.path.join(save_dir, "val_file_ids.npy")
    metrics_path = os.path.join(save_dir, "val_metrics.json")
    
    print(f"Saving logits {oof_path}...")
    np.save(oof_path, results)
    
    print(f"Saving IDs {ids_path}...")
    np.save(ids_path, np.array(val_ids))
    
    # Metrics
    pixel_acc, miou, ious = compute_metrics(cm)
    metrics = {
        "mIoU": float(miou),
        "pixel_acc": float(pixel_acc),
        "per_class_iou": {name: float(ious[i]) for i, name in enumerate(cfg.CLASS_NAMES)},
        "confusion_matrix": cm.tolist()
    }
    
    print(f"Saving metrics {metrics_path}...")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Done. Fold {fold} mIoU: {miou:.4f}")
    
    # ---------------------------------------------------------
    # 7. Generate Test Logits (Added Phase 8 Step D)
    # ---------------------------------------------------------
    print("\n--- Generating Test Assets ---")
    test_img_dir = os.path.join(cfg.TEST_DIR, "image")
    test_files = sorted([f for f in os.listdir(test_img_dir) if f.endswith(".png")])
    test_ids = [os.path.splitext(f)[0] for f in test_files]
    
    print(f"Test set: {len(test_ids)} images")
    
    test_ds = NYUDataset(
        image_paths=np.array([os.path.join(test_img_dir, f) for f in test_files]),
        label_paths=None, 
        depth_paths=np.array([os.path.join(cfg.TEST_DIR, "depth", f) for f in test_files]),
        cfg=cfg,
        transform=get_valid_transforms(cfg),
        is_train=False
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=max(1, int(cfg.BATCH_SIZE) * max(1, args.batch_mul)),
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    
    predictor_test = Predictor(model, test_loader, cfg.DEVICE, cfg)
    
    # Alloc Test Memory
    N_test = len(test_ids)
    print(f"Allocating Test Memory: ({N_test}, {C}, {H_std}, {W_std}) float16")
    test_results = np.zeros((N_test, C, H_std, W_std), dtype=np.float16)
    
    print("Running Test Inference...")
    iterator_test = predictor_test.predict_logits(tta_combs=tta_combs, return_details=False, return_logits=True)
    
    for i, logits in enumerate(tqdm(iterator_test, total=N_test)):
        # Shape Check
        if logits.shape[0] != C:
             if logits.shape[2] == C: logits = logits.transpose(2, 0, 1)
             
        test_results[i] = logits.astype(np.float16)
        
    # Save Test Artifacts
    test_out_npy = os.path.join(save_dir, "test_logits.npy")
    test_out_ids = os.path.join(save_dir, "test_file_ids.npy")
    
    print(f"Saving {test_out_npy}...")
    np.save(test_out_npy, test_results)
    
    print(f"Saving {test_out_ids}...")
    np.save(test_out_ids, np.array(test_ids))
    
    print("Test Logic Completed.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", type=str, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--save_dir", type=str, default=None, help="If None, saves to exp_dir/foldX")
    p.add_argument("--batch_mul", type=int, default=1)
    p.add_argument("--no_books_protect", action="store_true")
    p.add_argument("--split_mode", type=str, default=None, help="Force split mode (kfold/group)")
    
    args = p.parse_args()
    
    run_oof_inference(
        exp_dir=args.exp_dir,
        fold=args.fold,
        save_dir=args.save_dir,
        batch_mul=args.batch_mul,
        no_books_protect=args.no_books_protect,
        split_mode=args.split_mode
    )

if __name__ == "__main__":
    main()
