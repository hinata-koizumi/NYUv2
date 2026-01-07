
import os
import torch
import numpy as np
import argparse
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader

from nearest_final.data.dataset import NYUDataset
from nearest_final.data.transforms import get_valid_transforms
from nearest_final.engine.inference import Predictor
from nearest_final.model.meta_arch import SegFPN
from nearest_final.utils.misc import configure_runtime, seed_everything
from nearest_final.submit.utils import load_cfg_from_fold_dir, best_ckpt_path, safe_torch_load
from nearest_final.utils.metrics import update_confusion_matrix, compute_metrics

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", type=str, default="data/output/nearest_final")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--split_mode", type=str, default="kfold", choices=["kfold", "group"])
    args = p.parse_args()

    fold = args.fold
    exp_dir = args.exp_dir
    fold_dir = os.path.join(exp_dir, f"fold{fold}")
    
    print(f"--- Checking Consistency for Fold {fold} ---")
    
    cfg = load_cfg_from_fold_dir(fold_dir)
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

    # 2. Replicate Split Logic (Centralized)
    from nearest_final.data.fold_utils import get_split_files
    print("Generating splits via fold_utils...")
    
    # Ensure config has correct split mode
    if args.split_mode:
        overrides = {"SPLIT_MODE": args.split_mode}
        if args.split_mode == "group":
             overrides["SPLIT_BLOCK_SIZE"] = 50
        cfg = cfg.with_overrides(**overrides)

    _, file_ids = get_split_files(cfg, fold)
    print(f"Found {len(file_ids)} validation images ({cfg.SPLIT_MODE}).")
    
    label_dir = os.path.join(cfg.TRAIN_DIR, "label")
    depth_dir = os.path.join(cfg.TRAIN_DIR, "depth")
    img_dir = os.path.join(cfg.TRAIN_DIR, "image") # Added back
    
    img_paths = [os.path.join(img_dir, f"{fid}.png") for fid in file_ids]
    lbl_paths = [os.path.join(label_dir, f"{fid}.png") for fid in file_ids]
    dep_paths = [os.path.join(depth_dir, f"{fid}.png") for fid in file_ids]
    
    # Dataset
    ds = NYUDataset(
        image_paths=np.array(img_paths),
        label_paths=np.array(lbl_paths), # Has Labels!
        depth_paths=np.array(dep_paths),
        cfg=cfg,
        transform=get_valid_transforms(cfg),
        is_train=False
    )
    
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    
    # 3. Run Inference (Submit Recipe)
    predictor = Predictor(model, loader, cfg.DEVICE, cfg)
    
    # Use Full TTA + Books Protection (Official Recipe)
    tta_combs = list(cfg.TTA_COMBS)
    use_books_protect = bool(getattr(cfg, "INFER_TTA_BOOKS_PROTECT", False))
    print(f"Recipe: TTA={len(tta_combs)}, BooksProtect={use_books_protect}")
    
    cm = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=np.int64)
    
    iterator = predictor.predict_logits(tta_combs=tta_combs, return_details=False, return_logits=True)
    
    print("Running Inference...")
    for i, logits in enumerate(tqdm(iterator, total=len(ds))):
        # logits: (H, W, C) float32 ?? NO.
        # Updated infer_fold loop handles (H, W, C).
        # Let's check predict_logits:
        # If return_logits=True, it yields 'final_logits' which is (C, H, W).
        # WAIT! My infer_fold.py had logic: `if l.ndim == 3 and l.shape[2] == cfg.NUM_CLASSES: l = l.transpose(...)`
        # Because `predict_logits` logic I viewed earlier returned (C, H, W).
        # Let's Verify default return shape.
        
        # In engine/inference.py:
        # `prob_hw_c = final_probs.permute(1, 2, 0)...numpy()` -> (H, W, C)
        # `logits_hw_c = final_logits.permute(1, 2, 0)...numpy()` -> (H, W, C)
        # It yields `logits_hw_c` or `prob_hw_c`.
        # So it IS (H, W, C).
        
        # Prediction
        # Handle HWC vs CHW
        if logits.shape[0] == cfg.NUM_CLASSES:
             # CHW -> (C, H, W)
             pred = np.argmax(logits, axis=0).astype(np.uint8)
        elif logits.shape[2] == cfg.NUM_CLASSES:
             # HWC -> (H, W, C)
             pred = np.argmax(logits, axis=2).astype(np.uint8)
        else:
             print(f"Unknown shape {logits.shape}, assuming CHW")
             pred = np.argmax(logits, axis=0).astype(np.uint8)
        
        # Ground Truth
        # NYUDataset returns (x, y, meta). y is label.
        # We need raw label, but dataset transforms verify correctness?
        # Dataset returns transformed (resized/padded) label?
        # YES. is_train=False -> get_valid_transforms -> Resize + Pad.
        # But `predict_logits` UNPADS and RESIZES back to original.
        # So we need ORIGINAL label.
        
        # Start simplistic: Load original label from disk to be sure we match Submission Logic (Output=Original Size)
        orig_lbl_path = lbl_paths[i]
        orig_lbl = cv2.imread(orig_lbl_path, cv2.IMREAD_UNCHANGED)
        if orig_lbl.ndim == 3: orig_lbl = orig_lbl[:,:,0]
        
        # Check shapes
        if pred.shape != orig_lbl.shape:
            # Maybe predictor unpad failed?
            print(f"Shape Mismatch! pred={pred.shape}, gt={orig_lbl.shape}")
            # Resize GT to Pred? Or Pred to GT?
            # Submission requirement: output shape = original shape.
            # So Pred SHOULD match Orig GT.
            pass
            
        update_confusion_matrix(pred, orig_lbl, cfg.NUM_CLASSES, cfg.IGNORE_INDEX, cm)

    # 4. Compute Metrics
    pixel_acc, miou, ious = compute_metrics(cm)
    print("\n--- Results ---")
    print(f"Fold {fold} Validation consistency Check")
    print(f"mIoU: {miou:.4f}")
    print(f"Pixel Acc: {pixel_acc:.4f}")
    
    print("\nPer-Class IoU:")
    for i, name in enumerate(cfg.CLASS_NAMES):
        print(f"  {name:10s}: {ious[i]:.4f}")

if __name__ == "__main__":
    main()
