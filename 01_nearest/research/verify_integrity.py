
import os
import sys
import numpy as np
import torch
import cv2
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

# Adjust path
sys.path.append(os.getcwd())

from ..configs.base_config import Config
from ..submit.utils import load_cfg_from_fold_dir

def check_a_id_integrity():
    print("\n=== Check A: ID Integrity ===")
    
    # 1. Load Golden OOF
    golden_root = "nearest_final/golden_artifacts"
    ids_path = os.path.join(golden_root, "oof", "oof_file_ids.npy")
    
    if not os.path.exists(ids_path):
        # Fallback to local fold 0/1/etc logs if merged OOF not found?
        # User said "Generate Golden Artifacts" was running.
        # But we only ran for Fold 0 in D1 step!
        # So we DO NOT have full OOF yet. 
        # But we reported "Global OOF mIoU: 0.8727" from `eval_robust_cv.py`.
        # `eval_robust_cv` ran on-the-fly. It accumulated `global_cm`.
        # So we should check the logic in `eval_robust_cv.py` primarily, OR
        # Check the splits used in `eval_robust_cv.py`.
        
        print(f"[WARN] {ids_path} not found (Full merged OOF not generated yet).")
        print("Checking split logic directly used in `eval_robust_cv.py`...")
    
    # Re-simulate split logic from eval_robust_cv.py
    data_root = "/root/datasets/NYUv2/00_data"
    train_dir = os.path.join(data_root, "train")
    img_dir = os.path.join(train_dir, "image")
    all_images = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    full_paths = np.array([os.path.join(img_dir, f) for f in all_images])
    
    # Block Split
    def make_block_splits(image_paths, block_size=50, n_folds=5):
        files = sorted(list(image_paths))
        groups = [i // block_size for i in range(len(files))]
        gkf = GroupKFold(n_splits=n_folds)
        return list(gkf.split(files, groups=groups))
    
    splits = make_block_splits(full_paths)
    
    total_val = 0
    seen_val_indices = []
    
    for fold_idx in range(5):
        tr_idx, val_idx = splits[fold_idx]
        
        # Check intersection
        intersection = set(tr_idx) & set(val_idx)
        if len(intersection) > 0:
            print(f"[FAIL] Fold {fold_idx} has LEAKAGE! {len(intersection)} shared indices.")
        else:
            print(f"[PASS] Fold {fold_idx} Train/Val Disjoint.")
            
        seen_val_indices.extend(val_idx)
        total_val += len(val_idx)
    
    # Uniqueness
    unique_val = len(set(seen_val_indices))
    print(f"Total Validation Samples: {total_val}")
    print(f"Unique Validation Samples: {unique_val}")
    print(f"Total Images Available: {len(all_images)}")
    
    if total_val == unique_val == len(all_images):
        print("[PASS] Full coverage, no duplicates.")
    else:
        print("[FAIL] Mismatch in coverage/uniqueness.")

def simple_iou(pred, gt, num_classes=13):
    # flattened
    inter = np.zeros(num_classes)
    union = np.zeros(num_classes)
    
    for c in range(num_classes):
        p_mask = (pred == c)
        g_mask = (gt == c)
        
        # Valid mask (gt != 255) implicitly handled by not matching c?
        # No, if GT=255, it shouldn't count towards Union of Class C?
        # Usually GT=255 is excluded from ALL calculations.
        # But if pred=C and GT=255, is it False Positive?
        # Standard: Ignore 255 entirely.
        
        valid = (gt != 255)
        p_mask = p_mask & valid
        g_mask = g_mask & valid
        
        inter[c] = (p_mask & g_mask).sum()
        union[c] = (p_mask | g_mask).sum()
        
    return inter, union

def check_b_independent_miou():
    print("\n=== Check B: Independent mIoU Recalc (Fold 0) ===")
    # We will pick 10 random images from Fold 0 Validation and check mIoU
    # vs what our previous script claimed (or just typical values).
    # Since we can't easily match exactly what `eval_robust_cv` saw without running it,
    # let's run a mini-inference on Fold 0 and calc metric MANUALLY.
    
    base_dir = "/root/datasets/NYUv2/00_data"
    fold = 0
    ckpt_path = f"{base_dir}/output/nearest_final_fix/fold0/model_best.pth"
    
    # Split
    img_dir = os.path.join(base_dir, "train", "image")
    lbl_dir = os.path.join(base_dir, "train", "label")
    all_images = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    
    groups = [i // 50 for i in range(len(all_images))]
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(all_images, groups=groups))
    _, val_idx = splits[fold]
    
    # Load Model
    from ..model.meta_arch import build_model
    cfg = Config().with_overrides(DATA_ROOT=base_dir)
    model = build_model(cfg)
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    if "model" in ckpt: model.load_state_dict(ckpt["model"])
    else: model.load_state_dict(ckpt)
    model.to("cuda")
    model.eval()
    
    # Standard Transform (Linear Resize)
    from ..data.transforms import get_valid_transforms
    from ..data.dataset import NYUDataset
    from torch.utils.data import DataLoader
    
    val_files = [all_images[i] for i in val_idx]
    # Check simple subset (first 10)
    subset_files = val_files[:10] 
    
    print(f"Checking {len(subset_files)} samples manually...")
    
    ds = NYUDataset(
        image_paths=np.array([os.path.join(img_dir, f) for f in subset_files]),
        label_paths=np.array([os.path.join(lbl_dir, f) for f in subset_files]),
        depth_paths=np.array([os.path.join(base_dir, "train", "depth", f) for f in subset_files]),
        cfg=cfg,
        transform=get_valid_transforms(cfg),
        is_train=False
    )
    # Loader (Batch 1 for manual check)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    
    inter_accum = np.zeros(13)
    union_accum = np.zeros(13)
    
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if len(batch)==5: img, lbl, _, _, _ = batch
            else: img, lbl, _ = batch
            
            img = img.to("cuda")
            # Inference
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                logits = model(img)
                if isinstance(logits, tuple): logits = logits[0]
            
            # Pred (H, W) - Model Output is 720x960 usually?
            # Check shape
            # print(f"Logits shape: {logits.shape}") 
            # -> (1, 13, 720, 960)
            
            pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            
            # GT (H, W) - Loader returns RESIZED labels via `get_valid_transforms`.
            # If `eval_robust_cv` used the Loader's GT, it compared 720x960 against 720x960?
            # User requirement: "Evaluation resolution: 480x640 restored".
            # `eval_robust_cv.py` used:
            #   preds = torch.argmax(logits, dim=1).cpu().numpy()
            #   gts = lbls.cpu().numpy()
            #   update_confusion_matrix(preds[i], gts[i]...)
            # `lbls` from dataset are TRANSFROMED (Resized to 720x960).
            # So `eval_robust_cv` evaluated at 720x960!
            # BUT user requirement was "480x640 restored".
            
            # If we evaluate at 720x960 (interpolated GT), does it inflate score?
            # Usually yes, because boundary pixels are smoothed or matched easier? 
            # Or maybe not +0.16 boost.
            
            # Let's perform the STRICT check:
            # 1. Resize Pred to 480x640 (Nearest)
            # 2. Load Original GT (480x640)
            # 3. Calc IoU
            
            fid = subset_files[i]
            gt_orig = cv2.imread(os.path.join(lbl_dir, fid), cv2.IMREAD_UNCHANGED)
            if gt_orig.ndim==3: gt_orig = gt_orig[:,:,0] # (480, 640)
            
            # Resize Pred
            pred_480 = cv2.resize(pred, (640, 480), interpolation=cv2.INTER_NEAREST)
            
            inter, union = simple_iou(pred_480, gt_orig, 13)
            inter_accum += inter
            union_accum += union
            
            # Also calc "Lazy" IoU (720 vs 720) to see if that explains the gap
            lbl_720 = lbl.squeeze().cpu().numpy().astype(np.uint8)
            # ... (skip for brevity, focus on strict)
            
    iou_class = inter_accum / (union_accum + 1e-10)
    miou = np.nanmean(iou_class)
    print(f"\n[Strict Check 480x640] Mean IoU on {len(subset_files)} samples: {miou:.4f}")
    print(f"Class IoUs: {np.round(iou_class, 3)}")

def check_c_ablation_logic():
    print("\n=== Check C: Ablation Logic Review ===")
    # Visual check of `eval_depth_impact.py` code content
    script_path = "analysis/eval_depth_impact.py"
    with open(script_path, "r") as f:
        code = f.read()
        
    print("Checking 'Depth Zero' implementation...")
    if 'imgs[:, 3, :, :] = 0.0' in code:
        print("[PASS] Depth Zero sets channel 3 to 0.0")
    else:
        print("[FAIL] Depth Zero logic suspicious.")
        
    print("Checking 'Depth Shuffle' implementation...")
    # Logic in previous file: idx = torch.randperm(imgs.shape[0]); imgs[:, 3, ...] = imgs[idx, 3, ...]
    # This is BATCH shuffle (Image Swap).
    # User said: "Image unit shuffle... distance gradient might remain".
    # User suggestion: "Pixel shuffle".
    # But for "Check C", we are verifying what WAS run.
    # What was run was Batch Shuffle.
    # User analysis: "If depth works, shuffle should drop more."
    # If Batch Shuffle was used, maybe random images are "good enough" proxies for depth?
    # (e.g. floor is always at bottom).
    print("[INFO] Shuffle was Batch-based (Image Swap). User warned this might be weak.")

def main():
    check_a_id_integrity()
    check_b_independent_miou()
    check_c_ablation_logic()

if __name__ == "__main__":
    main()
