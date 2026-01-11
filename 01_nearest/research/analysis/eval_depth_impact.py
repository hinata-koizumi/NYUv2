
import os
import sys
import numpy as np
import torch
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold

sys.path.append(os.getcwd())

from ..data.dataset import NYUDataset
from ..data.transforms import get_valid_transforms
from ..model.meta_arch import build_model
from ..configs.base_config import Config
from ..utils.metrics import update_confusion_matrix, compute_metrics

# Re-use split logic
def make_block_splits(image_paths, block_size=50, n_folds=5, seed=42):
    files = sorted(list(image_paths))
    groups = [i // block_size for i in range(len(files))]
    gkf = GroupKFold(n_splits=n_folds)
    return list(gkf.split(files, groups=groups))

def get_boundary_mask(label, ignore_index=255, thickness=3):
    """
    Generates a mask enabling pixels within 'thickness' of the valid/ignore boundary.
    """
    # Valid mask: 1 where label is VALID, 0 where IGNORE
    valid = (label != ignore_index).astype(np.uint8)
    
    # Edges
    # Dilate valid area -> grows into invalid
    # Erode valid area -> shrinks away from invalid
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2*thickness+1, 2*thickness+1))
    dilated = cv2.dilate(valid, kernel)
    eroded = cv2.erode(valid, kernel)
    
    # Boundary band = (Dilated - Eroded) INTERSECT Valid
    # We only care about performance in the VALID region near the edge.
    # So we want pixels that are:
    # 1. Valid (we have GT)
    # 2. Near Invalid (eroded is 0)
    
    # 'eroded' has 0 near the boundary (inside valid).
    # 'valid' has 1 at the boundary.
    # So valid - eroded gives the inner frame of the valid region.
    
    inner_edge = valid - eroded
    
    # If we also care about "valid pixels that were invalid in depth but valid in label"?
    # The user says "Valid/Invalid boundary depth contamination".
    # Usually this means the edge of the projection.
    # Assume 'valid' here refers to the Label Validity (which often follows crop).
    # We will measure on the "Inner Edge" of the valid mask.
    
    return inner_edge.astype(bool)

def main():
    EXP_NAME = "nearest_final_fix"
    FOLD = 0 # Use Fold 0 for ablation
    DATA_ROOT = "/root/datasets/NYUv2/00_data"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"--> Starting Depth Impact Analysis (Fold {FOLD})")
    
    # --- Load Model ---
    cfg = Config().with_overrides(DATA_ROOT=DATA_ROOT)
    ckpt_path = os.path.join(DATA_ROOT, "output", EXP_NAME, f"fold{FOLD}", "model_best.pth")
    
    model = build_model(cfg)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    
    # --- Load Data ---
    train_dir = os.path.join(DATA_ROOT, "train")
    img_dir = os.path.join(train_dir, "image")
    all_images = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    full_paths = np.array([os.path.join(img_dir, f) for f in all_images])
    
    # Split
    splits = make_block_splits(full_paths)
    _, val_idx = splits[FOLD]
    
    # Use subset for speed if needed, but user said "Re-eval". Full valid is safer.
    # val_idx = val_idx[:100] # Debug
    
    lbl_dir = os.path.join(train_dir, "label")
    dep_dir = os.path.join(train_dir, "depth")
    full_lbl = np.array([os.path.join(lbl_dir, f) for f in all_images])
    full_dep = np.array([os.path.join(dep_dir, f) for f in all_images])
    
    ds = NYUDataset(
        full_paths[val_idx],
        full_lbl[val_idx],
        full_dep[val_idx],
        cfg,
        transform=get_valid_transforms(cfg),
        is_train=False
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)
    
    # --- Define Modes ---
    modes = ["Normal", "DepthZero", "DepthShuffle"]
    results = {}
    
    # Storage for Boundary Check (only for Normal mode usually, or compare?)
    # User says "B2. Halo fix... measure mIoU in boundary band".
    # This implies we want to see if it's GOOD.
    # We can calculate it during the 'Normal' run.
    
    boundary_cm = np.zeros((13, 13), dtype=np.int64)
    
    for mode in modes:
        print(f"\nRunning Mode: {mode}")
        cm = np.zeros((13, 13), dtype=np.int64)
        
        with torch.no_grad():
            for batch in tqdm(loader):
                imgs, lbls, _ = batch[:3] # Handle 3 or 5 return
                
                # Ablation Logic
                if mode == "DepthZero":
                    imgs[:, 3, :, :] = 0.0
                elif mode == "DepthShuffle":
                    # Shuffle across batch? Or spatial noise?
                    # "Depth shuffle (Noise)" usually implies breaking correlation.
                    # Let's shuffle the batch indices for the depth channel.
                    idx = torch.randperm(imgs.shape[0])
                    imgs[:, 3, :, :] = imgs[idx, 3, :, :]
                    
                imgs = imgs.to(DEVICE)
                lbls = lbls.to(DEVICE)
                
                with torch.amp.autocast("cuda", enabled=(DEVICE=="cuda"), dtype=torch.bfloat16):
                    logits = model(imgs)
                    if isinstance(logits, tuple): logits = logits[0]
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                gts = lbls.cpu().numpy()
                
                # Metric Update
                for i in range(preds.shape[0]):
                    update_confusion_matrix(preds[i], gts[i], 13, 255, cm)
                    
                    if mode == "Normal":
                        # Boundary Logic
                        # preds[i] is (H, W), gts[i] is (H, W)
                        mask_b = get_boundary_mask(gts[i], ignore_index=255, thickness=3)
                        # Filter by mask
                        p_flat = preds[i][mask_b]
                        g_flat = gts[i][mask_b]
                        
                        if len(p_flat) > 0:
                            # Ad-hoc update
                             # Cannot use update_confusion_matrix directly as it expects rectangular shape or flattened? 
                             # It flattens internally.
                             update_confusion_matrix(p_flat, g_flat, 13, 255, boundary_cm)

        _, miou, class_iou = compute_metrics(cm)
        results[mode] = {
            "miou": miou,
            "table": class_iou[9],
            "books": class_iou[1]
        }
        print(f"[{mode}] mIoU: {miou:.4f} (Table: {class_iou[9]:.4f}, Books: {class_iou[1]:.4f})")
    
    print("\n--- B1. Depth Ablation Report ---")
    base = results["Normal"]["miou"]
    for m in modes:
        diff = results[m]["miou"] - base
        print(f"{m}: {results[m]['miou']:.4f} (Delta: {diff:+.4f})")
        
    print("\n--- B2. Halo (Boundary) Report (Normal Mode) ---")
    _, b_miou, b_class_iou = compute_metrics(boundary_cm)
    print(f"Boundary Band mIoU: {b_miou:.4f}")
    print(f"Boundary Table: {b_class_iou[9]:.4f}")
    print(f"Boundary Books: {b_class_iou[1]:.4f}")

if __name__ == "__main__":
    main()
