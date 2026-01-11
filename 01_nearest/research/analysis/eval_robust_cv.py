
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold

# Ensure project root is in path
sys.path.append(os.getcwd())

from ..data.dataset import NYUDataset
from ..data.transforms import get_valid_transforms
from ..model.meta_arch import build_model
from ..configs.base_config import Config
from ..utils.metrics import update_confusion_matrix, compute_metrics

def make_block_splits(image_paths, block_size=50, n_folds=5, seed=42):
    """
    Generates Group K-Fold splits based on sequential blocks.
    Assumes image_paths are sorted by time/frame order.
    """
    files = sorted(list(image_paths))
    groups = [i // block_size for i in range(len(files))]
    
    gkf = GroupKFold(n_splits=n_folds)
    # GroupKFold is deterministic based on groups, shuffle is not used in split() default
    return list(gkf.split(files, groups=groups))

def main():
    # --- Settings ---
    EXP_NAME = "nearest_final_fix" # Target Experiment
    OUTPUT_ROOT = "/root/datasets/NYUv2/00_data/output"
    DATA_ROOT = "/root/datasets/NYUv2/00_data"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"--> Starting Robust CV Eval for {EXP_NAME} on {DEVICE}")
    
    # --- Data Setup ---
    train_dir = os.path.join(DATA_ROOT, "train")
    img_dir = os.path.join(train_dir, "image")
    lbl_dir = os.path.join(train_dir, "label")
    dep_dir = os.path.join(train_dir, "depth")
    
    all_images = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    full_img_paths = np.array([os.path.join(img_dir, f) for f in all_images])
    full_lbl_paths = np.array([os.path.join(lbl_dir, f) for f in all_images])
    full_dep_paths = np.array([os.path.join(dep_dir, f) for f in all_images])
    
    # Splits
    splits = make_block_splits(full_img_paths)
    
    # Global metrics accumulators
    global_cm = np.zeros((13, 13), dtype=np.int64)
    fold_stats = []
    
    cfg = Config().with_overrides(DATA_ROOT=DATA_ROOT)
    
    # --- Fold Loop ---
    for fold_idx in range(5):
        print(f"\n--- Evaluating Fold {fold_idx} ---")
        
        # 1. Identify Validation Set
        _, val_idx = splits[fold_idx]
        print(f"Validation Samples: {len(val_idx)}")
        
        # 2. Checkpoint
        ckpt_path = os.path.join(OUTPUT_ROOT, EXP_NAME, f"fold{fold_idx}", "model_best.pth")
        if not os.path.exists(ckpt_path):
            print(f"[WARN] Checkpoint not found: {ckpt_path}. Skipping Fold.")
            fold_stats.append(None)
            continue
            
        # 3. Model
        model = build_model(cfg)
        try:
             import warnings
             with warnings.catch_warnings():
                 warnings.filterwarnings("ignore")
                 ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
             if "model" in ckpt:
                 model.load_state_dict(ckpt["model"])
             else:
                 model.load_state_dict(ckpt)
        except Exception as e:
            print(f"[ERR] Failed to load info: {e}")
            fold_stats.append(None)
            continue
            
        model.to(DEVICE)
        model.eval()
        
        # 4. Loader
        ds = NYUDataset(
            full_img_paths[val_idx],
            full_lbl_paths[val_idx],
            full_dep_paths[val_idx],
            cfg,
            transform=get_valid_transforms(cfg),
            is_train=False
        )
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
        
        # 5. Inference
        fold_cm = np.zeros((13, 13), dtype=np.int64)
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Fold {fold_idx}"):
                # Unpack
                if len(batch) == 5:
                    imgs, lbls, _, _, meta = batch
                else:
                    imgs, lbls, meta = batch
                
                imgs = imgs.to(DEVICE)
                lbls = lbls.to(DEVICE)
                
                with torch.amp.autocast("cuda", enabled=(DEVICE=="cuda"), dtype=torch.bfloat16):
                    logits = model(imgs)
                    if isinstance(logits, tuple): logits = logits[0]
                
                # Correct Evaluation: Resize Logits -> Argmax -> Compare 480x640 GT
                # 1. Resize Logits (B, C, H, W) -> (480, 640) Bilinear
                import torch.nn.functional as F
                
                # logits is (B, 13, 720, 960)
                # Resize to (480, 640)
                logits_480 = F.interpolate(logits.float(), size=(480, 640), mode='bilinear', align_corners=False)
                
                # 2. Argmax
                preds_480 = torch.argmax(logits_480, dim=1).cpu().numpy()
                
                import cv2
            
                for i in range(preds_480.shape[0]): # Batch loop
                    # Get ID
                    fid = meta["file_id"][i]
                    
                    # Load Original GT
                    gt_path = os.path.join(cfg.DATA_ROOT, "train", "label", fid)
                    if not os.path.exists(gt_path):
                         print(f"Missing GT: {gt_path}")
                         continue
                    
                    gt_orig = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
                    if gt_orig is None: continue
                    if gt_orig.ndim == 3: gt_orig = gt_orig[:, :, 0]
                    
                    # Ensure GT is exactly 480x640 (NYUv2 usually is, but safety check)
                    if gt_orig.shape != (480, 640):
                        # print(f"GT shape mistmatch {gt_orig.shape}, resizing GT to 480x640 nearest")
                        gt_orig = cv2.resize(gt_orig, (640, 480), interpolation=cv2.INTER_NEAREST)

                    # Update CM
                    # preds_480[i] is already 480x640
                    update_confusion_matrix(preds_480[i].astype(np.uint8), gt_orig, cfg.NUM_CLASSES, cfg.IGNORE_INDEX, fold_cm)
                    update_confusion_matrix(preds_480[i].astype(np.uint8), gt_orig, cfg.NUM_CLASSES, cfg.IGNORE_INDEX, global_cm)
        
        # 6. Fold Metrics
        _, miou, class_iou = compute_metrics(fold_cm)
        print(f"Fold {fold_idx} mIoU: {miou:.4f}")
        print(f"Fold {fold_idx} Books: {class_iou[1]:.4f}, Table: {class_iou[9]:.4f}")
        
        fold_stats.append({
            "miou": miou,
            "books": class_iou[1],
            "table": class_iou[9],
            "class_iou": class_iou
        })
        
        # Cleaning
        del model
        del loader
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Summary ---
    print("\n" + "="*30)
    print("A1. Robust CV Metrics Summary")
    print("="*30)
    
    valid_folds = [s for s in fold_stats if s is not None]
    if len(valid_folds) > 0:
        mious = [s["miou"] for s in valid_folds]
        books = [s["books"] for s in valid_folds]
        tables = [s["table"] for s in valid_folds]
        
        print(f"Folds Completed: {len(valid_folds)}")
        print(f"Mean mIoU: {np.mean(mious):.4f} +/- {np.std(mious):.4f}")
        print(f"Mean Books: {np.mean(books):.4f} +/- {np.std(books):.4f}")
        print(f"Mean Table: {np.mean(tables):.4f} +/- {np.std(tables):.4f}")
        
        # Global OOF
        _, global_miou, global_class_iou = compute_metrics(global_cm)
        print(f"Global OOF mIoU: {global_miou:.4f}")
        print(f"Global OOF Books: {global_class_iou[1]:.4f}")
        print(f"Global OOF Table: {global_class_iou[9]:.4f}")
    else:
        print("No successful folds.")

if __name__ == "__main__":
    main()
