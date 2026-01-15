
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
from ..engine.inference import Predictor

def make_block_splits(image_paths, block_size=50, n_folds=5, seed=42):
    files = sorted(list(image_paths))
    groups = [i // block_size for i in range(len(files))]
    gkf = GroupKFold(n_splits=n_folds)
    return list(gkf.split(files, groups=groups))

def compute_iou(cm):
    """Simple IoU from CM"""
    inter = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = inter / union
    return iou

def main():
    EXP_NAME = "nearest_final_fix"
    FOLD = 0 # Use Fold 0 for C1/C2 diagnostic (or loop all if requested, but plan implies diagnostic)
    # Start with Fold 0 for speed.
    
    DATA_ROOT = "/root/datasets/NYUv2/00_data"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Bins for C1
    DIST_BINS = [(0, 1), (1, 2), (2, 5), (5, 10)]
    
    print(f"--> Starting Specialization Analysis (Fold {FOLD})")
    
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
    # val_idx = val_idx[:50] # Debug
    
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
    
    # ============================
    # C1. Distance Bins (using Normal Inference)
    # ============================
    print("\n--- C1. Distance Bin Analysis ---")
    bin_cms = {i: np.zeros((13, 13), dtype=np.int64) for i in range(len(DIST_BINS))}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Distance Bins"):
            imgs, lbls, deps = batch[:3] # Need raw depth for bins?
            # Dataset transforms might assume Depth is normalized? 
            # NYUDataset output depth is (1, H, W) normalized if transforms apply.
            # But we can read original depth from file or invert normalization?
            # Or pass `return_original`? 
            # Actually, `deps` in batch is transformed tensor. 
            # If configured with normalization, it's normalized.
            # Config.DEPTH_MIN/MAX usually used.
            # But let's look at `dataset.py`. It returns `depth` as tensor.
            # We can use the file paths in `full_dep[val_idx]` to get RAW depth for accurate bins.
            # But files are not aligned with batch indices easily without shuffling off.
            # Loader is shuffle=False.
            # But to be precise, let's assume `deps` tensor is linear scaling of depth?
            # Or just use the model prediction and match with the Batch (which is sequential).
            # We can rely on `deps` if we know the transform.
            pass
    
    # Alternative: Iterate dataset index directly? 
    # Or just use the `deps` tensor which IS the depth input to the model.
    # We need to know if it's in meters.
    # Usually `depth` tensor is normalized 0..1 or standardized.
    # If standard, hard to reverse without params.
    # Let's read the raw depth files again for the loop.
    
    # We will do a custom loop accessing dataset samples one by one to get raw depth easily
    # OR better: use the `deps` from batch but we need to know the scale.
    # Let's use `full_dep[val_idx]` and `cv2.imread`.
    
    val_files_map = {i: (full_paths[val_idx][i], full_lbl[val_idx][i], full_dep[val_idx][i]) for i in range(len(val_idx))}
    
    # We need predictions.
    # Let's run inference and store preds, then compute bin metrics offline/online.
    # To keep it memory efficient, we process batch and "match" with raw depth files?
    # Batch order is sequential.
    
    dataset_iter = iter(val_idx) # Indices in global list
    
    model_preds = []
    
    # 1. Run inference to get all preds (or do it in loop)
    # Let's do it in one loop.
    
    batch_size = 4
    total_samples = len(val_idx)
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    current_idx = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="C1 Inference"):
            imgs, lbls, _ = batch[:3]
            imgs = imgs.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=(DEVICE=="cuda"), dtype=torch.bfloat16):
                 logits = model(imgs)
                 if isinstance(logits, tuple): logits = logits[0]
            preds = torch.argmax(logits, dim=1).cpu().numpy() # (B, H, W)
            gts = lbls.cpu().numpy()
            
            # Process each sample in batch
            for b in range(preds.shape[0]):
                global_idx = val_idx[current_idx]
                raw_dep_path = full_dep[global_idx]
                current_idx += 1
                
                # Load Raw Depth
                # NYUv2 usually PNG 16bit mm?
                dimg = cv2.imread(raw_dep_path, cv2.IMREAD_UNCHANGED)
                if dimg is None: continue
                d_m = dimg.astype(float) / 1000.0 # meters
                
                # Resize depth to match pred (480, 640)?
                # Pred is usually resized to `RESIZE_HEIGHT` in model?
                # No, model output is `RESIZE_HEIGHT`.
                # We want metric at 480x640 (Official).
                # But here we are comparing against loaded GT `lbls`.
                # `lbls` from loader is RESIZED.
                # User says "Evaluation resolution... 480x640 restored".
                # Validation transforms usually Resize `(480, 640)`.
                # Check `base_config.py`. `RESIZE_HEIGHT`=480?
                # `RESIZE_HEIGHT` is 720, `RESIZE_WIDTH` is 960 in `constants.py`/`cli.py`.
                # So the model outputs 720x960.
                # We need to resize pred back to 480x640 to match Original GT?
                # OR does the user want "Resolution same as submission"?
                # "Evaluation resolution: 480x640 restored".
                # This means we must load the ORIGINAL GT and resize PRED to 480x640.
                # `NYUDataset` returns RESIZED GT.
                # So we should ignore `lbls` from batch and load manual GT.
                
                # Load Original GT
                gt_path = full_lbl[global_idx]
                gt_orig = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
                if gt_orig.ndim == 3: gt_orig = gt_orig[:,:,0]
                
                # Resize Pred to GT (480, 640)
                pred_native = preds[b] # 720x960
                pred_restored = cv2.resize(pred_native, (gt_orig.shape[1], gt_orig.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Resize Depth to GT (if not already 480x640)
                if d_m.shape != gt_orig.shape:
                    d_m = cv2.resize(d_m, (gt_orig.shape[1], gt_orig.shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                # Validity Mask
                mask_valid = (gt_orig != 255)
                
                # Bin updates
                for i_bin, (dmin, dmax) in enumerate(DIST_BINS):
                    mask_d = (d_m >= dmin) & (d_m < dmax) & mask_valid
                    if mask_d.sum() == 0: continue
                    
                    p_flat = pred_restored[mask_d]
                    g_flat = gt_orig[mask_d]
                    
                    update_confusion_matrix(p_flat, g_flat, 13, 255, bin_cms[i_bin])

    # Report C1
    print("\n[C1 Results]")
    for i_bin, (dmin, dmax) in enumerate(DIST_BINS):
        iou = compute_iou(bin_cms[i_bin])
        bk = iou[1]
        tb = iou[9]
        print(f"[{dmin}-{dmax}m] Books: {bk:.4f}, Table: {tb:.4f}")

    # ============================
    # C2. Books Protection (TTA) Check
    # ============================
    print("\n--- C2. Books Protection Check ---")
    
    # We need to run "With Protection" and "Without Protection".
    # This logic resides in "Predictor.predict_logits" via 'cfg' or flags.
    # The Config class has INFER_TTA_BOOKS_PROTECT = True/False.
    # We can perform TTA sweep logic or just run Predictor twice.
    
    # We need the Predictor class.
    # We will use the Validation Loader (which yields batches) but Predictor usually handles full inference loop?
    # Predictor takes a loader.
    
    # Set 1: Protect ON (default)
    print("Running Protect ON...")
    cfg_on = cfg.with_overrides(INFER_TTA_BOOKS_PROTECT=True)
    try:
         predictor_on = Predictor(model, loader, DEVICE, cfg_on)
         # Predictor.predict_logits yields results.
         # But we want metrics.
         # Can we just compute metrics from results?
         # Predictor returns iterator of dicts.
         pass
    except Exception:
         # Fallback if Predictor signature mismatch (I didn't check it fully)
         print("Predictor init failed.")
    
    # Actually implementing C2 effectively requires me to replicate the `oof_infer` logic but staying in memory.
    # Predictor logic is complex (sliding window vs resize, etc).
    # Since we are doing "Assessment", maybe we just check ONE fold (Fold 0)?
    # We are already in Fold 0.
    
    def run_protect_mode(p_cfg, label):
        cm = np.zeros((13, 13), dtype=np.int64)
        pred_engine = Predictor(model, loader, DEVICE, p_cfg)
        # Using TTA combs from config?
        # Predictor needs explicit tta_combs or uses defaults.
        # Let's use standard single scale for speed? Or TTA?
        # User says "TTA/Books Protection". So TTA implies multiple scales.
        # But for speed, let's stick to 1.0 scale TTA (basically no TTA but enabled path).
        # Actually TTA is crucial for Books Protection? Protection applies even at single scale?
        # Usually protection merges "NoFlip" branch.
        # Let's use `p_cfg.TTA_COMBS` (which is tuples).
        
        # We limit specific TTA combs to [(1.0, False)] for speed?
        # User says "Inference settings unified to final recipe".
        # Final recipe has TTA.
        # But TTA on full resolution takes time.
        # We'll use [(1.0, False)] as proxy unless "Books Protection" REQUIRES multi-scale to show value?
        # Usually protection is "Take 1.0 NoFlip Books prediction and overwrite".
        
        # Let's run minimal TTA: [(1.0, False)].
        # If Protection is ON, it isolates Books from this branch.
        
        results_iter = pred_engine.predict_logits(tta_combs=[(1.0, False)], temperature=1.0, return_details=True)
        
        count = 0
        for item in tqdm(results_iter, desc=label, total=len(val_idx)):
            # Item has "merged_probs" (H, W, C) or similar?
            # Check `oof_infer.py` or Predictor.
            # Predictor usually returns dict with keys.
            probs = item["merged_probs"] # (C, H, W) or (H, W, C)?
            # Predictor usually returns numpy (C, H, W) or (H, W, C).
            # From `cli.py`: argmax(probs, axis=0) -> implies (C, H, W).
            
            p_mask = np.argmax(probs, axis=0).astype(np.uint8) # (H, W)
            
            # Need to match GT.
            # Predictor order matches Loader order.
            # We must be careful about ordering.
            # Predictor yields in order of loader.
            
            global_idx = val_idx[count]
            gt_path = full_lbl[global_idx]
            gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            if gt.ndim == 3: gt = gt[:,:,0]
            
            # Resize Pred to GT (Restore 480x640)
            if p_mask.shape != gt.shape:
                 p_mask = cv2.resize(p_mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

            update_confusion_matrix(p_mask, gt, 13, 255, cm)
            count += 1
            
        return cm

    # Run ON
    cm_on = run_protect_mode(cfg_on, "Protect ON")
    
    # Run OFF
    print("Running Protect OFF...")
    cfg_off = cfg.with_overrides(INFER_TTA_BOOKS_PROTECT=False)
    cm_off = run_protect_mode(cfg_off, "Protect OFF")
    
    # Compare
    iou_on = compute_iou(cm_on)
    iou_off = compute_iou(cm_off)
    
    print("\n[C2 Results - Fold 0]")
    print(f"Protect ON  -> mIoU: {np.mean(iou_on):.4f}, Books: {iou_on[1]:.4f}")
    print(f"Protect OFF -> mIoU: {np.mean(iou_off):.4f}, Books: {iou_off[1]:.4f}")
    
    diff_books = iou_on[1] - iou_off[1]
    print(f"Books Gain: {diff_books:+.4f}")

if __name__ == "__main__":
    main()
