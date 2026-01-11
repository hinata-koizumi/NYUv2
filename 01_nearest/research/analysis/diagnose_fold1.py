
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold

sys.path.append(os.getcwd())

from ..data.dataset import NYUDataset
from ..data.transforms import get_valid_transforms
from ..model.meta_arch import build_model
from ..configs.base_config import Config
from ..utils.metrics import update_confusion_matrix

def analyze_fold1():
    print("--> Starting Fold 1 Books Diagnosis...")
    
    # 1. Setup
    cfg = Config().with_overrides(DATA_ROOT="/root/datasets/NYUv2/00_data")
    DEVICE = "cuda"
    
    # 2. Get Fold 1 Validation Set
    img_dir = os.path.join(cfg.DATA_ROOT, "train", "image")
    all_images = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    full_paths = np.array([os.path.join(img_dir, f) for f in all_images])
    
    # Replicate Split Logic
    files = sorted(list(full_paths))
    groups = [i // 50 for i in range(len(files))]
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(files, groups=groups))
    _, val_idx = splits[1] # Fold 1
    
    print(f"Fold 1 Validation Size: {len(val_idx)}")
    
    # Dataset
    lbl_dir = os.path.join(cfg.DATA_ROOT, "train", "label")
    dep_dir = os.path.join(cfg.DATA_ROOT, "train", "depth")
    
    val_files = [all_images[i] for i in val_idx]
    
    ds = NYUDataset(
        image_paths=np.array([os.path.join(img_dir, f) for f in val_files]),
        label_paths=np.array([os.path.join(lbl_dir, f) for f in val_files]),
        depth_paths=np.array([os.path.join(dep_dir, f) for f in val_files]),
        cfg=cfg,
        transform=get_valid_transforms(cfg),
        is_train=False
    )
    
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)
    
    # 3. Model
    ckpt_path = "/root/datasets/NYUv2/00_data/output/nearest_final_fix/fold1/model_best.pth"
    model = build_model(cfg)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if "model" in ckpt: model.load_state_dict(ckpt["model"])
    else: model.load_state_dict(ckpt)
    model.to(DEVICE)
    model.eval()
    
    # 4. Metrics
    # Books Class ID = 1
    # Confusion Matrix: Rows=GT, Cols=Pred
    cm_books = np.zeros(13, dtype=np.int64) # Where GT=Books went
    
    books_depths = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Diagnosing"):
            if len(batch) == 5:
                imgs, lbls, depths, _, meta = batch
            else:
                imgs, lbls, meta = batch
                depths = None # Should not happen if we requested depth paths
            
            imgs = imgs.to(DEVICE)
            
            # Predict
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                logits = model(imgs)
                if isinstance(logits, tuple): logits = logits[0]
            
            # Resize Logits to 480x640 (Strict Protocol)
            logits = F.interpolate(logits.float(), size=(480, 640), mode='bilinear', align_corners=False)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Iterate batch
            for i in range(preds.shape[0]):
                fid = meta["file_id"][i]
                
                # Load GT (Label + Depth)
                gt_path = os.path.join(lbl_dir, fid)
                dep_path = os.path.join(dep_dir, fid)
                
                gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
                if gt.ndim==3: gt=gt[:,:,0]
                
                raw_depth = cv2.imread(dep_path, cv2.IMREAD_UNCHANGED) # uint16 mm
                
                # Mask where GT is Books (1)
                books_mask = (gt == 1)
                
                if books_mask.sum() == 0:
                    continue
                    
                # 1. Confusion Analysis
                # Get predictions at these locations
                pred_at_books = preds[i][books_mask]
                # Bin count
                counts = np.bincount(pred_at_books, minlength=13)
                if len(counts) > 13: counts = counts[:13] # Ignore anything weird
                cm_books += counts
                
                # 2. Depth Analysis
                # Get depths at these locations (mm -> m)
                depth_at_books = raw_depth[books_mask].astype(np.float32) / 1000.0
                books_depths.extend(depth_at_books.tolist())
    
    # --- Report ---
    print("\n=== Fold 1 Books Diagnosis Report ===")
    
    # 1. Depth Distribution
    depths = np.array(books_depths)
    print(f"Total Books Pixels: {len(depths)}")
    if len(depths) > 0:
        p25, p50, p75 = np.percentile(depths, [25, 50, 75])
        mean_d = np.mean(depths)
        print(f"Depth Stats: Mean={mean_d:.2f}m, Median={p50:.2f}m, IQR=[{p25:.2f}, {p75:.2f}]")
        print(f"Ratio > 3m: {(depths > 3.0).mean() * 100:.2f}%")
        print(f"Ratio > 5m: {(depths > 5.0).mean() * 100:.2f}%")
    
    # 2. Confusion
    print("\nConfusion Distribution (Where does GT=Books go?):")
    total_books_gt = cm_books.sum()
    class_names = getattr(cfg, "CLASS_NAMES", ["wall", "books", "chair", "floor", "ceiling", "table", "objects", "window", "furniture", "picture", "door", "screen", "person"]) # Fallback if not found in cfg
    
    # Sort by count desc
    idxs = np.argsort(cm_books)[::-1]
    
    for idx in idxs:
        count = cm_books[idx]
        ratio = count / total_books_gt * 100
        if ratio < 0.1: continue # Skip noise
        name = class_names[idx] if idx < len(class_names) else str(idx)
        print(f"-> {name}: {ratio:.2f}%")

if __name__ == "__main__":
    analyze_fold1()
