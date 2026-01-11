
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from ..data.dataset import NYUDataset
from ..data.transforms import get_valid_transforms
from ..model.meta_arch import build_model
from ..utils.metrics import update_confusion_matrix, compute_metrics
from ..configs.base_config import Config

def _unpack_batch(batch):
    if len(batch) == 5:
        return batch # x, y, meta, dt, dv
    return batch[0], batch[1], batch[2], None, None

def tta_predict(model, x, mode="bilinear"):
    """
    Standard TTA (Scale 1.0, No Flip) but with controllable interpolation.
    """
    # x is (B, C, H, W)
    # We want to emulate the TTA resize if scale != 1.0.
    # But here we just use 1.0. 
    # Wait, if scale=1.0, TTA function in trainer DOES NO INTERPOLATION.
    # It only interpolates if scale != 1.0.
    
    # Check trainer.py:
    # if scale != 1.0: x_aug = F.interpolate(...)
    
    # So if we use default TTA (1.0), there is NO Interpolation happening at inference time (except in Data Loading).
    # Data Loading (Valid) uses A.Resize(LINEAR).
    
    # So the corruption happens at Data Loading for Validation.
    # transforms.py: "A.Resize(..., interpolation=cv2.INTER_LINEAR)"
    
    # So "Inference-Time" verification means:
    # We must change the TRANSFORM, not the model/TTA code (unless using TTA scales).
    
    # User Request "C2": "Existing weights... Depth Nearest Resize Input... Inference Only".
    # So we need to create a Validation Loader with NEAREST Resize.
    pass

def main():
    # Setup
    target_fold = 1
    chkPT = "/root/datasets/NYUv2/00_data/output/nearest_final/fold1/model_best.pth"
    
    if not os.path.exists(chkPT):
        print("Checkpoint not found.")
        return

    cfg = Config().with_overrides(DATA_ROOT="/root/datasets/NYUv2/00_data")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Model
    model = build_model(cfg)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        ckpt = torch.load(chkPT, map_location=device, weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    
    # Load Keys
    # We need to recreate the split. 
    # Just load ALL images and subset to simple N for speed, or try to respect split.
    # For "Minimal Verify", just taking first 50 images of "train" dir (assuming they have labels) is fine.
    # Or properly load "valid" split if possible?
    
    # Let's just load a subset of 50 images from TRAIN directory as "Validation".
    # It doesn't matter if it's train or val split, we just want to see delta in mIoU.
    image_dir = os.path.join(cfg.TRAIN_DIR, "image")
    label_dir = os.path.join(cfg.TRAIN_DIR, "label")
    depth_dir = os.path.join(cfg.TRAIN_DIR, "depth")
    images = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])[:100] # 100 samples
    
    X_img = np.array([os.path.join(image_dir, f) for f in images])
    X_lbl = np.array([os.path.join(label_dir, f) for f in images])
    X_dep = np.array([os.path.join(depth_dir, f) for f in images])
    
    # Run 1: Baseline (Linear Resize in Transform)
    # transforms.py uses A.Resize(..., interpolation=cv2.INTER_LINEAR)
    print(">>> Run 1: Baseline (Linear Resize)...")
    ds_linear = NYUDataset(
        X_img, X_lbl, X_dep, cfg,
        transform=get_valid_transforms(cfg), # Default is Linear
        is_train=False
    )
    dl_linear = DataLoader(ds_linear, batch_size=4, shuffle=False, num_workers=2)
    miou_linear = evaluate(model, dl_linear, device, cfg, "Linear")
    
    # Run 2: Experiment (Nearest Resize)
    # We need to construct a transform that uses NEAREST.
    print(">>> Run 2: Experiment (Nearest Resize)...")
    
    import albumentations as A
    import cv2
    
    # Custom Transform: Resize using Nearest for BOTH Image and Mask? 
    # Usually Image uses Linear. Mask uses Nearest.
    # Depth should use Nearest.
    # In transforms.py, 'depth' is 'image' target -> Linear.
    # We want 'depth' -> Nearest.
    
    # Workaround: Define specific transform where 'depth' uses Nearest behavior.
    # A.Resize allows interpolation.
    # If we want ONLY depth to be Nearest, we have to split them?
    # Albumentations Resize interpolation applies to 'image'. 'mask' is usually Nearest.
    # 'depth' is configured as 'image' in transforms.py.
    
    # Hack: Change additional_targets to treat 'depth' as 'mask' temporarily?
    # If 'depth' is 'mask', A.Resize uses interpolation=cv2.INTER_NEAREST (default for mask).
    
    # Let's define the tricky transform manually.
    h, w = cfg.RESIZE_HEIGHT, cfg.RESIZE_WIDTH
    min_h, min_w = ((h+31)//32)*32, ((w+31)//32)*32
    
    # We redefine targets so 'depth' is treated as 'mask' (Nearest) OR we explicitly manage it.
    # Easiest way in this script: Change additional_targets.
    
    algo_nearest = A.Compose([
        A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR), # Image uses Linear
        A.PadIfNeeded(min_height=min_h, min_width=min_w, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=255)
    ], additional_targets={"depth": "mask", "depth_valid": "mask"}) # Force depth to be mask -> Nearest
    
    ds_nearest = NYUDataset(
        X_img, X_lbl, X_dep, cfg,
        transform=algo_nearest,
        is_train=False
    )
    dl_nearest = DataLoader(ds_nearest, batch_size=4, shuffle=False, num_workers=2)
    miou_nearest = evaluate(model, dl_nearest, device, cfg, "Nearest")
    
    print("-" * 30)
    print(f"Baseline (Linear Depth): mIoU = {miou_linear:.4f}")
    print(f"Experiment (Nearest Depth): mIoU = {miou_nearest:.4f}")
    print(f"Delta: {miou_nearest - miou_linear:.4f}")

def evaluate(model, loader, device, cfg, desc):
    cm = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=np.int64)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            x, y, meta, _, _ = _unpack_batch(batch)
            x = x.to(device)
            y = y.to(device)
            
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                logits = model(x)
                if isinstance(logits, tuple): logits = logits[0]
            
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            gt = y.cpu().numpy()
            
            for i in range(pred.shape[0]):
                cm = update_confusion_matrix(pred[i], gt[i], cfg.NUM_CLASSES, cfg.IGNORE_INDEX, cm)
                
    _, miou, _ = compute_metrics(cm)
    return miou

if __name__ == "__main__":
    main()
