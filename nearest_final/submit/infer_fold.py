
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import NYUDataset
from ..data.transforms import get_valid_transforms
from ..engine.inference import Predictor
from ..model.meta_arch import SegFPN
from ..utils.misc import configure_runtime, seed_everything, worker_init_fn
from .utils import load_cfg_from_fold_dir, best_ckpt_path, safe_torch_load

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", type=str, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--batch_mul", type=int, default=1)
    p.add_argument("--no_books_protect", action="store_true", help="Disable Books Protection (Override config)")
    args = p.parse_args()

    exp_dir = args.exp_dir
    fold = args.fold
    out_path = args.out_path
    
    fold_dir = os.path.join(exp_dir, f"fold{fold}")
    if not os.path.exists(fold_dir):
        raise FileNotFoundError(f"Fold dir {fold_dir} not found.")

    cfg = load_cfg_from_fold_dir(fold_dir)
    # Apply CLI overrides
    if args.no_books_protect:
        cfg = cfg.with_overrides(INFER_TTA_BOOKS_PROTECT=False)
    seed_everything(cfg.SEED)
    configure_runtime(cfg)

    # Load Model
    ckpt_path = best_ckpt_path(fold_dir)
    print(f"Loading {ckpt_path}...")
    
    model = SegFPN(num_classes=cfg.NUM_CLASSES, in_channels=cfg.IN_CHANNELS, use_depth_aux=bool(getattr(cfg, "USE_DEPTH_AUX", False)))
    state = safe_torch_load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(cfg.DEVICE)
    model.eval()

    # Load Test Data
    test_image_dir = os.path.join(cfg.TEST_DIR, "image")
    test_depth_dir = os.path.join(cfg.TEST_DIR, "depth")
    image_files = sorted([os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir) if f.endswith(".png")])
    depth_files = []
    for img_p in image_files:
        base = os.path.basename(img_p)
        d_p = os.path.join(test_depth_dir, base)
        depth_files.append(d_p if os.path.exists(d_p) else None)
        
    ds = NYUDataset(
        image_paths=np.array(image_files),
        label_paths=None,
        depth_paths=np.array(depth_files),
        cfg=cfg,
        transform=get_valid_transforms(cfg),
        is_train=False
    )
    
    loader = DataLoader(
        ds, 
        batch_size=max(1, int(cfg.BATCH_SIZE) * max(1, args.batch_mul)),
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )

    # TTA Settings (Books Protection is handled inside Predictor if config flag is set)
    tta_combs = list(cfg.TTA_COMBS)
    
    # Run Inference
    predictor = Predictor(model, loader, cfg.DEVICE, cfg)
    
    print(f"Running Inference Fold {fold} (TTA={len(tta_combs)})...")
    
    # Pre-allocate output array to avoid memory spike during stacking
    # Get shape from first sample
    try:
        s0 = ds[0]
        # Valid dataset returns (img, mask, meta)
        meta0 = s0[2]
        orig_h, orig_w = int(meta0.get("orig_h", 480)), int(meta0.get("orig_w", 640))
    except Exception as e:
        print(f"Warning: Could not determine shape from ds[0], defaulting to 480x640. Error: {e}")
        orig_h, orig_w = 480, 640

    N = len(ds)
    C = cfg.NUM_CLASSES
    print(f"Allocating Memory: ({N}, {C}, {orig_h}, {orig_w}) float16")
    results = np.zeros((N, C, orig_h, orig_w), dtype=np.float16)

    # We iterate and fill
    # We iterate and fill
    with torch.no_grad():
        iterator = predictor.predict_logits(tta_combs=tta_combs, temperature=1.0, return_details=False, return_logits=True)
        for i, item in tqdm(enumerate(iterator), total=N):
            # item is directly the logits array (H, W, C) float32 because return_logits=True
            l = item
            if l is None:
                raise ValueError("Predictor returned None (config mismatch for Books Protection?)")
            
            if l.ndim == 3 and l.shape[2] == cfg.NUM_CLASSES:
                 # HWC -> CHW
                 l = l.transpose(2, 0, 1)
            
            if l.shape[1] != orig_h or l.shape[2] != orig_w:
                # Should not happen if _unpad_and_resize works
                pass
            results[i] = l.astype(np.float16)
            
    # Save
    print(f"Saving {out_path} shape={results.shape} dtype={results.dtype}...")
    np.save(out_path, results)

if __name__ == "__main__":
    main()
