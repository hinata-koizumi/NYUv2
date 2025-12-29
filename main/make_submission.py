import os
import sys
import numpy as np
import torch
import zipfile
import cv2
from tqdm import tqdm

sys.path.append(os.getcwd())

from configs.base_config import Config
from src.data.dataset import NYUDataset
from src.data.transforms import get_valid_transforms
from src.data.adapters import get_adapter
from src.model.meta_arch import SegFPN
from src.engine.inference import Predictor

def load_predictor(fold_idx, cfg, device):
    """Load predictor for a specific fold with clear error messages."""
    # Path: data/outputs/ExpName/foldK/model_best.pth
    ckpt_path = os.path.join("data", "outputs", cfg.EXP_NAME, f"fold{fold_idx}", "model_best.pth")
    
    if not os.path.exists(ckpt_path):
        # Provide helpful error message with expected path
        print(f"⚠️  Checkpoint not found for fold {fold_idx}")
        print(f"    Expected path: {ckpt_path}")
        print(f"    Available folds in {os.path.join('data', 'outputs', cfg.EXP_NAME)}:")
        exp_dir = os.path.join("data", "outputs", cfg.EXP_NAME)
        if os.path.exists(exp_dir):
            folds = [d for d in os.listdir(exp_dir) if d.startswith("fold")]
            if folds:
                print(f"    {', '.join(sorted(folds))}")
            else:
                print(f"    (no fold directories found)")
        else:
            print(f"    (experiment directory does not exist)")
        return None
    
    print(f"Loading fold {fold_idx} from {ckpt_path}...")
    
    # Model Init
    model = SegFPN(num_classes=cfg.NUM_CLASSES, in_channels=cfg.IN_CHANNELS).to(device)
    
    # Load State
    # trainer.py saves ema.ema.state_dict() directly to model_best.pth
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
         # weights_only available in recent torch
         state = torch.load(ckpt_path, map_location=device)
         
    model.load_state_dict(state)
    model.eval()
    
    # Create Predictor
    # Pass model as 'ema_model' so use_ema=True triggers logic using this model.
    predictor = Predictor(model, loader=None, device=device, cfg=cfg, ema_model=model)
    return predictor

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--folds", type=int, default=None, help="Number of folds to use")
    parser.add_argument("--tta_flip", action="store_true", help="Enable Flip TTA")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA weights")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config()
    
    # Validate config (Commit 6: fail-fast on invalid config)
    cfg.validate()
    
    if args.exp_name:
        cfg.EXP_NAME = args.exp_name
        
    device = torch.device(cfg.DEVICE)
    
    # Update Config based on args if needed (though we pass explicit ly to functions mostly)
    n_folds = args.folds if args.folds is not None else cfg.N_FOLDS
    
    print(f"Running Inference on device: {device}")
    
    # 1. Prepare Data
    # Using cfg.TEST_DIR
    test_img_dir = os.path.join(cfg.TEST_DIR, "image")
    test_dep_dir = os.path.join(cfg.TEST_DIR, "depth")
    
    if not os.path.exists(test_img_dir):
        print(f"Error: Test image directory not found: {test_img_dir}")
        return

    images = sorted([f for f in os.listdir(test_img_dir) if f.endswith(".png")])
    if len(images) == 0:
        print(f"❌ No images found in test directory: {test_img_dir}")
        print(f"   Directory contents: {os.listdir(test_img_dir) if os.path.exists(test_img_dir) else '(does not exist)'}")
        return
        
    img_paths = np.array([os.path.join(test_img_dir, f) for f in images])
    
    # Depth
    if os.path.exists(test_dep_dir):
        dep_paths = np.array([os.path.join(test_dep_dir, f) for f in images])
    else:
        print("Warning: Test depth directory not found! Using zeros/None.")
        dep_paths = None # Dataset handles None
    
    # Dataset
    # ensure adapter is set
    adapter = get_adapter(cfg)
    
    test_ds = NYUDataset(
        image_paths=img_paths,
        label_paths=None, 
        depth_paths=dep_paths,
        cfg=cfg,
        transform=get_valid_transforms(cfg),
        adapter=adapter,
        enable_smart_crop=False
    )
    
    print(f"Test Dataset Size: {len(test_ds)}")
    
    # 2. Load Predictors
    predictors = []
    # Try all defined folds
    for k in range(n_folds):
        p = load_predictor(k, cfg, device)
        if p is not None:
            predictors.append(p)
            
    if not predictors:
        print("No predictors loaded! Aborting.")
        return

    print(f"Ensembling {len(predictors)} models.")

    # 3. Inference Loop
    submission_preds = []
    
    for i in tqdm(range(len(test_ds)), desc="Predicting", unit="img"):
        sample = test_ds[i] 
        
        sum_p = None
        
        for p in predictors:
            # Predict
            # Returns: (C, H_orig, W_orig)
            prob = p.predict_proba_one(
                sample, 
                use_ema=args.use_ema, 
                amp=getattr(cfg, "USE_AMP", False), 
                tta_flip=args.tta_flip # Enable TTA Flip as requested
            )
            
            if sum_p is None:
                sum_p = prob
            else:
                sum_p += prob
        
        # Average
        avg_prob = sum_p / len(predictors)
        
        # Argmax
        label = np.argmax(avg_prob, axis=0).astype(np.uint8)
        submission_preds.append(label)
        
    # Stack
    submission = np.stack(submission_preds, axis=0) # (N, H, W)
    print(f"Submission Shape: {submission.shape}")
    
    # 4. Save
    np.save("submission.npy", submission)
    print("Saved submission.npy")
    
    # 5. Zip
    with zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write("submission.npy", arcname="submission.npy")
    print("Saved submission.zip")

if __name__ == "__main__":
    main()
