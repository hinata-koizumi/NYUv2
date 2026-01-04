
import os
import torch
import cv2
import numpy as np
import sys

# Ensure we can import from the directory
sys.path.append("/Users/koizumihinata/NYUv2")

from exp101_segformer.configs.base_config import Config
from exp101_segformer.model.meta_arch import build_model
from exp101_segformer.data.dataset import NYUDataset

def check_exp101():
    print("Settings Exp101 Config...")
    cfg = Config(
        IN_CHANNELS=4,
        BATCH_SIZE=2,
        NUM_WORKERS=0
    )
    
    print(f"Exp Name: {cfg.EXP_NAME}")
    print(f"Depth Interp: {cfg.DEPTH_INTERPOLATION}")
    assert cfg.EXP_NAME == "exp101_segformer"
    assert cfg.DEPTH_INTERPOLATION == "linear"
    
    # 1. Check Dataset
    print("\n--- Checking Dataset ---")
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_lbl = np.random.randint(0, 13, (480, 640), dtype=np.uint8)
    dummy_depth = np.random.randint(0, 10000, (480, 640), dtype=np.uint16)
    
    os.makedirs("tmp_data_exp101", exist_ok=True)
    cv2.imwrite("tmp_data_exp101/test_rgb.png", dummy_img)
    cv2.imwrite("tmp_data_exp101/test_lbl.png", dummy_lbl)
    cv2.imwrite("tmp_data_exp101/test_depth.png", dummy_depth)
    
    ds = NYUDataset(
        image_paths=np.array(["tmp_data_exp101/test_rgb.png"]),
        label_paths=np.array(["tmp_data_exp101/test_lbl.png"]),
        depth_paths=np.array(["tmp_data_exp101/test_depth.png"]),
        cfg=cfg,
        is_train=True
    )
    
    batch = ds[0]
    x = batch[0]
    print(f"Input Shape: {x.shape}")
    assert x.shape[0] == 4, "Input must have 4 channels"
    
    # 2. Check Model
    print("\n--- Checking Model ---")
    model = build_model(cfg)
    print(f"Model Class: {type(model).__name__}")
    assert type(model).__name__ == "SegFormer", "Model must be SegFormer by default"
    
    model.eval()
    with torch.no_grad():
        x_batch = x.unsqueeze(0) # [1, 4, H, W]
        print(f"Forward pass input: {x_batch.shape}")
        out = model(x_batch)
        
        if isinstance(out, tuple):
             seg, depth = out
             print(f"Output: Seg {seg.shape}, Depth {depth.shape}")
        else:
             print(f"Output: Seg {out.shape}")
             
    print("\nSUCCESS: Exp101 (Isolated) dry run completed.")

if __name__ == "__main__":
    check_exp101()
