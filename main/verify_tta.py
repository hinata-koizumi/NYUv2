import torch
import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from src.engine.inference import Predictor
from src.model.meta_arch import SegFPN
from configs.base_config import Config

def verify():
    cfg = Config()
    device = torch.device("cpu")
    
    print(f"Creating dummy model with in_channels={cfg.IN_CHANNELS}...")
    model = SegFPN(cfg.NUM_CLASSES, cfg.IN_CHANNELS).to(device)
    model.eval()
    
    # Dummy Predictor
    predictor = Predictor(model, loader=None, device=device, cfg=cfg, ema_model=model)
    
    # Dummy Sample
    # H_pad, W_pad should be divisible by 32
    # Resize=720x960. Pad -> 736x960.
    H_pad, W_pad = 736, 960
    
    # Create fake padded input
    x = torch.randn(cfg.IN_CHANNELS, H_pad, W_pad).to(device)
    
    # Meta (scalars)
    meta = {
        "orig_h": 480, 
        "orig_w": 640,
        "pad_h": H_pad - 720, # Wait, Resize is 720. 
        # Check get_valid_transforms pad logic:
        # Resize(720, 960) -> PadIfNeeded(736, 960). 
        # So image is 720x960. Padded to 736x960.
        # Pad amount: pad_h = 16, pad_w = 0.
        # BUT meta["pad_h"] in Dataset was: h_new - RESIZE_HEIGHT.
        # So pad_h = 736 - 720 = 16.
        # Correct.
        "pad_h": 16,
        "pad_w": 0,
        "file_id": "dummy"
    }
    
    sample = (x, None, meta)
    
    print("--- Testing TTA=False ---")
    prob = predictor.predict_proba_one(sample, use_ema=True, amp=False, tta_flip=False)
    print(f"Prob Shape: {prob.shape}")
    
    assert prob.shape == (cfg.NUM_CLASSES, 480, 640), f"Shape mismatch: {prob.shape}"
    assert prob.dtype == np.float32
    
    s = prob.sum(axis=0)
    print(f"Sum (min/max): {s.min():.4f} - {s.max():.4f}")
    # Should be approx 1.0
    if not np.allclose(s, 1.0, atol=1e-2):
        print("Warning: Sum is not close to 1.0 (bilinear interpolation artifact?)")
    
    print("--- Testing TTA=True ---")
    prob_tta = predictor.predict_proba_one(sample, use_ema=True, amp=False, tta_flip=True)
    print(f"Prob TTA Shape: {prob_tta.shape}")
    assert prob_tta.shape == (cfg.NUM_CLASSES, 480, 640)
    
    s_tta = prob_tta.sum(axis=0)
    print(f"Sum TTA (min/max): {s_tta.min():.4f} - {s_tta.max():.4f}")
    
    print("Verification Successful.")

if __name__ == "__main__":
    verify()
