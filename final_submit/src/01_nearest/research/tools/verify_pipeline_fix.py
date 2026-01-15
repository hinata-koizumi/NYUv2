
import cv2
import numpy as np
import albumentations as A
import sys
import os

# Add cwd to path for imports
sys.path.append(os.getcwd())

def test_transform_fix():
    print("Testing Transform Fix (No Interpolation)...")
    
    # 1. Setup
    from ..data.transforms import get_train_transforms, ADDITIONAL_TARGETS
    from ..configs.base_config import Config
    
    cfg = Config()
    t = get_train_transforms(cfg)
    
    print(f"Targets: {ADDITIONAL_TARGETS}")
    if ADDITIONAL_TARGETS.get("depth") != "mask":
        print("FAIL: Depth target is not 'mask'")
        return
        
    # 2. Data
    # Create an image with a sharp depth transition
    # 100x100. Left 50 is 0. Right 50 is 10.0.
    depth = np.zeros((100, 100), dtype=np.float32)
    depth[:, 50:] = 10.0
    
    # Valid mask matches
    valid = np.zeros((100, 100), dtype=np.float32)
    valid[:, 50:] = 1.0
    
    # Image (dummy)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # 3. Apply Transform 
    found_mix = False
    n_rotated = 0
    
    for i in range(100):
        # Apply transform
        res = t(image=img, mask=valid, depth=depth, depth_valid=valid)
        d_aug = res["depth"]
        # v_aug = res["depth_valid"] # This will have 255 padding
        
        # Check if rotation happened (if 10.0 moved)
        # Original 10.0 at [:, 50:].
        # If aug changed it, we might see it elsewhere or rotated.
        
        # 4. Check for Halo / Interpolation
        # Halo = Value between 0 and 10.0 (exclusive).
        # Linear interp would produce 5.0, 2.5 etc at boundary.
        # Nearest should produce only 0.0, 10.0, and Padding (255.0).
        
        uniques = np.unique(d_aug)
        
        # Valid values are ONLY: 0.0, 10.0, 255.0
        allowed = [0.0, 10.0, 255.0]
        
        # Floating point tolerance
        is_bad = False
        for u in uniques:
            # Check if close to any allowed
            if not any(np.isclose(u, a, atol=1e-5) for a in allowed):
                is_bad = True
                print(f"FAIL: Found interpolated value: {u}")
                
        if is_bad:
            found_mix = True
            break
            
    if not found_mix:
        print("PASS: No interpolated values found in 100 iterations.")
    else:
        print("FAIL: Interpolation detected!")

if __name__ == "__main__":
    test_transform_fix()
