
import os
import cv2
import numpy as np
import glob
from tqdm import tqdm

# Hardcoded constants from Config/Constants
DEPTH_MIN = 0.6
DEPTH_MAX = 10.0
SCALE = 0.75 

def verify_single_image(img_path, depth_path):
    if not os.path.exists(depth_path): return None

    # Load Depth (mm)
    d_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if d_mm is None: return None
    d_m = d_mm.astype(np.float32) / 1000.0
    
    # 1. Resize (Train uses Nearest for Dynamic Resize)
    h, w = d_m.shape
    h_small, w_small = int(h * SCALE), int(w * SCALE)
    d_small = cv2.resize(d_m, (w_small, h_small), interpolation=cv2.INTER_NEAREST)
    
    # 2. Pad (to Crop Size) -> Introduces 0s
    crop_h, crop_w = 576, 768
    pad_h = max(0, crop_h - h_small)
    pad_w = max(0, crop_w - w_small)
    d_padded = cv2.copyMakeBorder(d_small, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0.0)
    
    # 3. Augmentation (Rotate)
    center = (crop_w // 2, crop_h // 2)
    angle = 5.0
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Linear (Current)
    d_linear = cv2.warpAffine(d_padded, M, (crop_w, crop_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    
    # Nearest (Target)
    d_nearest = cv2.warpAffine(d_padded, M, (crop_w, crop_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    
    mask_nearest = (d_nearest > 0).astype(np.float32)
    valid_region = (mask_nearest > 0.5)

    # Boundary Calculation (Where Valid meets Invalid in Nearest)
    kernel = np.ones((3,3), np.uint8)
    m_uint = mask_nearest.astype(np.uint8)
    # Dilate adds border outside, Erode removes border inside. 
    # Boundary is (Dilate - Erode) which has both inner and outer edge.
    # We care about the valid part (Inner Edge), where corruption happens.
    boundary_mask = (cv2.dilate(m_uint, kernel) - cv2.erode(m_uint, kernel)) > 0
    boundary_valid = boundary_mask & valid_region

    diff = np.abs(d_linear - d_nearest)
    
    # Halo Definition:
    # 1. Under-range (< 0.6)
    # 2. Deviation (> 0.2)
    halo = valid_region & ((d_linear < DEPTH_MIN) | (diff > 0.2))
    
    return {
        "valid_pixels": np.sum(valid_region),
        "total_boundary": np.sum(boundary_valid),
        "halo_on_boundary": np.sum(halo & boundary_valid)
    }

def main():
    root = "/root/datasets/NYUv2/00_data/train"
    image_dir = os.path.join(root, "image")
    depth_dir = os.path.join(root, "depth")
    images = sorted(glob.glob(os.path.join(image_dir, "*.png")))[:50]
    
    total_valid = 0
    total_bound = 0
    total_halo_on_bound = 0
    
    print("Verifying Augmentation Artifacts (Pad -> Rotate)...")
    
    for p in tqdm(images):
        res = verify_single_image(p, os.path.join(depth_dir, os.path.basename(p)))
        if res:
            total_valid += res["valid_pixels"]
            total_bound += res["total_boundary"]
            total_halo_on_bound += res["halo_on_boundary"]
            
    print(f"Total Valid: {total_valid}")
    print(f"Total Boundary (Inner Edge): {total_bound}")
    print(f"Corrupted Boundary Pixels: {total_halo_on_bound}")
    if total_bound > 0:
        print(f"Ratio: {total_halo_on_bound / total_bound * 100:.2f}%")
        
    if total_bound > 0 and (total_halo_on_bound/total_bound > 0.5):
        print("CONCLUSION: FACT CONFIRMED. >50% of boundary is corrupted.")
    else:
        print("CONCLUSION: Low impact.")

if __name__ == "__main__":
    main()
