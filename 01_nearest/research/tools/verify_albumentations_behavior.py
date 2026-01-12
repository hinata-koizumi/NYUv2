
import numpy as np
import albumentations as A
import cv2

def main():
    print("Verifying Albumentations Behavior for Depth-as-Mask...")
    
    # 1. Setup Dummy Data
    h, w = 100, 100
    # Depth map: float32, range 0.6 to 10.0
    depth = np.random.uniform(0.6, 10.0, (h, w)).astype(np.float32)
    # Put a specific value to check preservation
    depth[50, 50] = 5.123456 
    
    # 2. Define Transform treating depth as MASK
    # Note: mask_value=255 is standard in this repo for labels
    t = A.Compose([
        A.Resize(height=50, width=50, interpolation=cv2.INTER_NEAREST), # Resize
        A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.0, rotate_limit=45, p=1.0, 
                          border_mode=cv2.BORDER_CONSTANT, mask_value=255), # Rotate with padding
    ], additional_targets={"depth": "mask"})
    
    # 3. Apply
    res = t(image=np.zeros((h,w,3), dtype=np.uint8), depth=depth)
    d_res = res["depth"]
    
    # 4. Check Dtype
    print(f"Original Dtype: {depth.dtype}")
    print(f"Result Dtype: {d_res.dtype}")
    
    if d_res.dtype != np.float32:
        print("FAIL: Dtype changed! Albumentations cast float mask to int?")
    else:
        print("PASS: Dtype preserved.")

    # 5. Check Value Preservation (Center pixel should be approx same if not lost to rot)
    # Hard to tracking exact pixel after rotate, but we can check unique values.
    # If it was cast to int, we lose decimals.
    # Check if there are non-integer values
    is_float_vals = np.any((d_res % 1) > 0.0)
    print(f"Has decimal values: {is_float_vals}")
    
    if not is_float_vals:
        print("FAIL: All values are integers? Precision lost!")
    else:
        print("PASS: Decimal values preserved.")
        
    # 6. Check Padding Value
    # We used mask_value=255.
    # If depth is treated as mask, padding should be 255.
    # Check corners (likely padding after 45 deg rotate + resize)
    uniq = np.unique(d_res)
    if 255.0 in uniq:
        print("INFO: Padding value 255.0 found in depth map.")
    else:
        print(f"INFO: Padding value 255.0 NOT found. Unique vals sample: {uniq[:5]}")
        # Maybe 0?
        if 0.0 in uniq:
            print("INFO: Padding value 0.0 found.")

    # 7. Check if changing logic works
    # We want: Depth -> Nearest, but Padding -> 0.0 (ideal) or handled.
    # If padding is 255.0, dataset.py needs to handle it.
    # Dataset.py cleans 'valid_mask' (255->0), but usually not 'depth'.
    # Does 'depth' padding matter if valid_mask says it's invalid?
    # Yes, if we use depth raw values somewhere else.
    # But for input 4th channel, we mask by valid_mask.
    
if __name__ == "__main__":
    main()
