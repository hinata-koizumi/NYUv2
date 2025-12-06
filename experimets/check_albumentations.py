
import albumentations as A
import numpy as np
import cv2

def test_resize_interpolation():
    # Create a dummy mask with distinct values
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 20:40] = 1
    mask[60:80, 60:80] = 2
    
    # Resize to something that would cause interpolation artifacts if linear is used
    transform = A.Compose([
        A.Resize(height=50, width=50)
    ])
    
    augmented = transform(image=mask, mask=mask) # Passing mask as image to see what happens if treated as image, and as mask
    
    resized_mask = augmented['mask']
    
    print(f"Original unique values: {np.unique(mask)}")
    print(f"Resized mask unique values: {np.unique(resized_mask)}")
    
    if len(np.unique(resized_mask)) > 3:
        print("FAIL: Interpolation artifacts detected (likely Linear/Cubic used).")
    else:
        print("PASS: No interpolation artifacts (likely Nearest used).")

if __name__ == "__main__":
    test_resize_interpolation()
