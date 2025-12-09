import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from exp060 import NYUDataset, Config, get_valid_transforms

def calculate_weights():
    cfg = Config()
    
    # Paths
    label_dir = os.path.join(cfg.DATA_ROOT, 'label')
    label_files = sorted(os.listdir(label_dir))
    label_paths = [os.path.join(label_dir, f) for f in label_files]
    
    # We use all labels for calculation or just train? 
    # Usually train set only. But simpler to iterate all or just reuse split logic.
    # Let's import train_test_split to be accurate.
    from sklearn.model_selection import train_test_split
    train_idx, _ = train_test_split(
        range(len(label_paths)),
        test_size=0.2,
        random_state=cfg.SEED,
        shuffle=True
    )
    train_labels = [label_paths[i] for i in train_idx]
    
    # Dataset just for labels
    # We don't need images/depths for this, but Dataset expects them.
    # We can pass dummy list of same length or just instantiate with actual paths but only read labels.
    # Actually, let's just manually iterate files to be faster and not depend on transforms.
    
    print(f"Calculating weights from {len(train_labels)} training masks...")
    
    pixel_counts = np.zeros(cfg.NUM_CLASSES, dtype=np.int64)
    total_pixels = 0
    
    import cv2
    
    for p in tqdm(train_labels):
        label = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if label.ndim == 3:
            label = label[:, :, 0]
            
        # Ignore mask
        mask = (label != cfg.IGNORE_INDEX)
        valid_pixels = label[mask]
        
        counts = np.bincount(valid_pixels, minlength=cfg.NUM_CLASSES)
        pixel_counts += counts
        total_pixels += counts.sum()
        
    print("Pixel Counts:", pixel_counts)
    
    # Frequency
    freq = pixel_counts / total_pixels
    
    # Median Frequency Balancing
    # w_c = median_freq / freq_c
    median_freq = np.median(freq)
    weights = median_freq / (freq + 1e-10) # Avoid div by zero
    
    # User requested square root smoothing: w_c = w_c ** 0.5
    weights_sqrt = weights ** 0.5
    
    print("-" * 20)
    print("Computed Weights (MFB):")
    print(weights)
    print("-" * 20)
    print("Computed Weights (Sqrt MFB):")
    print(weights_sqrt)
    print("-" * 20)
    
    # Format for copy-paste
    print("weights_list = [")
    for w in weights_sqrt:
        print(f"    {w:.4f},")
    print("]")

if __name__ == "__main__":
    calculate_weights()
