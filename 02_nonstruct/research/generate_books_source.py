"""
generate_books_source.py
Scan all training labels and identify images containing 'books' (class 1).
Save the IDs to config.DATA_DIR/ids/books_source_ids.txt
"""

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import sys
sys.path.append("/root/datasets/NYUv2/02_nonstruct"); sys.path.append("/root/datasets/NYUv2")
from configs import default as config

def main():
    # Load all train IDs
    if not os.path.exists(config.TRAIN_IDS_FILE):
        print(f"Error: {config.TRAIN_IDS_FILE} does not exist.")
        return

    with open(config.TRAIN_IDS_FILE, 'r') as f:
        train_ids = [line.strip() for line in f.readlines()]
    
    print(f"Scanning {len(train_ids)} training images for books...")
    
    books_ids = []
    first_debug = False
    
    for img_id in tqdm(train_ids):
        # TRAIN_IDS_FILE contains filenames like "000002.png"
        lbl_path = os.path.join(config.DATA_DIR, "train/label", img_id)
        
        if not first_debug:
            print(f"DEBUG: First path check: {lbl_path}")
            print(f"DEBUG: Exists: {os.path.exists(lbl_path)}")
            first_debug = True
            
        if not os.path.exists(lbl_path):
            continue
            
        try:
            lbl = np.array(Image.open(lbl_path))
            # Unique check
            if 1 in np.unique(lbl):
                books_ids.append(img_id)
        except Exception as e:
            print(f"Error loading {lbl_path}: {e}")
            continue
            
    print(f"Found {len(books_ids)} images containing books.")
    
    save_path = os.path.join(config.DATA_DIR, "ids/books_source_ids.txt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        for i in books_ids:
            f.write(i + "\n")
            
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    main()
