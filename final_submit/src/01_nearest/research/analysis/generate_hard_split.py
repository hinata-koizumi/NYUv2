
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
import json

def generate_hard_split():
    print("--> Generating Hard Split (Embedding Clustering)...")
    
    # Settings
    DATA_ROOT = "/root/datasets/NYUv2/00_data"
    OUTPUT_PATH = "/root/datasets/NYUv2/00_data/hard_split_manifest.json"
    N_CLUSTERS = 75 # Target ~10 images per cluster (795 total)
    N_FOLDS = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Image List
    img_dir = os.path.join(DATA_ROOT, "train", "image")
    all_images = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    print(f"Total Images: {len(all_images)}")
    
    # 2. Embedding Model (ResNet18)
    # We use a simple pretrained model to capture scene texture/structure
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity() # Remove classification head
    model.to(DEVICE)
    model.eval()
    
    # 3. Dataset/Loader
    class SimpleDataset(Dataset):
        def __init__(self, root, files):
            self.root = root
            self.files = files
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        def __len__(self): return len(self.files)
        
        def __getitem__(self, idx):
            path = os.path.join(self.root, self.files[idx])
            img = Image.open(path).convert("RGB")
            return self.transform(img), self.files[idx]
            
    ds = SimpleDataset(img_dir, all_images)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    
    # 4. Extract Embeddings
    embeddings = []
    file_list = []
    
    print("Extracting features...")
    with torch.no_grad():
        for imgs, fnames in loader:
            imgs = imgs.to(DEVICE)
            feats = model(imgs) # (B, 512)
            embeddings.append(feats.cpu().numpy())
            file_list.extend(fnames)
            
    embeddings = np.concatenate(embeddings, axis=0) # (795, 512)
    # Normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # 5. Clustering (KMeans)
    print(f"Clustering into {N_CLUSTERS} scenes...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    groups = kmeans.fit_predict(embeddings)
    
    # 6. Make Splits
    gkf = GroupKFold(n_splits=N_FOLDS)
    folds = {}
    
    # Sklearn split returns indices
    # We want to save {file_id: fold_idx} or similar.
    # Actually, let's save a manifest list.
    
    dummy_X = np.zeros(len(file_list))
    
    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(dummy_X, groups=groups)):
        # val_idx are the indices for this fold's validation
        for idx in val_idx:
            fname = file_list[idx]
            folds[fname] = fold_i
            
    # Verify
    print("Fold Counts:")
    for i in range(N_FOLDS):
        count = sum(1 for v in folds.values() if v == i)
        print(f"Fold {i}: {count}")
        
    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "strategy": "resnet18_kmeans_75",
            "folds": folds, # {filename: fold_idx}
            "groups": {f: int(g) for f, g in zip(file_list, groups)}
        }, f, indent=2)
        
    print(f"Saved Hard Split Manifest to {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_hard_split()
