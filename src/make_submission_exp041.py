import os
import json
import numpy as np
import torch
import cv2
import zipfile
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
import segmentation_models_pytorch as smp

# Import Config from exp041
from exp041 import Config

# --- Test Dataset ---
class TestDataset(Dataset):
    def __init__(self, image_paths, depth_paths, cfg):
        self.image_paths = image_paths
        self.depth_paths = depth_paths
        self.cfg = cfg
        self.transform = A.Compose(
            [
                A.Resize(height=cfg.IMAGE_SIZE[0], width=cfg.IMAGE_SIZE[1]),
            ],
            additional_targets={
                "depth": "image",
            },
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # --- RGB ---
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # H,W,3

        # --- Depth ---
        depth = cv2.imread(self.depth_paths[idx], cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / 1000.0  # mm -> m

        dmin = self.cfg.DEPTH_MIN
        dmax = self.cfg.DEPTH_MAX
        depth = np.clip(depth, dmin, dmax)

        # Encoding (Log)
        encoding = getattr(self.cfg, "DEPTH_ENCODING", "linear")
        if encoding == "log":
             eps = 1e-6
             depth_log = np.log(depth + eps)
             log_min = np.log(dmin)
             log_max = np.log(dmax)
             depth = (depth_log - log_min) / (log_max - log_min)
        else:
             # Fallback if config changed, but exp041 is log
             depth = (depth - dmin) / (dmax - dmin)

        # --- Transform ---
        augmented = self.transform(image=image, depth=depth)
        image = augmented["image"]
        depth = augmented["depth"]

        # --- Tensor & Normalize ---
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(self.cfg.MEAN).view(-1, 1, 1)
        std = torch.tensor(self.cfg.STD).view(-1, 1, 1)
        image = (image - mean) / std

        depth = np.clip(depth, 0.0, 1.0)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()
        image = torch.cat([image, depth_tensor], dim=0)

        return image, self.image_paths[idx]

def main():
    cfg = Config()
    
    # Override device if needed, or stick to Config
    device = cfg.DEVICE
    print(f"Using device: {device}")

    # --- Data Paths ---
    # Assuming data/test/image and data/test/depth exist based on README/previous logic
    # The user said data/test exists now? No, wait.
    # Previous `list_dir data` showed `test` directory exists.
    test_root = os.path.join("data", "test") 
    image_dir = os.path.join(test_root, 'image')
    depth_dir = os.path.join(test_root, 'depth')
    
    # Check if dirs exist
    if not os.path.exists(image_dir) or not os.path.exists(depth_dir):
        print(f"Error: Test directories not found at {image_dir} or {depth_dir}")
        return

    image_files = sorted(os.listdir(image_dir))
    depth_files = sorted(os.listdir(depth_dir))

    image_paths = [os.path.join(image_dir, f) for f in image_files]
    depth_paths = [os.path.join(depth_dir, f) for f in depth_files]

    print(f"Test samples: {len(image_paths)}")

    # --- Dataset & Loader ---
    test_dataset = TestDataset(image_paths, depth_paths, cfg)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False, # Important: keep order or use filenames to sort later if needed
        num_workers=2,
        pin_memory=True
    )

    # --- Model ---
    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights=None, # Loading trained weights
        in_channels=4,
        classes=cfg.NUM_CLASSES
    )
    model.to(device)

    # --- Load Weights ---
    weight_path = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")
    if not os.path.exists(weight_path):
        print(f"Error: Model weights not found at {weight_path}")
        return
    
    print(f"Loading weights from {weight_path}")
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # --- Inference ---
    all_preds = []
    
    with torch.no_grad():
        for images, paths in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1) # [B, H, W]
            all_preds.append(preds.cpu().numpy())

    # Stack predictions
    submission = np.concatenate(all_preds, axis=0) # [N, H, W]
    submission = submission.astype(np.uint8) # 0-12 fits in uint8
    
    print(f"Submission shape: {submission.shape}")

    # --- Save .npy ---
    save_path = os.path.join(cfg.OUTPUT_DIR, "submission.npy")
    np.save(save_path, submission)
    print(f"Saved submission to {save_path}")

    # --- Create Zip ---
    zip_path = os.path.join(cfg.OUTPUT_DIR, "submission.zip")
    print(f"Creating zip file at {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        # 1. submission.npy
        zf.write(save_path, arcname="submission.npy")
        
        # 2. Model weights (.pt) - instructions say .pt, our file is .pth. 
        # Usually checking extension doesn't matter, but let's rename or keep as is.
        # "テストに使用した .pt 重み" -> "Weights used for test (.pt)"
        # I'll include model_best.pth
        zf.write(weight_path, arcname="model_best.pth")
        
        # 3. Notebook/Script
        # "ノートブック" -> "Notebook". Since we have scripts, I'll include the relevant python scripts.
        zf.write("src/exp041.py", arcname="exp041.py")
        zf.write("src/make_submission_exp041.py", arcname="make_submission_exp041.py")
    
    print("Zip creation completed.")
    
    # Verify zip
    with zipfile.ZipFile(zip_path, 'r') as zf:
        print("\nZip contents:")
        for info in zf.infolist():
            print(f"- {info.filename} : {info.file_size} bytes")

if __name__ == "__main__":
    main()
