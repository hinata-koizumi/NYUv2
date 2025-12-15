import os
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import segmentation_models_pytorch as smp
from tqdm import tqdm
import cv2

# --- Configuration (Hardcoded for Inference) ---
class Config:
    SEED = 42
    RESIZE_HEIGHT = 600
    RESIZE_WIDTH = 800
    CROP_SIZE = (512, 512) 
    NUM_CLASSES = 13
    IGNORE_INDEX = 255
    DATA_ROOT = 'data/train'
    
    # Models to Ensemble
    MODEL_PATHS = [
        "data/outputs/exp068_kfold_f0_smartcrop/model_best.pth",
        "data/outputs/exp068_kfold_f1_smartcrop/model_best.pth",
        "data/outputs/exp068_kfold_f2_smartcrop/model_best.pth",
        "data/outputs/exp068_kfold_f3_smartcrop/model_best.pth",
        "data/outputs/exp068_kfold_f4_smartcrop/model_best.pth"
    ]
    
    # Validation Target
    VALID_FOLD = 0 # "train's 20% valid" -> Fold 0
    
    MEAN = [0.525, 0.443, 0.400]
    STD = [0.281, 0.281, 0.293]
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_valid_transforms(cfg):
    return A.Compose(
        [
            A.Resize(height=cfg.CROP_SIZE[0], width=cfg.CROP_SIZE[1]),
        ],
        additional_targets={
            "depth": "image",
            "depth_target": "image",
        },
    )

class NYUDataset(Dataset):
    def __init__(self, image_paths, label_paths, depth_paths=None, transform=None, cfg=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.depth_paths = depth_paths
        self.transform = transform
        self.cfg = cfg

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(self.label_paths[idx], cv2.IMREAD_UNCHANGED)
        if label.ndim == 3:
            label = label[:, :, 0]

        # For inference validation, we just need Image and Label via Transforms
        # Depth is needed for model input format but not strictly for Seg valid (unless model requires it)
        # Our model is MultiTaskDeepLab(in_channels=4). So we NEED depth input.
        
        depth_input = None
        if self.depth_paths is not None:
             raw_depth = cv2.imread(self.depth_paths[idx], cv2.IMREAD_UNCHANGED)
             raw_depth = raw_depth.astype(np.float32) / 1000.0
             # We use fixed min/max from exp068 generic
             dmin, dmax = 0.71, 10.0
             raw_depth = np.clip(raw_depth, dmin, dmax)
             inv = 1.0 / raw_depth
             inv_min = 1.0 / dmax
             inv_max = 1.0 / dmin
             depth_input = (inv - inv_min) / (inv_max - inv_min)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=label, depth=depth_input)
            image = augmented["image"]
            label = augmented["mask"]
            depth_input = augmented["depth"]

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(self.cfg.MEAN).view(-1, 1, 1)
        std = torch.tensor(self.cfg.STD).view(-1, 1, 1)
        image = (image - mean) / std

        if depth_input is not None:
            depth_input = np.clip(depth_input, 0.0, 1.0)
            d_tensor = torch.from_numpy(depth_input).unsqueeze(0).float()
            image = torch.cat([image, d_tensor], dim=0)

        label = torch.from_numpy(label).long()
        return image, label

class MultiTaskDeepLab(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.backbone = smp.DeepLabV3Plus(
            encoder_name="resnet101",
            encoder_weights=None, # Loading weights manually
            in_channels=in_channels,
            classes=num_classes,
        )
        self.depth_head = nn.Conv2d(256, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        features = self.backbone.encoder(x)
        decoder_out = self.backbone.decoder(features)
        seg_logits = self.backbone.segmentation_head(decoder_out)
        return seg_logits

def compute_metrics(confusion_matrix):
    pixel_acc = np.diag(confusion_matrix).sum() / (confusion_matrix.sum() + 1e-10)
    intersection = np.diag(confusion_matrix)
    union = confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - intersection
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = intersection / union
    miou = np.nanmean(iou)
    return pixel_acc, miou, iou

def update_confusion_matrix(preds, labels, num_classes, ignore_index, existing_matrix=None):
    if existing_matrix is None:
        existing_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    preds = preds.flatten()
    labels = labels.flatten()
    mask = labels != ignore_index
    preds = preds[mask]
    labels = labels[mask]
    cm = np.bincount(
        num_classes * labels + preds,
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    existing_matrix += cm
    return existing_matrix

def main():
    cfg = Config()
    seed_everything(cfg.SEED)
    
    print(f"Device: {cfg.DEVICE}")
    print(f"Target Valid Fold: {cfg.VALID_FOLD}")
    
    # 1. Load Data
    df = pd.read_csv('train_folds.csv')
    valid_df = df[df['fold'] == cfg.VALID_FOLD].reset_index(drop=True)
    
    image_dir = os.path.join(cfg.DATA_ROOT, 'image')
    label_dir = os.path.join(cfg.DATA_ROOT, 'label')
    depth_dir = os.path.join(cfg.DATA_ROOT, 'depth')
    
    valid_images = [os.path.join(image_dir, i) for i in valid_df['image_id']]
    valid_labels = [os.path.join(label_dir, i) for i in valid_df['image_id']]
    valid_depths = [os.path.join(depth_dir, i) for i in valid_df['image_id']]
    
    print(f"Validation Size: {len(valid_images)}")
    
    dataset = NYUDataset(
        valid_images, valid_labels, depth_paths=valid_depths,
        transform=get_valid_transforms(cfg), cfg=cfg
    )
    
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2) # Adjust batch size if needed
    
    # 2. Load Models
    models = []
    for path in cfg.MODEL_PATHS:
        if not os.path.exists(path):
            print(f"Warning: Model not found at {path}. Skipping.")
            continue
        
        m = MultiTaskDeepLab(num_classes=cfg.NUM_CLASSES, in_channels=4)
        # Load state dict
        state = torch.load(path, map_location=cfg.DEVICE)
        m.load_state_dict(state)
        m.to(cfg.DEVICE)
        m.eval()
        models.append(m)
        print(f"Loaded: {path}")
        
    if not models:
        print("No models loaded!")
        return

    # 3. Inference
    confusion_matrix = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=np.int64)
    
    print("Running Ensemble Inference...")
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(cfg.DEVICE)
            
            # Aggregate Logits
            avg_logits = None
            
            for m in models:
                logits = m(images) # [B, C, H, W]
                if avg_logits is None:
                    avg_logits = logits
                else:
                    avg_logits += logits
            
            avg_logits /= len(models)
            preds = torch.argmax(avg_logits, dim=1)
            
            # Update Metrics
            confusion_matrix = update_confusion_matrix(
                preds.cpu().numpy(),
                labels.numpy(),
                cfg.NUM_CLASSES,
                cfg.IGNORE_INDEX,
                confusion_matrix
            )
            
    pixel_acc, miou, class_iou = compute_metrics(confusion_matrix)
    print("\n--- Ensemble Results ---")
    print(f"Pixel Acc: {pixel_acc:.4f}")
    print(f"mIoU:      {miou:.4f}")
    print("------------------------")
    print("Class IoUs:")
    classes = ["Bed", "Books", "Ceiling", "Chair", "Floor", "Furniture", "Objects", "Picture", "Sofa", "Table", "TV", "Wall", "Window"]
    for cls_name, iou_val in zip(classes, class_iou):
        print(f"{cls_name}: {iou_val:.4f}")

if __name__ == '__main__':
    main()
