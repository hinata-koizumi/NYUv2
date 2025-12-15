import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

# --- Configuration ---
class Config:
    EXP_NAME = "exp064_ensemble_search"
    SEED = 42
    IMAGE_SIZE = (480, 640)
    BATCH_SIZE = 8
    NUM_CLASSES = 13
    IGNORE_INDEX = 255
    
    # Depth range for Dataset (same as training)
    DEPTH_MIN = 0.71
    DEPTH_MAX = 10.0
    
    # Normalization (RGB)
    MEAN = [0.525, 0.443, 0.400]
    STD = [0.281, 0.281, 0.293]

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    DATA_ROOT = 'data/train'
    
    # Model Paths
    MODEL_PATH_060 = "data/outputs/exp060_multitask_resnet101/model_best.pth"
    MODEL_PATH_062 = "data/outputs/exp062_class_weighted_resnet101/model_best.pth"
    MODEL_PATH_063 = "data/outputs/exp063_boundary_aware_resnet101/model_best.pth"
    
    # Ensemble Weights to Evaluate
    WEIGHT_SETS = [
        (1.0, 0.0, 0.0),   # baseline: 060
        (0.6, 0.2, 0.2),   # Candidate 1
        (0.5, 0.25, 0.25), # Smooth
        (0.5, 0.5, 0.0),   # 060 + 062
        (0.5, 0.0, 0.5),   # 060 + 063
        (0.4, 0.3, 0.3),   # More spice
        (0.0, 1.0, 0.0),   # 062 only
        (0.0, 0.0, 1.0),   # 063 only
    ]

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# --- Transforms ---
def get_valid_transforms(cfg):
    return A.Compose(
        [
            A.Resize(height=cfg.IMAGE_SIZE[0], width=cfg.IMAGE_SIZE[1]),
        ],
        additional_targets={
            "depth": "image",
            "depth_target": "image",
        },
    )

# --- Dataset ---
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
        # --- RGB ---
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- Label ---
        label = cv2.imread(self.label_paths[idx], cv2.IMREAD_UNCHANGED)
        if label.ndim == 3:
            label = label[:, :, 0]

        # --- Depth ---
        depth_input = None
        depth_target = None
        
        if self.depth_paths is not None:
            raw_depth = cv2.imread(self.depth_paths[idx], cv2.IMREAD_UNCHANGED)
            raw_depth = raw_depth.astype(np.float32) / 1000.0  # mm -> m
            dmin = self.cfg.DEPTH_MIN
            dmax = self.cfg.DEPTH_MAX
            raw_depth = np.clip(raw_depth, dmin, dmax)

            # 1. Input Depth (Inverse Encoding)
            inv = 1.0 / raw_depth
            inv_min = 1.0 / dmax
            inv_max = 1.0 / dmin
            depth_input = (inv - inv_min) / (inv_max - inv_min)

            # 2. Target Depth (Linear Encoding)
            depth_target = (raw_depth - dmin) / (dmax - dmin)

        # --- Albumentations ---
        if self.transform is not None:
            augmented = self.transform(
                image=image,
                mask=label,
                depth=depth_input,
                depth_target=depth_target
            )
            image = augmented["image"]
            label = augmented["mask"]
            depth_input = augmented["depth"]
            depth_target = augmented["depth_target"]

        # --- Tensor Conversion ---
        # RGB
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(self.cfg.MEAN).view(-1, 1, 1)
        std = torch.tensor(self.cfg.STD).view(-1, 1, 1)
        image = (image - mean) / std

        # Depth Input
        if depth_input is not None:
            depth_input = np.clip(depth_input, 0.0, 1.0)
            d_tensor = torch.from_numpy(depth_input).unsqueeze(0).float()
            image = torch.cat([image, d_tensor], dim=0)

        # Label
        label = torch.from_numpy(label).long()

        return image, label

# --- Model ---
class MultiTaskDeepLab(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.backbone = smp.DeepLabV3Plus(
            encoder_name="resnet101",
            encoder_weights=None, # Loading custom weights anyway
            in_channels=in_channels,
            classes=num_classes,
        )
        self.depth_head = nn.Conv2d(
            in_channels=256, 
            out_channels=1,
            kernel_size=3,
            padding=1,
        )
    
    def forward(self, x):
        features = self.backbone.encoder(x)
        decoder_out = self.backbone.decoder(features)
        seg_logits = self.backbone.segmentation_head(decoder_out)
        return seg_logits, None # We only care about seg logits for ensemble

# --- Metrics ---
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

def validate_ensemble(models, loader, device, cfg):
    for m in models:
        m.eval()
    
    weight_sets = cfg.WEIGHT_SETS
    cms = [np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=np.int64) for _ in weight_sets]
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating Ensemble", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.cpu().numpy() # [B, H, W]

            # Forward pass for all models
            # Model 0: Exp060
            logits0, _ = models[0](images)
            # Model 1: Exp062
            logits1, _ = models[1](images)
            # Model 2: Exp063
            logits2, _ = models[2](images)
            
            logits_list = [logits0, logits1, logits2]
            
            # Evaluate each weight set
            for k, weights in enumerate(weight_sets):
                w0, w1, w2 = weights
                # Linear combination of logits
                ens_logits = w0 * logits_list[0] + w1 * logits_list[1] + w2 * logits_list[2]
                
                preds = torch.argmax(ens_logits, dim=1).cpu().numpy()
                
                cms[k] = update_confusion_matrix(
                    preds, labels, cfg.NUM_CLASSES, cfg.IGNORE_INDEX, cms[k]
                )

    # Report Stats
    best_miou = -1
    best_weights = None
    best_class_iou = None

    print("\n--- Ensemble Results ---")
    print(f"{'Weights (060, 062, 063)':<25} | {'mIoU':<8} | {'PixAcc':<8}")
    print("-" * 50)
    
    for k, weights in enumerate(weight_sets):
        pix_acc, miou, class_iou = compute_metrics(cms[k])
        print(f"{str(weights):<25} | {miou:.5f}  | {pix_acc:.5f}")
        
        if miou > best_miou:
            best_miou = miou
            best_weights = weights
            best_class_iou = class_iou
            
    print("-" * 50)
    print(f"Best Weights: {best_weights} with mIoU: {best_miou:.5f}")
    if best_class_iou is not None:
        print(f"Best Class IoU: {[float(f'{x:.4f}') for x in best_class_iou]}")
        # Save best metrics to file for reference
        results = {
            "best_weights": best_weights,
            "best_miou": float(best_miou),
            "class_iou": [float(x) for x in best_class_iou]
        }
        with open(os.path.join("data", "outputs", cfg.EXP_NAME, "best_ensemble_metrics.json"), 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Saved best metrics to data/outputs/{cfg.EXP_NAME}/best_ensemble_metrics.json")
    
    return best_weights, best_miou

def load_model(path, device, cfg):
    model = MultiTaskDeepLab(num_classes=cfg.NUM_CLASSES, in_channels=4)
    # Check if path exists
    if not os.path.exists(path):
        print(f"Warning: Model path not found: {path}")
    else:
        print(f"Loading weights from {path}")
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    return model

def main():
    cfg = Config()
    seed_everything(cfg.SEED)
    
    print(f"Using device: {cfg.DEVICE}")
    
    # --- Data ---
    image_dir = os.path.join(cfg.DATA_ROOT, 'image')
    label_dir = os.path.join(cfg.DATA_ROOT, 'label')
    depth_dir = os.path.join(cfg.DATA_ROOT, 'depth')
    
    image_files = sorted(os.listdir(image_dir))
    label_files = sorted(os.listdir(label_dir))
    depth_files = sorted(os.listdir(depth_dir))
    
    image_paths = [os.path.join(image_dir, f) for f in image_files]
    label_paths = [os.path.join(label_dir, f) for f in label_files]
    depth_paths = [os.path.join(depth_dir, f) for f in depth_files]
    
    # Same split as training
    _, valid_idx = train_test_split(
        range(len(image_paths)),
        test_size=0.2,
        random_state=cfg.SEED,
        shuffle=True
    )
    
    valid_images = [image_paths[i] for i in valid_idx]
    valid_labels = [label_paths[i] for i in valid_idx]
    valid_depths = [depth_paths[i] for i in valid_idx]
    
    print(f"Valid size: {len(valid_images)}")
    
    valid_dataset = NYUDataset(
        valid_images,
        valid_labels,
        depth_paths=valid_depths,
        transform=get_valid_transforms(cfg),
        cfg=cfg
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    
    # --- Load Models ---
    print("Loading models...")
    model060 = load_model(cfg.MODEL_PATH_060, cfg.DEVICE, cfg)
    model062 = load_model(cfg.MODEL_PATH_062, cfg.DEVICE, cfg)
    model063 = load_model(cfg.MODEL_PATH_063, cfg.DEVICE, cfg)
    
    models = [model060, model062, model063]
    
    # --- Run Ensemble Search ---
    validate_ensemble(models, valid_loader, cfg.DEVICE, cfg)

if __name__ == '__main__':
    main()
