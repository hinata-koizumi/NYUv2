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
    EXP_NAME = "exp065_ensemble_tta"
    SEED = 42
    IMAGE_SIZE = (480, 640)
    BATCH_SIZE = 8
    NUM_CLASSES = 13
    IGNORE_INDEX = 255
    
    # Depth range for Dataset
    DEPTH_MIN = 0.71
    DEPTH_MAX = 10.0
    
    # Normalization (RGB)
    MEAN = [0.525, 0.443, 0.400]
    STD = [0.281, 0.281, 0.293]

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    DATA_ROOT = 'data/train'
    
    MODEL_PATH_060 = "data/outputs/exp060_multitask_resnet101/model_best.pth"
    MODEL_PATH_062 = "data/outputs/exp062_class_weighted_resnet101/model_best.pth"
    MODEL_PATH_063 = "data/outputs/exp063_boundary_aware_resnet101/model_best.pth"
    
    # Target Ensemble Weights (Likely candidate)
    ENSEMBLE_WEIGHTS = (0.6, 0.2, 0.2)

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

# --- Transforms & Dataset (Same as exp064) ---
def get_valid_transforms(cfg):
    return A.Compose(
        [
            A.Resize(height=cfg.IMAGE_SIZE[0], width=cfg.IMAGE_SIZE[1]),
        ],
        additional_targets={
            "depth": "image",
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

        depth_input = None
        
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

        if self.transform is not None:
            augmented = self.transform(
                image=image,
                mask=label,
                depth=depth_input
            )
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

# --- Model ---
class MultiTaskDeepLab(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.backbone = smp.DeepLabV3Plus(
            encoder_name="resnet101",
            encoder_weights=None, 
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
        return seg_logits, None 

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

# --- TTA Function ---
def predict_flip_only(model, images):
    with torch.no_grad():
        images_flip = torch.flip(images, dims=[3])  # width dimension
        logits_flip, _ = model(images_flip)
        logits_flip = torch.flip(logits_flip, dims=[3])  # flip back
    return logits_flip

def validate_tta_comparison(models, loader, device, cfg):
    for m in models:
        m.eval()
    
    # 0: No TTA, 1: With TTA
    cm_no_tta = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=np.int64)
    cm_tta = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=np.int64)
    
    w0, w1, w2 = cfg.ENSEMBLE_WEIGHTS
    print(f"Comparing TTA vs No-TTA with weights: {cfg.ENSEMBLE_WEIGHTS}")
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating TTA", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.cpu().numpy()

            # --- No TTA Pass ---
            l0, _ = models[0](images)
            l1, _ = models[1](images)
            l2, _ = models[2](images)
            
            ens_no_tta = w0 * l0 + w1 * l1 + w2 * l2
            preds_no_tta = torch.argmax(ens_no_tta, dim=1).cpu().numpy()
            cm_no_tta = update_confusion_matrix(preds_no_tta, labels, cfg.NUM_CLASSES, cfg.IGNORE_INDEX, cm_no_tta)

            # --- TTA Pass (Reuse No TTA) ---
            # Calculate flip logits only
            l0_flip = predict_flip_only(models[0], images)
            l1_flip = predict_flip_only(models[1], images)
            l2_flip = predict_flip_only(models[2], images)
            
            # Average with normal logits
            l0_tta = 0.5 * (l0 + l0_flip)
            l1_tta = 0.5 * (l1 + l1_flip)
            l2_tta = 0.5 * (l2 + l2_flip)
            
            ens_tta = w0 * l0_tta + w1 * l1_tta + w2 * l2_tta
            preds_tta = torch.argmax(ens_tta, dim=1).cpu().numpy()
            cm_tta = update_confusion_matrix(preds_tta, labels, cfg.NUM_CLASSES, cfg.IGNORE_INDEX, cm_tta)

    # Report
    print("\n--- TTA Comparison Results ---")
    
    p_acc, miou, iou = compute_metrics(cm_no_tta)
    print(f"No TTA   | mIoU: {miou:.5f} | PixAcc: {p_acc:.5f}")
    
    p_acc_tta, miou_tta, iou_tta = compute_metrics(cm_tta)
    print(f"With TTA | mIoU: {miou_tta:.5f} | PixAcc: {p_acc_tta:.5f}")
    
    diff = miou_tta - miou
    print(f"Diff     | mIoU: {diff:+.5f}")
    
    if diff > 0.003:
        print(">> TTA is EFFECTIVE (diff > 0.003)")
    else:
        print(">> TTA impact is small")
    
    return {
        "exp_name": cfg.EXP_NAME,
        "ensemble_weights": list(cfg.ENSEMBLE_WEIGHTS),
        "no_tta": {"miou": float(miou), "pixacc": float(p_acc), "class_iou": np.nan_to_num(iou).tolist()},
        "with_tta": {"miou": float(miou_tta), "pixacc": float(p_acc_tta), "class_iou": np.nan_to_num(iou_tta).tolist()},
        "diff_miou": float(diff),
    }

def load_model(path, device, cfg):
    model = MultiTaskDeepLab(num_classes=cfg.NUM_CLASSES, in_channels=4)
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
    
    image_dir = os.path.join(cfg.DATA_ROOT, 'image')
    label_dir = os.path.join(cfg.DATA_ROOT, 'label')
    depth_dir = os.path.join(cfg.DATA_ROOT, 'depth')
    
    image_files = sorted(os.listdir(image_dir))
    label_files = sorted(os.listdir(label_dir))
    depth_files = sorted(os.listdir(depth_dir))
    
    image_paths = [os.path.join(image_dir, f) for f in image_files]
    label_paths = [os.path.join(label_dir, f) for f in label_files]
    depth_paths = [os.path.join(depth_dir, f) for f in depth_files]
    
    # Same split
    _, valid_idx = train_test_split(
        range(len(image_paths)), test_size=0.2, random_state=cfg.SEED, shuffle=True
    )
    
    valid_images = [image_paths[i] for i in valid_idx]
    valid_labels = [label_paths[i] for i in valid_idx]
    valid_depths = [depth_paths[i] for i in valid_idx]
    
    print(f"Valid size: {len(valid_images)}")
    
    valid_dataset = NYUDataset(
        valid_images, valid_labels, depth_paths=valid_depths,
        transform=get_valid_transforms(cfg), cfg=cfg
    )
    
    valid_loader = DataLoader(
        valid_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, 
        num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn
    )
    
    # Load Models
    print("Loading models...")
    models = [
        load_model(cfg.MODEL_PATH_060, cfg.DEVICE, cfg),
        load_model(cfg.MODEL_PATH_062, cfg.DEVICE, cfg),
        load_model(cfg.MODEL_PATH_063, cfg.DEVICE, cfg),
    ]
    
    # Run
    results = validate_tta_comparison(models, valid_loader, cfg.DEVICE, cfg)
    
    # Save results
    output_dir = os.path.join("data", "outputs", cfg.EXP_NAME)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "metrics.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {output_path}")

if __name__ == '__main__':
    main()
