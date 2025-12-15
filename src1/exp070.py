import os
import random
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

# --- Configuration ---
class Config:
    EXP_NAME = "exp070_rgbd_normals_resnet101"
    SEED = 42
    IMAGE_SIZE = (480, 640) # Height, Width
    EPOCHS = 30
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_CLASSES = 13
    IGNORE_INDEX = 255

    # Depth range
    DEPTH_MIN = 0.71  # m
    DEPTH_MAX = 10.0  # m

    # Normalization constants (RGB)
    MEAN = [0.525, 0.443, 0.400] # R: 133.88/255, G: 112.97/255, B: 102.11/255
    STD = [0.281, 0.281, 0.293]  # R: 71.74/255, G: 71.53/255, B: 74.75/255

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    DATA_ROOT = 'data/train'
    OUTPUT_DIR = os.path.join("data", "outputs", EXP_NAME)

    @classmethod
    def to_dict(cls):
        d = {}
        for k, v in cls.__dict__.items():
            if k.startswith('__') or k == 'DEVICE':
                continue
            if isinstance(v, (classmethod, staticmethod, type)):
                continue
            if callable(v):
                continue
            d[k] = v
        return d

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
def get_train_transforms(cfg):
    return A.Compose(
        [
            A.Resize(height=cfg.IMAGE_SIZE[0], width=cfg.IMAGE_SIZE[1]),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=cfg.IGNORE_INDEX
            ),
            A.RandomCrop(
                height=cfg.IMAGE_SIZE[0],
                width=cfg.IMAGE_SIZE[1],
                p=0.5,
            ),
        ],
        additional_targets={
            "depth": "image",
            "normals": "image", # 3ch normal map treated as image
        },
    )

def get_valid_transforms(cfg):
    return A.Compose(
        [
            A.Resize(height=cfg.IMAGE_SIZE[0], width=cfg.IMAGE_SIZE[1]),
        ],
        additional_targets={
            "depth": "image",
            "normals": "image",
        },
    )

# --- Helper: Surface Normals Computation ---
def compute_surface_normals(depth):
    """
    Compute surface normals from depth map.
    depth: [H, W] float32 array (meters)
    Returns: [H, W, 3] float32 array, values in [0, 1] (mapped from [-1, 1])
    """
    # 1. Compute gradients (dz/dx, dz/dy)
    # Using central differences with padding for borders
    
    # Pad depth map to handle borders
    # (Using edge replication or constant usually OK, let's use replication for continuity)
    depth_pad = np.pad(depth, ((1, 1), (1, 1)), mode='edge')
    
    # Central difference: f(x+1) - f(x-1)
    # Scale factor? 
    # Technically dz/dx = (z(x+1) - z(x-1)) / 2.
    # But since we just want direction, the scale 2 doesn't matter much BEFORE normalization.
    # However, to be somewhat consistent with pixel units vs meters, we might care.
    # But standard practice for simple approx is just this difference.
    
    dzdy = depth_pad[2:, 1:-1] - depth_pad[:-2, 1:-1] # Y gradient
    dzdx = depth_pad[1:-1, 2:] - depth_pad[1:-1, :-2] # X gradient

    # 2. Construct normal vector (-dzdx, -dzdy, 1)
    # This approximate normal points towards camera (+Z).
    # N = (-dz/dx, -dz/dy, 1) if Z axis points out/in.
    # Let's assume standard camera coords where Z is depth.
    
    # We stack along the last axis to make it (H, W, 3) image-like
    # -dzdx, -dzdy, 1
    ones = np.ones_like(depth)
    normal = np.stack([-dzdx, -dzdy, ones], axis=2) # (H, W, 3)

    # 3. Normalize vector
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal = normal / (norm + 1e-6) # [-1, 1] range

    # 4. Map to [0, 1]
    # n_new = (n + 1) / 2
    normal = (normal * 0.5) + 0.5 
    
    return normal.astype(np.float32)

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # H,W,3

        # --- Label ---
        label = cv2.imread(self.label_paths[idx], cv2.IMREAD_UNCHANGED)
        if label is None:
            raise ValueError(f"Label not found: {self.label_paths[idx]}")
        if label.ndim == 3:
            label = label[:, :, 0]

        # --- Depth & Normals ---
        depth_input = None
        normals = None

        if self.depth_paths is not None:
            raw_depth = cv2.imread(self.depth_paths[idx], cv2.IMREAD_UNCHANGED)
            if raw_depth is None:
                raise ValueError(f"Depth not found: {self.depth_paths[idx]}")

            raw_depth = raw_depth.astype(np.float32) / 1000.0  # mm -> m
            dmin = self.cfg.DEPTH_MIN
            dmax = self.cfg.DEPTH_MAX
            
            # Clip raw depth for calculations
            raw_depth_clipped = np.clip(raw_depth, dmin, dmax)
            
            # Identify invalid depth (0 or close to 0) to mask normals later
            valid_mask = (raw_depth > 1e-3)

            # 1. Compute Normals (before encoding/transforms, using metric depth)
            normals = compute_surface_normals(raw_depth_clipped) # H,W,3 in [0,1]
            
            # Set normals at invalid depth pixels to neutral value (0.5, 0.5, 0.5) which corresponds to vector 0
            # This prevents undefined/noisy normals from missing depth data
            normals[~valid_mask] = 0.5

            # 2. Prepare Inverse Depth Input
            inv = 1.0 / raw_depth_clipped
            inv_min = 1.0 / dmax
            inv_max = 1.0 / dmin
            depth_input = (inv - inv_min) / (inv_max - inv_min) # H,W in [0,1]

        # --- Albumentations ---
        if self.transform is not None:
            augmented = self.transform(
                image=image,
                mask=label,
                depth=depth_input,
                normals=normals
            )
            image = augmented["image"]     # H,W,3
            label = augmented["mask"]      # H,W
            depth_input = augmented["depth"] # H,W
            normals = augmented["normals"]   # H,W,3

        # --- Tensor Conversion ---
        # RGB
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(self.cfg.MEAN).view(-1, 1, 1)
        std = torch.tensor(self.cfg.STD).view(-1, 1, 1)
        image = (image - mean) / std

        # Concatenate: RGB (3) + Depth (1) + Normals (3) = 7 Channels
        concat_list = [image]
        
        if depth_input is not None:
            depth_input = np.clip(depth_input, 0.0, 1.0)
            d_tensor = torch.from_numpy(depth_input).unsqueeze(0).float() # [1,H,W]
            concat_list.append(d_tensor)
        
        if normals is not None:
            normals = np.clip(normals, 0.0, 1.0)
            n_tensor = torch.from_numpy(normals).permute(2, 0, 1).float() # [3,H,W]
            concat_list.append(n_tensor)

        image = torch.cat(concat_list, dim=0) # [7, H, W]

        # Label
        label = torch.from_numpy(label).long()

        return image, label

# --- Utils for Logging ---
def save_config(cfg, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(cfg.to_dict(), f, indent=4)

def init_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'train_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'valid_loss', 'valid_mIoU', 'valid_pixel_acc'])
    return log_path

def log_metrics(log_path, epoch, train_loss, valid_loss, valid_miou, valid_pixel_acc):
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, valid_loss, valid_miou, valid_pixel_acc])

def save_visualizations(images, labels, preds, output_dir, epoch, cfg):
    sample_dir = os.path.join(output_dir, 'samples')
    os.makedirs(sample_dir, exist_ok=True)

    mean = np.array(cfg.MEAN)
    std = np.array(cfg.STD)

    for i in range(min(3, len(images))):
        img_tensor = images[i]
        
        # 1. RGB
        rgb = img_tensor[:3, :, :].cpu().numpy().transpose(1, 2, 0)
        rgb = (rgb * std + mean) * 255.0
        rgb = rgb.astype(np.uint8)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # 2. Depth (Channel 3)
        depth = img_tensor[3, :, :].cpu().numpy()
        d_vis = (depth * 255.0).astype(np.uint8)
        d_vis = cv2.cvtColor(d_vis, cv2.COLOR_GRAY2BGR)

        # 3. Normals (Channels 4,5,6)
        normals = img_tensor[4:7, :, :].cpu().numpy().transpose(1, 2, 0) # H,W,3 in [0,1]
        n_vis = (normals * 255.0).astype(np.uint8)
        n_vis = cv2.cvtColor(n_vis, cv2.COLOR_RGB2BGR) # Convert to BGR for opencv

        # GT
        gt = labels[i].cpu().numpy().astype(np.uint8)
        gt_vis = (gt * 20).astype(np.uint8)
        gt_vis = cv2.cvtColor(gt_vis, cv2.COLOR_GRAY2BGR)

        # Pred
        pred = preds[i]
        pred_vis = (pred * 20).astype(np.uint8)
        pred_vis = cv2.cvtColor(pred_vis, cv2.COLOR_GRAY2BGR)

        # Concat
        concat_img = np.hstack([rgb, d_vis, n_vis, gt_vis, pred_vis])
        cv2.imwrite(os.path.join(sample_dir, f'epoch_{epoch}_sample_{i}.png'), concat_img)

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

# --- Training ---
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    return running_loss / len(loader)

def validate(model, loader, criterion, device, num_classes, ignore_index):
    model.eval()
    running_loss = 0.0
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    last_batch_images = None
    last_batch_labels = None
    last_batch_preds = None

    with torch.no_grad():
        pbar = tqdm(loader, desc="Valid", leave=False)
        for images, labels in pbar:
            images_dev = images.to(device)
            labels_dev = labels.to(device)
            outputs = model(images_dev)
            loss = criterion(outputs, labels_dev)
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            preds_np = preds.cpu().numpy()
            labels_np = labels_dev.cpu().numpy()
            confusion_matrix = update_confusion_matrix(
                preds_np, labels_np, num_classes, ignore_index, confusion_matrix
            )
            last_batch_images = images
            last_batch_labels = labels
            last_batch_preds = preds_np

    pixel_acc, miou, class_iou = compute_metrics(confusion_matrix)
    return running_loss / len(loader), pixel_acc, miou, class_iou, last_batch_images, last_batch_labels, last_batch_preds

def main():
    cfg = Config()
    seed_everything(cfg.SEED)

    print(f"Using device: {cfg.DEVICE}")
    print(f"Output directory: {cfg.OUTPUT_DIR}")

    save_config(cfg, cfg.OUTPUT_DIR)
    log_path = init_logger(cfg.OUTPUT_DIR)

    # --- Data Paths ---
    image_dir = os.path.join(cfg.DATA_ROOT, 'image')
    label_dir = os.path.join(cfg.DATA_ROOT, 'label')
    depth_dir = os.path.join(cfg.DATA_ROOT, 'depth')

    image_files = sorted(os.listdir(image_dir))
    label_files = sorted(os.listdir(label_dir))
    depth_files = sorted(os.listdir(depth_dir))

    image_paths = [os.path.join(image_dir, f) for f in image_files]
    label_paths = [os.path.join(label_dir, f) for f in label_files]
    depth_paths = [os.path.join(depth_dir, f) for f in depth_files]

    train_idx, valid_idx = train_test_split(
        range(len(image_paths)),
        test_size=0.2,
        random_state=cfg.SEED,
        shuffle=True
    )

    train_images = [image_paths[i] for i in train_idx]
    train_labels = [label_paths[i] for i in train_idx]
    train_depths = [depth_paths[i] for i in train_idx]

    valid_images = [image_paths[i] for i in valid_idx]
    valid_labels = [label_paths[i] for i in valid_idx]
    valid_depths = [depth_paths[i] for i in valid_idx]

    print(f"Train size: {len(train_images)}, Valid size: {len(valid_images)}")

    train_dataset = NYUDataset(
        train_images,
        train_labels,
        depth_paths=train_depths,
        transform=get_train_transforms(cfg),
        cfg=cfg,
    )

    valid_dataset = NYUDataset(
        valid_images,
        valid_labels,
        depth_paths=valid_depths,
        transform=get_valid_transforms(cfg),
        cfg=cfg,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    # --- Model ---
    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=7, # RGB(3) + Depth(1) + Normals(3)
        classes=cfg.NUM_CLASSES
    )
    model.to(cfg.DEVICE)

    # --- Loss, Optimizer ---
    criterion = nn.CrossEntropyLoss(
        ignore_index=cfg.IGNORE_INDEX
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.EPOCHS
    )

    # --- Train Loop ---
    best_miou = 0.0

    for epoch in range(1, cfg.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{cfg.EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE)
        valid_loss, pixel_acc, miou, class_iou, vis_imgs, vis_lbls, vis_preds = validate(
            model, valid_loader, criterion, cfg.DEVICE, cfg.NUM_CLASSES, cfg.IGNORE_INDEX
        )

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
        print(f"Pixel Acc: {pixel_acc:.4f} | mIoU: {miou:.4f}")

        log_metrics(log_path, epoch, train_loss, valid_loss, miou, pixel_acc)

        if epoch % 5 == 0 or miou > best_miou:
            save_visualizations(vis_imgs, vis_lbls, vis_preds, cfg.OUTPUT_DIR, epoch, cfg)

        if miou > best_miou:
            best_miou = miou
            print(f"New best mIoU: {best_miou:.4f}. Saving model...")
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, 'model_best.pth'))
            best_metrics = {
                'best_epoch': epoch,
                'best_val_mIoU': float(best_miou),
                'class_iou': [float(x) for x in class_iou]
            }
            with open(os.path.join(cfg.OUTPUT_DIR, 'best_metrics.json'), 'w') as f:
                json.dump(best_metrics, f, indent=4)

if __name__ == '__main__':
    main()
