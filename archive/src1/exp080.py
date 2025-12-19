import os
import random
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    EXP_NAME = "exp080_multitask_edge_resnet101"
    SEED = 42
    IMAGE_SIZE = (480, 640) # Height, Width
    EPOCHS = 30
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_CLASSES = 13
    IGNORE_INDEX = 255

    # Loss Weights
    DEPTH_LOSS_LAMBDA = 0.1
    EDGE_LOSS_LAMBDA = 1.0

    # Depth range
    DEPTH_MIN = 0.71  # m
    DEPTH_MAX = 10.0  # m
    
    # Normalization constants (RGB)
    MEAN = [0.525, 0.443, 0.400]
    STD = [0.281, 0.281, 0.293]

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
            "depth": "image",         # Input depth
            "depth_target": "image",  # Target depth
            "edge_target": "mask",    # Edge map (binary), use mask to ensure proper interpolation (nearest)
        },
    )

def get_valid_transforms(cfg):
    return A.Compose(
        [
            A.Resize(height=cfg.IMAGE_SIZE[0], width=cfg.IMAGE_SIZE[1]),
        ],
        additional_targets={
            "depth": "image",
            "depth_target": "image",
            "edge_target": "mask",
        },
    )

# --- Helper: Edge Map Generation ---
def generate_edge_map(label, ignore_index=255):
    """
    Generate binary edge map from label map.
    label: [H, W] numpy array (int)
    Returns: [H, W] float32 array (0.0 or 1.0)
    """
    # Create a copy to modify
    lbl = label.copy()
    
    # Mask ignore regions
    mask_ignore = (lbl == ignore_index)
    
    # Temporarily fill ignore regions to avoid edges with 255
    # Filling with a common class (e.g. 0) or neighbor might still create edges
    # but at least prevents 255 from being treated as a distinct class.
    # Simple strategy: fill with 0.
    lbl[mask_ignore] = 0
    
    lbl = lbl.astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    
    eroded = cv2.erode(lbl, kernel, iterations=1)
    
    # Edge is where label is different from eroded version
    edge = (lbl != eroded)
    
    # Remove edges that fall within ignore regions (we don't want supervision there)
    edge[mask_ignore] = 0
    
    return edge.astype(np.float32)

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
        if label is None:
            raise ValueError(f"Label not found: {self.label_paths[idx]}")
        if label.ndim == 3:
            label = label[:, :, 0]

        # Generate Edge Map (Before transform, on raw label)
        # Note: If we do it after transform (on augmented label), it might be better 
        # because resizing/rotation interpolates labels.
        # BUT: Albumentations handles masks correctly.
        # Let's compute it here on the fly from the raw label.
        edge_target = generate_edge_map(label)

        # --- Depth ---
        depth_input = None
        depth_target = None
        
        if self.depth_paths is not None:
            raw_depth = cv2.imread(self.depth_paths[idx], cv2.IMREAD_UNCHANGED)
            if raw_depth is None:
                raise ValueError(f"Depth not found: {self.depth_paths[idx]}")

            raw_depth = raw_depth.astype(np.float32) / 1000.0  # mm -> m
            dmin = self.cfg.DEPTH_MIN
            dmax = self.cfg.DEPTH_MAX
            
            raw_depth = np.clip(raw_depth, dmin, dmax)

            # 1. Input Depth (Inverse)
            inv = 1.0 / raw_depth
            inv_min = 1.0 / dmax
            inv_max = 1.0 / dmin
            depth_input = (inv - inv_min) / (inv_max - inv_min)

            # 2. Target Depth (Linear)
            depth_target = (raw_depth - dmin) / (dmax - dmin)

        # --- Albumentations ---
        if self.transform is not None:
            augmented = self.transform(
                image=image,
                mask=label,                   # Seg Mask
                edge_target=edge_target,      # Edge Mask
                depth=depth_input,            # Depth Input
                depth_target=depth_target     # Depth Target
            )
            image = augmented["image"]
            label = augmented["mask"]
            edge_target = augmented["edge_target"]
            depth_input = augmented["depth"]
            depth_target = augmented["depth_target"]

        # --- Tensor Conversion ---
        # RGB
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(self.cfg.MEAN).view(-1, 1, 1)
        std = torch.tensor(self.cfg.STD).view(-1, 1, 1)
        image = (image - mean) / std

        # Depth Input -> Append to Image
        if depth_input is not None:
            depth_input = np.clip(depth_input, 0.0, 1.0)
            d_tensor = torch.from_numpy(depth_input).unsqueeze(0).float()
            image = torch.cat([image, d_tensor], dim=0)

        # Depth Target -> Separate Tensor
        if depth_target is not None:
            depth_target = torch.from_numpy(depth_target).unsqueeze(0).float()
            
        # Edge Target -> Separate Tensor
        # Albumentations might return it as [H, W]
        edge_target = torch.from_numpy(edge_target).unsqueeze(0).float() # [1, H, W]
        
        # Label
        label = torch.from_numpy(label).long()

        return image, label, depth_target, edge_target

# --- Model ---
class MultiTaskDeepLabEdge(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.backbone = smp.DeepLabV3Plus(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
        decoder_channels = 256
        
        # Depth Head
        self.depth_head = nn.Conv2d(
            in_channels=decoder_channels, 
            out_channels=1,
            kernel_size=3,
            padding=1,
        )
        
        # Edge Head
        self.edge_head = nn.Conv2d(
            in_channels=decoder_channels,
            out_channels=1,
            kernel_size=3,
            padding=1,
        )
    
    def forward(self, x):
        features = self.backbone.encoder(x)
        decoder_out = self.backbone.decoder(features)
        
        seg_logits = self.backbone.segmentation_head(decoder_out)
        depth_pred = self.depth_head(decoder_out)
        edge_pred = self.edge_head(decoder_out)
        
        return seg_logits, depth_pred, edge_pred

# --- Utils ---
def save_config(cfg, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(cfg.to_dict(), f, indent=4)

def init_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'train_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 
                         'train_loss', 'train_seg', 'train_depth', 'train_edge',
                         'valid_loss', 'valid_seg', 'valid_depth', 'valid_edge',
                         'valid_mIoU', 'valid_pixel_acc'])
    return log_path

def log_metrics(log_path, epoch, t_loss, t_seg, t_d, t_e, v_loss, v_seg, v_d, v_e, v_miou, v_acc):
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, t_loss, t_seg, t_d, t_e, v_loss, v_seg, v_d, v_e, v_miou, v_acc])

def save_visualizations(images, labels, depth_targets, edge_targets, seg_preds, depth_preds, edge_preds, output_dir, epoch, cfg):
    sample_dir = os.path.join(output_dir, 'samples')
    os.makedirs(sample_dir, exist_ok=True)
    
    mean = np.array(cfg.MEAN)
    std = np.array(cfg.STD)

    for i in range(min(3, len(images))):
        # RGB
        img_tensor = images[i]
        rgb = img_tensor[:3, :, :].cpu().numpy().transpose(1, 2, 0)
        rgb = (rgb * std + mean) * 255.0
        rgb = rgb.astype(np.uint8)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # Targets
        d_gt = depth_targets[i].squeeze().cpu().numpy()
        d_gt_vis = (d_gt * 255.0).astype(np.uint8)
        d_gt_vis = cv2.cvtColor(d_gt_vis, cv2.COLOR_GRAY2BGR)

        e_gt = edge_targets[i].squeeze().cpu().numpy()
        e_gt_vis = (e_gt * 255.0).astype(np.uint8)
        e_gt_vis = cv2.cvtColor(e_gt_vis, cv2.COLOR_GRAY2BGR)

        gt = labels[i].cpu().numpy().astype(np.uint8)
        gt_vis = (gt * 20).astype(np.uint8)
        gt_vis = cv2.cvtColor(gt_vis, cv2.COLOR_GRAY2BGR)

        # Preds
        d_pred = depth_preds[i].squeeze().cpu().numpy()
        d_pred = np.clip(d_pred, 0, 1)
        d_pred_vis = (d_pred * 255.0).astype(np.uint8)
        d_pred_vis = cv2.cvtColor(d_pred_vis, cv2.COLOR_GRAY2BGR)

        # Edge Pred (Sigmoid -> 0-1)
        e_pred = torch.sigmoid(edge_preds[i]).squeeze().cpu().numpy()
        e_pred_vis = (e_pred * 255.0).astype(np.uint8)
        e_pred_vis = cv2.cvtColor(e_pred_vis, cv2.COLOR_GRAY2BGR)

        seg_pred = seg_preds[i].cpu().numpy().astype(np.uint8)
        seg_pred_vis = (seg_pred * 20).astype(np.uint8)
        seg_pred_vis = cv2.cvtColor(seg_pred_vis, cv2.COLOR_GRAY2BGR)

        # Layout
        # Row 1: RGB | Seg GT | Seg Pred
        # Row 2: Edge GT | Edge Pred | Depth GT | Depth Pred
        # Concatenating all: RGB | Seg(GT/Pred) | Edge(GT/Pred) | Depth(GT/Pred)
        concat_img = np.hstack([rgb, gt_vis, seg_pred_vis, e_gt_vis, e_pred_vis, d_gt_vis, d_pred_vis])
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
def train_one_epoch(model, loader, criterion_seg, criterion_edge, optimizer, device, cfg):
    model.train()
    total_loss_avg = 0.0
    seg_loss_avg = 0.0
    depth_loss_avg = 0.0
    edge_loss_avg = 0.0
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels, depth_targets, edge_targets in pbar:
        images = images.to(device)
        labels = labels.to(device)
        depth_targets = depth_targets.to(device)
        edge_targets = edge_targets.to(device)

        optimizer.zero_grad()
        seg_logits, depth_pred, edge_pred = model(images)

        # 1. Segmentation Loss
        loss_seg = criterion_seg(seg_logits, labels)

        # 2. Depth Loss (masked)
        valid_mask_d = (depth_targets > 1e-4).float()
        loss_depth_raw = F.l1_loss(depth_pred, depth_targets, reduction='none')
        loss_depth = (loss_depth_raw * valid_mask_d).sum() / (valid_mask_d.sum() + 1e-6)

        # 3. Edge Loss (BCE)
        loss_edge = criterion_edge(edge_pred, edge_targets)

        # Combined Loss
        loss = loss_seg + (cfg.DEPTH_LOSS_LAMBDA * loss_depth) + (cfg.EDGE_LOSS_LAMBDA * loss_edge)

        loss.backward()
        optimizer.step()

        total_loss_avg += loss.item()
        seg_loss_avg += loss_seg.item()
        depth_loss_avg += loss_depth.item()
        edge_loss_avg += loss_edge.item()
        
        pbar.set_postfix({'loss': loss.item(), 'seg': loss_seg.item(), 'edge': loss_edge.item()})

    n = len(loader)
    return total_loss_avg / n, seg_loss_avg / n, depth_loss_avg / n, edge_loss_avg / n

def validate(model, loader, criterion_seg, criterion_edge, device, cfg):
    model.eval()
    total_loss_avg = 0.0
    seg_loss_avg = 0.0
    depth_loss_avg = 0.0
    edge_loss_avg = 0.0
    
    confusion_matrix = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=np.int64)
    
    last_vals = {} # Store last batch for vis

    with torch.no_grad():
        pbar = tqdm(loader, desc="Valid", leave=False)
        for images, labels, depth_targets, edge_targets in pbar:
            images = images.to(device)
            labels = labels.to(device)
            depth_targets = depth_targets.to(device)
            edge_targets = edge_targets.to(device)

            seg_logits, depth_pred, edge_pred = model(images)
            
            # Losses
            loss_seg = criterion_seg(seg_logits, labels)
            
            valid_mask_d = (depth_targets > 1e-4).float()
            loss_depth_raw = F.l1_loss(depth_pred, depth_targets, reduction='none')
            loss_depth = (loss_depth_raw * valid_mask_d).sum() / (valid_mask_d.sum() + 1e-6)
            
            loss_edge = criterion_edge(edge_pred, edge_targets)
            
            loss = loss_seg + (cfg.DEPTH_LOSS_LAMBDA * loss_depth) + (cfg.EDGE_LOSS_LAMBDA * loss_edge)

            total_loss_avg += loss.item()
            seg_loss_avg += loss_seg.item()
            depth_loss_avg += loss_depth.item()
            edge_loss_avg += loss_edge.item()

            # Metrics (Seg Only)
            preds = torch.argmax(seg_logits, dim=1)
            confusion_matrix = update_confusion_matrix(
                preds.cpu().numpy(), 
                labels.cpu().numpy(), 
                cfg.NUM_CLASSES, 
                cfg.IGNORE_INDEX, 
                confusion_matrix
            )
            
            last_vals['images'] = images
            last_vals['labels'] = labels
            last_vals['depth_targets'] = depth_targets
            last_vals['edge_targets'] = edge_targets
            last_vals['seg_preds'] = preds
            last_vals['depth_preds'] = depth_pred
            last_vals['edge_preds'] = edge_pred

    pixel_acc, miou, class_iou = compute_metrics(confusion_matrix)
    
    n = len(loader)
    return (total_loss_avg / n, seg_loss_avg / n, depth_loss_avg / n, edge_loss_avg / n,
            pixel_acc, miou, class_iou,
            last_vals['images'], last_vals['labels'], last_vals['depth_targets'], last_vals['edge_targets'],
            last_vals['seg_preds'], last_vals['depth_preds'], last_vals['edge_preds'])

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
    model = MultiTaskDeepLabEdge(num_classes=cfg.NUM_CLASSES, in_channels=4)
    model.to(cfg.DEVICE)

    # --- Optimizer ---
    criterion_seg = nn.CrossEntropyLoss(ignore_index=cfg.IGNORE_INDEX)
    criterion_edge = nn.BCEWithLogitsLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

    best_miou = 0.0

    for epoch in range(1, cfg.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{cfg.EPOCHS}")

        t_loss, t_seg, t_d, t_e = train_one_epoch(
            model, train_loader, criterion_seg, criterion_edge, optimizer, cfg.DEVICE, cfg
        )
        v_loss, v_seg, v_d, v_e, pixel_acc, miou, class_iou, vis_img, vis_lbl, vis_d_gt, vis_e_gt, vis_s_p, vis_d_p, vis_e_p = validate(
            model, valid_loader, criterion_seg, criterion_edge, cfg.DEVICE, cfg
        )

        scheduler.step()

        print(f"Train Loss: {t_loss:.4f} (Seg: {t_seg:.3f}, D: {t_d:.3f}, E: {t_e:.3f})")
        print(f"Valid Loss: {v_loss:.4f} (Seg: {v_seg:.3f}, D: {v_d:.3f}, E: {v_e:.3f})")
        print(f"Pixel Acc: {pixel_acc:.4f} | mIoU: {miou:.4f}")

        log_metrics(log_path, epoch, t_loss, t_seg, t_d, t_e, v_loss, v_seg, v_d, v_e, miou, pixel_acc)

        if epoch % 5 == 0 or miou > best_miou:
            save_visualizations(vis_img, vis_lbl, vis_d_gt, vis_e_gt, vis_s_p, vis_d_p, vis_e_p, cfg.OUTPUT_DIR, epoch, cfg)

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
