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
from sklearn.model_selection import KFold
from tqdm import tqdm
import cv2

# --- Configuration ---
class Config:
    EXP_NAME = "exp093_fpn_convnextb_smartcrop"
    SEED = 42
    
    # Train Image Strategy
    RESIZE_HEIGHT = 600
    RESIZE_WIDTH = 800
    CROP_SIZE = (512, 512) # H, W
    
    # Smart Crop Params
    SMART_CROP_PROB = 0.5
    # Small Objects: Books(1), Chair(3), Objects(6), Picture(7), TV(10)
    # 13 classes: 0:Bed, 1:Books, 2:Ceiling, 3:Chair, 4:Floor, 5:Furniture, 
    # 6:Objects, 7:Picture, 8:Sofa, 9:Table, 10:TV, 11:Wall, 12:Window
    SMALL_OBJ_IDS = [1, 3, 6, 7, 10]
    
    EPOCHS = 50
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_CLASSES = 13
    IGNORE_INDEX = 255

    # Fold
    N_FOLDS = 5
    FOLD = 0 

    # Loss Weight
    DEPTH_LOSS_LAMBDA = 0.1

    # Depth range
    DEPTH_MIN = 0.71  # m
    DEPTH_MAX = 10.0  # m
    
    # Normalization constants (RGB)
    MEAN = [0.485, 0.456, 0.406] # ConvNeXt/ImageNet default usually
    STD = [0.229, 0.224, 0.225]

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    DATA_ROOT = '/Users/koizumihinata/NYUv2/data/train'
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
def get_pre_crop_transforms(cfg):
    """Transforms applied BEFORE cropping."""
    return A.Compose(
        [
            # 1. Resize Larger
            A.Resize(height=cfg.RESIZE_HEIGHT, width=cfg.RESIZE_WIDTH),
            
            # 2. Augmentations
            A.HorizontalFlip(p=0.5),
            
            # Random Scale 
            # Note: If we scale down too much, image might be smaller than CROP_SIZE
            # We must Pad if needed.
            A.ShiftScaleRotate(
                shift_limit=0.0, 
                scale_limit=0.2, # 0.8 - 1.2
                rotate_limit=0,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=cfg.IGNORE_INDEX,
                p=0.5
            ),
            
            # Brightness / Contrast 
            A.RandomBrightnessContrast(
                brightness_limit=0.1, 
                contrast_limit=0.1, 
                p=0.5
            ),
            
            # Ensure image is at least CROP_SIZE
            A.PadIfNeeded(
                min_height=cfg.CROP_SIZE[0],
                min_width=cfg.CROP_SIZE[1],
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=cfg.IGNORE_INDEX
            )
        ],
        additional_targets={
            # "depth": "image",  # Removed to prevent leak
            "depth_target": "image",  
        },
    )

def get_valid_transforms(cfg):
    return A.Compose(
        [
            # Resize to Crop Size for consistency 
            A.Resize(height=cfg.CROP_SIZE[0], width=cfg.CROP_SIZE[1]),
        ],
        additional_targets={
            "depth": "image",
            "depth_target": "image",
        },
    )

# --- Smart Crop Logic ---
def smart_crop(image, label, depth_input, depth_target, crop_height, crop_width, target_ids, prob=0.5):
    """
    Performs cropping. 
    With probability `prob`, tries to center the crop around a pixel belonging to `target_ids`.
    Otherwise (or if target not found), performs random crop.
    """
    h, w = label.shape[:2]
    
    # Ensure image is large enough (should be handled by PadIfNeeded, but safety check)
    if h < crop_height or w < crop_width:
        # Should not happen if Pre-Crop transforms are correct
        # Fallback to Resize if needed or just return center crop of what we have (risky)
        # But let's assume PadIfNeeded worked.
        pass

    # Ranges for top-left corner
    max_y = h - crop_height
    max_x = w - crop_width
    
    if max_y < 0 or max_x < 0:
         # This effectively means image is smaller than crop, just return as is (or crop max possible)
         # For simplicity, let's just default to center crop logic if dimensions are weird
         top = max(0, (h - crop_height)//2)
         left = max(0, (w - crop_width)//2)
         # Override crop size to fit
         actual_h = min(h, crop_height)
         actual_w = min(w, crop_width)
         return (image[top:top+actual_h, left:left+actual_w], 
                 label[top:top+actual_h, left:left+actual_w],
                 depth_input[top:top+actual_h, left:left+actual_w] if depth_input is not None else None,
                 depth_target[top:top+actual_h, left:left+actual_w] if depth_target is not None else None)

    # Decision: Smart or Random?
    do_smart = (random.random() < prob)
    
    top, left = -1, -1
    
    if do_smart:
        # localized Find coordinates of target classes
        mask = np.isin(label, target_ids)
        if mask.any():
            y_indices, x_indices = np.where(mask)
            # Pick a random pixel
            idx = random.randint(0, len(y_indices) - 1)
            cy, cx = y_indices[idx], x_indices[idx]
            
            # We want the crop to contain (cy, cx).
            # The top-left (y, x) must satisfy:
            # y <= cy < y + crop_height  =>  cy - crop_height + 1 <= y <= cy
            # Also 0 <= y <= max_y
            
            min_t = max(0, cy - crop_height + 1)
            max_t = min(max_y, cy)
            
            min_l = max(0, cx - crop_width + 1)
            max_l = min(max_x, cx)
            
            # It's possible ranges are valid.
            if min_t <= max_t and min_l <= max_l:
                top = random.randint(min_t, max_t)
                left = random.randint(min_l, max_l)
            else:
                # Should NOT happen if crop fit in image
                do_smart = False
        else:
            do_smart = False
            
    if not do_smart or top == -1:
        # Random Crop
        top = random.randint(0, max_y)
        left = random.randint(0, max_x)
        
    # Perform Crop
    img_crop = image[top:top+crop_height, left:left+crop_width]
    lbl_crop = label[top:top+crop_height, left:left+crop_width]
    
    d_in_crop = None
    if depth_input is not None:
        d_in_crop = depth_input[top:top+crop_height, left:left+crop_width]
        
    d_tgt_crop = None
    if depth_target is not None:
        d_tgt_crop = depth_target[top:top+crop_height, left:left+crop_width]
        
    return img_crop, lbl_crop, d_in_crop, d_tgt_crop


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
            
            # Clip raw depth
            raw_depth = np.clip(raw_depth, dmin, dmax)

            # 1. Input Depth (Inverse Encoding) -> REMOVED (Leak)
            # depth_input = (inv - inv_min) / (inv_max - inv_min)

            # 2. Target Depth (Linear Encoding)
            depth_target = (raw_depth - dmin) / (dmax - dmin)

        # --- Transforms & Cropping ---
        if self.transform is not None:
            # Apply transforms
            augmented = self.transform(
                image=image,
                mask=label,
                # depth=depth_input, # Removed
                depth_target=depth_target
            )
            image = augmented["image"]        
            label = augmented["mask"]         
            # depth_input = augmented["depth"]  # Removed
            depth_target = augmented["depth_target"]
            
            # Check if we need to Crop (Only for Train)
            # We can check image size. If > Crop Size, then we Crop.
            h, w = image.shape[:2]
            ch, cw = self.cfg.CROP_SIZE
            
            # If image matches crop size (Valid), skip
            if h == ch and w == cw:
                pass
            else:
                # We assume this is training and we need to crop
                # (Since Pre-Crop transforms Pad to >= Crop Size)
                image, label, _, depth_target = smart_crop(
                    image, label, None, depth_target, 
                    ch, cw, 
                    self.cfg.SMALL_OBJ_IDS, 
                    self.cfg.SMART_CROP_PROB
                )


        # --- Tensor Conversion ---
        # RGB
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(self.cfg.MEAN).view(-1, 1, 1)
        std = torch.tensor(self.cfg.STD).view(-1, 1, 1)
        # Depth Input -> REMOVED
        # if depth_input is not None:
        #    depth_input = np.clip(depth_input, 0.0, 1.0)
        #    d_tensor = torch.from_numpy(depth_input).unsqueeze(0).float() # [1,H,W]
        #    image = torch.cat([image, d_tensor], dim=0) # [4,H,W]

        # Depth Target -> Separate Tensor
        if depth_target is not None:
            depth_target = torch.from_numpy(depth_target).unsqueeze(0).float()
        
        # Label
        label = torch.from_numpy(label).long()

        return image, label, depth_target

# --- Model ---
class MultiTaskFPN(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        # Use ConvNeXt Base encoder
        # Check available encoders in SMP if uncertain, but usually 'tu-convnext_base' works
        self.backbone = smp.FPN(
            encoder_name="tu-convnext_base", 
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
        
        # FPN decoder output channels. 
        # Typically 256
        try:
            decoder_channels = self.backbone.decoder.out_channels
        except AttributeError:
             # Fallback if attribute missing (older versions?)
             # Try inspecting segmentation head
             decoder_channels = self.backbone.segmentation_head[0].in_channels

        self.depth_head = nn.Conv2d(
            in_channels=decoder_channels, 
            out_channels=1,
            kernel_size=3,
            padding=1,
        )
    
    def forward(self, x):
        features = self.backbone.encoder(x)
        decoder_out = self.backbone.decoder(features)
        
        # smp.FPN decoder output is passed to segmentation_head
        # We can also pass it to our depth_head
        
        seg_logits = self.backbone.segmentation_head(decoder_out)
        depth_pred = self.depth_head(decoder_out)
        
        # Upsample depth to match input size (segmentation head does this internally)
        depth_pred = F.interpolate(depth_pred, size=seg_logits.shape[2:], mode='bilinear', align_corners=False)
        
        return seg_logits, depth_pred

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
        writer.writerow(['epoch', 'train_loss', 'train_loss_seg', 'train_loss_depth', 
                         'valid_loss', 'valid_loss_seg', 'valid_loss_depth',
                         'valid_mIoU', 'valid_pixel_acc'])
    return log_path

def log_metrics(log_path, epoch, t_loss, t_seg, t_depth, v_loss, v_seg, v_depth, v_miou, v_acc):
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, t_loss, t_seg, t_depth, v_loss, v_seg, v_depth, v_miou, v_acc])

def save_visualizations(images, labels, depth_targets, seg_preds, depth_preds, output_dir, epoch, cfg):
    sample_dir = os.path.join(output_dir, 'samples')
    os.makedirs(sample_dir, exist_ok=True)
    
    mean = np.array(cfg.MEAN)
    std = np.array(cfg.STD)

    for i in range(min(3, len(images))):
        img_tensor = images[i]
        rgb = img_tensor[:3, :, :].cpu().numpy().transpose(1, 2, 0)
        rgb = (rgb * std + mean) * 255.0
        rgb = rgb.astype(np.uint8)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        d_gt = depth_targets[i].squeeze().cpu().numpy()
        d_gt_vis = (d_gt * 255.0).astype(np.uint8)
        d_gt_vis = cv2.cvtColor(d_gt_vis, cv2.COLOR_GRAY2BGR)

        d_pred = depth_preds[i].squeeze().cpu().numpy()
        d_pred = np.clip(d_pred, 0, 1)
        d_pred_vis = (d_pred * 255.0).astype(np.uint8)
        d_pred_vis = cv2.cvtColor(d_pred_vis, cv2.COLOR_GRAY2BGR)

        gt = labels[i].cpu().numpy().astype(np.uint8)
        gt_vis = (gt * 20).astype(np.uint8)
        gt_vis = cv2.cvtColor(gt_vis, cv2.COLOR_GRAY2BGR)

        pred = seg_preds[i].cpu().numpy().astype(np.uint8)
        pred_vis = (pred * 20).astype(np.uint8)
        pred_vis = cv2.cvtColor(pred_vis, cv2.COLOR_GRAY2BGR)

        concat_img = np.hstack([rgb, d_gt_vis, d_pred_vis, gt_vis, pred_vis])
        cv2.imwrite(os.path.join(sample_dir, f'epoch_{epoch}_sample_{i}.png'), concat_img)

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
def train_one_epoch(model, loader, criterion_seg, optimizer, device, cfg):
    model.train()
    total_loss_avg = 0.0
    seg_loss_avg = 0.0
    depth_loss_avg = 0.0
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels, depth_targets in pbar:
        images = images.to(device)
        labels = labels.to(device)
        depth_targets = depth_targets.to(device)

        optimizer.zero_grad()
        seg_logits, depth_pred = model(images)

        # 1. Segmentation Loss
        loss_seg = criterion_seg(seg_logits, labels)

        # 2. Depth Loss (masked)
        valid_mask = (depth_targets > 1e-4).float()
        loss_depth_raw = F.l1_loss(depth_pred, depth_targets, reduction='none')
        loss_depth = (loss_depth_raw * valid_mask).sum() / (valid_mask.sum() + 1e-6)

        # Combined Loss
        loss = loss_seg + cfg.DEPTH_LOSS_LAMBDA * loss_depth

        loss.backward()
        optimizer.step()

        total_loss_avg += loss.item()
        seg_loss_avg += loss_seg.item()
        depth_loss_avg += loss_depth.item()
        
        pbar.set_postfix({'loss': loss.item(), 'd_loss': loss_depth.item()})

    n = len(loader)
    return total_loss_avg / n, seg_loss_avg / n, depth_loss_avg / n

def validate(model, loader, criterion_seg, device, cfg):
    model.eval()
    total_loss_avg = 0.0
    seg_loss_avg = 0.0
    depth_loss_avg = 0.0
    
    confusion_matrix = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=np.int64)
    
    last_batch_images = None
    last_batch_labels = None
    last_batch_depth_targets = None
    last_batch_seg_preds = None
    last_batch_depth_preds = None

    with torch.no_grad():
        pbar = tqdm(loader, desc="Valid", leave=False)
        for images, labels, depth_targets in pbar:
            images = images.to(device)
            labels = labels.to(device)
            depth_targets = depth_targets.to(device)

            seg_logits, depth_pred = model(images)
            
            loss_seg = criterion_seg(seg_logits, labels)
            
            valid_mask = (depth_targets > 1e-4).float()
            loss_depth_raw = F.l1_loss(depth_pred, depth_targets, reduction='none')
            loss_depth = (loss_depth_raw * valid_mask).sum() / (valid_mask.sum() + 1e-6)
            
            loss = loss_seg + cfg.DEPTH_LOSS_LAMBDA * loss_depth

            total_loss_avg += loss.item()
            seg_loss_avg += loss_seg.item()
            depth_loss_avg += loss_depth.item()

            preds = torch.argmax(seg_logits, dim=1)
            confusion_matrix = update_confusion_matrix(
                preds.cpu().numpy(), 
                labels.cpu().numpy(), 
                cfg.NUM_CLASSES, 
                cfg.IGNORE_INDEX, 
                confusion_matrix
            )
            
            last_batch_images = images
            last_batch_labels = labels
            last_batch_depth_targets = depth_targets
            last_batch_seg_preds = preds
            last_batch_depth_preds = depth_pred

    pixel_acc, miou, class_iou = compute_metrics(confusion_matrix)
    
    n = len(loader)
    return (total_loss_avg / n, seg_loss_avg / n, depth_loss_avg / n, 
            pixel_acc, miou, class_iou,
            last_batch_images, last_batch_labels, last_batch_depth_targets, 
            last_batch_seg_preds, last_batch_depth_preds)

def main():
    cfg = Config()
    seed_everything(cfg.SEED)

    print(f"Using device: {cfg.DEVICE}")
    print(f"Total Folds: {cfg.N_FOLDS}")
    base_output_dir = cfg.OUTPUT_DIR
    print(f"Base Output directory: {base_output_dir}")

    image_dir = os.path.join(cfg.DATA_ROOT, 'image')
    label_dir = os.path.join(cfg.DATA_ROOT, 'label')
    depth_dir = os.path.join(cfg.DATA_ROOT, 'depth')

    image_files = sorted(os.listdir(image_dir))
    label_files = sorted(os.listdir(label_dir))
    depth_files = sorted(os.listdir(depth_dir))

    image_paths = np.array([os.path.join(image_dir, f) for f in image_files])
    label_paths = np.array([os.path.join(label_dir, f) for f in label_files])
    depth_paths = np.array([os.path.join(depth_dir, f) for f in depth_files])

    kf = KFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    
    # Store all splits first to ensure consistency (though random_state handles it)
    fold_splits = list(kf.split(image_paths))

    for fold_idx in range(cfg.N_FOLDS):
        print(f"\n{'='*40}")
        print(f"   STARTING FOLD {fold_idx}/{cfg.N_FOLDS - 1}")
        print(f"{'='*40}")
        
        cfg.FOLD = fold_idx
        # Setup specific output directory for this fold
        fold_output_dir = os.path.join(base_output_dir, f"fold{fold_idx}")
        cfg.OUTPUT_DIR = fold_output_dir # Update config for utils that use it
        
        save_config(cfg, fold_output_dir)
        log_path = init_logger(fold_output_dir)

        train_idx, valid_idx = fold_splits[fold_idx]

        train_images = image_paths[train_idx]
        train_labels = label_paths[train_idx]
        train_depths = depth_paths[train_idx]

        valid_images = image_paths[valid_idx]
        valid_labels = label_paths[valid_idx]
        valid_depths = depth_paths[valid_idx]

        print(f"Train size: {len(train_images)}, Valid size: {len(valid_images)}")

        train_dataset = NYUDataset(
            train_images,
            train_labels,
            depth_paths=train_depths,
            transform=get_pre_crop_transforms(cfg),
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
            drop_last=True 
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

        # Re-initialize model/optimizer/scheduler for each fold
        model = MultiTaskFPN(num_classes=cfg.NUM_CLASSES, in_channels=3) # Changed from 4 to 3
        model.to(cfg.DEVICE)

        criterion_seg = nn.CrossEntropyLoss(ignore_index=cfg.IGNORE_INDEX)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

        best_miou = 0.0

        for epoch in range(1, cfg.EPOCHS + 1):
            print(f"\n[Fold {fold_idx}] Epoch {epoch}/{cfg.EPOCHS}")

            t_loss, t_seg, t_depth = train_one_epoch(model, train_loader, criterion_seg, optimizer, cfg.DEVICE, cfg)
            v_loss, v_seg, v_depth, pixel_acc, miou, class_iou, vis_img, vis_lbl, vis_d_tgt, vis_seg_p, vis_d_p = validate(
                model, valid_loader, criterion_seg, cfg.DEVICE, cfg
            )

            scheduler.step()

            print(f"Train Loss: {t_loss:.4f} (Seg: {t_seg:.4f}, Depth: {t_depth:.4f})")
            print(f"Valid Loss: {v_loss:.4f} (Seg: {v_seg:.4f}, Depth: {v_depth:.4f})")
            print(f"Pixel Acc: {pixel_acc:.4f} | mIoU: {miou:.4f}")

            log_metrics(log_path, epoch, t_loss, t_seg, t_depth, v_loss, v_seg, v_depth, miou, pixel_acc)

            if epoch % 5 == 0 or miou > best_miou:
                save_visualizations(vis_img, vis_lbl, vis_d_tgt, vis_seg_p, vis_d_p, fold_output_dir, epoch, cfg)

            if miou > best_miou:
                best_miou = miou
                print(f"New best mIoU: {best_miou:.4f} (Fold {fold_idx}). Saving model...")
                torch.save(model.state_dict(), os.path.join(fold_output_dir, 'model_best.pth'))
                best_metrics = {
                    'best_epoch': epoch,
                    'best_val_mIoU': float(best_miou),
                    'class_iou': [float(x) for x in class_iou]
                }
                with open(os.path.join(fold_output_dir, 'best_metrics.json'), 'w') as f:
                    json.dump(best_metrics, f, indent=4)
    
    print("\nAll folds completed.")

if __name__ == '__main__':
    main()
