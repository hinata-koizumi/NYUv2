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
import copy

# --- Configuration ---
class Config:
    EXP_NAME = "exp100_2_pspnet"
    SEED = 42
    
    # Train Image Strategy (High Resolution - 正攻法)
    # RESIZE = 720 x 960 (高解像度)
    RESIZE_HEIGHT = 720
    RESIZE_WIDTH = 960
    # CROP: (576, 768) or (640, 640) - より大きなcrop sizeで高解像度の利点を活かす
    CROP_SIZE = (576, 768)  # H, W - アスペクト比を保持したcrop
    
    # Smart Crop Params
    SMART_CROP_PROB = 0.5
    SMALL_OBJ_IDS = [1, 3, 6, 7, 10]
    
    EPOCHS = 50
    BATCH_SIZE = 4  # 3090なら4-6で試す
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_CLASSES = 13
    IGNORE_INDEX = 255
    
    # EMA Settings (Exponential Moving Average)
    EMA_DECAY = 0.999

    # Fold
    N_FOLDS = 5
    FOLD = 0 
    
    # Loss Weight
    DEPTH_LOSS_LAMBDA = 0.1
    USE_DEPTH_LOSS = True
    
    # Depth range
    DEPTH_MIN = 0.71  # m
    DEPTH_MAX = 10.0  # m
    
    # Normalization constants (RGB)
    MEAN = [0.485, 0.456, 0.406] 
    STD = [0.229, 0.224, 0.225]

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    DATA_ROOT = 'data/train'
    
    # TTA Settings (6-8パターン)
    # scales: 0.5, 0.75, 1.0, 1.25, 1.5 × flip
    TTA_COMBS = [
        (0.5, False), (0.5, True),
        (0.75, False), (0.75, True),
        (1.0, False),  (1.0, True),
        (1.25, False), (1.25, True),
        (1.5, False), (1.5, True)
    ]
    
    # Temp Sweep for OOF (全foldでスイープしてベストTを決定)
    TEMPERATURES = [0.7, 0.8, 0.9, 1.0]

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

# --- Utils and Helpers ---
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

class ModelEMA:
    """Exponential Moving Average for model weights"""
    def __init__(self, model, decay):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for param in self.ema.parameters():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for ema_v, model_v in zip(self.ema.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(self.decay * ema_v + (1. - self.decay) * model_v)

# --- Transforms ---
def get_pre_crop_transforms(cfg):
    return A.Compose(
        [
            A.Resize(height=cfg.RESIZE_HEIGHT, width=cfg.RESIZE_WIDTH),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0, 
                scale_limit=0.2, 
                rotate_limit=0,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=cfg.IGNORE_INDEX,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, 
                contrast_limit=0.1, 
                p=0.5
            ),
            A.PadIfNeeded(
                min_height=cfg.CROP_SIZE[0],
                min_width=cfg.CROP_SIZE[1],
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=cfg.IGNORE_INDEX
            )
        ],
        additional_targets={"depth": "image", "depth_target": "image"},
    )

def get_valid_transforms(cfg):
    return A.Compose(
        [
            A.Resize(height=cfg.CROP_SIZE[0], width=cfg.CROP_SIZE[1]),
        ],
        additional_targets={"depth": "image", "depth_target": "image"},
    )

# --- Smart Crop ---
def smart_crop(image, label, depth_input, depth_target, crop_height, crop_width, target_ids, prob=0.5):
    h, w = label.shape[:2]
    max_y = h - crop_height
    max_x = w - crop_width
    
    if max_y < 0 or max_x < 0:
         top = max(0, (h - crop_height)//2)
         left = max(0, (w - crop_width)//2)
         actual_h = min(h, crop_height)
         actual_w = min(w, crop_width)
         return (image[top:top+actual_h, left:left+actual_w], 
                 label[top:top+actual_h, left:left+actual_w],
                 None,
                 depth_target[top:top+actual_h, left:left+actual_w] if depth_target is not None else None)

    do_smart = (random.random() < prob)
    top, left = -1, -1
    
    if do_smart:
        mask = np.isin(label, target_ids)
        if mask.any():
            y_indices, x_indices = np.where(mask)
            idx = random.randint(0, len(y_indices) - 1)
            cy, cx = y_indices[idx], x_indices[idx]
            min_t = max(0, cy - crop_height + 1)
            max_t = min(max_y, cy)
            min_l = max(0, cx - crop_width + 1)
            max_l = min(max_x, cx)
            if min_t <= max_t and min_l <= max_l:
                top = random.randint(min_t, max_t)
                left = random.randint(min_l, max_l)
            else:
                do_smart = False
        else:
            do_smart = False
            
    if not do_smart or top == -1:
        top = random.randint(0, max_y)
        left = random.randint(0, max_x)
        
    img_crop = image[top:top+crop_height, left:left+crop_width]
    lbl_crop = label[top:top+crop_height, left:left+crop_width]
    
    d_tgt_crop = None
    if depth_target is not None:
        d_tgt_crop = depth_target[top:top+crop_height, left:left+crop_width]
        
    return img_crop, lbl_crop, None, d_tgt_crop

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
        # Load Image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load Label
        label = cv2.imread(self.label_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Load Raw Depth for Target (if needed) and Input
        # Note: Input depth needs specific preprocess (Inverse 0-1)
        depth_path = self.depth_paths[idx] if self.depth_paths is not None else None
        
        # Precompute Input Depth (Inverse + Normalized)
        raw_depth = None
        d_input_vis = np.zeros_like(label, dtype=np.float32) # Default if no depth
        
        if depth_path is not None:
             # Load 16-bit depth (mm)
             raw_d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
             raw_depth = raw_d / 1000.0  # m
             
             # Inverse Depth for Input
             # clip 0.71 ~ 10.0
             d_clamped = np.clip(raw_depth, self.cfg.DEPTH_MIN, self.cfg.DEPTH_MAX)
             inv_d = 1.0 / d_clamped
             inv_min = 1.0 / self.cfg.DEPTH_MAX
             inv_max = 1.0 / self.cfg.DEPTH_MIN
             
             # Normalize to 0-1
             norm_inv_d = (inv_d - inv_min) / (inv_max - inv_min)
             d_input_vis = np.clip(norm_inv_d, 0.0, 1.0).astype(np.float32)
         
        image_f = image.astype(np.float32)
        # Stack
        img_4ch = np.dstack([image_f, d_input_vis]) # [H, W, 4], float32
        
        # Prepare Depth Target (for Loss)
        if raw_depth is not None:
            depth_target = raw_depth.astype(np.float32)
        else:
            depth_target = np.zeros_like(label, dtype=np.float32)
            
        # Transform
        if self.transform:
            augmented = self.transform(image=img_4ch, mask=label, depth_target=depth_target)
            img_4ch = augmented['image']
            label = augmented['mask']
            depth_target = augmented['depth_target']
            
            # Smart Crop check
            h, w = img_4ch.shape[:2]
            ch, cw = self.cfg.CROP_SIZE
            if not(h == ch and w == cw):
                 img_4ch, label, _, depth_target = smart_crop(
                     img_4ch, label, None, depth_target, ch, cw,
                     self.cfg.SMALL_OBJ_IDS, self.cfg.SMART_CROP_PROB
                 )
        
        # Split RGB and Depth
        image_rgb = img_4ch[:, :, :3] # 0-255 float
        image_depth = img_4ch[:, :, 3] # 0-1 float
        
        # Tensor Conversion & Norm
        image_rgb = image_rgb / 255.0 # to 0-1
        
        # Manual Norm with Mean/Std
        mean = np.array(self.cfg.MEAN, dtype=np.float32)
        std = np.array(self.cfg.STD, dtype=np.float32)
        image_rgb = (image_rgb - mean) / std
        
        # To Tensor [C, H, W]
        image_rgb = torch.from_numpy(image_rgb).permute(2, 0, 1).float()
        image_depth = torch.from_numpy(image_depth).unsqueeze(0).float()
        
        # Cat -> [4, H, W]
        image_tensor = torch.cat([image_rgb, image_depth], dim=0)
        
        label_tensor = torch.from_numpy(label).long()
        depth_target_tensor = torch.from_numpy(depth_target).float()
        
        return image_tensor, label_tensor, depth_target_tensor

# --- Model ---
class MultiTaskPSPNet(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.backbone = smp.PSPNet(
            encoder_name="tu-convnext_base", 
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
        try:
            decoder_channels = self.backbone.segmentation_head[0].in_channels
        except Exception as e:
            print(f"Warning: Could not get decoder channels from segmentation_head: {e}")
            # PSPNet default decoder channels
            # Usually 512
            decoder_channels = 512
            
        self.depth_head = nn.Conv2d(in_channels=decoder_channels, out_channels=1, kernel_size=3, padding=1)
    
    def forward(self, x):
        features = self.backbone.encoder(x)
        decoder_out = self.backbone.decoder(features) # No unpacking for PSPNet either typically
        seg_logits = self.backbone.segmentation_head(decoder_out)
        depth_pred = self.depth_head(decoder_out)
        
        # Resize to input size
        if depth_pred.shape[-2:] != seg_logits.shape[-2:]:
             depth_pred = F.interpolate(depth_pred, size=seg_logits.shape[-2:], mode='bilinear', align_corners=False)
             
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

# --- TTA Logic ---
def tta_inference(model, image, cfg, temperature=1.0):
    # Convert torch tensor to numpy array if needed
    mean_cpu = torch.tensor(cfg.MEAN).view(1, 1, 3)
    std_cpu = torch.tensor(cfg.STD).view(1, 1, 3)
    
    if torch.is_tensor(image):
        img_rgb = image[:3, :, :].permute(1, 2, 0).cpu() # H,W,3
        img_d = image[3, :, :].unsqueeze(2).cpu()        # H,W,1
        img_rgb = img_rgb * std_cpu + mean_cpu
        
        img_rgb = (img_rgb.numpy() * 255.0).astype(np.uint8)
        img_d = (img_d.numpy() * 255.0).astype(np.uint8) 
        
        img_rgb_np = img_rgb
        img_d_np = img_d.numpy() 
        
    else:
        pass

    image_f = img_rgb_np.astype(np.float32)
    image_combined = np.dstack([image_f, img_d_np])
    
    h_base, w_base = image_combined.shape[:2]
    accumulated_probs = None
    count = 0
    mean = torch.tensor(cfg.MEAN).view(1, 1, 3).to(cfg.DEVICE)
    std = torch.tensor(cfg.STD).view(1, 1, 3).to(cfg.DEVICE)
    
    for scale, flip in cfg.TTA_COMBS:
        h_new = int(h_base * scale)
        w_new = int(w_base * scale)
        h_new = (h_new // 32) * 32
        w_new = (w_new // 32) * 32
        
        img_aug = cv2.resize(image_combined, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
        if flip: img_aug = cv2.flip(img_aug, 1)
            
        img_tensor = torch.from_numpy(img_aug).float().to(cfg.DEVICE)
        
        rgb = img_tensor[:, :, :3] / 255.0
        d = img_tensor[:, :, 3:4] 
        
        rgb = (rgb - mean) / std
        
        img_input = torch.cat([rgb, d], dim=2) # [H, W, 4]
        img_input = img_input.permute(2, 0, 1).unsqueeze(0) # [1, 4, H, W]
        
        with torch.no_grad():
            seg_logits, _ = model(img_input)
            seg_probs = F.softmax(seg_logits / temperature, dim=1)
            
        seg_probs = seg_probs.squeeze(0).permute(1, 2, 0).cpu().numpy()
        if flip: seg_probs = cv2.flip(seg_probs, 1)
        seg_probs = cv2.resize(seg_probs, (w_base, h_base), interpolation=cv2.INTER_LINEAR)
        
        if accumulated_probs is None:
            accumulated_probs = seg_probs
        else:
            accumulated_probs += seg_probs
        count += 1
        
    return accumulated_probs / count

# --- Training & Validation Loops ---
def train_one_epoch(model, ema_model, loader, criterion_seg, optimizer, device, cfg):
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

        # Loss Calculation
        loss_seg = criterion_seg(seg_logits, labels)
        
        loss_depth = torch.tensor(0.0, device=device)
        if cfg.USE_DEPTH_LOSS:
            valid_mask = (depth_targets > 1e-4).float()
            if depth_pred.shape[1] == 1:
                depth_pred = depth_pred.squeeze(1)
            loss_depth_raw = F.l1_loss(depth_pred, depth_targets, reduction='none')
            loss_depth = (loss_depth_raw * valid_mask).sum() / (valid_mask.sum() + 1e-6)
            loss = loss_seg + cfg.DEPTH_LOSS_LAMBDA * loss_depth
        else:
            loss = loss_seg

        loss.backward()
        optimizer.step()
        
        if ema_model is not None:
            ema_model.update(model)

        total_loss_avg += loss.item()
        seg_loss_avg += loss_seg.item()
        if cfg.USE_DEPTH_LOSS:
            depth_loss_avg += loss_depth.item()
            
        pbar.set_postfix({'loss': loss.item(), 'd_loss': loss_depth.item()})

    n = len(loader)
    return total_loss_avg/n, seg_loss_avg/n, depth_loss_avg/n

def validate(model, loader, criterion_seg, device, cfg):
    model.eval()
    total_loss_avg = 0.0
    seg_loss_avg = 0.0
    depth_loss_avg = 0.0
    confusion_matrix = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=np.int64)

    with torch.no_grad():
        pbar = tqdm(loader, desc="Valid", leave=False)
        for images, labels, depth_targets in pbar:
            images = images.to(device)
            labels = labels.to(device)
            depth_targets = depth_targets.to(device)

            seg_logits, depth_pred = model(images)
            loss_seg = criterion_seg(seg_logits, labels)
            
            valid_mask = (depth_targets > 1e-4).float()
            if depth_pred.shape[1] == 1:
                 depth_pred = depth_pred.squeeze(1)
            loss_depth_raw = F.l1_loss(depth_pred, depth_targets, reduction='none')
            loss_depth = (loss_depth_raw * valid_mask).sum() / (valid_mask.sum() + 1e-6)
            
            if cfg.USE_DEPTH_LOSS:
                loss = loss_seg + cfg.DEPTH_LOSS_LAMBDA * loss_depth
            else:
                loss = loss_seg

            total_loss_avg += loss.item()
            seg_loss_avg += loss_seg.item()
            depth_loss_avg += loss_depth.item()

            preds = torch.argmax(seg_logits, dim=1)
            confusion_matrix = update_confusion_matrix(
                preds.cpu().numpy(), labels.cpu().numpy(), cfg.NUM_CLASSES, cfg.IGNORE_INDEX, confusion_matrix
            )
            
    pixel_acc, miou, class_iou = compute_metrics(confusion_matrix)
    n = len(loader)
    return total_loss_avg/n, seg_loss_avg/n, depth_loss_avg/n, pixel_acc, miou, class_iou

def validate_tta_sweep(model, dataset, cfg):
    """TTA + Temperature sweep for a single fold"""
    model.eval()
    print(f"Starting TTA Sweep. Combs: {len(cfg.TTA_COMBS)}, Temps: {cfg.TEMPERATURES}")
    
    best_iou = 0
    best_temp = 1.0
    results = {}

    for t in cfg.TEMPERATURES:
        print(f"Evaluating Temperature T={t}...")
        confusion_matrix = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=np.int64)
        
        for i in tqdm(range(len(dataset)), desc=f"TTA T={t}"):
             image, label, _ = dataset[i]
             if torch.is_tensor(label):
                 label = label.cpu().numpy()
             
             avg_probs = tta_inference(model, image, cfg, temperature=t)
             pred_label = np.argmax(avg_probs, axis=2)
             confusion_matrix = update_confusion_matrix(pred_label, label, cfg.NUM_CLASSES, cfg.IGNORE_INDEX, confusion_matrix)
        
        _, miou, _ = compute_metrics(confusion_matrix)
        print(f"  Result T={t}: mIoU = {miou:.4f}")
        results[t] = miou
        
        if miou > best_iou:
            best_iou = miou
            best_temp = t
            
    print(f"Sweep Best: T={best_temp} with mIoU {best_iou:.4f}")
    return best_temp, best_iou, results

def compute_oof_metrics(all_fold_results, cfg):
    print("\n" + "="*60)
    print("Computing OOF Metrics")
    print("="*60)
    
    oof_results = {}
    for temp in cfg.TEMPERATURES:
        oof_confusion_matrix = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=np.int64)
        for fold_result in all_fold_results:
            if f'confusion_matrix_T{temp}' in fold_result:
                oof_confusion_matrix += fold_result[f'confusion_matrix_T{temp}']
        
        _, oof_miou, _ = compute_metrics(oof_confusion_matrix)
        oof_results[temp] = oof_miou
        print(f"OOF mIoU @ T={temp}: {oof_miou:.4f}")
    
    best_oof_temp = max(oof_results.items(), key=lambda x: x[1])[0]
    best_oof_miou = oof_results[best_oof_temp]
    
    print(f"\nBest OOF Temperature: T={best_oof_temp} with mIoU {best_oof_miou:.4f}")
    print("="*60)
    return best_oof_temp, best_oof_miou, oof_results

def main():
    cfg = Config()
    seed_everything(cfg.SEED)
    cfg.OUTPUT_DIR = os.path.join("data", "outputs", cfg.EXP_NAME)

    image_dir = os.path.join(cfg.DATA_ROOT, 'image')
    label_dir = os.path.join(cfg.DATA_ROOT, 'label')
    depth_dir = os.path.join(cfg.DATA_ROOT, 'depth')
    
    # Fix paths
    if not os.path.exists(cfg.DATA_ROOT) and os.path.exists('/Users/koizumihinata/NYUv2/' + cfg.DATA_ROOT):
        prefix = '/Users/koizumihinata/NYUv2/'
        cfg.DATA_ROOT = prefix + cfg.DATA_ROOT
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
    fold_splits = list(kf.split(image_paths))

    all_fold_results = []
    fold_tta_results = {}

    for fold_idx in range(cfg.N_FOLDS):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx} / {cfg.N_FOLDS - 1}")
        print(f"{'='*60}")
        
        cfg.FOLD = fold_idx
        fold_output_dir = os.path.join(cfg.OUTPUT_DIR, f"fold{fold_idx}")
        save_config(cfg, fold_output_dir)
        log_path = init_logger(fold_output_dir)
        
        train_idx, valid_idx = fold_splits[fold_idx]
        
        train_dataset = NYUDataset(
            image_paths[train_idx], label_paths[train_idx], depth_paths[train_idx],
            transform=get_pre_crop_transforms(cfg), cfg=cfg
        )
        valid_dataset = NYUDataset(
            image_paths[valid_idx], label_paths[valid_idx], depth_paths[valid_idx],
            transform=get_valid_transforms(cfg), cfg=cfg
        )
        valid_dataset_tta = NYUDataset(
           image_paths[valid_idx], label_paths[valid_idx], depth_paths[valid_idx],
           transform=None, cfg=cfg 
        )

        train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        model = MultiTaskPSPNet(num_classes=cfg.NUM_CLASSES, in_channels=4)
        model.to(cfg.DEVICE)
        
        ema_model = ModelEMA(model, cfg.EMA_DECAY)

        criterion_seg = nn.CrossEntropyLoss(ignore_index=cfg.IGNORE_INDEX)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

        best_miou = 0.0

        for epoch in range(1, cfg.EPOCHS + 1):
            t_loss, t_seg, t_depth = train_one_epoch(model, ema_model, train_loader, criterion_seg, optimizer, cfg.DEVICE, cfg)
            
            # Validation with EMA Model
            v_loss, v_seg, v_depth, pixel_acc, miou, class_iou = validate(ema_model.ema, valid_loader, criterion_seg, cfg.DEVICE, cfg)
            
            scheduler.step()
            
            print(f"Epoch {epoch}/{cfg.EPOCHS}: Train Loss {t_loss:.4f}, Valid mIoU {miou:.4f} (EMA)")
            log_metrics(log_path, epoch, t_loss, t_seg, t_depth, v_loss, v_seg, v_depth, miou, pixel_acc)
            
            if miou > best_miou:
                best_miou = miou
                torch.save(ema_model.ema.state_dict(), os.path.join(fold_output_dir, 'model_best.pth'))
                
        # End of Fold
        print(f"\nFold {fold_idx} Training Done. Running TTA Sweep on Best Model...")
        best_model = MultiTaskPSPNet(num_classes=cfg.NUM_CLASSES, in_channels=4)
        best_model.load_state_dict(torch.load(os.path.join(fold_output_dir, 'model_best.pth'), map_location=cfg.DEVICE))
        best_model.to(cfg.DEVICE)
        
        fold_best_temp, fold_best_iou, fold_temp_results = validate_tta_sweep(best_model, valid_dataset_tta, cfg)
        fold_tta_results[fold_idx] = {
            'best_temp': fold_best_temp,
            'best_iou': fold_best_iou,
            'temp_results': fold_temp_results
        }
        
        fold_result = {}
        for temp in cfg.TEMPERATURES:
            confusion_matrix = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=np.int64)
            for i in range(len(valid_dataset_tta)):
                image, label, _ = valid_dataset_tta[i]
                if torch.is_tensor(label):
                    label = label.cpu().numpy()
                avg_probs = tta_inference(best_model, image, cfg, temperature=temp)
                pred_label = np.argmax(avg_probs, axis=2)
                confusion_matrix = update_confusion_matrix(pred_label, label, cfg.NUM_CLASSES, cfg.IGNORE_INDEX, confusion_matrix)
            fold_result[f'confusion_matrix_T{temp}'] = confusion_matrix
        
        all_fold_results.append(fold_result)
        
        # --- Phase B Prep: Save Confusion Matrix & Class IoU ---
        best_cm = fold_result[f'confusion_matrix_T{fold_best_temp}']
        np.save(os.path.join(fold_output_dir, f'fold{fold_idx}_confusion.npy'), best_cm)
        
        _, _, best_class_iou = compute_metrics(best_cm)
        # -----------------------------------------------------
        
        print(f"Fold {fold_idx} Summary: Best Temp={fold_best_temp}, mIoU={fold_best_iou:.4f}")
        
        with open(os.path.join(fold_output_dir, 'tta_results.json'), 'w') as f:
            json.dump({
                'best_temp': fold_best_temp,
                'best_iou': float(fold_best_iou),
                'class_iou': best_class_iou.tolist(), # Added for Phase B
                'temp_results': {str(k): float(v) for k, v in fold_temp_results.items()}
            }, f, indent=2)

    best_oof_temp, best_oof_miou, oof_results = compute_oof_metrics(all_fold_results, cfg)
    
    summary = {
        'exp_name': cfg.EXP_NAME,
        'best_oof_temp': float(best_oof_temp),
        'best_oof_miou': float(best_oof_miou),
        'oof_results': {str(k): float(v) for k, v in oof_results.items()},
        'fold_results': {}
    }
    
    for fold_idx, fold_result in fold_tta_results.items():
        summary['fold_results'][f'fold{fold_idx}'] = {
            'best_temp': float(fold_result['best_temp']),
            'best_iou': float(fold_result['best_iou']),
            'temp_results': {str(k): float(v) for k, v in fold_result['temp_results'].items()}
        }
    
    summary_path = os.path.join(cfg.OUTPUT_DIR, 'oof_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best OOF Temperature: T={best_oof_temp}")
    print(f"Best OOF mIoU: {best_oof_miou:.4f}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
