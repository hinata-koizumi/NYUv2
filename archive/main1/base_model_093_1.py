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
    EXP_NAME = "exp093_1_fullpower_ema_tta"
    SEED = 42
    
    # Train Image Strategy (Full Power)
    # 案A (Safe): 720x960
    RESIZE_HEIGHT = 720
    RESIZE_WIDTH = 960
    # CROP: (640, 640)
    CROP_SIZE = (640, 640) # H, W
    
    # Smart Crop Params
    SMART_CROP_PROB = 0.5
    SMALL_OBJ_IDS = [1, 3, 6, 7, 10]
    
    EPOCHS = 50
    BATCH_SIZE = 4 # Reduced for higher resolution
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_CLASSES = 13
    IGNORE_INDEX = 255
    
    # EMA Settings
    EMA_DECAY = 0.999

    # Fold
    N_FOLDS = 5
    FOLD = 0 

    # Loss Weight
    DEPTH_LOSS_LAMBDA = 0.1

    # Depth range
    DEPTH_MIN = 0.71  # m
    DEPTH_MAX = 10.0  # m
    
    # Normalization constants (RGB)
    MEAN = [0.485, 0.456, 0.406] 
    STD = [0.229, 0.224, 0.225]

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    DATA_ROOT = 'data/train' # Relative path assumed based on context or absolute

    # TTA Settings (Full Power)
    # scales = [0.75, 1.0, 1.25]
    # flip = [False, True]
    TTA_COMBS = [
        (0.75, False), (0.75, True),
        (1.0, False),  (1.0, True),
        (1.25, False), (1.25, True)
    ]
    
    # Temp Sweep for OOF (or Fold Validation)
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
        additional_targets={"depth_target": "image"},
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
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(self.label_paths[idx], cv2.IMREAD_UNCHANGED)
        if label.ndim == 3: label = label[:, :, 0]

        depth_target = None
        if self.depth_paths is not None:
            raw_depth = cv2.imread(self.depth_paths[idx], cv2.IMREAD_UNCHANGED)
            raw_depth = raw_depth.astype(np.float32) / 1000.0 
            dmin, dmax = self.cfg.DEPTH_MIN, self.cfg.DEPTH_MAX
            raw_depth = np.clip(raw_depth, dmin, dmax)
            depth_target = (raw_depth - dmin) / (dmax - dmin)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=label, depth_target=depth_target)
            image = augmented["image"]        
            label = augmented["mask"]         
            depth_target = augmented["depth_target"]
            
            # Smart Crop (Training only logic usually)
            h, w = image.shape[:2]
            ch, cw = self.cfg.CROP_SIZE
            if not(h == ch and w == cw):
                image, label, _, depth_target = smart_crop(
                    image, label, None, depth_target, ch, cw, 
                    self.cfg.SMALL_OBJ_IDS, self.cfg.SMART_CROP_PROB
                )

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if depth_target is not None:
            depth_target = torch.from_numpy(depth_target).unsqueeze(0).float()
        label = torch.from_numpy(label).long()

        return image, label, depth_target

# --- Model ---
class MultiTaskFPN(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.backbone = smp.FPN(
            encoder_name="tu-convnext_base", 
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
        try:
            decoder_channels = self.backbone.decoder.out_channels
        except AttributeError:
            decoder_channels = self.backbone.segmentation_head[0].in_channels

        self.depth_head = nn.Conv2d(in_channels=decoder_channels, out_channels=1, kernel_size=3, padding=1)
    
    def forward(self, x):
        features = self.backbone.encoder(x)
        decoder_out = self.backbone.decoder(features)
        seg_logits = self.backbone.segmentation_head(decoder_out)
        depth_pred = self.depth_head(decoder_out)
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
    h_base, w_base = image.shape[:2]
    accumulated_probs = None
    count = 0
    mean = torch.tensor(cfg.MEAN).view(1, 1, 3).to(cfg.DEVICE)
    std = torch.tensor(cfg.STD).view(1, 1, 3).to(cfg.DEVICE)
    
    for scale, flip in cfg.TTA_COMBS:
        h_new = int(h_base * scale)
        w_new = int(w_base * scale)
        # Check alignment, though typically convnext handles random sizes well.
        # But for UNet-like decoders, factors of 32 are safe.
        h_new = (h_new // 32) * 32
        w_new = (w_new // 32) * 32
        
        img_aug = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
        if flip: img_aug = cv2.flip(img_aug, 1)
            
        img_tensor = torch.from_numpy(img_aug).float().to(cfg.DEVICE) / 255.0
        img_tensor = (img_tensor - mean) / std
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        
        with torch.no_grad():
            seg_logits, _ = model(img_tensor)
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

        loss_seg = criterion_seg(seg_logits, labels)
        valid_mask = (depth_targets > 1e-4).float()
        loss_depth_raw = F.l1_loss(depth_pred, depth_targets, reduction='none')
        loss_depth = (loss_depth_raw * valid_mask).sum() / (valid_mask.sum() + 1e-6)
        loss = loss_seg + cfg.DEPTH_LOSS_LAMBDA * loss_depth

        loss.backward()
        optimizer.step()
        
        # Update EMA
        if ema_model is not None:
            ema_model.update(model)

        total_loss_avg += loss.item()
        seg_loss_avg += loss_seg.item()
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
            loss_depth_raw = F.l1_loss(depth_pred, depth_targets, reduction='none')
            loss_depth = (loss_depth_raw * valid_mask).sum() / (valid_mask.sum() + 1e-6)
            loss = loss_seg + cfg.DEPTH_LOSS_LAMBDA * loss_depth

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
    model.eval()
    print(f"Starting TTA Sweep. Combs: {len(cfg.TTA_COMBS)}, Temps: {cfg.TEMPERATURES}")
    
    # Pre-calculate base predictions (Standard TTA Logic) to avoid re-running model for every temperature?
    # No, temperature scaling happens on logits.
    # To be efficient, we should save logits for all TTA variations and then combine.
    # But for simplicity/memory, we can just run looped inference for each temperature or nested.
    
    # Optimization: Run Logic ONE time per image per TTA augment, save logits.
    # But logits are large. (13, H, W).
    # Let's just do the simple loop for each temperature for now, unless it's too slow.
    # User said "OOF mIoU sweep".
    
    best_iou = 0
    best_temp = 1.0
    
    results = {}

    for t in cfg.TEMPERATURES:
        print(f"Evaluating Temperature T={t}...")
        confusion_matrix = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=np.int64)
        
        for i in tqdm(range(len(dataset)), desc=f"TTA T={t}"):
             image, label, _ = dataset[i]
             # image: [H, W, 3] numpy
             
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
    return best_temp, best_iou

def main():
    cfg = Config()
    seed_everything(cfg.SEED)
    cfg.OUTPUT_DIR = os.path.join("data", "outputs", cfg.EXP_NAME)

    image_dir = os.path.join(cfg.DATA_ROOT, 'image')
    label_dir = os.path.join(cfg.DATA_ROOT, 'label')
    depth_dir = os.path.join(cfg.DATA_ROOT, 'depth')
    
    # Fix paths if using Full Path vs Relative
    if not os.path.exists(cfg.DATA_ROOT) and os.path.exists('/Users/koizumihinata/NYUv2/' + cfg.DATA_ROOT):
        # Adjust for local env if needed or just assume correct CWD
        cfg.DATA_ROOT = '/Users/koizumihinata/NYUv2/' + cfg.DATA_ROOT
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

    for fold_idx in range(cfg.N_FOLDS):
        # Optional: Run only Fold 0 for now as '093' usually implies refining one fold or all.
        # User prompt implies "Train", so we should loop folds.
        # But for "OOF sweep" typically we train all 5. 
        # I'll loop 5 folds.
        
        print(f"\n======== FOLD {fold_idx} ========")
        cfg.FOLD = fold_idx
        fold_output_dir = os.path.join(cfg.OUTPUT_DIR, f"fold{fold_idx}")
        save_config(cfg, fold_output_dir)
        log_path = init_logger(fold_output_dir)
        
        train_idx, valid_idx = fold_splits[fold_idx]
        
        # Datasets
        train_dataset = NYUDataset(
            image_paths[train_idx], label_paths[train_idx], depth_paths[train_idx],
            transform=get_pre_crop_transforms(cfg), cfg=cfg
        )
        valid_dataset = NYUDataset(
            image_paths[valid_idx], label_paths[valid_idx], depth_paths[valid_idx],
            transform=get_valid_transforms(cfg), cfg=cfg
        )
        # For TTA Sweep (need raw images)
        # We can reuse valid_dataset but we need a "Test" dataset that returns raw image.
        # Let's create a separate TTA dataset or modify behavior.
        # Simple hack: Reuse NYUDataset but with a transform that doesn't normalize if doing that in TTA loop?
        # The Validate loop uses standard valid_loader (normalized).
        # TTA sweep uses manual loop.
        # We'll use a special TTA dataset with NO transform (return raw) for the Sweep phase.
        valid_dataset_tta = NYUDataset(
           image_paths[valid_idx], label_paths[valid_idx], depth_paths[valid_idx],
           transform=None, cfg=cfg 
        ) # Returns cv2 image (BGR), handled in __getitem__ to RGB. Correct.

        train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        # Model & EMA
        model = MultiTaskFPN(num_classes=cfg.NUM_CLASSES, in_channels=3)
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
            
            print(f"Epoch {epoch}: Train Loss {t_loss:.4f}, Valid mIoU {miou:.4f} (EMA)")
            log_metrics(log_path, epoch, t_loss, t_seg, t_depth, v_loss, v_seg, v_depth, miou, pixel_acc)
            
            if miou > best_miou:
                best_miou = miou
                torch.save(ema_model.ema.state_dict(), os.path.join(fold_output_dir, 'model_best.pth'))
                
        # End of Fold - Run TTA Sweep on Best Model
        print(f"Fold {fold_idx} Training Done. Running TTA Sweep on Best Model...")
        best_model = MultiTaskFPN(num_classes=cfg.NUM_CLASSES, in_channels=3)
        best_model.load_state_dict(torch.load(os.path.join(fold_output_dir, 'model_best.pth')))
        best_model.to(cfg.DEVICE)
        
        fold_best_temp, fold_best_iou = validate_tta_sweep(best_model, valid_dataset_tta, cfg)
        print(f"Fold {fold_idx} Optimized: Temp={fold_best_temp}, mIoU={fold_best_iou:.4f}")

if __name__ == '__main__':
    main()
