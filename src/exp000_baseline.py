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
    EXP_NAME = "exp000_baseline"
    SEED = 42
    IMAGE_SIZE = (480, 640) # Height, Width
    EPOCHS = 30
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_CLASSES = 13
    IGNORE_INDEX = 255
    
    # Normalization constants
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
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# --- Dataset ---
class NYUDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Read Image (RGB)
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read Label (Grayscale)
        label = cv2.imread(self.label_paths[idx], cv2.IMREAD_UNCHANGED)
        
        # Ensure label is 2D
        if label.ndim == 3:
            label = label[:, :, 0]
            
        # Sanity check
        if label is None:
             raise ValueError(f"Label not found or unable to read: {self.label_paths[idx]}")
        
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']
            
        return image, label.long()

def get_transforms(cfg):
    return A.Compose([
        A.Resize(height=cfg.IMAGE_SIZE[0], width=cfg.IMAGE_SIZE[1]),
        A.Normalize(mean=cfg.MEAN, std=cfg.STD),
        ToTensorV2(),
    ])

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
    # Save a few samples: input (denormalized), gt, pred
    sample_dir = os.path.join(output_dir, 'samples')
    os.makedirs(sample_dir, exist_ok=True)
    
    # Denormalize image
    mean = np.array(cfg.MEAN)
    std = np.array(cfg.STD)
    
    for i in range(min(3, len(images))): # Save up to 3 samples
        img = images[i].cpu().numpy().transpose(1, 2, 0) # C, H, W -> H, W, C
        img = (img * std + mean) * 255.0
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        gt = labels[i].cpu().numpy().astype(np.uint8)
        # Map GT to checkable colors (simple grayscale or color map)
        # For simplicity, scaling 0-12 to 0-255 roughly for visibility, or use a proper palette if needed.
        # Here just mapping to grayscale for quick check: value * 20
        gt_vis = (gt * 20).astype(np.uint8) 
        
        pred = preds[i] # already numpy from validate function
        pred_vis = (pred * 20).astype(np.uint8)
        
        # Concat horizontally
        # Make GT and Pred 3-channel for concatenation
        gt_vis = cv2.cvtColor(gt_vis, cv2.COLOR_GRAY2BGR)
        pred_vis = cv2.cvtColor(pred_vis, cv2.COLOR_GRAY2BGR)
        
        # Handle ignore index in GT visualization (make it red or black)
        # If ignore index is 255, it might be white or black depending on above mult.
        # Let's clean it up slightly:
        
        concat_img = np.hstack([img, gt_vis, pred_vis])
        cv2.imwrite(os.path.join(sample_dir, f'epoch_{epoch}_sample_{i}.png'), concat_img)

# --- Metrics ---
def compute_metrics(confusion_matrix):
    # confusion_matrix: [num_classes, num_classes] (Rows: True, Cols: Pred)
    
    # Pixel Accuracy
    pixel_acc = np.diag(confusion_matrix).sum() / (confusion_matrix.sum() + 1e-10)
    
    # IoU
    intersection = np.diag(confusion_matrix)
    union = confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - intersection
    
    # Handle division by zero (empty union) -> NaN
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = intersection / union
    
    # mIoU: Mean of valid IoUs (ignoring NaNs)
    miou = np.nanmean(iou)
    
    return pixel_acc, miou, iou

def update_confusion_matrix(preds, labels, num_classes, ignore_index, existing_matrix=None):
    # preds: [B, H, W] or flattened
    # labels: [B, H, W] or flattened
    
    if existing_matrix is None:
        existing_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        
    preds = preds.flatten()
    labels = labels.flatten()
    
    mask = labels != ignore_index
    preds = preds[mask]
    labels = labels[mask]
    
    # Fast histogram
    # Ensure inputs are integers within range
    # Note: We rely on caller to ensure labels are < num_classes (except ignore_index)
    
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
    
    # For visualization, keep the last batch or a few images
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
            
            preds = torch.argmax(outputs, dim=1) # Keep on GPU for a moment if needed, or move to CPU
            preds_np = preds.cpu().numpy()
            labels_np = labels_dev.cpu().numpy()
            
            # Update confusion matrix
            confusion_matrix = update_confusion_matrix(
                preds_np, labels_np, num_classes, ignore_index, confusion_matrix
            )
            
            # Save for visualization (last batch)
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
    
    # Initialize Logging
    save_config(cfg, cfg.OUTPUT_DIR)
    log_path = init_logger(cfg.OUTPUT_DIR)
    
    # Data Preparation
    image_dir = os.path.join(cfg.DATA_ROOT, 'image')
    label_dir = os.path.join(cfg.DATA_ROOT, 'label')
    
    image_files = sorted(os.listdir(image_dir))
    label_files = sorted(os.listdir(label_dir))
    
    image_paths = [os.path.join(image_dir, f) for f in image_files]
    label_paths = [os.path.join(label_dir, f) for f in label_files]
    
    # Split
    train_idx, valid_idx = train_test_split(
        range(len(image_paths)), 
        test_size=0.2, 
        random_state=cfg.SEED, 
        shuffle=True
    )
    
    train_images = [image_paths[i] for i in train_idx]
    train_labels = [label_paths[i] for i in train_idx]
    valid_images = [image_paths[i] for i in valid_idx]
    valid_labels = [label_paths[i] for i in valid_idx]
    
    print(f"Train size: {len(train_images)}, Valid size: {len(valid_images)}")
    
    train_dataset = NYUDataset(train_images, train_labels, transform=get_transforms(cfg))
    valid_dataset = NYUDataset(valid_images, valid_labels, transform=get_transforms(cfg))
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    
    # Model
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=cfg.NUM_CLASSES
    )
    model.to(cfg.DEVICE)
    
    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.IGNORE_INDEX)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    
    # Training Loop
    best_miou = 0.0
    
    for epoch in range(1, cfg.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{cfg.EPOCHS}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE)
        valid_loss, pixel_acc, miou, class_iou, vis_imgs, vis_lbls, vis_preds = validate(
            model, valid_loader, criterion, cfg.DEVICE, cfg.NUM_CLASSES, cfg.IGNORE_INDEX
        )
        
        scheduler.step()
        
        # Log to Console
        print(f"Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
        print(f"Pixel Acc: {pixel_acc:.4f} | mIoU: {miou:.4f}")
        
        # Log to CSV
        log_metrics(log_path, epoch, train_loss, valid_loss, miou, pixel_acc)
        
        # Save Visualization (every 5 epochs and best)
        if epoch % 5 == 0 or miou > best_miou:
            save_visualizations(vis_imgs, vis_lbls, vis_preds, cfg.OUTPUT_DIR, epoch, cfg)
        
        # Save Best Model & Metrics
        if miou > best_miou:
            best_miou = miou
            print(f"New best mIoU: {best_miou:.4f}. Saving model...")
            
            # Save Model
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, 'model_best.pth'))
            
            # Save Best Metrics
            best_metrics = {
                'best_epoch': epoch,
                'best_val_mIoU': float(best_miou),
                'class_iou': [float(x) for x in class_iou]
            }
            with open(os.path.join(cfg.OUTPUT_DIR, 'best_metrics.json'), 'w') as f:
                json.dump(best_metrics, f, indent=4)

if __name__ == '__main__':
    main()
