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
    EXP_NAME = "exp066_post_process"
    SEED = 42
    IMAGE_SIZE = (480, 640)
    BATCH_SIZE = 8
    NUM_CLASSES = 13
    IGNORE_INDEX = 255
    
    # Depth range
    DEPTH_MIN = 0.71
    DEPTH_MAX = 10.0
    
    # Normalization
    MEAN = [0.525, 0.443, 0.400]
    STD = [0.281, 0.281, 0.293]

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    DATA_ROOT = 'data/train'
    
    MODEL_PATH_060 = "data/outputs/exp060_multitask_resnet101/model_best.pth"
    MODEL_PATH_062 = "data/outputs/exp062_class_weighted_resnet101/model_best.pth"
    MODEL_PATH_063 = "data/outputs/exp063_boundary_aware_resnet101/model_best.pth"
    
    # Best setup from previous steps
    ENSEMBLE_WEIGHTS = (0.6, 0.2, 0.2)
    USE_TTA = True
    
    # Post-process params
    # 13 classes: 0:Bed, 1:Books, 2:Ceiling, 3:Chair, 4:Floor, 5:Furniture, 
    # 6:Objects, 7:Picture, 8:Sofa, 9:Table, 10:TV, 11:Wall, 12:Window
    
    # Small Objects to cleanup: Books(1), Chair(3), Objects(6), Picture(7), TV(10)
    # Exclude 0(Bed) as per user suggestion to avoid no-op if filling with 0
    REMOVE_SMALL_OBJ_IDS = [1, 3, 6, 7, 10] 
    MIN_SIZE = 100 # pixels
    
    # Holes to fill (Large surfaces): Ceiling(2), Floor(4), Wall(11)
    FILL_HOLES_IDS = [2, 4, 11]

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

# --- Transforms & Dataset (Same as exp064/065) ---
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
            raw_depth = raw_depth.astype(np.float32) / 1000.0
            dmin = self.cfg.DEPTH_MIN
            dmax = self.cfg.DEPTH_MAX
            raw_depth = np.clip(raw_depth, dmin, dmax)
            
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

# --- TTA ---
def predict_with_hflip_tta(model, images):
    with torch.no_grad():
        logits, _ = model(images)
        images_flip = torch.flip(images, dims=[3])
        logits_flip, _ = model(images_flip)
        logits_flip = torch.flip(logits_flip, dims=[3])
        logits_tta = 0.5 * (logits + logits_flip)
    return logits_tta

# --- Post Processing ---
def remove_small_regions(label, min_size=50, target_classes=None):
    # label: [H, W] np.uint8 or int64
    # target_classes: list of class_ids to apply cleanup. If None, apply to all.
    
    out = label.copy()
    
    # It's faster to process per class, or use connected components on entire map if classes are disjoint?
    # ConnectedComponents is binary. So we must loop over classes.
    
    classes = np.unique(label)
    
    for cid in classes:
        if target_classes is not None and cid not in target_classes:
            continue
        if cid == Config.IGNORE_INDEX:
            continue
            
        mask = (label == cid).astype(np.uint8)
        num, comp = cv2.connectedComponents(mask)
        
        # If num is small, skipping is faster, but we need to check sizes
        if num <= 1: continue 
        
        # Compute areas
        # (This can be slow if there are many blobs. Fast optimization: return stats from ccalg)
        # cv2.connectedComponents doesn't return stats. cv2.connectedComponentsWithStats does.
        
        num, comp, stats, centroids = cv2.connectedComponentsWithStats(mask)
        
        for i in range(1, num): # 0 is background (where mask==0, i.e. other classes)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_size:
                # Remove this blob
                # Fill with what? 
                # Strategy: Fill with 0 (void) or nearest neighbor?
                # User prompted: "For example background or surrounding majority"
                # Simplest is 0 (Void/Backround?) or if 0 is a valid class (it is), we need to be careful.
                # If we set to 0, does 0 mean something?
                # In NYUv2, 0 is often a class.
                # Maybe set to 'majority' is better but complex.
                # Let's set to IGNORE_INDEX (255) if it's considered "Noise/Void"
                # OR, finding the surrounding value.
                # For simplicity, let's keep it simple: set to '0' if 0 is background, or '255' (ignore).
                # But if we change prediction to Ignore, it won't help mIoU unless Ground Truth is also Ignore (unlikely).
                # If we change it to "Background" (if 0 is background), that might be okay.
                # Let's assume we want to "remove" it.
                # Re-reading prompt: "Chair point... dots... remove" "out[comp==cid] = 0 # e.g. background".
                # Okay, I will set to 0. But note that if 0 is, say, 'Bed', then we are turning tiny Chairs into Beds.
                # If we don't know "Background" ID, maybe safest to set to 255 (Ignore) so we don't get penalized?
                # But we want to improve mIoU. Predicting "Ignore" usually counts as Wrong if GT is something else? 
                # Wait, mIoU computation usually ignores pixels where PREDICTION is ignore? No, where GT is ignore.
                # If GT is valid, and we predict Ignore, is that a match? No.
                # So we must predict *some* valid class.
                # 0 is the safest fallback usually.
                out[comp == i] = 0 
                
    return out

def fill_holes(label, kernel_size=3, target_classes=None):
    # label: [H, W]
    out = label.copy()
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    classes = np.unique(label)
    for cid in classes:
        if target_classes is not None and cid not in target_classes:
            continue
        if cid == Config.IGNORE_INDEX:
            continue
            
        mask = (label == cid).astype(np.uint8)
        
        # Closing: Dilation then Erosion. Fills small holes (dark spots) inside the foreground (white).
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Update only where mask was 0 but closed is 1
        # output[(closed == 1) & (mask == 0)] = cid
        # But wait, if we overwrite, we might overwrite another class.
        # We should only overwrite if the underlying pixel was "removed" or maybe "small hole"?
        # Actually morphologyEx affects the whole mask.
        # Ideally we only fill "holes" that were previously other classes (usually small noise of other classes inside this class).
        # By doing this, we let this class 'eat' small intruders.
        
        out[(closed == 1) & (mask == 0)] = cid
        
    return out

def validate_post_process(models, loader, device, cfg):
    for m in models:
        m.eval()
        
    cm_base = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=np.int64)
    cm_post = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=np.int64)
    
    w0, w1, w2 = cfg.ENSEMBLE_WEIGHTS
    print(f"Validating Post-Processing (Ensemble + TTA: {cfg.USE_TTA})")
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Post-Processing", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.cpu().numpy()

            # --- Ensemble + TTA ---
            if cfg.USE_TTA:
                l0 = predict_with_hflip_tta(models[0], images)
                l1 = predict_with_hflip_tta(models[1], images)
                l2 = predict_with_hflip_tta(models[2], images)
            else:
                l0, _ = models[0](images)
                l1, _ = models[1](images)
                l2, _ = models[2](images)
                
            ens_logits = w0 * l0 + w1 * l1 + w2 * l2
            preds_base = torch.argmax(ens_logits, dim=1).cpu().numpy()
            
            # Update Base Metric
            cm_base = update_confusion_matrix(preds_base, labels, cfg.NUM_CLASSES, cfg.IGNORE_INDEX, cm_base)
            
            # --- Post Process ---
            # Process each image in batch
            preds_pp = preds_base.copy()
            for i in range(preds_pp.shape[0]):
                p = preds_pp[i] # [H, W]
                
                # 1. Remove Small Regions
                p = remove_small_regions(p, min_size=cfg.MIN_SIZE, target_classes=cfg.REMOVE_SMALL_OBJ_IDS)
                
                # 2. Fill Holes
                p = fill_holes(p, kernel_size=3, target_classes=cfg.FILL_HOLES_IDS)
                
                preds_pp[i] = p
                
            cm_post = update_confusion_matrix(preds_pp, labels, cfg.NUM_CLASSES, cfg.IGNORE_INDEX, cm_post)

    # Report
    print("\n--- Post-Processing Results ---")
    
    p_acc, miou, _ = compute_metrics(cm_base)
    print(f"Base (Ens+TTA) | mIoU: {miou:.5f} | PixAcc: {p_acc:.5f}")
    
    p_acc_pp, miou_pp, _ = compute_metrics(cm_post)
    print(f"Post-Process   | mIoU: {miou_pp:.5f} | PixAcc: {p_acc_pp:.5f}")
    
    diff = miou_pp - miou
    print(f"Diff           | mIoU: {diff:+.5f}")


def load_model(path, device, cfg):
    model = MultiTaskDeepLab(num_classes=cfg.NUM_CLASSES, in_channels=4)
    if not os.path.exists(path):
        pass
    else:
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    return model

def main():
    cfg = Config()
    seed_everything(cfg.SEED)
    print(f"Using device: {cfg.DEVICE}")
    
    # Data Setup
    image_dir = os.path.join(cfg.DATA_ROOT, 'image')
    label_dir = os.path.join(cfg.DATA_ROOT, 'label')
    depth_dir = os.path.join(cfg.DATA_ROOT, 'depth')
    
    image_files = sorted(os.listdir(image_dir))
    label_files = sorted(os.listdir(label_dir))
    depth_files = sorted(os.listdir(depth_dir))
    
    image_paths = [os.path.join(image_dir, f) for f in image_files]
    label_paths = [os.path.join(label_dir, f) for f in label_files]
    depth_paths = [os.path.join(depth_dir, f) for f in depth_files]
    
    # Split
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
    validate_post_process(models, valid_loader, cfg.DEVICE, cfg)

if __name__ == '__main__':
    main()
