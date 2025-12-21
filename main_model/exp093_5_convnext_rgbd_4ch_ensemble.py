import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from tqdm import tqdm
import cv2
import segmentation_models_pytorch as smp


class Config:
    SEED = 42
    CROP_SIZE = (576, 768)  # H, W - matching exp093_5_convnext_rgbd_4ch
    NUM_CLASSES = 13
    IGNORE_INDEX = 255
    DATA_ROOT = "data/test"  # images only, no labels required
    
    # Depth range (matching training config)
    DEPTH_MIN = 0.71  # m
    DEPTH_MAX = 10.0  # m
    
    # Model paths for exp093_5_convnext_rgbd_4ch
    MODEL_PATHS = [
        f"data/outputs/exp093_5_convnext_rgbd_4ch/fold{i}/model_best.pth" for i in range(5)
    ]
    
    # OUTPUT_PATH and DIR will be set in main() via args
    OUTPUT_PATH = "data/outputs/exp093_5_convnext_rgbd_4ch/ensemble/submission.npy"
    OUTPUT_DIR = "data/outputs/exp093_5_convnext_rgbd_4ch/ensemble"
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    BATCH_SIZE = 2


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(cfg: Config):
    return A.Compose(
        [
            A.Resize(height=cfg.CROP_SIZE[0], width=cfg.CROP_SIZE[1]),
        ]
    )


class TestDataset(Dataset):
    def __init__(self, image_paths, depth_paths, transform, cfg: Config):
        self.image_paths = image_paths
        self.depth_paths = depth_paths
        self.transform = transform
        self.cfg = cfg

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load RGB image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load depth image
        depth_path = self.depth_paths[idx]
        raw_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if raw_depth is None:
            raise FileNotFoundError(f"Depth not found: {depth_path}")
        
        # Convert depth from mm to meters
        raw_depth = raw_depth.astype(np.float32) / 1000.0  # mm -> m
        
        # Clip depth
        raw_depth = np.clip(raw_depth, self.cfg.DEPTH_MIN, self.cfg.DEPTH_MAX)
        
        # Inverse depth encoding (matching training preprocessing)
        inv_d = 1.0 / raw_depth
        inv_min = 1.0 / self.cfg.DEPTH_MAX
        inv_max = 1.0 / self.cfg.DEPTH_MIN
        
        norm_inv_d = (inv_d - inv_min) / (inv_max - inv_min)
        norm_inv_d = np.clip(norm_inv_d, 0.0, 1.0)
        
        # Resize depth to match image size
        if norm_inv_d.shape != image.shape[:2]:
            norm_inv_d = cv2.resize(norm_inv_d, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Stack RGB and depth to create 4-channel image [H, W, 4]
        img_4ch = np.dstack([image, norm_inv_d])

        # Apply transforms
        if self.transform is not None:
            augmented = self.transform(image=img_4ch)
            img_4ch = augmented["image"]

        # Split RGB and depth
        image_rgb = img_4ch[:, :, :3]
        image_depth = img_4ch[:, :, 3:4]  # Keep as [H, W, 1] for stacking

        # Convert to tensor
        image_rgb = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        image_depth = torch.from_numpy(image_depth).permute(2, 0, 1).float()  # Already 0-1
        
        # Normalize RGB
        mean = torch.tensor(self.cfg.MEAN).view(-1, 1, 1)
        std = torch.tensor(self.cfg.STD).view(-1, 1, 1)
        image_rgb = (image_rgb - mean) / std
        
        # Concatenate RGB and depth to create 4-channel tensor [4, H, W]
        image_4ch = torch.cat([image_rgb, image_depth], dim=0)

        return image_4ch, img_path


class MultiTaskFPN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int):
        super().__init__()
        self.backbone = smp.FPN(
            encoder_name="tu-convnext_base",
            encoder_weights=None,
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


def load_models(cfg: Config):
    models = []
    for path in cfg.MODEL_PATHS:
        if not os.path.exists(path):
            print(f"[WARN] Model not found, skip: {path}")
            continue
        model = MultiTaskFPN(num_classes=cfg.NUM_CLASSES, in_channels=4)  # 4 channels for RGBD
        state = torch.load(path, map_location=cfg.DEVICE)
        # The model may have depth_head weights, but we only need segmentation for inference
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            print(f"[WARN] Unexpected keys ignored: {unexpected[:5]}...")  # Show first 5
        if missing:
            # Only acceptable missing keys are depth_head.* if we're not loading them
            unexpected_missing = [m for m in missing if not m.startswith("depth_head.")]
            if unexpected_missing:
                print(f"[WARN] Missing keys: {unexpected_missing[:5]}...")
        model.to(cfg.DEVICE)
        model.eval()
        models.append(model)
        print(f"[OK] Loaded: {path}")
    if not models:
        raise FileNotFoundError("No model weights loaded. Check MODEL_PATHS.")
    print(f"Loaded {len(models)} models for ensemble")
    return models


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/outputs/exp093_5_convnext_rgbd_4ch/ensemble/submission.npy", 
                       help="Path to save submission.npy")
    args = parser.parse_args()

    cfg = Config()
    cfg.OUTPUT_PATH = args.output
    cfg.OUTPUT_DIR = os.path.dirname(args.output)
    
    seed_everything(cfg.SEED)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Find test image and depth directories
    image_dir = None
    depth_dir = None
    candidates = [
        ("data/NYUv2/test/image", "data/NYUv2/test/depth"),
        ("data/test/image", "data/test/depth"),
        (os.path.join(cfg.DATA_ROOT, "image"), os.path.join(cfg.DATA_ROOT, "depth"))
    ]
    
    for img_candidate, depth_candidate in candidates:
        if os.path.isdir(img_candidate) and os.path.isdir(depth_candidate):
            image_dir = img_candidate
            depth_dir = depth_candidate
            print(f"Found data at: {image_dir} and {depth_dir}")
            break
            
    if image_dir is None or depth_dir is None:
        print("ERROR: Could not find dataset directories. Checked: ", candidates)
        image_dir = os.path.join(cfg.DATA_ROOT, "image")
        depth_dir = os.path.join(cfg.DATA_ROOT, "depth")

    if os.path.exists(image_dir) and os.path.exists(depth_dir):
        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        image_paths = [os.path.join(image_dir, f) for f in image_files]
        depth_paths = [os.path.join(depth_dir, f) for f in image_files]  # Assume same filenames
        print(f"Found {len(image_paths)} images")
    else:
        print(f"CRITICAL: Image or depth directory does not exist.")
        image_paths = []
        depth_paths = []

    if len(image_paths) == 0:
        raise ValueError("No images found for inference")

    dataset = TestDataset(image_paths, depth_paths, transform=get_transforms(cfg), cfg=cfg)
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    models = load_models(cfg)
    all_preds = []

    print("Starting ensemble inference...")
    with torch.no_grad():
        for images, paths in tqdm(loader, desc="Ensemble Inference"):
            images = images.to(cfg.DEVICE)
            avg_logits = None
            for m in models:
                seg_logits, _ = m(images)  # Get segmentation logits, ignore depth
                if avg_logits is None:
                    avg_logits = seg_logits
                else:
                    avg_logits += seg_logits
            avg_logits /= len(models)
            preds = torch.argmax(avg_logits, dim=1)  # [B, H, W]
            all_preds.append(preds.cpu().numpy())

    # [N, H, W]
    submission = np.concatenate(all_preds, axis=0)
    np.save(cfg.OUTPUT_PATH, submission.astype(np.uint8))
    print(f"Saved: {cfg.OUTPUT_PATH}  shape={submission.shape}")

    # Save run config for traceability
    with open(os.path.join(cfg.OUTPUT_DIR, "ensemble_config.json"), "w") as f:
        json.dump(
            {
                "model_paths": cfg.MODEL_PATHS,
                "data_root": image_dir,
                "depth_root": depth_dir,
                "mean": cfg.MEAN,
                "std": cfg.STD,
                "crop_size": cfg.CROP_SIZE,
                "batch_size": cfg.BATCH_SIZE,
                "num_models": len(models),
                "depth_min": cfg.DEPTH_MIN,
                "depth_max": cfg.DEPTH_MAX,
            },
            f,
            indent=4,
        )
    print("Ensemble inference completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRITICAL ERROR IN INFERENCE: {e}")
