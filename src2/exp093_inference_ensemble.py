import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from tqdm import tqdm
import cv2
import segmentation_models_pytorch as smp


class Config:
    SEED = 42
    CROP_SIZE = (512, 512)
    NUM_CLASSES = 13
    IGNORE_INDEX = 255
    DATA_ROOT = "data/test"  # images only, no labels required
    # Dynamic model path resolution
    # 1. Server/Zip mode: model_weights/foldX_model_best.pth
    # 2. Local mode: data/outputs/exp093_fpn_convnextb_smartcrop/foldX/model_best.pth
    _SERVER_PATH = "model_weights/fold{}_model_best.pth"
    _LOCAL_PATH = "data/outputs/exp093_fpn_convnextb_smartcrop/fold{}/model_best.pth"
    
    MODEL_PATHS = []
    for i in range(5):
        if os.path.exists(_SERVER_PATH.format(i)):
            MODEL_PATHS.append(_SERVER_PATH.format(i))
        elif os.path.exists(_LOCAL_PATH.format(i)):
            MODEL_PATHS.append(_LOCAL_PATH.format(i))
        else:
            # Fallback or keep server path to let load_models print warning
            MODEL_PATHS.append(_SERVER_PATH.format(i))
    # OUTPUT_PATH and DIR will be set in main() via args
    OUTPUT_PATH = "data/outputs/exp093_fpn_convnextb_smartcrop/ensemble/submission.npy"
    OUTPUT_DIR = "data/outputs/exp093_fpn_convnextb_smartcrop/ensemble"
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
    def __init__(self, image_paths, transform, cfg: Config):
        self.image_paths = image_paths
        self.transform = transform
        self.cfg = cfg

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(self.cfg.MEAN).view(-1, 1, 1)
        std = torch.tensor(self.cfg.STD).view(-1, 1, 1)
        image = (image - mean) / std

        return image, path


class MultiTaskFPN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int):
        super().__init__()
        self.backbone = smp.FPN(
            encoder_name="tu-convnext_base",
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
        )

    def forward(self, x):
        features = self.backbone.encoder(x)
        decoder_out = self.backbone.decoder(features)
        seg_logits = self.backbone.segmentation_head(decoder_out)
        return seg_logits


def load_models(cfg: Config):
    models = []
    for path in cfg.MODEL_PATHS:
        if not os.path.exists(path):
            print(f"[WARN] Model not found, skip: {path}")
            continue
        model = MultiTaskFPN(num_classes=cfg.NUM_CLASSES, in_channels=3)
        state = torch.load(path, map_location=cfg.DEVICE)
        # Drop depth head weights if present (training model had depth_head)
        filtered = {
            k: v for k, v in state.items()
            if not k.startswith("depth_head.")
        }
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if unexpected:
            print(f"[WARN] Unexpected keys ignored: {unexpected}")
        if missing:
            # Only acceptable missing keys are depth_head.* since we removed them
            unexpected_missing = [m for m in missing if not m.startswith("depth_head.")]
            if unexpected_missing:
                raise RuntimeError(f"Missing keys not related to depth_head: {unexpected_missing}")
        model.to(cfg.DEVICE)
        model.eval()
        models.append(model)
        print(f"[OK] Loaded: {path}")
    if not models:
        raise FileNotFoundError("No model weights loaded. Check MODEL_PATHS.")
    return models


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/outputs/exp093_fpn_convnextb_smartcrop/ensemble/submission.npy", help="Path to save submission.npy")
    args = parser.parse_args()

    cfg = Config()
    cfg.OUTPUT_PATH = args.output
    cfg.OUTPUT_DIR = os.path.dirname(args.output)
    
    seed_everything(cfg.SEED)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


    image_dir = None
    # Candidate paths for dataset root
    candidates = [
        "data/NYUv2/test/image", 
        "data/test/image", 
        os.path.join(cfg.DATA_ROOT, "image")
    ]
    
    for candidate in candidates:
        if os.path.isdir(candidate):
            image_dir = candidate
            print(f"Found data at: {image_dir}")
            break
            
    if image_dir is None:
        # Fallback to current behavior (will likely crash later but print error now)
        print("ERROR: Could not find dataset directory. Checked: ", candidates)
        image_dir = os.path.join(cfg.DATA_ROOT, "image") # Default back to force error if needed or use arg match relative

    if os.path.exists(image_dir):
        image_files = sorted(os.listdir(image_dir))
        image_paths = [os.path.join(image_dir, f) for f in image_files]
        print(f"Images: {len(image_paths)}")
    else:
        print(f"CRITICAL: Image directory {image_dir} does not exist.")
        image_paths = []


    dataset = TestDataset(image_paths, transform=get_transforms(cfg), cfg=cfg)
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    models = load_models(cfg)
    all_preds = []

    with torch.no_grad():
        for images, paths in tqdm(loader, desc="Ensemble Inference"):
            images = images.to(cfg.DEVICE)
            avg_logits = None
            for m in models:
                logits = m(images)
                if avg_logits is None:
                    avg_logits = logits
                else:
                    avg_logits += logits
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
                "data_root": cfg.DATA_ROOT,
                "mean": cfg.MEAN,
                "std": cfg.STD,
                "crop_size": cfg.CROP_SIZE,
                "batch_size": cfg.BATCH_SIZE,
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRITICAL ERROR IN INFERENCE: {e}")
        # Create a dummy submission file to allow score.py to run and possibly show a score of 0 or error log
        # This helps confirm if the script at least ran.
        cfg = Config()
        output_dir = os.path.dirname(cfg.OUTPUT_PATH)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        dummy_shape = (654, 512, 512) # Approximate shape for test set
        dummy = np.zeros(dummy_shape, dtype=np.uint8)
        np.save(cfg.OUTPUT_PATH, dummy)
        print(f"[FALLBACK] Saved dummy submission to {cfg.OUTPUT_PATH} due to error.")

