
import sys
from pathlib import Path

# Add src to path to potentially import, but for this standalone test we will copy necessary parts 
# to ensure it works without complex dependency management if imports fail.
sys.path.append(str(Path(__file__).parent.parent / "src"))

import argparse
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- Constants from pipeline-9.py ---
RGB_MEAN = torch.tensor([133.88, 112.97, 102.11]) / 255.0
RGB_STD = torch.tensor([71.74, 71.53, 74.75]) / 255.0
DEPTH_MIN = 0.71
DEPTH_MAX = 10.0

@dataclass
class Config:
    dataset_root: Path
    epochs: int = 200
    batch_size: int = 4
    num_workers: int = 0 # 0 for debugging/simplicity
    learning_rate: float = 1e-3 # Higher LR for overfitting
    weight_decay: float = 1e-4
    img_size: Tuple[int, int] = (480, 640)
    num_classes: int = 13
    ignore_index: int = 255
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    loss_ce_w: float = 0.7
    loss_lovasz_w: float = 0.3
    encoder_name: str = "timm-efficientnet-b4"

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def build_transforms(size: Tuple[int, int]) -> A.Compose:
    h, w = size
    # No augmentation for overfitting test, just resize and normalize
    return A.Compose(
        [
            A.Resize(height=h, width=w),
            ToTensorV2(transpose_mask=True),
        ],
        additional_targets={"depth": "image"},
    )

class NYUv2Segmentation(Dataset):
    def __init__(self, root: Path, ids: Sequence[str], transforms: A.Compose):
        self.root = root
        self.ids = list(ids)
        self.transforms = transforms
        self.images = []
        self.depths = []
        self.masks = []

        print(f"Loading {len(self.ids)} images...")
        for name in self.ids:
            img_path = self.root / "train" / "image" / name
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Missing image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.images.append(image)

            depth_path = self.root / "train" / "depth" / name
            depth = self._load_depth(depth_path)
            self.depths.append(depth)

            mask_path = self.root / "train" / "label" / name
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise FileNotFoundError(f"Missing label: {mask_path}")
            self.masks.append(mask)

    def _load_depth(self, path: Path) -> np.ndarray:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Missing depth: {path}")
        depth = depth.astype(np.float32)
        depth_m = depth / 1000.0
        depth_m = np.clip(depth_m, DEPTH_MIN, DEPTH_MAX)
        depth_log = np.log1p(depth_m) / np.log1p(DEPTH_MAX)
        return depth_log[..., None]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        depth = self.depths[idx]
        mask = self.masks[idx]

        data = self.transforms(image=image, depth=depth, mask=mask)
        
        image_t = data["image"].float()
        depth_t = data["depth"].float()
        mask_t = data["mask"].long()

        if image_t.max() > 1.5:
            image_t = image_t / 255.0

        rgb = (image_t[:3] - RGB_MEAN.view(3, 1, 1)) / RGB_STD.view(3, 1, 1)
        x = torch.cat([rgb, depth_t[:1]], dim=0)

        return x, mask_t

class IoUMetric:
    def __init__(self, num_classes: int, ignore_index: int):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds = preds.view(-1).cpu()
        targets = targets.view(-1).cpu()
        mask = targets != self.ignore_index
        preds = preds[mask]
        targets = targets[mask]
        if preds.numel() == 0:
            return
        k = (targets * self.num_classes + preds).long()
        bins = torch.bincount(k, minlength=self.num_classes ** 2).float()
        self.confusion += bins.view(self.num_classes, self.num_classes)

    def compute(self) -> float:
        intersection = torch.diag(self.confusion)
        union = self.confusion.sum(1) + self.confusion.sum(0) - intersection
        
        iou = intersection / (union + 1e-7)
        
        # Only compute mean over classes that are present in the union
        valid_classes = union > 0
        if valid_classes.sum() == 0:
            return 0.0
            
        miou = iou[valid_classes].mean().item()
        return miou

def lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard

def lovasz_softmax_flat(probs, labels):
    if probs.numel() == 0:
        return probs.sum()
    num_classes = probs.size(1)
    losses = []
    for c in range(num_classes):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        class_pred = probs[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    if len(losses) == 0:
        return torch.tensor(0.0, device=probs.device)
    return torch.stack(losses).mean()

def lovasz_softmax(probas, labels, ignore_index=255):
    if probas.numel() == 0:
        return probas.sum()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, probas.size(1))
    labels = labels.view(-1)
    valid = labels != ignore_index
    probas = probas[valid]
    labels = labels[valid]
    return lovasz_softmax_flat(probas, labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data"))
    args = parser.parse_args()

    config = Config(dataset_root=args.dataset_root)
    set_seed(config.seed)
    
    # Get first 4 images
    image_dir = config.dataset_root / "train" / "image"
    all_ids = sorted([p.name for p in image_dir.glob("*.png")])
    if not all_ids:
        print("No images found!")
        return
    
    subset_ids = all_ids[:4]
    print(f"Overfitting on: {subset_ids}")

    transforms = build_transforms(config.img_size)
    dataset = NYUv2Segmentation(config.dataset_root, subset_ids, transforms)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    model = smp.DeepLabV3Plus(
        encoder_name=config.encoder_name,
        encoder_weights="imagenet",
        in_channels=4,
        classes=config.num_classes,
    ).to(config.device)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=config.ignore_index)
    scaler = GradScaler(enabled=config.amp and config.device == "cuda")

    print("Starting training...")
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0
        
        for x, y in loader:
            x, y = x.to(config.device), y.to(config.device)
            
            with autocast(enabled=scaler.is_enabled()):
                logits = model(x)
                ce = ce_loss_fn(logits, y)
                lovasz = lovasz_softmax(F.softmax(logits, dim=1), y, ignore_index=config.ignore_index)
                loss = config.loss_ce_w * ce + config.loss_lovasz_w * lovasz
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()

        # Evaluation on the SAME data
        if epoch % 10 == 0:
            model.eval()
            metric = IoUMetric(config.num_classes, config.ignore_index)
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(config.device), y.to(config.device)
                    logits = model(x)
                    preds = logits.argmax(dim=1)
                    metric.update(preds, y)
            
            miou = metric.compute()
            print(f"Epoch {epoch}: Loss={epoch_loss:.4f}, mIoU={miou:.4f}")
            
            if miou > 0.95:
                print("SUCCESS: Model successfully overfitted (mIoU > 0.95)!")
                return

    print("FINISHED: Training loop ended.")

if __name__ == "__main__":
    main()
