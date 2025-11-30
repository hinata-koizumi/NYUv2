# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
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
from sklearn.model_selection import KFold
from tqdm import tqdm

# Dataset-specific normalization (Table 2)
RGB_MEAN = torch.tensor([133.88, 112.97, 102.11]) / 255.0
RGB_STD = torch.tensor([71.74, 71.53, 74.75]) / 255.0
# Depth range and log compression (Table 3)
DEPTH_MIN = 0.71
DEPTH_MAX = 10.0


@dataclass
class Config:
    dataset_root: Path
    output_dir: Path
    epochs: int = 60
    batch_size: int = 2
    accum_steps: int = 2
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    img_size: Tuple[int, int] = (480, 640)
    num_classes: int = 13
    ignore_index: int = 255
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    grad_clip: float = 1.0
    t0: int = 5
    t_mult: int = 2
    submission_filename: str = "submission.npy"
    # Simplified loss weights: CE + Lovasz only
    loss_ce_w: float = 0.7
    loss_lovasz_w: float = 0.3
    encoder_name: str = "timm-efficientnet-b4"
    n_splits: int = 5
    # Multi-scale TTA parameters
    tta_scales: List[float] = None  # If None, defaults to [0.75, 1.0, 1.25]

    def __post_init__(self):
        self.dataset_root = Path(self.dataset_root).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.tta_scales is None:
            self.tta_scales = [0.75, 1.0, 1.25]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def list_ids(root: Path, split: str) -> List[str]:
    image_dir = root / split / "image"
    return sorted([p.name for p in image_dir.glob("*.png")])


def build_transforms(size: Tuple[int, int], split: str) -> A.Compose:
    h, w = size

    if split == "train":
        # Simplified augmentation: reduced CoarseDropout, removed VerticalFlip
        return A.Compose(
            [
                A.Resize(height=h, width=w),
                A.HorizontalFlip(p=0.5),
                # VerticalFlip removed (室内シーンだと上下反転がやや不自然)
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(0.02, 0.02),
                    rotate=(-8, 8),
                    shear=(-4, 4),
                    fit_output=False,
                    p=0.3,
                ),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(p=0.3),
                # CoarseDropout: reduced probability from 0.35 to 0.2
                A.CoarseDropout(
                    max_holes=8,
                    max_height=48,
                    max_width=48,
                    min_holes=8,
                    min_height=48,
                    min_width=48,
                    fill_value=0,
                    mask_fill_value=255,
                    p=0.2,
                ),
                ToTensorV2(transpose_mask=True),
            ],
            additional_targets={"depth": "image"},
        )

    # Validation/Test transforms
    return A.Compose(
        [
            A.Resize(height=h, width=w),
            ToTensorV2(transpose_mask=True),
        ],
        additional_targets={"depth": "image"},
    )


class NYUv2Segmentation(Dataset):
    def __init__(
        self,
        root: Path,
        ids: Sequence[str],
        split: str,
        transforms: A.Compose,
        cache_data: bool = True,
    ):
        self.root = root
        self.ids = list(ids)
        self.split = split
        self.transforms = transforms
        self.cache_data = cache_data
        self.images = []
        self.depths = []
        self.masks = []

        if self.cache_data:
            print(f"Caching {len(self.ids)} images for {split}...")
            for name in tqdm(self.ids, desc=f"Loading {split}"):
                img_path = self.root / self.split / "image" / name
                image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(f"Missing image: {img_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.images.append(image)

                depth_path = self.root / self.split / "depth" / name
                depth = self._load_depth(depth_path)
                self.depths.append(depth)

                if self.split != "test":
                    mask_path = self.root / self.split / "label" / name
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                    if mask is None:
                        raise FileNotFoundError(f"Missing label: {mask_path}")
                    self.masks.append(mask)
                else:
                    self.masks.append(None)

    def __len__(self) -> int:
        return len(self.ids)

    def _load_depth(self, path: Path) -> np.ndarray:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Missing depth: {path}")
        depth = depth.astype(np.float32)
        depth_m = depth / 1000.0  # assume millimeters -> meters
        depth_m = np.clip(depth_m, DEPTH_MIN, DEPTH_MAX)
        depth_log = np.log1p(depth_m) / np.log1p(DEPTH_MAX)
        return depth_log[..., None]

    def __getitem__(self, idx: int):
        name = self.ids[idx]
        
        if self.cache_data:
            image = self.images[idx]
            depth = self.depths[idx]
            mask = self.masks[idx]
        else:
            image_path = self.root / self.split / "image" / name
            depth_path = self.root / self.split / "depth" / name
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Missing image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            depth = self._load_depth(depth_path)
            mask = None
            if self.split != "test":
                mask_path = self.root / self.split / "label" / name
                mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if mask is None:
                    raise FileNotFoundError(f"Missing label: {mask_path}")

        if mask is not None:
            data = self.transforms(image=image, depth=depth, mask=mask)
            mask_t = data["mask"].long()
        else:
            data = self.transforms(image=image, depth=depth)
        
        image_t = data["image"].float()
        depth_t = data["depth"].float()

        if image_t.max() > 1.5:
            image_t = image_t / 255.0

        # Normalize RGB
        rgb = (image_t[:3] - RGB_MEAN.view(3, 1, 1)) / RGB_STD.view(3, 1, 1)
        
        # 4-channel early fusion: concatenate RGB and depth
        x = torch.cat([rgb, depth_t[:1]], dim=0)  # [4, H, W]
        
        if mask is None:
            return x, name
        return x, mask_t, name


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

    def compute(self) -> Tuple[float, torch.Tensor]:
        intersection = torch.diag(self.confusion)
        union = self.confusion.sum(1) + self.confusion.sum(0) - intersection
        iou = intersection / (union + 1e-7)
        miou = iou.mean().item()
        return miou, iou


def lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax_flat(probs, labels, classes="present"):
    if probs.numel() == 0:
        return probs.sum()
    num_classes = probs.size(1)
    losses = []
    class_to_sum = list(range(num_classes)) if classes in ["all", "present"] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if classes == "present" and fg.sum() == 0:
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


def lovasz_softmax(probas, labels, classes="present", ignore_index=255):
    if probas.numel() == 0:
        return probas.sum()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, probas.size(1))
    labels = labels.view(-1)
    valid = labels != ignore_index
    probas = probas[valid]
    labels = labels[valid]
    return lovasz_softmax_flat(probas, labels, classes=classes)


def build_model(config: Config) -> nn.Module:
    """
    Build DeepLabV3+ with 4-channel input (RGB + Depth early fusion).
    Segmentation-only model (no depth/edge heads).
    """
    print(f"Building DeepLabV3+ with encoder: {config.encoder_name}")
    print(f"  Input channels: 4 (RGB + Depth early fusion)")
    
    model = smp.DeepLabV3Plus(
        encoder_name=config.encoder_name,
        encoder_weights="imagenet",
        in_channels=4,  # RGB (3) + Depth (1)
        classes=config.num_classes,
    )
    return model


def compute_class_weights(config: Config) -> torch.Tensor:
    """
    Compute class weights using inverse frequency.
    Rare class multiplier reduced from 2.0 to 1.5.
    """
    weight_cache = config.output_dir / "class_weights_inverse_freq.npy"
    if weight_cache.exists():
        weights = np.load(weight_cache)
        return torch.tensor(weights, dtype=torch.float32)

    counts = np.zeros(config.num_classes, dtype=np.float64)
    label_dir = config.dataset_root / "train" / "label"
    for path in label_dir.glob("*.png"):
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        mask = mask.astype(np.int64)
        valid = mask != config.ignore_index
        hist = np.bincount(mask[valid].ravel(), minlength=config.num_classes)
        counts += hist[: config.num_classes]

    min_count = counts[counts > 0].min() if (counts > 0).any() else 1.0
    weights = min_count / (counts + 1e-8)
    weights = weights / weights.mean()
    
    # Rare class multiplier reduced from 2.0 to 1.5
    rare_classes = [1, 7, 10]
    for rare_cls in rare_classes:
        if rare_cls < len(weights):
            weights[rare_cls] *= 1.5
    
    weights = weights / weights.mean()
    
    np.save(weight_cache, weights)
    print(f"Class weights: {weights}")
    return torch.tensor(weights, dtype=torch.float32)


def prepare_dataloaders(config: Config, train_ids: List[str], val_ids: List[str]):
    train_tf = build_transforms(config.img_size, "train")
    val_tf = build_transforms(config.img_size, "val")
    test_tf = build_transforms(config.img_size, "test")

    train_loader = None
    val_loader = None

    if len(train_ids) > 0:
        train_ds = NYUv2Segmentation(config.dataset_root, train_ids, "train", train_tf)
        # Simplified: use shuffle=True instead of WeightedRandomSampler
        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,  # Simple shuffle instead of weighted sampler
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

    if len(val_ids) > 0:
        val_ds = NYUv2Segmentation(config.dataset_root, val_ids, "train", val_tf)
        val_loader = DataLoader(
            val_ds,
            batch_size=max(1, config.batch_size // 2),
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    test_ids = list_ids(config.dataset_root, "test")
    test_ds = NYUv2Segmentation(config.dataset_root, test_ids, "test", test_tf)
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, val_loader, test_loader


def forward_loss(model, batch, device, ce_loss, config: Config):
    """
    Simplified loss: CE + Lovasz from epoch 1.
    No Dice, no depth/edge losses.
    """
    inputs, targets = batch
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    
    # Forward pass returns segmentation logits only
    seg_logits = model(inputs)
    
    # CE loss
    ce = ce_loss(seg_logits, targets)
    
    # Lovasz loss (from epoch 1, not delayed)
    lovasz = lovasz_softmax(F.softmax(seg_logits, dim=1), targets, ignore_index=config.ignore_index)
    
    # Total loss: 0.7 * CE + 0.3 * Lovasz
    total_loss = config.loss_ce_w * ce + config.loss_lovasz_w * lovasz
    
    return total_loss, seg_logits, targets


def evaluate(model, loader, device, ce_loss, config: Config):
    model.eval()
    metric = IoUMetric(config.num_classes, config.ignore_index)
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating")
        for batch in pbar:
            loss, logits, targets = forward_loss(model, (batch[0], batch[1]), device, ce_loss, config)
            preds = logits.argmax(dim=1)
            metric.update(preds, targets)
            bsz = targets.size(0)
            total_loss += loss.item() * bsz
            total_samples += bsz
            pbar.set_postfix({'loss': f'{total_loss/max(total_samples,1):.4f}'})
    miou, class_iou = metric.compute()
    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, miou, class_iou


def deepcopy_state_dict(state_dict: dict) -> dict:
    return {k: v.clone() for k, v in state_dict.items()}


def train(config: Config, fold_idx: int, train_ids: List[str], val_ids: List[str]):
    set_seed(config.seed + fold_idx)
    device = torch.device(config.device)
    train_loader, val_loader, _ = prepare_dataloaders(config, train_ids, val_ids)
    model = build_model(config).to(device)

    class_weights = compute_class_weights(config).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=config.ignore_index)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = GradScaler(enabled=config.amp and device.type == "cuda")
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.t0, T_mult=config.t_mult)

    best_miou = -1.0
    best_model_state = None
    best_class_iou = None
    
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        samples = 0
        optimizer.zero_grad(set_to_none=True)
        
        pbar = tqdm(train_loader, desc=f"Fold {fold_idx+1} Epoch {epoch:03d}")
        for step, batch in enumerate(pbar, start=1):
            with autocast(enabled=scaler.is_enabled()):
                loss, logits, targets = forward_loss(
                    model, (batch[0], batch[1]), device, ce_loss, config
                )
                loss = loss / config.accum_steps
            
            scaler.scale(loss).backward()
            
            if step % config.accum_steps == 0 or step == len(train_loader):
                if config.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                # Update scheduler at step level
                effective_step = step // config.accum_steps
                effective_steps_per_epoch = (len(train_loader) + config.accum_steps - 1) // config.accum_steps
                scheduler.step(epoch - 1 + effective_step / effective_steps_per_epoch)

            bsz = targets.size(0)
            epoch_loss += loss.item() * bsz * config.accum_steps
            samples += bsz
            
            # Update progress bar
            if step % config.accum_steps == 0 or step == len(train_loader):
                current_loss = epoch_loss / max(samples, 1)
                pbar.set_postfix({'loss': f'{current_loss:.4f}'})
            
        avg_train_loss = epoch_loss / max(samples, 1)
        val_loss, val_miou, class_iou = evaluate(model, val_loader, device, ce_loss, config)

        print(
            f"[Fold {fold_idx+1} | Epoch {epoch:03d}] train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_mIoU={val_miou:.4f}"
        )

        if val_miou > best_miou:
            best_miou = val_miou
            best_model_state = deepcopy_state_dict(model.state_dict())
            best_class_iou = class_iou.detach().clone()
            
            ckpt_path = config.output_dir / "checkpoints" / f"fold{fold_idx+1}_best.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  New best mIoU: {best_miou:.4f} (Saved)")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    if best_class_iou is not None:
        class_iou_path = config.output_dir / f"fold{fold_idx+1}_best_class_iou.npy"
        np.save(class_iou_path, best_class_iou.cpu().numpy())
    
    return model, best_miou


def run_inference(models: List[nn.Module], loader: DataLoader, config: Config) -> Path:
    """
    Multi-scale + flip TTA (Test Time Augmentation)
    - Scales: 0.75, 1.0, 1.25 (configurable via config.tta_scales)
    - Each scale: original + horizontal flip
    - All logits resized to original resolution and averaged
    """
    device = torch.device(config.device)
    for model in models:
        model.eval()

    scales = config.tta_scales
    use_amp = config.amp and device.type == "cuda"
    
    print(f"Ensembling {len(models)} models with multi-scale TTA...")
    print(f"  TTA scales: {scales}")
    print(f"  AMP enabled: {use_amp}")
    
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference (Multi-scale TTA)"):
            inputs, name = batch
            inputs_orig = inputs.to(device, non_blocking=True)

            # Get original size from tensor
            orig_h, orig_w = inputs_orig.shape[2:]
            
            # Accumulate logits by summing (memory efficient)
            sum_logits = None
            count = 0
            
            for scale in scales:
                # Resize inputs to scaled size
                scaled_h = int(orig_h * scale)
                scaled_w = int(orig_w * scale)
                
                inputs_scaled = F.interpolate(
                    inputs_orig,
                    size=(scaled_h, scaled_w),
                    mode='bilinear',
                    align_corners=False
                )
                
                # Original orientation
                for model in models:
                    with autocast(enabled=use_amp):
                        logits = model(inputs_scaled)
                    # Resize logits back to original size
                    logits_resized = F.interpolate(
                        logits,
                        size=(orig_h, orig_w),
                        mode='bilinear',
                        align_corners=False
                    )
                    if sum_logits is None:
                        sum_logits = logits_resized
                    else:
                        sum_logits = sum_logits + logits_resized
                    count += 1
                
                # Horizontal flip
                inputs_flip = torch.flip(inputs_scaled, dims=[3])
                
                for model in models:
                    with autocast(enabled=use_amp):
                        logits_flip = model(inputs_flip)
                    # Flip back logits
                    logits_flip = torch.flip(logits_flip, dims=[3])
                    # Resize logits back to original size
                    logits_flip_resized = F.interpolate(
                        logits_flip,
                        size=(orig_h, orig_w),
                        mode='bilinear',
                        align_corners=False
                    )
                    if sum_logits is None:
                        sum_logits = logits_flip_resized
                    else:
                        sum_logits = sum_logits + logits_flip_resized
                    count += 1
            
            # Average all logits
            avg_logits = sum_logits / count
            
            pred = avg_logits.argmax(dim=1).cpu().numpy()
            preds.append(pred)

    predictions = np.concatenate(preds, axis=0)
    save_path = config.output_dir / config.submission_filename
    np.save(save_path, predictions)
    print(f"Saved submission to {save_path}")
    return save_path


def parse_args():
    parser = argparse.ArgumentParser(description="NYUv2 Seg - Pipeline-9: Simplified 4ch Early Fusion")
    parser.add_argument("--dataset-root", type=Path, default=Path("data"), help="Path to data")
    parser.add_argument("--output-dir", type=Path, default=Path("data/output_pipeline9"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--submission-filename", type=str, default="submission.npy")
    parser.add_argument("--encoder-name", type=str, default="timm-efficientnet-b4")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--tta-scales", type=float, nargs="+", default=[0.75, 1.0, 1.25],
                        help="TTA scales for multi-scale inference (default: 0.75 1.0 1.25)")
    return parser.parse_args()


def main():
    args = parse_args()

    config = Config(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
        submission_filename=args.submission_filename,
        encoder_name=args.encoder_name,
        n_splits=args.n_splits,
        tta_scales=args.tta_scales,
    )

    all_ids = np.array(list_ids(config.dataset_root, "train"))
    kf = KFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    trained_models = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_ids)):
        fold_id = fold_idx + 1
        print(f"\n=== Fold {fold_id}/{config.n_splits} ===")
        
        ckpt_path = config.output_dir / "checkpoints" / f"fold{fold_id}_best.pt"
        if ckpt_path.exists():
            print(f"Loading existing checkpoint: {ckpt_path}")
            model = build_model(config).to(config.device)
            model.load_state_dict(torch.load(ckpt_path, map_location=config.device))
            trained_models.append(model)
            continue
            
        t_ids = all_ids[train_idx].tolist()
        v_ids = all_ids[val_idx].tolist()
        
        model, _ = train(config, fold_idx, t_ids, v_ids)
        trained_models.append(model)

    _, _, test_loader = prepare_dataloaders(config, [], [])
    run_inference(trained_models, test_loader, config)


if __name__ == "__main__":
    main()

