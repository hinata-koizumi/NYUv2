# -*- coding: utf-8 -*-
"""
High-performance NYUv2 semantic segmentation pipeline.

Key choices:
- Architecture: DeepLabV3+ (encoder: EfficientNetV2-L by default, switchable to MiT-B5 via --encoder-name) from segmentation_models_pytorch for strong contextual modeling and robust decoders. Supports 4-channel input (RGB + depth).
- Loss: Combo of class-weighted CrossEntropy (ignore_index=255), soft Dice, and late-stage Lovasz-Softmax to address heavy class imbalance and directly optimize IoU.
- Resolution: Native 480x640 (H,W) to avoid distortion; all augments preserve size.
- Normalization: Dataset-specific RGB mean/std; depth clipped to [0.71m, 10m], log-compressed.
- Inference: Ensembling of K-fold checkpoints with flip TTA.
"""
from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import albumentations as A
import cv2
import matplotlib.pyplot as plt
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
    batch_size: int = 4
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    val_ratio: float = 0.1
    img_size: Tuple[int, int] = (480, 640)  # (height, width) native resolution
    num_classes: int = 13
    ignore_index: int = 255
    use_depth: bool = True
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    grad_clip: float = 1.0
    t0: int = 5  # CosineAnnealingWarmRestarts initial period
    t_mult: int = 2
    save_every: int = 1
    submission_filename: str = "submission.npy"
    lovasz_start: int = 10  # start adding Lovasz loss in last N epochs
    loss_ce_w: float = 0.5
    loss_dice_w: float = 0.3
    loss_lovasz_w: float = 0.2
    encoder_name: str = "timm-efficientnetv2_l"
    n_splits: int = 5

    def __post_init__(self):
        self.dataset_root = Path(self.dataset_root).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # チェックポイント保存を無効化しているため、checkpointsフォルダは作成しない


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def list_ids(root: Path, split: str) -> List[str]:
    image_dir = root / split / "image"
    return sorted([p.name for p in image_dir.glob("*.png")])


def train_val_split(ids: Sequence[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    ids = list(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    val_size = max(1, int(len(ids) * val_ratio))
    val_ids = ids[:val_size]
    train_ids = ids[val_size:]
    return train_ids, val_ids


def build_transforms(size: Tuple[int, int], split: str) -> A.Compose:
    h, w = size
    if split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                # --- ↓↓↓ 重い処理をコメントアウト (CPU負荷軽減) ↓↓↓ ---
                # A.Affine(
                #     scale=(0.9, 1.1),
                #     translate_percent=(0.02, 0.02),
                #     rotate=(-8, 8),
                #     shear=(-4, 4),
                #     cval=0,
                #     mode=cv2.BORDER_REFLECT_101,
                #     fit_output=False,
                #     p=0.7,
                # ),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                # A.GaussianBlur(p=0.2),  # 重い処理をコメントアウト
                # --- ↑↑↑ ここまでコメントアウト ↑↑↑ ---
                # CoarseDropout は精度向上に効くので残す
                A.CoarseDropout(
                    max_holes=8,
                    max_height=48,
                    max_width=48,
                    fill_value=0,
                    mask_fill_value=255,
                    p=0.35,
                ),
                ToTensorV2(transpose_mask=True),
            ],
            additional_targets={"depth": "image"},
        )
    return A.Compose(
        [
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
        use_depth: bool = True,
        cache_data: bool = True,  # 追加: RAMにキャッシュするフラグ
    ):
        self.root = root
        self.ids = list(ids)
        self.split = split
        self.transforms = transforms
        self.use_depth = use_depth
        self.cache_data = cache_data
        self.images = []
        self.depths = []
        self.masks = []

        # --- 高速化: 最初に全部メモリに読み込む ---
        if self.cache_data:
            print(f"Caching {len(self.ids)} images for {split}...")
            for name in tqdm(self.ids, desc=f"Loading {split}"):
                # 画像読み込み
                img_path = self.root / self.split / "image" / name
                image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(f"Missing image: {img_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.images.append(image)

                # 深度読み込み
                depth_path = self.root / self.split / "depth" / name
                depth = self._load_depth(depth_path)
                self.depths.append(depth)

                # マスク読み込み (test以外)
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
        
        # キャッシュから取得
        if self.cache_data:
            image = self.images[idx]
            depth = self.depths[idx]
            mask = self.masks[idx]
        else:
            # 従来のディスク読み込み（フォールバック用）
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

        # Augmentation
        if mask is not None:
            data = self.transforms(image=image, depth=depth, mask=mask)
            mask_t = data["mask"].long()
        else:
            data = self.transforms(image=image, depth=depth)
        
        image_t = data["image"].float()
        depth_t = data["depth"].float()

        if image_t.max() > 1.5:
            image_t = image_t / 255.0

        # RGB Normalize
        rgb = (image_t[:3] - RGB_MEAN.view(3, 1, 1)) / RGB_STD.view(3, 1, 1)
        
        if self.use_depth:
            x = torch.cat([rgb, depth_t[:1]], dim=0)
        else:
            x = rgb

        if mask is None:
            return x, name

        return x, mask_t, name


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int, eps: float = 1e-7) -> torch.Tensor:
    num_classes = logits.shape[1]
    mask = targets != ignore_index
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    probs = torch.softmax(logits, dim=1)
    probs = probs * mask.unsqueeze(1)
    targets = targets.clamp(0, num_classes - 1)
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
    one_hot = one_hot * mask.unsqueeze(1)
    intersection = (probs * one_hot).sum(dim=(0, 2, 3))
    union = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    dice = 1.0 - (2 * intersection + eps) / (union + eps)
    return dice.mean()


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
    # Flatten
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, probas.size(1))
    labels = labels.view(-1)
    valid = labels != ignore_index
    probas = probas[valid]
    labels = labels[valid]
    return lovasz_softmax_flat(probas, labels, classes=classes)


def build_model(config: Config) -> nn.Module:
    model = smp.DeepLabV3Plus(
        encoder_name=config.encoder_name,
        encoder_weights="imagenet",
        in_channels=4 if config.use_depth else 3,
        classes=config.num_classes,
    )
    return model


def compute_class_weights(config: Config) -> torch.Tensor:
    weight_cache = config.output_dir / "class_weights.npy"
    if weight_cache.exists():
        weights = np.load(weight_cache)
        return torch.tensor(weights, dtype=torch.float32)

    counts = np.zeros(config.num_classes, dtype=np.float64)
    label_dir = config.dataset_root / "train" / "label"
    for path in sorted(label_dir.glob("*.png")):
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        mask = mask.astype(np.int64)
        valid = mask != config.ignore_index
        hist = np.bincount(mask[valid].ravel(), minlength=config.num_classes)
        counts += hist[: config.num_classes]

    eps = 1.1  # avoid log(1)
    weights = 1.0 / np.log(counts + eps)
    # Normalize to keep loss scale stable
    weights = weights / weights.mean()
    np.save(weight_cache, weights)
    return torch.tensor(weights, dtype=torch.float32)


def prepare_dataloaders(config: Config, train_ids: List[str], val_ids: List[str]):
    train_tf = build_transforms(config.img_size, "train")
    val_tf = build_transforms(config.img_size, "val")
    test_tf = build_transforms(config.img_size, "test")

    train_loader = None
    val_loader = None

    if len(train_ids) > 0:
        train_ds = NYUv2Segmentation(config.dataset_root, train_ids, "train", train_tf, use_depth=config.use_depth)
        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,  # エポック間の待機時間を削減
        )

    if len(val_ids) > 0:
        val_ds = NYUv2Segmentation(config.dataset_root, val_ids, "train", val_tf, use_depth=config.use_depth)
        val_loader = DataLoader(
            val_ds,
            batch_size=max(1, config.batch_size // 2),
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=True,  # エポック間の待機時間を削減
        )

    test_ids = list_ids(config.dataset_root, "test")
    test_ds = NYUv2Segmentation(config.dataset_root, test_ids, "test", test_tf, use_depth=config.use_depth)
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,  # エポック間の待機時間を削減
    )
    return train_loader, val_loader, test_loader


def forward_loss(model, batch, device, ce_loss, config: Config, epoch: int):
    inputs, targets = batch
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    logits = model(inputs)
    ce = ce_loss(logits, targets)
    dice = dice_loss(logits, targets, config.ignore_index)
    loss = config.loss_ce_w * ce + config.loss_dice_w * dice
    if epoch >= (config.epochs - config.lovasz_start):
        lovasz = lovasz_softmax(F.softmax(logits, dim=1), targets, ignore_index=config.ignore_index)
        loss = loss + config.loss_lovasz_w * lovasz
    return loss, logits, targets


def evaluate(model, loader, device, ce_loss, config: Config, epoch: int):
    model.eval()
    metric = IoUMetric(config.num_classes, config.ignore_index)
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            loss, logits, targets = forward_loss(model, (batch[0], batch[1]), device, ce_loss, config, epoch)
            preds = logits.argmax(dim=1)
            metric.update(preds, targets)
            bsz = targets.size(0)
            total_loss += loss.item() * bsz
            total_samples += bsz
    miou, class_iou = metric.compute()
    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, miou, class_iou


def find_existing_checkpoint(config: Config, fold_idx: int) -> Tuple[Path | None, int, float]:
    """既存のチェックポイントを検索して、最新のエポックとbest_miouを返す"""
    checkpoint_dir = config.output_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None, 0, -1.0
    
    # このフォールドの最良チェックポイントを検索
    pattern = f"best_fold{fold_idx+1}_epoch*_miou*.pt"
    checkpoints = list(checkpoint_dir.glob(pattern))
    if not checkpoints:
        return None, 0, -1.0
    
    # ファイル名からエポックとmIoUを抽出して、最新のものを取得
    best_ckpt = None
    best_epoch = 0
    best_miou = -1.0
    
    for ckpt in checkpoints:
        # ファイル名例: best_fold1_epoch010_miou0.5563.pt
        parts = ckpt.stem.split("_")
        try:
            epoch_part = [p for p in parts if p.startswith("epoch")][0]
            miou_part = [p for p in parts if p.startswith("miou")][0]
            epoch = int(epoch_part.replace("epoch", ""))
            miou = float(miou_part.replace("miou", ""))
            if epoch > best_epoch:
                best_epoch = epoch
                best_miou = miou
                best_ckpt = ckpt
        except (ValueError, IndexError):
            continue
    
    return best_ckpt, best_epoch, best_miou


def train(config: Config, fold_idx: int, train_ids: List[str], val_ids: List[str], resume_from_checkpoint: bool = True):
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
    global_step = 0
    start_epoch = 1

    # チェックポイントから再開（無効化：チェックポイント保存をしないため）
    # 常に最初から学習を開始

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        samples = 0
        start = time.time()
        for step, (inputs, targets, _) in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                loss, logits, targets = forward_loss(model, (inputs, targets), device, ce_loss, config, epoch)
            scaler.scale(loss).backward()
            if config.grad_clip is not None and config.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            scheduler.step(epoch - 1 + step / len(train_loader))

            bsz = targets.size(0)
            epoch_loss += loss.item() * bsz
            samples += bsz
        avg_train_loss = epoch_loss / max(samples, 1)

        val_loss, val_miou, class_iou = evaluate(model, val_loader, device, ce_loss, config, epoch)
        elapsed = time.time() - start

        print(
            f"[Fold {fold_idx+1} | Epoch {epoch:03d}] train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_mIoU={val_miou:.4f} time={elapsed:.1f}s"
        )

        if val_miou > best_miou:
            best_miou = val_miou
            # チェックポイント保存を無効化（メモリ内でモデルを保持）
            best_model_state = model.state_dict().copy()
            print(f"  New best mIoU: {best_miou:.4f} (not saving checkpoint)")

    # 最良のモデル状態をロード（メモリ内）
    if best_model_state is not None and best_miou > 0:
        model.load_state_dict(best_model_state)
        print(f"  Training completed. Best mIoU: {best_miou:.4f}")
    elif best_miou > 0:
        # best_model_stateがNoneでもbest_miouが設定されている場合は現在のモデルを使用
        print(f"  Training completed. Best mIoU: {best_miou:.4f} (using current model state)")
    else:
        raise RuntimeError(f"Fold {fold_idx + 1}: Training failed - no valid mIoU achieved.")
    
    # モデルオブジェクトとbest_miouを返す（チェックポイントパスではなく）
    return model, best_miou


def run_inference(models: List[nn.Module], loader: DataLoader, config: Config) -> Path:
    device = torch.device(config.device)
    # モデルをevalモードに設定
    for model in models:
        model.eval()
    if len(models) == 0:
        raise ValueError("No valid models provided for inference.")

    print(f"Ensembling {len(models)} models with TTA...")

    preds = []
    with torch.no_grad():
        for inputs, name in tqdm(loader, desc="Inference"):
            inputs = inputs.to(device, non_blocking=True)
            inputs_flip = torch.flip(inputs, dims=[3])

            avg_logits = None
            for model in models:
                logits = model(inputs)
                logits_flip = torch.flip(model(inputs_flip), dims=[3])
                batch_logits = (logits + logits_flip) / 2.0
                if avg_logits is None:
                    avg_logits = batch_logits
                else:
                    avg_logits += batch_logits

            avg_logits = avg_logits / len(models)
            pred = avg_logits.argmax(dim=1).cpu().numpy()
            preds.append(pred)

    predictions = np.concatenate(preds, axis=0)
    save_path = config.output_dir / config.submission_filename
    np.save(save_path, predictions)
    print(f"Saved submission to {save_path} with shape {predictions.shape}")
    return save_path


def visualize_sample(model, dataset: Dataset, config: Config, idx: int = 0, save_path: Path | None = None):
    device = torch.device(config.device)
    model.eval()
    sample = dataset[idx]
    if len(sample) == 3:
        inputs, mask, name = sample
    else:
        inputs, name = sample
        mask = None

    with torch.no_grad():
        pred = model(inputs.unsqueeze(0).to(device)).argmax(dim=1).squeeze(0).cpu()

    rgb = inputs[:3].cpu() * RGB_STD.view(3, 1, 1) + RGB_MEAN.view(3, 1, 1)
    rgb = torch.clamp(rgb, 0, 1).permute(1, 2, 0).numpy()
    depth = inputs[3].cpu().numpy() if inputs.shape[0] > 3 else None

    fig, axs = plt.subplots(1, 4 if depth is not None else 3, figsize=(16, 5))
    axs[0].imshow(rgb)
    axs[0].set_title(f"Image {name}")
    axs[0].axis("off")
    if depth is not None:
        axs[1].imshow(depth, cmap="magma")
        axs[1].set_title("Depth (norm)")
        axs[1].axis("off")
    axs[-2].imshow(mask.cpu().numpy(), cmap="tab20", vmin=0, vmax=config.num_classes) if mask is not None else axs[-2].imshow(
        np.zeros_like(pred.cpu().numpy()), cmap="gray"
    )
    axs[-2].set_title("GT" if mask is not None else "GT (missing)")
    axs[-2].axis("off")
    axs[-1].imshow(pred.cpu().numpy(), cmap="tab20", vmin=0, vmax=config.num_classes)
    axs[-1].set_title("Prediction")
    axs[-1].axis("off")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Visualization saved to {save_path}")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="NYUv2 semantic segmentation - DeepLabV3+ strong baseline")
    parser.add_argument("--dataset-root", type=Path, default=Path("data"), help="Path to data directory containing train/ and test/")
    parser.add_argument("--output-dir", type=Path, default=Path("data/output"), help="Directory to save checkpoints and submission.npy")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--no-depth", action="store_true", help="Disable depth channel (RGB only)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--submission-filename", type=str, default="submission.npy")
    parser.add_argument("--visualize", action="store_true", help="Save a visualization on the validation set after training")
    parser.add_argument("--lovasz-start", type=int, default=10, help="Start Lovasz-Softmax after last N epochs")
    parser.add_argument(
        "--encoder-name",
        type=str,
        default="timm-efficientnetv2_l",
        help="Backbone encoder for DeepLabV3+ (e.g., timm-efficientnetv2_l, mit_b5)",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Number of folds for K-Fold cross validation")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume training from existing checkpoints (default: True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start training from scratch, ignoring existing checkpoints")
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
        val_ratio=args.val_ratio,
        use_depth=not args.no_depth,
        device=args.device,
        seed=args.seed,
        submission_filename=args.submission_filename,
        lovasz_start=args.lovasz_start,
        encoder_name=args.encoder_name,
        n_splits=args.n_splits,
    )
    print(f"Using device: {config.device}")

    all_ids = np.array(list_ids(config.dataset_root, "train"))
    kf = KFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    trained_models: List[nn.Module] = []

    last_val_ids: List[str] = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_ids)):
        print(f"\n=== Training Fold {fold_idx + 1}/{config.n_splits} ===")
        t_ids = all_ids[train_idx].tolist()
        v_ids = all_ids[val_idx].tolist()
        last_val_ids = v_ids
        
        # チェックポイント保存を無効化しているため、常に最初から学習
        model, fold_miou = train(config, fold_idx, t_ids, v_ids, resume_from_checkpoint=False)
        if model is None:
            raise RuntimeError(f"Fold {fold_idx + 1} failed: Training did not produce a model.")
        trained_models.append(model)
        print(f"Fold {fold_idx + 1} Best mIoU: {fold_miou:.4f}")

    # 推論前にモデルの存在を確認
    if len(trained_models) == 0:
        raise RuntimeError("No trained models found for inference. Training may have failed.")
    if len(trained_models) < config.n_splits:
        print(f"Warning: Only {len(trained_models)}/{config.n_splits} folds have trained models.")
        print(f"Proceeding with available models.")

    # Build test loader once
    _, _, test_loader = prepare_dataloaders(config, [], [])
    submission_path = run_inference(trained_models, test_loader, config)

    if args.visualize and len(last_val_ids) > 0:
        viz_model = trained_models[-1]
        _, val_loader, _ = prepare_dataloaders(config, [], last_val_ids)
        if val_loader is not None:
            visualize_sample(viz_model, val_loader.dataset, config, idx=0, save_path=config.output_dir / "qualitative.png")

    print(f"Done. Submission saved to {submission_path}")


if __name__ == "__main__":
    main()
