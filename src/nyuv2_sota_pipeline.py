# -*- coding: utf-8 -*-
"""
High-performance NYUv2 semantic segmentation pipeline.

Key choices:
- Architecture: DeepLabV3+ (encoder: ResNet101, ImageNet init) from segmentation_models_pytorch for strong contextual modeling and robust decoders. Supports 4-channel input (RGB + depth).
- Loss: Combo of CrossEntropy (ignore_index=255) and soft Dice to handle class imbalance and stabilize IoU.
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
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass
class Config:
    dataset_root: Path
    output_dir: Path
    epochs: int = 60
    batch_size: int = 6
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    val_ratio: float = 0.1
    img_size: Tuple[int, int] = (512, 512)  # (height, width) per README
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

    def __post_init__(self):
        self.dataset_root = Path(self.dataset_root).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)


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
                A.RandomResizedCrop(height=h, width=w, scale=(0.7, 1.0), ratio=(0.9, 1.1), interpolation=cv2.INTER_LINEAR),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=10,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.7,
                ),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussianBlur(p=0.2),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=48,
                    max_width=48,
                    fill_value=0,
                    mask_fill_value=255,
                    p=0.35,
                ),
                A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR),
                ToTensorV2(transpose_mask=True),
            ],
            additional_targets={"depth": "image"},
        )
    return A.Compose(
        [
            A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR),
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
    ):
        self.root = root
        self.ids = list(ids)
        self.split = split
        self.transforms = transforms
        self.use_depth = use_depth

    def __len__(self) -> int:
        return len(self.ids)

    def _load_depth(self, path: Path) -> np.ndarray:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Missing depth: {path}")
        depth = depth.astype(np.float32)
        depth = np.clip(depth, 0, 10000) / 10000.0  # scale to 0-1
        return depth[..., None]

    def __getitem__(self, idx: int):
        name = self.ids[idx]
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

        data = self.transforms(image=image, depth=depth, mask=mask) if mask is not None else self.transforms(
            image=image, depth=depth
        )
        image_t: torch.Tensor = data["image"].float()
        depth_t: torch.Tensor = data["depth"].float()

        if image_t.max() > 1.5:
            image_t = image_t / 255.0
        if depth_t.max() > 1.5:
            depth_t = depth_t / 255.0

        # Normalize RGB only; depth stays in [0,1]
        rgb = (image_t[:3] - IMAGENET_MEAN) / IMAGENET_STD
        if self.use_depth:
            x = torch.cat([rgb, depth_t[:1]], dim=0)
        else:
            x = rgb

        if mask is None:
            return x, name

        mask_t: torch.Tensor = data["mask"].long()
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


def build_model(config: Config) -> nn.Module:
    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=4 if config.use_depth else 3,
        classes=config.num_classes,
    )
    return model


def prepare_dataloaders(config: Config):
    all_ids = list_ids(config.dataset_root, "train")
    train_ids, val_ids = train_val_split(all_ids, config.val_ratio, config.seed)

    train_tf = build_transforms(config.img_size, "train")
    val_tf = build_transforms(config.img_size, "val")
    test_tf = build_transforms(config.img_size, "test")

    train_ds = NYUv2Segmentation(config.dataset_root, train_ids, "train", train_tf, use_depth=config.use_depth)
    val_ds = NYUv2Segmentation(config.dataset_root, val_ids, "train", val_tf, use_depth=config.use_depth)
    test_ids = list_ids(config.dataset_root, "test")
    test_ds = NYUv2Segmentation(config.dataset_root, test_ids, "test", test_tf, use_depth=config.use_depth)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, config.batch_size // 2),
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def forward_loss(model, batch, device, ce_loss, config: Config):
    inputs, targets = batch
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    logits = model(inputs)
    loss = 0.6 * ce_loss(logits, targets) + 0.4 * dice_loss(logits, targets, config.ignore_index)
    return loss, logits, targets


def evaluate(model, loader, device, ce_loss, config: Config):
    model.eval()
    metric = IoUMetric(config.num_classes, config.ignore_index)
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            loss, logits, targets = forward_loss(model, (batch[0], batch[1]), device, ce_loss, config)
            preds = logits.argmax(dim=1)
            metric.update(preds, targets)
            bsz = targets.size(0)
            total_loss += loss.item() * bsz
            total_samples += bsz
    miou, class_iou = metric.compute()
    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, miou, class_iou


def train(config: Config):
    set_seed(config.seed)
    device = torch.device(config.device)
    train_loader, val_loader, test_loader = prepare_dataloaders(config)
    model = build_model(config).to(device)

    ce_loss = nn.CrossEntropyLoss(ignore_index=config.ignore_index)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = GradScaler(enabled=config.amp and device.type == "cuda")
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.t0, T_mult=config.t_mult)

    best_miou = -1.0
    best_path = None
    history = []
    global_step = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        samples = 0
        start = time.time()
        for step, (inputs, targets, _) in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                loss, logits, targets = forward_loss(model, (inputs, targets), device, ce_loss, config)
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

        val_loss, val_miou, class_iou = evaluate(model, val_loader, device, ce_loss, config)
        elapsed = time.time() - start
        history.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": val_loss, "val_miou": val_miou})

        print(
            f"[Epoch {epoch:03d}] train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_mIoU={val_miou:.4f} time={elapsed:.1f}s"
        )

        if val_miou > best_miou:
            best_miou = val_miou
            best_path = config.output_dir / "checkpoints" / f"best_epoch{epoch:03d}_miou{val_miou:.4f}.pt"
            torch.save({"model": model.state_dict(), "config": config}, best_path)
            print(f"  Saved new best checkpoint to {best_path}")
        if epoch % config.save_every == 0:
            last_path = config.output_dir / "checkpoints" / f"last_epoch{epoch:03d}.pt"
            torch.save({"model": model.state_dict(), "config": config}, last_path)

    return model, best_path, best_miou, test_loader


def run_inference(model, checkpoint: Path | None, loader: DataLoader, config: Config) -> Path:
    device = torch.device(config.device)
    if checkpoint is not None and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model"])
    model.eval()

    preds = []
    names = []
    with torch.no_grad():
        for inputs, name in loader:
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.append(pred)
            names.extend(name)
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

    rgb = inputs[:3].cpu() * IMAGENET_STD + IMAGENET_MEAN
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
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--no-depth", action="store_true", help="Disable depth channel (RGB only)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--submission-filename", type=str, default="submission.npy")
    parser.add_argument("--visualize", action="store_true", help="Save a visualization on the validation set after training")
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
    )
    print(f"Using device: {config.device}")
    model, best_ckpt, best_miou, test_loader = train(config)
    print(f"Best validation mIoU: {best_miou:.4f}")

    submission_path = run_inference(model, best_ckpt, test_loader, config)
    if args.visualize:
        _, val_loader, _ = prepare_dataloaders(config)
        visualize_sample(model, val_loader.dataset, config, idx=0, save_path=config.output_dir / "qualitative.png")
    print(f"Done. Submission saved to {submission_path}")


if __name__ == "__main__":
    main()
