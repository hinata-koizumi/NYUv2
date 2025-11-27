# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import time
from tqdm import tqdm
import numpy as np
import cv2
import albumentations as A
from PIL import Image
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import VisionDataset
from torch.cuda.amp import autocast, GradScaler
from albumentations.pytorch import ToTensorV2
from dataclasses import dataclass
from typing import List, Tuple
import random
import hashlib

"""# DataLoader"""

DEPTH_MIN = 0.71
DEPTH_MAX = 10.0
GEO_CHANNELS = 4  # depth_log + normalized depth + dx + dy


# カラーマップ生成関数：セグメンテーションの可視化用
def colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

# NYUv2データセット：RGB画像、セグメンテーション、深度、法線マップを提供するデータセット
class NYUv2(VisionDataset):
    """NYUv2 dataset

    Args:
        root (string): Root directory path.
        split (string, optional): 'train' for training set, and 'test' for test set. Default: 'train'.
        target_type (string, optional): Type of target to use, ``semantic``, ``depth``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """
    def __init__(self,
                 root,
                 split='train',
                 include_depth=False,
                 transform=None,
                 target_transform=None,
                 ):
        super(NYUv2, self).__init__(root, transform=transform, target_transform=target_transform)

        # データセットの基本設定
        assert(split in ('train', 'test'))
        self.root = str(root)
        self.split = split
        self.include_depth = include_depth

        # 画像ファイルのパスリストを作成
        img_names = os.listdir(os.path.join(self.root, self.split, 'image'))
        img_names.sort()
        images_dir = os.path.join(self.root, self.split, 'image')
        self.images = [os.path.join(images_dir, name) for name in img_names]

        label_dir = os.path.join(self.root, self.split, 'label')
        self.labels = []
        if self.split == 'train':
            self.labels = [os.path.join(label_dir, name) for name in img_names]
            self.targets = self.labels

        depth_dir = os.path.join(self.root, self.split, 'depth')
        self.depths = [os.path.join(depth_dir, name) for name in img_names]
        self.rare_class_ids = (1, 7, 10)
        self.contains_rare = self._compute_contains_rare() if self.split == 'train' else [False] * len(self.images)

    def _load_depth(self, path: str) -> np.ndarray:
        depth = np.array(Image.open(path), dtype=np.float32)
        depth_m = depth / 1000.0  # assume millimeters -> meters
        depth_m = np.clip(depth_m, DEPTH_MIN, DEPTH_MAX)
        depth_log = np.log1p(depth_m) / np.log1p(DEPTH_MAX)
        return depth_log

    def _build_geo_channels(self, depth_tensor: torch.Tensor) -> torch.Tensor:
        """
        depth_tensor: [1, H, W] torch tensor (0-1 range, log depth)
        returns: [4, H, W] torch tensor with depth, normalized depth, dx, dy
        """
        depth_np = depth_tensor.squeeze(0).cpu().numpy()
        d_norm = (depth_np - depth_np.mean()) / (depth_np.std() + 1e-6)
        d_norm = np.clip(d_norm, -5.0, 5.0)

        dx = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)

        dx = dx / (np.max(np.abs(dx)) + 1e-6)
        dy = dy / (np.max(np.abs(dy)) + 1e-6)
        dx = np.clip(dx, -1.0, 1.0)
        dy = np.clip(dy, -1.0, 1.0)

        geo = np.stack([depth_np, d_norm, dx, dy], axis=0)
        return torch.from_numpy(geo).float()

    def _compute_contains_rare(self) -> List[bool]:
        flags: List[bool] = []
        if not self.labels:
            return [False] * len(self.images)
        print("Analyzing rare class presence for sampler...")
        for label_path in tqdm(self.labels, desc="Rare class scan"):
            try:
                mask = np.array(Image.open(label_path))
            except FileNotFoundError:
                flags.append(False)
                continue
            unique_vals = np.unique(mask)
            has_rare = any(int(cls) in self.rare_class_ids for cls in unique_vals)
            flags.append(has_rare)
        return flags

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        image_np = np.array(image)
        depth_np = None
        depth_tensor = None
        geo = None
        mask = None
        target = None

        if self.include_depth:
            depth_np = self._load_depth(self.depths[idx])[..., None]

        if self.split == 'train':
            mask = np.array(Image.open(self.targets[idx]))

        if self.transform is not None:
            if self.include_depth and mask is not None:
                transformed = self.transform(image=image_np, depth=depth_np, mask=mask)
            elif self.include_depth:
                transformed = self.transform(image=image_np, depth=depth_np)
            elif mask is not None:
                transformed = self.transform(image=image_np, mask=mask)
            else:
                transformed = self.transform(image=image_np)

            image_tensor = transformed["image"]
            if self.include_depth and "depth" in transformed:
                depth_tensor = transformed["depth"]
                geo = self._build_geo_channels(depth_tensor)
            if mask is not None and "mask" in transformed:
                target = transformed["mask"].long()
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            if self.include_depth and depth_np is not None:
                depth_tensor = torch.from_numpy(depth_np).permute(2, 0, 1).float()
                geo = self._build_geo_channels(depth_tensor)
            if mask is not None:
                target = torch.from_numpy(mask).long()

        if self.include_depth and geo is None:
            raise RuntimeError("Failed to compute depth geometry features.")

        if self.split == 'test':
            if self.include_depth:
                return image_tensor, geo
            return image_tensor
        if self.include_depth:
            return image_tensor, geo, target

        return image_tensor, target

    def __len__(self):
        return len(self.images)

"""# Model Section

"""

# 2つの畳み込み層とバッチ正規化、ReLUを含むブロック
# UNetの各層で使用される基本的な畳み込みブロック
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

# UNetモデル：エンコーダ・デコーダ構造のセグメンテーションモデル
class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # エンコーダ部分：特徴量の抽出と空間サイズの縮小
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)

        # デコーダ部分：特徴量の統合と空間サイズの復元
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = DoubleConv(512 + 256, 256)
        self.dec2 = DoubleConv(256 + 128, 128)
        self.dec1 = DoubleConv(128 + 64, 64)

        # 最終層：クラス数に応じた出力チャネルに変換
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # エンコーダパス：特徴抽出とダウンサンプリング
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # デコーダパス：特徴統合とアップサンプリング（スキップ接続を使用）
        d3 = self.dec3(torch.cat([self.up(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))

        return self.final(d1)

"""# Train and Valid"""

# config
@dataclass
class TrainingConfig:
    dataset_root: Path
    output_dir: Path

    # データ関連
    batch_size: int = 32
    num_workers: int = 4

    # モデル関連
    include_depth: bool = True
    in_channels: int = 7
    num_classes: int = 13  # NYUv2データセットの場合

    # 学習関連
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    max_train_steps: int | None = None
    max_test_steps: int | None = None

    # デバイス設定
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # チェックポイント関連
    checkpoint_dir: Path | None = None
    save_interval: int = 5  # エポックごとのモデル保存間隔

    # データ拡張・前処理関連
    image_size: Tuple[int, int] = (256, 256)
    submission_filename: str = "submission.npy"
    use_rare_sampler: bool = False

    def __post_init__(self):
        self.dataset_root = Path(self.dataset_root).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.in_channels = 3 + (GEO_CHANNELS if self.include_depth else 0)

def set_seed(seed: int) -> None:
    """
    シードを固定する．
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA が使用できないため、CPU で実行します。")
        return "cpu"
    return device_arg


def parse_args():
    default_root = (Path(__file__).resolve().parent.parent / "data")
    default_output = default_root / "output"
    parser = argparse.ArgumentParser(description="NYUv2 セマンティックセグメンテーション Baseline")
    parser.add_argument("--dataset-root", type=Path, default=default_root,
                        help="NYUv2 データセットのルートパス")
    parser.add_argument("--output-dir", type=Path, default=default_output,
                        help="モデルや submission.npy を保存するディレクトリ")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--no-depth", action="store_true",
                        help="深度マップを入力に含めない")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--save-interval", type=int, default=5,
                        help="何エポックごとにチェックポイントを保存するか")
    parser.add_argument("--submission-filename", type=str, default="submission.npy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-steps", type=int, default=None,
                        help="デバッグ用: 1エポックあたりの更新回数を制限する")
    parser.add_argument("--max-test-steps", type=int, default=None,
                        help="デバッグ用: 推論時のバッチ数を制限する")
    parser.add_argument(
        "--use-rare-sampler",
        action="store_true",
        help="Use weighted sampler that oversamples images containing rare classes."
    )
    return parser.parse_args()


def verify_dataset_structure(dataset_root: Path, include_depth: bool) -> None:
    required_dirs = [
        dataset_root / "train" / "image",
        dataset_root / "train" / "label",
        dataset_root / "test" / "image",
    ]
    if include_depth:
        required_dirs.extend([
            dataset_root / "train" / "depth",
            dataset_root / "test" / "depth",
        ])
    missing = [path for path in required_dirs if not path.exists()]
    if missing:
        missing_str = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(f"以下のディレクトリが見つかりませんでした:\n{missing_str}")


def build_transforms(size: Tuple[int, int], split: str) -> A.Compose:
    h, w = size

    if split == "train":
        return A.Compose(
            [
                # Multi-scale crop that handles scale + crop + resize in one step
                A.RandomResizedCrop(
                    height=h,
                    width=w,
                    scale=(0.6, 1.4),
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(0.02, 0.02),
                    rotate=(-8, 8),
                    shear=(-4, 4),
                    cval=0,
                    mode=cv2.BORDER_REFLECT_101,
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
                A.CoarseDropout(
                    holes=8,
                    height=48,
                    width=48,
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


def compute_class_weights_inverse_frequency(config: TrainingConfig) -> torch.Tensor:
    """逆頻度ベースのクラス重みを計算"""
    weight_cache = config.output_dir / "class_weights_inverse_freq.npy"
    if weight_cache.exists():
        weights = np.load(weight_cache)
        return torch.tensor(weights, dtype=torch.float32)

    counts = np.zeros(config.num_classes, dtype=np.float64)
    label_dir = config.dataset_root / "train" / "label"
    for path in label_dir.glob("*.png"):
        mask = np.array(Image.open(path))
        mask = mask.astype(np.int64)
        valid = mask != 255  # ignore_index
        hist = np.bincount(mask[valid].ravel(), minlength=config.num_classes)
        counts += hist[:config.num_classes]

    min_count = counts[counts > 0].min() if (counts > 0).any() else 1.0
    weights = min_count / (counts + 1e-8)
    weights = weights / weights.mean()
    
    rare_classes = [1, 7, 10]
    for rare_cls in rare_classes:
        if rare_cls < len(weights):
            weights[rare_cls] *= 2.0
    
    weights = weights / weights.mean()
    
    np.save(weight_cache, weights)
    print(f"Class weights (inverse frequency): {weights}")
    return torch.tensor(weights, dtype=torch.float32)


def compute_sample_weights(dataset: NYUv2, config: TrainingConfig, rare_classes: list = [1, 7, 10]) -> np.ndarray:
    """Rareクラスを含む画像に高い重みを割り当て"""
    img_names_str = "|".join(sorted([os.path.basename(p) for p in dataset.images]))
    cache_hash = hashlib.md5(img_names_str.encode()).hexdigest()[:8]
    cache_path = config.output_dir / f"sample_weights_cache_{cache_hash}.npy"
    
    if cache_path.exists():
        sample_weights = np.load(cache_path)
        print(f"Loaded sample weights from cache: {cache_path.name}")
        print(f"Sample weights: min={sample_weights.min():.2f}, max={sample_weights.max():.2f}, mean={sample_weights.mean():.2f}")
        return sample_weights
    
    sample_weights = np.ones(len(dataset), dtype=np.float32)
    rare_classes_array = np.array([cls for cls in rare_classes if cls < config.num_classes], dtype=np.int64)
    rare_max = rare_classes_array.max() if len(rare_classes_array) > 0 else 0
    
    print("Computing sample weights based on rare class presence...")
    for idx in tqdm(range(len(dataset)), desc="Analyzing samples"):
        label_path = dataset.labels[idx]
        mask = np.array(Image.open(label_path))
        mask_flat = mask.ravel()
        max_val = max(mask_flat.max(), rare_max) + 1
        counts = np.bincount(mask_flat, minlength=int(max_val))
        rare_count = np.sum(counts[rare_classes_array] > 0)
        
        if rare_count > 0:
            sample_weights[idx] = 1.0 + rare_count * 2.0
        else:
            sample_weights[idx] = 1.0
    
    sample_weights = sample_weights / sample_weights.mean() * len(dataset)
    np.save(cache_path, sample_weights)
    print(f"Saved sample weights to cache: {cache_path.name}")
    print(f"Sample weights: min={sample_weights.min():.2f}, max={sample_weights.max():.2f}, mean={sample_weights.mean():.2f}")
    return sample_weights


def build_dataloaders(config: TrainingConfig):
    verify_dataset_structure(config.dataset_root, config.include_depth)
    train_transform = build_transforms(config.image_size, "train")
    test_transform = build_transforms(config.image_size, "test")

    train_dataset = NYUv2(
        root=config.dataset_root,
        split='train',
        include_depth=config.include_depth,
        transform=train_transform,
    )

    test_dataset = NYUv2(
        root=config.dataset_root,
        split='test',
        include_depth=config.include_depth,
        transform=test_transform
    )

    pin_memory = config.device == "cuda"
    sampler = None
    if config.use_rare_sampler:
        weights = [3.0 if flag else 1.0 for flag in train_dataset.contains_rare]
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        print("Using rare-class sampler with weight=3.0.")
    else:
        sample_weights = compute_sample_weights(train_dataset, config)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory
    )
    return train_loader, test_loader


def create_model(config: TrainingConfig):
    device = torch.device(config.device)
    model = UNet(in_channels=config.in_channels, num_classes=config.num_classes).to(device)
    return model


def train_model(config: TrainingConfig,
                model: nn.Module,
                train_loader: DataLoader,
                criterion,
                optimizer) -> Path:
    device = torch.device(config.device)
    scaler = GradScaler(enabled=device.type == "cuda")
    latest_checkpoint = None
    model.train()
    for epoch in range(config.epochs):
        total_loss = 0.0
        num_samples = 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}") as pbar:
            for step, batch in enumerate(pbar, start=1):
                if config.include_depth:
                    image, geo, label = batch
                    image = image.to(device, non_blocking=True)
                    geo = geo.to(device, non_blocking=True)
                    inputs = torch.cat((image, geo), dim=1)
                else:
                    image, label = batch
                    inputs = image.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=scaler.is_enabled()):
                    pred = model(inputs)
                    loss = criterion(pred, label)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                batch_size = label.size(0)
                total_loss += loss.item() * batch_size
                num_samples += batch_size
                pbar.set_postfix(loss=total_loss / max(num_samples, 1))

                if config.max_train_steps is not None and step >= config.max_train_steps:
                    print(f"Stopping early after {step} steps (debug mode).")
                    break

        if ((epoch + 1) % config.save_interval == 0) or (epoch + 1 == config.epochs):
            current_time = time.strftime("%Y%m%d%H%M%S")
            ckpt_name = f"model_epoch_{epoch+1}_{current_time}.pt"
            latest_checkpoint = config.checkpoint_dir / ckpt_name
            torch.save(model.state_dict(), latest_checkpoint)
            print(f"Saved checkpoint to {latest_checkpoint}")

    if latest_checkpoint is None:
        raise RuntimeError("チェックポイントが保存されませんでした。save_interval の設定を確認してください。")
    return latest_checkpoint


def generate_predictions(config: TrainingConfig,
                         model: nn.Module,
                         test_loader: DataLoader,
                         submission_path: Path) -> Path:
    device = torch.device(config.device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_loader, desc="Generating predictions"), start=1):
            if config.include_depth:
                image, geo = batch
                image = image.to(device, non_blocking=True)
                geo = geo.to(device, non_blocking=True)
                inputs = torch.cat((image, geo), dim=1)
            else:
                image = batch[0] if isinstance(batch, (tuple, list)) else batch
                inputs = image.to(device, non_blocking=True)

            output = model(inputs)
            pred = output.argmax(dim=1)
            predictions.append(pred.cpu())

            if config.max_test_steps is not None and step >= config.max_test_steps:
                print(f"Stopping prediction early after {step} steps (debug mode).")
                break

    predictions = torch.cat(predictions, dim=0).numpy()
    np.save(submission_path, predictions)
    print(f"Predictions saved to {submission_path}")
    return submission_path


def main():
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)
    config = TrainingConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        include_depth=not args.no_depth,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        image_size=(args.image_height, args.image_width),
        save_interval=args.save_interval,
        submission_filename=args.submission_filename,
        max_train_steps=args.max_train_steps,
        max_test_steps=args.max_test_steps,
        use_rare_sampler=args.use_rare_sampler,
    )

    print(f"Using device: {config.device}")
    train_loader, test_loader = build_dataloaders(config)
    model = create_model(config)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    class_weights = compute_class_weights_inverse_frequency(config)
    device_obj = torch.device(config.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device_obj), ignore_index=255)

    checkpoint_path = train_model(config, model, train_loader, criterion, optimizer)
    state_dict = torch.load(checkpoint_path, map_location=device_obj)
    model.load_state_dict(state_dict)

    submission_path = config.output_dir / config.submission_filename
    generate_predictions(config, model, test_loader, submission_path)
    print(f"Training and inference complete. Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
