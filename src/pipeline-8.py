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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import KFold
from tqdm import tqdm
import hashlib

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
    use_depth: bool = True  # B2 experiment requires depth (fixed)
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    grad_clip: float = 1.0
    t0: int = 5
    t_mult: int = 2
    submission_filename: str = "submission.npy"
    lovasz_start: int = 10
    loss_ce_w: float = 0.5
    loss_dice_w: float = 0.3
    loss_lovasz_w: float = 0.2
    loss_depth_w: float = 0.3  # Weight for Depth Reconstruction Loss (Seg main, Depth auxiliary)
    loss_edge_w: float = 0.3  # Weight for Edge Detection Loss (μ ~ 0.2-0.5)
    encoder_name: str = "timm-efficientnet-b7"
    n_splits: int = 5
    use_rare_sampler: bool = False
    # Depth gating parameters
    depth_gate_alpha: float = 0.5  # Weight for depth gate: feat = feat * (1 + alpha * gate)
    depth_gate_channels: int = 1  # Number of channels for depth gate (1 or few)
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
        # Modified: Fixed arguments for newer albumentations versions to avoid UserWarnings
        return A.Compose(
            [
                A.Resize(height=h, width=w),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                # Affine parameters simplified to avoid version conflicts
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(0.02, 0.02),
                    rotate=(-8, 8),
                    shear=(-4, 4),
                    # cval/mode removed to rely on defaults (usually 0 padding) which is safe
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
                # CoarseDropout updated to use max_holes/max_height/max_width
                A.CoarseDropout(
                    max_holes=8,
                    max_height=48,
                    max_width=48,
                    min_holes=8,      # Ensure consistent size
                    min_height=48,
                    min_width=48,
                    fill_value=0,
                    mask_fill_value=255,
                    p=0.35,
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
        use_depth: bool = True,
        cache_data: bool = True,
    ):
        self.root = root
        self.ids = list(ids)
        self.split = split
        self.transforms = transforms
        self.use_depth = use_depth
        self.cache_data = cache_data
        self.rare_class_ids = (1, 7, 10)
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

        self.contains_rare = self._compute_contains_rare()

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

        rgb = (image_t[:3] - RGB_MEAN.view(3, 1, 1)) / RGB_STD.view(3, 1, 1)
        if self.use_depth:
            # Return RGB and depth separately for depth gating (no geo needed for B2)
            if mask is None:
                return (rgb, depth_t[:1]), name
            return (rgb, depth_t[:1]), mask_t, name
        else:
            x = rgb
            if mask is None:
                return x, name
            return x, mask_t, name

    def _compute_contains_rare(self) -> List[bool]:
        if self.split == "test":
            return [False] * len(self.ids)
        flags: List[bool] = []
        for idx, name in enumerate(self.ids):
            mask = None
            if self.cache_data and idx < len(self.masks):
                mask = self.masks[idx]
            if mask is None:
                mask_path = self.root / self.split / "label" / name
                mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                flags.append(False)
                continue
            unique_vals = np.unique(mask)
            has_rare = any(int(cls) in self.rare_class_ids for cls in unique_vals)
            flags.append(has_rare)
        return flags


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
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, probas.size(1))
    labels = labels.view(-1)
    valid = labels != ignore_index
    probas = probas[valid]
    labels = labels[valid]
    return lovasz_softmax_flat(probas, labels, classes=classes)


def generate_edge_mask(labels: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """
    Generate Canny-like edge masks from segmentation labels.
    Uses morphological operations to detect semantic boundaries.
    
    Args:
        labels: [B, H, W] segmentation labels
        ignore_index: index to ignore
    
    Returns:
        edge_mask: [B, 1, H, W] binary edge mask (0 or 1)
    """
    B, H, W = labels.shape
    device = labels.device
    
    # Create edge masks for each sample
    edge_masks = []
    
    for b in range(B):
        label = labels[b]  # [H, W]
        
        # Create valid mask (ignore invalid regions)
        valid_mask = (label != ignore_index).float()
        
        # Compute gradients using Sobel-like operators
        # Pad label to handle boundaries
        label_padded = F.pad(label.unsqueeze(0).unsqueeze(0).float(), (1, 1, 1, 1), mode='replicate')
        
        # Horizontal gradient (detect vertical edges)
        dx = label_padded[:, :, 1:-1, 2:] - label_padded[:, :, 1:-1, :-2]  # [1, 1, H, W]
        # Vertical gradient (detect horizontal edges)
        dy = label_padded[:, :, 2:, 1:-1] - label_padded[:, :, :-2, 1:-1]  # [1, 1, H, W]
        
        # Edge exists where gradient is non-zero
        edge = ((dx.abs() > 0) | (dy.abs() > 0)).float().squeeze(0)  # [1, H, W]
        
        # Apply valid mask (set edges in invalid regions to 0)
        edge = edge * valid_mask.unsqueeze(0)
        
        edge_masks.append(edge)
    
    # Stack all edge masks
    edge_mask = torch.stack(edge_masks, dim=0)  # [B, 1, H, W]
    
    return edge_mask


# === Depth Gating Module ===
class DepthGateModule(nn.Module):
    """
    Generate attention gate from depth features.
    Features: log-depth, depth gradient magnitude (edge-like regions)
    """
    def __init__(self, gate_channels: int = 1):
        super().__init__()
        self.gate_channels = gate_channels
        
        # Register Sobel kernels as buffers for efficiency
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        
    def compute_depth_gradient_magnitude(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Compute absolute value of depth gradient (edge-like regions).
        depth: [B, 1, H, W]
        returns: [B, 1, H, W] gradient magnitude
        """
        # Use registered buffers (automatically moved to correct device/dtype)
        sobel_x = self.sobel_x.to(dtype=depth.dtype)
        sobel_y = self.sobel_y.to(dtype=depth.dtype)
        
        dx = F.conv2d(depth, sobel_x, padding=1)
        dy = F.conv2d(depth, sobel_y, padding=1)
        
        # Gradient magnitude
        grad_mag = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        # Normalize per sample (not batch-wide) for stability
        B, _, H, W = grad_mag.shape
        grad_mag_flat = grad_mag.view(B, -1)
        max_per_sample = grad_mag_flat.max(dim=1, keepdim=True).values  # [B, 1]
        max_per_sample = max_per_sample.view(B, 1, 1, 1)
        grad_mag = grad_mag / (max_per_sample + 1e-6)
        return grad_mag
    
    def forward(self, depth: torch.Tensor, target_size: Tuple[int, int] = None) -> torch.Tensor:
        """
        Generate depth gate from depth tensor.
        depth: [B, 1, H, W] log-depth normalized to 0-1
        target_size: (H, W) for resizing gate if needed
        returns: [B, gate_channels, H', W'] attention gate
        """
        B, C, H, W = depth.shape
        
        # Resize depth to target size if specified
        if target_size is not None and (H, W) != target_size:
            depth = F.interpolate(depth, size=target_size, mode='bilinear', align_corners=False)
            H, W = target_size
        
        # Feature 1: log-depth (normalized)
        log_depth = depth  # Already log-normalized
        
        # Feature 2: depth gradient magnitude (edge-like regions)
        grad_mag = self.compute_depth_gradient_magnitude(depth)
        
        # Combine features: [log_depth, grad_mag]
        # For gate_channels=1, we can combine them or use one
        if self.gate_channels == 1:
            # Combine: weighted sum or product
            # Using weighted combination: emphasize edges and depth
            gate = 0.5 * log_depth + 0.5 * grad_mag
        else:
            # Stack multiple channels
            gate = torch.cat([log_depth, grad_mag], dim=1)
            if self.gate_channels > 2:
                # Add more depth-based features if needed
                # For now, just repeat or add variations
                gate = gate[:, :self.gate_channels]
        
        # Clamp gate to [0, 1] range (sigmoid is too restrictive)
        # log_depth and grad_mag are already ~0-1, so weighted sum is also ~0-1
        gate = torch.clamp(gate, 0.0, 1.0)
        
        return gate


# === Custom DeepLabV3+ Decoder Wrapper ===
class CustomDeepLabV3PlusDecoder(nn.Module):
    """
    Wrapper for DeepLabV3PlusDecoder that uses a different feature index for high-res features.
    For EfficientNet-B7 with output_stride=32, we need to use features[3] instead of features[2].
    Also adjusts block1 to handle the correct number of input channels.
    """
    def __init__(self, base_decoder, high_res_idx=3, encoder_channels=None):
        super().__init__()
        self.base_decoder = base_decoder
        self.high_res_idx = high_res_idx
        
        # Adjust block1 if needed (for different input channels)
        if encoder_channels is not None and high_res_idx < len(encoder_channels):
            highres_in_channels = encoder_channels[high_res_idx]
            highres_out_channels = 48  # proposed by authors of paper
            # Replace block1 if channel count differs
            if base_decoder.block1[0].in_channels != highres_in_channels:
                self.block1 = nn.Sequential(
                    nn.Conv2d(
                        highres_in_channels, highres_out_channels, kernel_size=1, bias=False
                    ),
                    nn.BatchNorm2d(highres_out_channels),
                    nn.ReLU(),
                )
            else:
                self.block1 = base_decoder.block1
        else:
            self.block1 = base_decoder.block1
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # Use features[-1] for ASPP (same as base decoder)
        aspp_features = self.base_decoder.aspp(features[-1])
        aspp_features = self.base_decoder.up(aspp_features)
        
        # Use features[high_res_idx] for high-res features (instead of features[2])
        high_res_features = self.block1(features[self.high_res_idx])
        
        # Concatenate and process
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.base_decoder.block2(concat_features)
        return fused_features


# === Encoder with Depth Gating ===
class DepthGatedEncoder(nn.Module):
    """
    Encoder that applies depth gates to intermediate RGB features.
    Wraps base encoder and applies depth gates to each output feature map.
    """
    def __init__(self, base_encoder, depth_gate_module: DepthGateModule, alpha: float = 0.5):
        super().__init__()
        self.base_encoder = base_encoder
        self.depth_gate = depth_gate_module
        self.alpha = alpha  # Weight for depth gate: feat = feat * (1 + alpha * gate)
        
    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass with depth gating applied to encoder output features.
        rgb: [B, 3, H, W]
        depth: [B, 1, H, W] log-depth
        returns: List of encoder features with depth gates applied
        """
        # Forward through base encoder
        encoder_features = self.base_encoder(rgb)
        
        # Handle both list and tuple (smp encoders may return either)
        if not isinstance(encoder_features, (list, tuple)):
            encoder_features = [encoder_features]
        
        # Apply depth gates to each feature map
        gated_features = []
        for feat in encoder_features:
            # Generate depth gate for this resolution
            _, _, H, W = feat.shape
            depth_gate = self.depth_gate(depth, target_size=(H, W))
            
            # Apply gate: feat = feat * (1 + alpha * gate)
            if depth_gate.shape[1] == 1:
                # Broadcast single channel gate to all feature channels
                gate_expanded = depth_gate.expand_as(feat)
            else:
                # For multi-channel gates, use first channel
                gate_expanded = depth_gate[:, 0:1, :, :].expand_as(feat)
            
            # Apply gating
            gated_feat = feat * (1.0 + self.alpha * gate_expanded)
            gated_features.append(gated_feat)
        
        return gated_features


# === Multi-Task Model with Depth Gating ===
class DepthGatedDeepLabV3Plus(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        encoder_name = config.encoder_name
        if encoder_name == "timm-efficientnet-l2":
            encoder_weights = "noisy-student"
        else:
            encoder_weights = "imagenet"
        
        # Create base encoder (RGB only, 3 channels)
        self.base_encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights,
        )
        
        # Depth gate module
        self.depth_gate = DepthGateModule(gate_channels=config.depth_gate_channels)
        
        # Create gated encoder wrapper
        self.gated_encoder = DepthGatedEncoder(
            self.base_encoder,
            self.depth_gate,
            alpha=config.depth_gate_alpha
        )
        
        # Decoder (DeepLabV3+ decoder)
        # For EfficientNet-B7 with output_stride=32, we need to use features[3] instead of features[2]
        # Use output_stride=16 to get 4x upsampling (15x20 -> 60x80) to match features[3]
        decoder_out_channels = 256
        base_decoder = smp.decoders.deeplabv3.decoder.DeepLabV3PlusDecoder(
            encoder_channels=self.base_encoder.out_channels,
            encoder_depth=5,
            out_channels=decoder_out_channels,
            atrous_rates=(12, 24, 36),
            output_stride=16,  # Use 16 to get 4x upsampling
            aspp_separable=False,
            aspp_dropout=0.5,
        )
        # Wrap decoder to use features[3] instead of features[2] for high-res features
        self.decoder = CustomDeepLabV3PlusDecoder(
            base_decoder, 
            high_res_idx=3,
            encoder_channels=self.base_encoder.out_channels
        )
        
        # Segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(decoder_out_channels, decoder_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_out_channels, config.num_classes, kernel_size=1),
        )
        
        # Depth reconstruction head
        self.depth_head = nn.Sequential(
            nn.Conv2d(decoder_out_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1)
        )
        
        # Edge detection head (1 channel for binary edge map)
        self.edge_head = nn.Sequential(
            nn.Conv2d(decoder_out_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        
        self.config = config

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor):
        """
        Forward pass with depth gating.
        rgb: [B, 3, H, W]
        depth: [B, 1, H, W] log-depth
        """
        # Encoder with depth gating
        encoder_features = self.gated_encoder(rgb, depth)
        
        # Decoder expects a list of features
        decoder_output = self.decoder(encoder_features)
        
        # Segmentation head
        seg_logits = self.segmentation_head(decoder_output)
        
        # Depth reconstruction head
        depth_recon = self.depth_head(decoder_output)
        
        # Edge detection head
        edge_logits = self.edge_head(decoder_output)
        
        # Resize to input size
        if seg_logits.shape[2:] != rgb.shape[2:]:
            seg_logits = F.interpolate(
                seg_logits,
                size=rgb.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        if depth_recon.shape[2:] != rgb.shape[2:]:
            depth_recon = F.interpolate(
                depth_recon,
                size=rgb.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        if edge_logits.shape[2:] != rgb.shape[2:]:
            edge_logits = F.interpolate(
                edge_logits,
                size=rgb.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        return seg_logits, depth_recon, edge_logits


def build_model(config: Config) -> nn.Module:
    print(f"Building Depth-Gated DeepLabV3+ with encoder: {config.encoder_name}")
    print(f"  Depth gate alpha: {config.depth_gate_alpha}")
    print(f"  Depth gate channels: {config.depth_gate_channels}")
    model = DepthGatedDeepLabV3Plus(config)
    return model


def compute_class_weights(config: Config) -> torch.Tensor:
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
    
    rare_classes = [1, 7, 10]
    for rare_cls in rare_classes:
        if rare_cls < len(weights):
            weights[rare_cls] *= 2.0
    
    weights = weights / weights.mean()
    
    np.save(weight_cache, weights)
    print(f"Class weights: {weights}")
    return torch.tensor(weights, dtype=torch.float32)


def compute_sample_weights(dataset: NYUv2Segmentation, config: Config, rare_classes: list = [1, 7, 10]) -> np.ndarray:
    ids_str = "|".join(sorted(dataset.ids))
    cache_hash = hashlib.md5(ids_str.encode()).hexdigest()[:8]
    cache_path = config.output_dir / f"sample_weights_cache_{cache_hash}.npy"
    
    if cache_path.exists():
        return np.load(cache_path)
    
    sample_weights = np.ones(len(dataset), dtype=np.float32)
    rare_classes_array = np.array([cls for cls in rare_classes if cls < config.num_classes], dtype=np.int64)
    
    print("Computing sample weights...")
    for idx in tqdm(range(len(dataset))):
        if dataset.cache_data and idx < len(dataset.masks) and dataset.masks[idx] is not None:
            mask = dataset.masks[idx]
        else:
            name = dataset.ids[idx]
            mask_path = dataset.root / dataset.split / "label" / name
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
        
        mask_flat = mask.ravel()
        max_val = max(mask_flat.max(), rare_classes_array.max() if len(rare_classes_array)>0 else 0) + 1
        counts = np.bincount(mask_flat, minlength=int(max_val))
        if len(rare_classes_array) > 0 and np.sum(counts[rare_classes_array] > 0) > 0:
            sample_weights[idx] = 1.0 + np.sum(counts[rare_classes_array] > 0) * 2.0
        else:
            sample_weights[idx] = 1.0
    
    # Normalize to mean=1.0 (WeightedRandomSampler uses ratios, not absolute values)
    sample_weights = sample_weights / sample_weights.mean()
    np.save(cache_path, sample_weights)
    return sample_weights


def prepare_dataloaders(config: Config, train_ids: List[str], val_ids: List[str]):
    train_tf = build_transforms(config.img_size, "train")
    val_tf = build_transforms(config.img_size, "val")
    test_tf = build_transforms(config.img_size, "test")

    train_loader = None
    val_loader = None

    if len(train_ids) > 0:
        train_ds = NYUv2Segmentation(config.dataset_root, train_ids, "train", train_tf, use_depth=config.use_depth)
        if config.use_rare_sampler:
            weights = [3.0 if flag else 1.0 for flag in train_ds.contains_rare]
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        else:
            sample_weights = compute_sample_weights(train_ds, config)
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        
        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

    if len(val_ids) > 0:
        val_ds = NYUv2Segmentation(config.dataset_root, val_ids, "train", val_tf, use_depth=config.use_depth)
        val_loader = DataLoader(
            val_ds,
            batch_size=max(1, config.batch_size // 2),
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    test_ids = list_ids(config.dataset_root, "test")
    test_ds = NYUv2Segmentation(config.dataset_root, test_ids, "test", test_tf, use_depth=config.use_depth)
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, val_loader, test_loader


def forward_loss(model, batch, device, ce_loss, config: Config, epoch: int):
    if config.use_depth:
        inputs_tuple, targets = batch
        rgb, depth = inputs_tuple
        rgb = rgb.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
    else:
        inputs, targets = batch
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        rgb = inputs
        depth = None
    
    # Forward pass returns tuple (seg, depth, edge)
    if config.use_depth:
        seg_logits, depth_recon, edge_logits = model(rgb, depth)
    else:
        # Fallback for no-depth mode (shouldn't happen in this experiment)
        seg_logits, depth_recon, edge_logits = model(rgb, torch.zeros_like(rgb[:, :1]))
    
    # 1. Segmentation Losses
    ce = ce_loss(seg_logits, targets)
    dice = dice_loss(seg_logits, targets, config.ignore_index)
    total_loss = config.loss_ce_w * ce + config.loss_dice_w * dice
    
    if epoch >= (config.epochs - config.lovasz_start):
        lovasz = lovasz_softmax(F.softmax(seg_logits, dim=1), targets, ignore_index=config.ignore_index)
        total_loss = total_loss + config.loss_lovasz_w * lovasz

    # 2. Depth Reconstruction Loss
    if config.use_depth and depth_recon is not None:
        # Use log-depth as target
        depth_target = depth
        # L1 Loss for depth reconstruction
        depth_loss = F.l1_loss(depth_recon, depth_target)
        total_loss = total_loss + config.loss_depth_w * depth_loss
    
    # 3. Edge Detection Loss
    if edge_logits is not None:
        # Generate edge masks from segmentation labels
        edge_target = generate_edge_mask(targets, ignore_index=config.ignore_index)  # [B, 1, H, W]
        
        # Binary Cross Entropy Loss with logits
        # Use pos_weight to handle class imbalance (edges are sparse)
        pos_weight = torch.tensor([3.0], device=device)  # Give more weight to edge pixels
        edge_loss = F.binary_cross_entropy_with_logits(
            edge_logits, 
            edge_target, 
            pos_weight=pos_weight
        )
        total_loss = total_loss + config.loss_edge_w * edge_loss
    
    return total_loss, seg_logits, targets


def evaluate(model, loader, device, ce_loss, config: Config, epoch: int):
    model.eval()
    metric = IoUMetric(config.num_classes, config.ignore_index)
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating")
        for batch in pbar:
            # forward_loss handles use_depth internally
            loss, logits, targets = forward_loss(model, (batch[0], batch[1]), device, ce_loss, config, epoch)
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
                    model, (batch[0], batch[1]), device, ce_loss, config, epoch
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
        val_loss, val_miou, class_iou = evaluate(model, val_loader, device, ce_loss, config, epoch)

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
    E1: Multi-scale + flip TTA (Test Time Augmentation)
    - Scales: 0.75, 1.0, 1.25 (configurable via config.tta_scales)
    - Each scale: original + horizontal flip
    - All logits resized to original resolution and averaged
    
    Note: To use existing B2 models, set --output-dir to the B2 output directory
    (e.g., --output-dir data/output_B2) so it can find the checkpoints.
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
            if config.use_depth:
                inputs_tuple, name = batch
                rgb_orig, depth_orig = inputs_tuple
                rgb_orig = rgb_orig.to(device, non_blocking=True)
                depth_orig = depth_orig.to(device, non_blocking=True)
            else:
                inputs, name = batch
                rgb_orig = inputs.to(device, non_blocking=True)
                depth_orig = torch.zeros_like(rgb_orig[:, :1])

            # Get original size from tensor (safer than config.img_size)
            orig_h, orig_w = rgb_orig.shape[2:]
            
            # Accumulate logits by summing (memory efficient)
            sum_logits = None
            count = 0
            
            for scale in scales:
                # Resize RGB and depth to scaled size
                scaled_h = int(orig_h * scale)
                scaled_w = int(orig_w * scale)
                
                rgb_scaled = F.interpolate(
                    rgb_orig,
                    size=(scaled_h, scaled_w),
                    mode='bilinear',
                    align_corners=False
                )
                depth_scaled = F.interpolate(
                    depth_orig,
                    size=(scaled_h, scaled_w),
                    mode='bilinear',
                    align_corners=False
                )
                
                # Original orientation
                for model in models:
                    with autocast(enabled=use_amp):
                        logits, _, _ = model(rgb_scaled, depth_scaled)  # Unpack seg, depth, edge
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
                rgb_flip = torch.flip(rgb_scaled, dims=[3])
                depth_flip = torch.flip(depth_scaled, dims=[3])
                
                for model in models:
                    with autocast(enabled=use_amp):
                        logits_flip, _, _ = model(rgb_flip, depth_flip)  # Unpack seg, depth, edge
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
    parser = argparse.ArgumentParser(description="NYUv2 Seg - Experiment E1: Multi-scale + Flip TTA")
    parser.add_argument("--dataset-root", type=Path, default=Path("data"), help="Path to data")
    parser.add_argument("--output-dir", type=Path, default=Path("data/output_E1"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    # Note: B2 experiment requires depth (no --no-depth option)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--submission-filename", type=str, default="submission.npy")
    parser.add_argument("--encoder-name", type=str, default="timm-efficientnet-b7")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--depth-gate-alpha", type=float, default=0.5, 
                        help="Weight for depth gate: feat = feat * (1 + alpha * gate)")
    parser.add_argument("--depth-gate-channels", type=int, default=1,
                        help="Number of channels for depth gate (1 or few)")
    parser.add_argument("--edge-loss-w", type=float, default=0.3,
                        help="Weight for edge detection loss (μ ~ 0.2-0.5)")
    parser.add_argument("--tta-scales", type=float, nargs="+", default=[0.75, 1.0, 1.25],
                        help="TTA scales for multi-scale inference (default: 0.75 1.0 1.25)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check encoder support
    unsupported_encoders = {
        # Mapped legacy encoder
        "timm-efficientnetv2_l": "timm-efficientnet-b7",
        "timm-efficientnet-l2": "timm-efficientnet-b7",
    }
    if args.encoder_name in unsupported_encoders:
        args.encoder_name = unsupported_encoders[args.encoder_name]

    config = Config(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        use_depth=True,  # B2 experiment requires depth
        device=args.device,
        seed=args.seed,
        submission_filename=args.submission_filename,
        encoder_name=args.encoder_name,
        n_splits=args.n_splits,
        depth_gate_alpha=args.depth_gate_alpha,
        depth_gate_channels=args.depth_gate_channels,
        loss_edge_w=args.edge_loss_w,
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

