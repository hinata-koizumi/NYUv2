import os
import random
import json
import csv
import time
import glob
import zipfile
import copy
import re
import hashlib
import platform
from typing import Optional, Tuple, List, Dict

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold
from tqdm import tqdm


# =========================
# Config
# =========================
class Config:
    EXP_NAME = "exp096_convnext_rgbd_4ch"
    SEED = 42

    # Image sizes
    RESIZE_HEIGHT = 720
    RESIZE_WIDTH = 960
    CROP_SIZE = (576, 768)  # (H, W)

    # Smart crop
    SMART_CROP_PROB = 0.5
    SMALL_OBJ_IDS = [1, 3, 6, 7, 10]

    # Train
    EPOCHS = 180
    WARMUP_EPOCHS = 5
    BATCH_SIZE = 6
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Early stopping / checkpoints
    EARLY_STOPPING_PATIENCE = 30
    EARLY_STOPPING_MIN_DELTA = 0.0003
    SAVE_TOP_K = 5
    SAVE_START_EPOCH = 20

    # LR schedule
    ETA_MIN = 1e-6
    # LR_SCHEDULE:
    # - "cosine_drop": cosine over full training (after warmup) + one-time LR_DROP at LR_DROP_EPOCH
    #                 (this matches the original behavior; good strong baseline)
    # - "cosine_restarts_full": cosine warm restarts right after warmup (can plateau earlier if T0 is small)
    # - "cosine_drop_then_restarts": baseline until LR_DROP_EPOCH, then cosine warm restarts with peak at (lr_at_drop * LR_DROP_FACTOR)
    LR_SCHEDULE = "cosine_drop"
    # If using cosine_drop
    LR_DROP_EPOCH = 40
    LR_DROP_FACTOR = 0.3
    # If using cosine_restarts_full / cosine_drop_then_restarts (cycle length in epochs)
    COSINE_RESTART_T0 = 40
    COSINE_RESTART_T_MULT = 1

    # Task
    NUM_CLASSES = 13
    IGNORE_INDEX = 255
    IN_CHANNELS = 4  # RGB + normalized inv-depth

    # Safety checks (to avoid silent accuracy drops)
    # - STRICT_DEPTH_FOR_TRAIN: if depth file is missing/unreadable during train/valid, raise error.
    #   (Test-time depth can be missing -> allowed)
    STRICT_DEPTH_FOR_TRAIN = True
    # - SANITY_CHECK_FIRST_N: validate label value range for the first N samples per Dataset instance.
    #   Helps catch wrong label encoding/mapping early (e.g., values outside [0, NUM_CLASSES-1]).
    SANITY_CHECK_FIRST_N = 20

    # EMA
    EMA_DECAY = 0.999

    # CV
    N_FOLDS = 5

    # Loss
    SEG_LOSS = "ce_dice"      # "ce" | "ce_dice"
    DICE_WEIGHT = 0.5

    # Depth input preprocessing
    DEPTH_MIN = 0.6
    DEPTH_MAX = 10.0

    # RGB normalization
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # TTA
    TTA_COMBS = [
        (0.75, False), (0.75, True),
        (1.0, False),  (1.0, True),
        (1.25, False), (1.25, True),
        (1.5, False),  (1.5, True),
    ]
    # Temperature scaling sweep for TTA evaluation.
    # If you're sure about a fixed value, you can set e.g. [0.7].
    TEMPERATURES = [0.6, 0.7, 0.8, 1.0]

    DATA_ROOT = "data"
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    TEST_DIR = os.path.join(DATA_ROOT, "test")

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    @classmethod
    def to_dict(cls) -> dict:
        d = {}
        for k, v in cls.__dict__.items():
            if k.startswith("__") or k in ("DEVICE",):
                continue
            if callable(v) or isinstance(v, (classmethod, staticmethod, type)):
                continue
            d[k] = v
        return d


# =========================
# Repro / EMA
# =========================
def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    seed = int(np.random.get_state()[1][0]) + int(worker_id)
    np.random.seed(seed)
    random.seed(seed)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = float(decay)
        for p in self.ema.parameters():
            p.requires_grad = False

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                model_v = msd[k]
                if not torch.is_floating_point(model_v):
                    ema_v.copy_(model_v)
                else:
                    ema_v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)


# =========================
# Albumentations helpers
# =========================
def safe_shift_scale_rotate(cfg: Config) -> A.BasicTransform:
    # Albumentations warning is fine; keeping for backwards compatibility
    try:
        return A.ShiftScaleRotate(
            shift_limit=0.0,
            scale_limit=0.2,
            rotate_limit=0,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=cfg.IGNORE_INDEX,
            p=0.5,
        )
    except TypeError:
        return A.ShiftScaleRotate(
            shift_limit=0.0,
            scale_limit=0.2,
            rotate_limit=0,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=cfg.IGNORE_INDEX,
            p=0.5,
        )


def safe_pad_if_needed(cfg: Config, min_height: int, min_width: int) -> A.BasicTransform:
    try:
        return A.PadIfNeeded(
            min_height=min_height,
            min_width=min_width,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=cfg.IGNORE_INDEX,
            position="center",
        )
    except TypeError:
        return A.PadIfNeeded(
            min_height=min_height,
            min_width=min_width,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=cfg.IGNORE_INDEX,
            position="center",
        )


def get_train_transforms(cfg: Config) -> A.Compose:
    # NOTE: PadIfNeeded(CROP_SIZE) after Resize(720x960) does nothing -> removed.
    return A.Compose(
        [
            A.Resize(height=cfg.RESIZE_HEIGHT, width=cfg.RESIZE_WIDTH),
            A.HorizontalFlip(p=0.5),
            safe_shift_scale_rotate(cfg),
        ],
        additional_targets={"depth": "image", "depth_valid": "mask"},
    )


def get_color_transforms(_: Config) -> A.Compose:
    return A.Compose(
        [A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5)]
    )


def get_valid_transforms(cfg: Config) -> A.Compose:
    h_pad = ((cfg.RESIZE_HEIGHT + 31) // 32) * 32
    w_pad = ((cfg.RESIZE_WIDTH + 31) // 32) * 32
    return A.Compose(
        [
            A.Resize(height=cfg.RESIZE_HEIGHT, width=cfg.RESIZE_WIDTH),
            safe_pad_if_needed(cfg, min_height=h_pad, min_width=w_pad),
        ],
        additional_targets={"depth": "image", "depth_valid": "mask"},
    )


# =========================
# Smart crop
# =========================
def smart_crop(
    image: np.ndarray,
    label: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    crop_h: int,
    crop_w: int,
    target_ids: List[int],
    prob: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = label.shape[:2]
    max_y = h - crop_h
    max_x = w - crop_w

    # If too small, center crop as much as possible
    if max_y < 0 or max_x < 0:
        top = max(0, (h - crop_h) // 2)
        left = max(0, (w - crop_w) // 2)
        return (
            image[top : top + min(h, crop_h), left : left + min(w, crop_w)],
            label[top : top + min(h, crop_h), left : left + min(w, crop_w)],
            depth[top : top + min(h, crop_h), left : left + min(w, crop_w)],
            valid[top : top + min(h, crop_h), left : left + min(w, crop_w)],
        )

    do_smart = (random.random() < prob)
    top = left = -1

    if do_smart:
        mask = np.isin(label, target_ids)
        if mask.any():
            ys, xs = np.where(mask)
            k = random.randint(0, len(ys) - 1)
            cy, cx = ys[k], xs[k]
            min_t = max(0, cy - crop_h + 1)
            max_t = min(max_y, cy)
            min_l = max(0, cx - crop_w + 1)
            max_l = min(max_x, cx)
            if min_t <= max_t and min_l <= max_l:
                top = random.randint(min_t, max_t)
                left = random.randint(min_l, max_l)

    if top < 0:
        top = random.randint(0, max_y)
        left = random.randint(0, max_x)

    return (
        image[top : top + crop_h, left : left + crop_w],
        label[top : top + crop_h, left : left + crop_w],
        depth[top : top + crop_h, left : left + crop_w],
        valid[top : top + crop_h, left : left + crop_w],
    )


# =========================
# Dataset (RGB + invDepth -> 4ch)
# =========================
class NYUDataset(Dataset):
    def __init__(
        self,
        image_paths: np.ndarray,
        label_paths: Optional[np.ndarray],
        depth_paths: Optional[np.ndarray],
        cfg: Config,
        transform: Optional[A.Compose] = None,
        color_transform: Optional[A.Compose] = None,
        enable_smart_crop: bool = False,
        return_raw_for_tta: bool = False,  # returns (H,W,4) float32 for cv2-based TTA
    ):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.depth_paths = depth_paths
        self.cfg = cfg
        self.transform = transform
        self.color_transform = color_transform
        self.enable_smart_crop = enable_smart_crop
        self.return_raw_for_tta = return_raw_for_tta
        self._sanity_remaining = int(getattr(cfg, "SANITY_CHECK_FIRST_N", 0))

        self.mean = torch.tensor(cfg.MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(cfg.STD, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_rgb(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_label(self, path: str, shape_hw: Tuple[int, int]) -> np.ndarray:
        if path is None:
            return np.zeros(shape_hw, dtype=np.uint8)
        lbl = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if lbl is None:
            raise FileNotFoundError(f"Label not found: {path}")
        if lbl.ndim == 3:
            lbl = lbl[:, :, 0]
        return lbl

    def _load_depth_mm(self, path: Optional[str], shape_hw: Tuple[int, int], *, strict: bool) -> np.ndarray:
        if path is None or str(path) in ("", "None"):
            return np.zeros(shape_hw, dtype=np.float32)
        d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if d is None:
            if strict:
                raise FileNotFoundError(f"Depth not found or unreadable: {path}")
            return np.zeros(shape_hw, dtype=np.float32)
        return d.astype(np.float32)

    def __getitem__(self, idx: int):
        img_path = str(self.image_paths[idx])
        img = self._load_rgb(img_path)

        h, w = img.shape[:2]
        lbl = None
        if self.label_paths is not None:
            lbl = self._load_label(str(self.label_paths[idx]), (h, w))
        else:
            lbl = np.zeros((h, w), dtype=np.uint8)

        strict_depth = bool(getattr(self.cfg, "STRICT_DEPTH_FOR_TRAIN", False)) and (self.label_paths is not None)
        raw_depth_mm = self._load_depth_mm(
            None if self.depth_paths is None else self.depth_paths[idx],
            (h, w),
            strict=strict_depth,
        )

        valid = (raw_depth_mm > 0).astype(np.float32)
        depth_m = np.clip(raw_depth_mm / 1000.0, self.cfg.DEPTH_MIN, self.cfg.DEPTH_MAX)

        if self.transform is not None:
            aug = self.transform(image=img, mask=lbl, depth=depth_m, depth_valid=valid)
            img = aug["image"]
            lbl = aug["mask"]
            depth_m = aug["depth"]
            valid = aug["depth_valid"].astype(np.float32)
            # IMPORTANT:
            # depth_valid is treated as a "mask" in Albumentations, so padding/border fill may become
            # IGNORE_INDEX (=255). Do NOT let 255 be treated as valid=1.
            valid = ((valid > 0.5) & (valid < 1.5)).astype(np.float32)

            # Safety: transforms may introduce fill=0 into depth; re-clip to the expected metric range.
            depth_m = np.clip(depth_m, self.cfg.DEPTH_MIN, self.cfg.DEPTH_MAX)

        # Smart crop only for train (needs labels) and only when output is tensor path
        if (
            self.enable_smart_crop
            and (self.label_paths is not None)
            and (not self.return_raw_for_tta)
            and (self.cfg.CROP_SIZE is not None)
        ):
            ch, cw = self.cfg.CROP_SIZE
            if img.shape[0] > ch or img.shape[1] > cw:
                img, lbl, depth_m, valid = smart_crop(
                    image=img,
                    label=lbl,
                    depth=depth_m,
                    valid=valid,
                    crop_h=ch,
                    crop_w=cw,
                    target_ids=self.cfg.SMALL_OBJ_IDS,
                    prob=self.cfg.SMART_CROP_PROB,
                )

        # Color aug (RGB only)
        if (self.color_transform is not None) and (self.label_paths is not None) and (not self.return_raw_for_tta):
            img = self.color_transform(image=img)["image"]

        # Build 4ch input
        img_f = img.astype(np.float32) / 255.0

        inv = np.zeros_like(depth_m, dtype=np.float32)
        m = (valid > 0.5) & (depth_m > 0)
        inv[m] = 1.0 / depth_m[m]

        min_inv = 1.0 / self.cfg.DEPTH_MAX
        max_inv = 1.0 / self.cfg.DEPTH_MIN
        inv_norm = np.zeros_like(inv, dtype=np.float32)
        inv_norm[m] = (inv[m] - min_inv) / (max_inv - min_inv)
        inv_norm = np.clip(inv_norm, 0.0, 1.0)

        if self.return_raw_for_tta:
            rgb_norm = (img_f - np.array(self.cfg.MEAN, np.float32)) / np.array(self.cfg.STD, np.float32)
            x = np.dstack([rgb_norm, inv_norm]).astype(np.float32)  # (H,W,4)
            return x, lbl, valid

        rgb_t = torch.from_numpy(img_f).permute(2, 0, 1).float()
        rgb_t = (rgb_t - self.mean) / self.std
        d_t = torch.from_numpy(inv_norm).unsqueeze(0).float()
        x = torch.cat([rgb_t, d_t], dim=0)  # (4,H,W)

        if self.label_paths is None:
            return x, os.path.basename(img_path)

        if self._sanity_remaining > 0:
            self._sanity_remaining -= 1
            m_lbl = (lbl != self.cfg.IGNORE_INDEX)
            if np.any(m_lbl):
                mn = int(np.min(lbl[m_lbl]))
                mx = int(np.max(lbl[m_lbl]))
                if mn < 0 or mx >= int(self.cfg.NUM_CLASSES):
                    raise ValueError(
                        f"Label value out of range in {img_path}: min={mn}, max={mx}, "
                        f"expected within [0, {int(self.cfg.NUM_CLASSES) - 1}] (ignore={self.cfg.IGNORE_INDEX})."
                    )

        y = torch.from_numpy(lbl).long()
        return x, y


# =========================
# Model (Seg only)
# =========================
class SegFPN(nn.Module):
    _printed_4ch_init_msg = False

    def __init__(self, num_classes: int, in_channels: int):
        super().__init__()
        self.net = smp.FPN(
            encoder_name="tu-convnext_base",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
        self._init_4th_channel_if_needed(in_channels)

    def _init_4th_channel_if_needed(self, in_channels: int) -> None:
        if in_channels != 4:
            return

        enc = self.net.encoder
        # timm convnext usually has stem[0]
        if hasattr(enc, "stem") and isinstance(enc.stem[0], nn.Conv2d):
            conv = enc.stem[0]
        elif hasattr(enc, "conv1") and isinstance(enc.conv1, nn.Conv2d):
            conv = enc.conv1
        else:
            return

        if conv.weight.shape[1] != 4:
            return

        with torch.no_grad():
            conv.weight[:, 3, :, :] = conv.weight[:, :3, :, :].mean(dim=1)
        if not SegFPN._printed_4ch_init_msg:
            print("SegFPN: initialized 4th channel weights = mean(RGB)")
            SegFPN._printed_4ch_init_msg = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.net.encoder(x)
        dec = self.net.decoder(feats)
        logits = self.net.segmentation_head(dec)
        logits = F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)
        return logits


# =========================
# Loss (CE + Dice(present only))
# =========================
def _flatten_probas_and_targets(
    probas: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # probas: (B,C,H,W) after softmax
    # targets: (B,H,W)
    b, c, h, w = probas.shape
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, c)
    targets = targets.contiguous().view(-1)
    valid = targets != ignore_index
    return probas[valid], targets[valid]


def soft_dice_loss_present_only(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    probas = F.softmax(logits, dim=1)
    probas_flat, targets_flat = _flatten_probas_and_targets(probas, targets, ignore_index)

    if probas_flat.numel() == 0:
        return logits.new_tensor(0.0)

    targets_oh = F.one_hot(targets_flat, num_classes=num_classes).to(dtype=probas_flat.dtype)

    intersection = (probas_flat * targets_oh).sum(dim=0)
    cardinality = probas_flat.sum(dim=0) + targets_oh.sum(dim=0)
    dice = (2.0 * intersection + eps) / (cardinality + eps)

    present = targets_oh.sum(dim=0) > 0
    if not present.any():
        return logits.new_tensor(0.0)

    return 1.0 - dice[present].mean()


class CombinedSegLoss(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ignore_index = cfg.IGNORE_INDEX
        self.num_classes = cfg.NUM_CLASSES
        self.mode = cfg.SEG_LOSS
        self.dice_weight = float(cfg.DICE_WEIGHT)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, ignore_index=self.ignore_index)
        if self.mode == "ce":
            return ce
        if self.mode == "ce_dice":
            dice = soft_dice_loss_present_only(
                logits=logits,
                targets=targets,
                num_classes=self.num_classes,
                ignore_index=self.ignore_index,
            )
            return ce + self.dice_weight * dice
        raise ValueError(f"Unknown SEG_LOSS: {self.mode}")


# =========================
# Metrics
# =========================
def update_confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int, ignore_index: int, cm: np.ndarray) -> np.ndarray:
    preds = preds.flatten()
    labels = labels.flatten()
    m = labels != ignore_index
    preds = preds[m]
    labels = labels[m]
    cm += np.bincount(num_classes * labels + preds, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return cm


def compute_metrics(cm: np.ndarray) -> Tuple[float, float, np.ndarray]:
    pixel_acc = np.diag(cm).sum() / (cm.sum() + 1e-10)
    inter = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = inter / union
    miou = np.nanmean(iou)
    return pixel_acc, miou, iou


# =========================
# LR schedule
# =========================
def get_lr_for_epoch(
    epoch: int,
    warmup_epochs: int,
    max_epochs: int,
    base_lr: float,
    eta_min: float,
    lr_drop_epoch: Optional[int],
    lr_drop_factor: float,
    lr_schedule: str = "cosine_drop",
    cosine_restart_t0: int = 40,
    cosine_restart_t_mult: int = 1,
) -> float:
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        mult = float(epoch) / float(warmup_epochs)
    else:
        if lr_schedule in ("cosine_restarts_full", "cosine_restarts"):
            # Epoch index starting at 0 right after warmup
            t = max(0, int(epoch - warmup_epochs - 1))
            t0 = max(1, int(cosine_restart_t0))
            tm = max(1, int(cosine_restart_t_mult))

            # find current cycle length and position within cycle
            cycle_len = t0
            cycle_start = 0
            while t >= cycle_start + cycle_len:
                cycle_start += cycle_len
                cycle_len *= tm

            t_in = t - cycle_start
            progress = float(t_in) / float(max(1, cycle_len))
            cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
            alpha = float(eta_min) / float(base_lr)
            mult = alpha + (1.0 - alpha) * cosine_factor
        elif lr_schedule == "cosine_drop_then_restarts":
            # Before LR_DROP_EPOCH: baseline cosine over full training (after warmup)
            # After LR_DROP_EPOCH: cosine restarts with peak at (lr_at_drop * LR_DROP_FACTOR)
            drop_ep = int(lr_drop_epoch) if lr_drop_epoch is not None else None
            if drop_ep is None or epoch <= drop_ep:
                denom = max(1, (max_epochs - warmup_epochs))
                progress = float(epoch - warmup_epochs) / float(denom)
                progress = min(max(progress, 0.0), 1.0)
                cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
                alpha = float(eta_min) / float(base_lr)
                mult = alpha + (1.0 - alpha) * cosine_factor
            else:
                # lr_max is the baseline lr at drop epoch, scaled by LR_DROP_FACTOR
                denom = max(1, (max_epochs - warmup_epochs))
                progress_drop = float(drop_ep - warmup_epochs) / float(denom)
                progress_drop = min(max(progress_drop, 0.0), 1.0)
                cosine_factor_drop = 0.5 * (1.0 + np.cos(np.pi * progress_drop))
                alpha = float(eta_min) / float(base_lr)
                mult_drop = alpha + (1.0 - alpha) * cosine_factor_drop
                lr_max = float(base_lr) * float(mult_drop) * float(lr_drop_factor)

                # Restarts start right after drop epoch
                t = max(0, int(epoch - drop_ep - 1))
                t0 = max(1, int(cosine_restart_t0))
                tm = max(1, int(cosine_restart_t_mult))

                cycle_len = t0
                cycle_start = 0
                while t >= cycle_start + cycle_len:
                    cycle_start += cycle_len
                    cycle_len *= tm
                t_in = t - cycle_start
                progress = float(t_in) / float(max(1, cycle_len))
                cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))

                # lerp between lr_max and eta_min directly (avoid coupling to base_lr shape)
                lr = float(eta_min) + (lr_max - float(eta_min)) * float(cosine_factor)
                return float(lr)
        else:
            # default: single cosine over the whole training (after warmup)
            denom = max(1, (max_epochs - warmup_epochs))
            progress = float(epoch - warmup_epochs) / float(denom)
            progress = min(max(progress, 0.0), 1.0)
            cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
            alpha = float(eta_min) / float(base_lr)
            mult = alpha + (1.0 - alpha) * cosine_factor

    lr = float(base_lr) * float(mult)
    if lr_schedule == "cosine_drop" and lr_drop_epoch is not None and epoch >= int(lr_drop_epoch):
        lr *= float(lr_drop_factor)
    return lr


# =========================
# Train / Valid
# =========================
def train_one_epoch(model, ema: ModelEMA, loader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    total = 0.0
    seg_total = 0.0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()
        ema.update(model)

        total += loss.item()
        seg_total += loss.item()

    n = max(1, len(loader))
    return total / n, seg_total / n


@torch.no_grad()
def validate(model, loader, criterion, device, cfg: Config) -> Tuple[float, float, float, float]:
    model.eval()
    total = 0.0
    seg_total = 0.0
    cm = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=np.int64)

    for x, y in tqdm(loader, desc="Valid", leave=False):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total += loss.item()
        seg_total += loss.item()

        preds = torch.argmax(logits, dim=1)
        cm = update_confusion_matrix(
            preds.cpu().numpy(),
            y.cpu().numpy(),
            cfg.NUM_CLASSES,
            cfg.IGNORE_INDEX,
            cm,
        )

    pixel_acc, miou, _ = compute_metrics(cm)
    n = max(1, len(loader))
    return total / n, seg_total / n, pixel_acc, miou


# =========================
# TTA (cv2-based)
# =========================
@torch.no_grad()
def tta_inference(model: nn.Module, x_hw4: np.ndarray, cfg: Config, temperature: float) -> np.ndarray:
    """
    x_hw4: (H, W, 4) float32, already normalized as training input.
    returns: (H, W, C) avg probabilities
    """
    model.eval()
    h, w = x_hw4.shape[:2]
    acc = np.zeros((h, w, cfg.NUM_CLASSES), dtype=np.float32)
    cnt = 0

    for scale, flip in cfg.TTA_COMBS:
        h_new = max(32, (int(h * scale) // 32) * 32)
        w_new = max(32, (int(w * scale) // 32) * 32)

        img = cv2.resize(x_hw4, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
        if flip:
            img = cv2.flip(img, 1)

        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(cfg.DEVICE)
        logits = model(t)
        probs = F.softmax(logits / float(temperature), dim=1)

        p = probs.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        if flip:
            p = cv2.flip(p, 1)
        p = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)

        acc += p
        cnt += 1

    return acc / max(1, cnt)


@torch.no_grad()
def validate_tta_sweep(model: nn.Module, dataset_tta: NYUDataset, cfg: Config) -> Tuple[float, float, Dict[float, float]]:
    best_temp = 1.0
    best_miou = -1.0
    results = {}

    for t in cfg.TEMPERATURES:
        cm = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=np.int64)

        for i in tqdm(range(len(dataset_tta)), desc=f"TTA T={t}", leave=False):
            x, y, _valid = dataset_tta[i]
            probs = tta_inference(model, x, cfg, temperature=float(t))
            pred = np.argmax(probs, axis=2).astype(np.int64)
            cm = update_confusion_matrix(pred, y.astype(np.int64), cfg.NUM_CLASSES, cfg.IGNORE_INDEX, cm)

        _, miou, _ = compute_metrics(cm)
        results[float(t)] = float(miou)

        if miou > best_miou:
            best_miou = float(miou)
            best_temp = float(t)

    return best_temp, best_miou, results


# =========================
# Logging / checkpoints
# =========================
def save_config(cfg: Config, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)


def save_run_info(out_dir: str, keys: List[str], cfg: Config) -> None:
    """
    Persist run metadata to make accuracy regressions diagnosable:
    - dataset filename list digest (detects data/split changes)
    - library versions / device info (detects environment changes)
    """
    os.makedirs(out_dir, exist_ok=True)
    h = hashlib.md5("\n".join(keys).encode("utf-8")).hexdigest()

    gpu_name = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "unknown"

    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": getattr(torch, "__version__", "unknown"),
        "opencv": getattr(cv2, "__version__", "unknown"),
        "numpy": getattr(np, "__version__", "unknown"),
        "smp": getattr(smp, "__version__", "unknown"),
        "device": str(cfg.DEVICE),
        "gpu_name": gpu_name,
        "keys_count": int(len(keys)),
        "keys_md5": h,
    }
    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)


def init_logger(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "train_log.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "lr", "train_loss", "valid_loss", "valid_miou", "valid_pixel_acc"])
    return path


def log_metrics(path: str, epoch: int, lr: float, tr_loss: float, va_loss: float, miou: float, acc: float) -> None:
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([epoch, lr, tr_loss, va_loss, miou, acc])


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = None
        self.bad = 0
        self.stop = False

    def __call__(self, score: float) -> None:
        if self.best is None:
            self.best = float(score)
            return
        if float(score) < self.best + self.min_delta:
            self.bad += 1
            if self.bad >= self.patience:
                self.stop = True
        else:
            self.best = float(score)
            self.bad = 0


class CheckpointManager:
    def __init__(self, save_dir: str, top_k: int):
        self.save_dir = save_dir
        self.top_k = int(top_k)
        self.items: List[Tuple[float, str]] = []
        os.makedirs(save_dir, exist_ok=True)
        self._prune_existing()

    def _prune_existing(self) -> None:
        """
        Prune checkpoints from previous runs so they don't accumulate forever.
        Keeps only top_k by mIoU (parsed from filename).
        """
        pat = re.compile(r"^model_epoch(\d+)_miou([0-9.]+)\.pth$")
        found: List[Tuple[float, str]] = []

        try:
            files = os.listdir(self.save_dir)
        except FileNotFoundError:
            return

        for fn in files:
            m = pat.match(fn)
            if not m:
                continue
            miou = float(m.group(2))
            path = os.path.join(self.save_dir, fn)
            found.append((miou, path))

        found.sort(key=lambda x: x[0], reverse=True)
        self.items = found[: self.top_k]

        for _miou, p in found[self.top_k :]:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass

        # Clean up obvious temporaries from interrupted saves
        for fn in files:
            if fn.endswith(".tmp") or fn.endswith(".partial"):
                try:
                    os.remove(os.path.join(self.save_dir, fn))
                except FileNotFoundError:
                    pass

    def _atomic_torch_save(self, state_dict: dict, path: str) -> None:
        tmp = path + ".tmp"
        try:
            torch.save(state_dict, tmp)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def save(self, model: nn.Module, epoch: int, miou: float) -> None:
        path = os.path.join(self.save_dir, f"model_epoch{epoch}_miou{miou:.4f}.pth")
        self._atomic_torch_save(model.state_dict(), path)

        self.items.append((float(miou), path))
        self.items.sort(key=lambda x: x[0], reverse=True)

        while len(self.items) > self.top_k:
            _, p = self.items.pop()
            if os.path.exists(p):
                os.remove(p)


# =========================
# Submission
# =========================
def collect_fold_weights(output_dir: str, n_folds: int) -> List[str]:
    paths = []
    for f in range(n_folds):
        p = os.path.join(output_dir, f"fold{f}", "model_best.pth")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing weights: {p}")
        paths.append(p)
    return paths


@torch.no_grad()
def make_submission_npy(
    weight_paths: List[str],
    test_image_dir: str,
    test_depth_dir: str,
    output_path: str,
    cfg: Config,
    temperature: float,
) -> None:
    image_files = sorted(glob.glob(os.path.join(test_image_dir, "*.png")))
    if len(image_files) == 0:
        raise FileNotFoundError("No test images found.")

    depth_files = []
    for img_p in image_files:
        base = os.path.basename(img_p)
        d_p = os.path.join(test_depth_dir, base)
        depth_files.append(d_p if os.path.exists(d_p) else None)

    test_ds = NYUDataset(
        image_paths=np.array(image_files),
        label_paths=None,
        depth_paths=np.array(depth_files),
        cfg=cfg,
        transform=get_valid_transforms(cfg),
        color_transform=None,
        enable_smart_crop=False,
        return_raw_for_tta=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0 if cfg.DEVICE.type == "mps" else 2,
        pin_memory=True,
    )

    sample_x, _ = test_ds[0]
    H_pad, W_pad = int(sample_x.shape[1]), int(sample_x.shape[2])
    N = len(image_files)

    memmap_dir = os.path.dirname(output_path) or "."
    os.makedirs(memmap_dir, exist_ok=True)
    mm_path = os.path.join(memmap_dir, f"probs_accum_{int(time.time())}.dat")
    acc = np.memmap(mm_path, dtype="float16", mode="w+", shape=(N, H_pad, W_pad, cfg.NUM_CLASSES))
    acc[:] = 0.0
    acc.flush()

    for wp in weight_paths:
        model = SegFPN(num_classes=cfg.NUM_CLASSES, in_channels=cfg.IN_CHANNELS)
        state = torch.load(wp, map_location="cpu")
        model.load_state_dict(state if not isinstance(state, dict) else state)
        model.to(cfg.DEVICE)
        model.eval()

        idx = 0
        for x, _names in tqdm(test_loader, desc=f"Infer {os.path.basename(wp)}", leave=False):
            x = x.to(cfg.DEVICE)
            logits = model(x)
            probs = F.softmax(logits / float(temperature), dim=1)

            x_flip = torch.flip(x, dims=[3])
            logits_flip = model(x_flip)
            probs_flip = F.softmax(logits_flip / float(temperature), dim=1)
            probs_flip = torch.flip(probs_flip, dims=[3])

            probs = (probs + probs_flip) / 2.0
            p = probs.cpu().numpy().transpose(0, 2, 3, 1).astype(np.float16)

            b = p.shape[0]
            acc[idx : idx + b] += p
            idx += b

        if idx != N:
            raise RuntimeError(f"Inference count mismatch: got {idx}, expected {N}")
        acc.flush()

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    crop_h, crop_w = cfg.RESIZE_HEIGHT, cfg.RESIZE_WIDTH
    start_y = max(0, (H_pad - crop_h) // 2)
    start_x = max(0, (W_pad - crop_w) // 2)

    preds = []
    for i in tqdm(range(N), desc="Argmax & Save", leave=False):
        p = acc[i].astype(np.float32) / float(len(weight_paths))
        p = p[start_y : start_y + crop_h, start_x : start_x + crop_w, :]
        preds.append(np.argmax(p, axis=2).astype(np.uint8))

    preds = np.array(preds)
    np.save(output_path.replace(".npy", ""), preds)

    del acc
    if os.path.exists(mm_path):
        os.remove(mm_path)


def build_submission_zip(zip_path: str, submission_npy_path: str) -> None:
    if not os.path.exists(submission_npy_path):
        raise FileNotFoundError(f"Not found: {submission_npy_path}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(submission_npy_path, arcname="tmp/submission.npy")


# =========================
# Main
# =========================
def main():
    cfg = Config()
    seed_everything(cfg.SEED)

    output_dir = os.path.join("data", "outputs", cfg.EXP_NAME)
    os.makedirs(output_dir, exist_ok=True)

    # collect aligned train files by basename
    image_dir = os.path.join(cfg.TRAIN_DIR, "image")
    label_dir = os.path.join(cfg.TRAIN_DIR, "label")
    depth_dir = os.path.join(cfg.TRAIN_DIR, "depth")

    images = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    labels = sorted([f for f in os.listdir(label_dir) if f.endswith(".png")])
    depths = sorted([f for f in os.listdir(depth_dir) if f.endswith(".png")])

    img_map = {f: os.path.join(image_dir, f) for f in images}
    lbl_map = {f: os.path.join(label_dir, f) for f in labels}
    dep_map = {f: os.path.join(depth_dir, f) for f in depths}

    keys = sorted(list(set(img_map.keys()) & set(lbl_map.keys()) & set(dep_map.keys())))
    if len(keys) == 0:
        raise ValueError("No common files found in train/image,label,depth")

    X_img = np.array([img_map[k] for k in keys])
    X_lbl = np.array([lbl_map[k] for k in keys])
    X_dep = np.array([dep_map[k] for k in keys])

    # Save run metadata once per experiment (same across folds)
    save_run_info(output_dir, keys, cfg)

    kf = KFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    splits = list(kf.split(X_img))

    all_fold_cm = []
    fold_tta_summary = {}

    for fold_idx in range(cfg.N_FOLDS):
        print("\n" + "=" * 60)
        print(f"FOLD {fold_idx} / {cfg.N_FOLDS - 1}")
        print("=" * 60)

        fold_dir = os.path.join(output_dir, f"fold{fold_idx}")
        save_config(cfg, fold_dir)
        log_path = init_logger(fold_dir)

        tr_idx, va_idx = splits[fold_idx]

        train_ds = NYUDataset(
            image_paths=X_img[tr_idx],
            label_paths=X_lbl[tr_idx],
            depth_paths=X_dep[tr_idx],
            cfg=cfg,
            transform=get_train_transforms(cfg),
            color_transform=get_color_transforms(cfg),
            enable_smart_crop=True,
            return_raw_for_tta=False,
        )
        valid_ds = NYUDataset(
            image_paths=X_img[va_idx],
            label_paths=X_lbl[va_idx],
            depth_paths=X_dep[va_idx],
            cfg=cfg,
            transform=get_valid_transforms(cfg),
            color_transform=None,
            enable_smart_crop=False,
            return_raw_for_tta=False,
        )
        valid_ds_tta = NYUDataset(
            image_paths=X_img[va_idx],
            label_paths=X_lbl[va_idx],
            depth_paths=X_dep[va_idx],
            cfg=cfg,
            transform=get_valid_transforms(cfg),
            color_transform=None,
            enable_smart_crop=False,
            return_raw_for_tta=True,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

        model = SegFPN(num_classes=cfg.NUM_CLASSES, in_channels=cfg.IN_CHANNELS).to(cfg.DEVICE)
        ema = ModelEMA(model, cfg.EMA_DECAY)

        criterion = CombinedSegLoss(cfg)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        ckpt = CheckpointManager(fold_dir, top_k=cfg.SAVE_TOP_K)
        early = EarlyStopping(cfg.EARLY_STOPPING_PATIENCE, cfg.EARLY_STOPPING_MIN_DELTA)

        best_miou = -1.0
        lr_prev = None

        for epoch in range(1, cfg.EPOCHS + 1):
            lr_now = get_lr_for_epoch(
                epoch=epoch,
                warmup_epochs=cfg.WARMUP_EPOCHS,
                max_epochs=cfg.EPOCHS,
                base_lr=cfg.LEARNING_RATE,
                eta_min=cfg.ETA_MIN,
                lr_drop_epoch=cfg.LR_DROP_EPOCH,
                lr_drop_factor=cfg.LR_DROP_FACTOR,
                lr_schedule=str(getattr(cfg, "LR_SCHEDULE", "cosine_drop")),
                cosine_restart_t0=int(getattr(cfg, "COSINE_RESTART_T0", 40)),
                cosine_restart_t_mult=int(getattr(cfg, "COSINE_RESTART_T_MULT", 1)),
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            if (
                str(getattr(cfg, "LR_SCHEDULE", "cosine_drop")) == "cosine_drop"
                and (lr_prev is not None)
                and (cfg.LR_DROP_EPOCH is not None)
                and epoch == cfg.LR_DROP_EPOCH
            ):
                ratio = lr_now / (lr_prev + 1e-30)
                print(f"[LR_DROP_CHECK] epoch{epoch-1}->{epoch}: lr ratio={ratio:.6g} (target {cfg.LR_DROP_FACTOR})")
            lr_prev = lr_now

            tr_loss, _ = train_one_epoch(model, ema, train_loader, criterion, optimizer, cfg.DEVICE)
            va_loss, _, acc, miou = validate(ema.ema, valid_loader, criterion, cfg.DEVICE, cfg)

            print(f"Epoch {epoch}/{cfg.EPOCHS}: lr={lr_now:.6g}, TrainLoss={tr_loss:.4f}, Valid mIoU={miou:.4f} (EMA)")
            log_metrics(log_path, epoch, lr_now, tr_loss, va_loss, miou, acc)

            if epoch >= cfg.SAVE_START_EPOCH:
                ckpt.save(ema.ema, epoch, miou)

            if miou > best_miou:
                best_miou = miou
                # atomic save to avoid leaving a corrupted partial file on disk-full/interruption
                best_path = os.path.join(fold_dir, "model_best.pth")
                ckpt._atomic_torch_save(ema.ema.state_dict(), best_path)

            early(miou)
            if early.stop:
                print(f"Early stop at epoch {epoch}")
                break

        # TTA sweep for this fold
        best_model_path = os.path.join(fold_dir, "model_best.pth")
        best_model = SegFPN(num_classes=cfg.NUM_CLASSES, in_channels=cfg.IN_CHANNELS)
        best_model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
        best_model.to(cfg.DEVICE)
        fold_best_temp, fold_best_miou, fold_temp_results = validate_tta_sweep(best_model, valid_ds_tta, cfg)

        fold_tta_summary[f"fold{fold_idx}"] = {
            "best_temp": fold_best_temp,
            "best_miou": fold_best_miou,
            "temp_results": {str(k): float(v) for k, v in fold_temp_results.items()},
        }
        print(f"Fold {fold_idx} TTA: best_temp={fold_best_temp}, miou={fold_best_miou:.4f}")

        # store confusion for OOF temp selection (only for temps in cfg.TEMPERATURES)
        # (optional) if you want true OOF-temp selection, you can compute cm here again.
        all_fold_cm.append(None)

    # Pick a single temperature for submission based on mean mIoU across folds.
    # (Not a true OOF aggregation, but better than ignoring the sweep result.)
    temp_scores = {float(t): [] for t in cfg.TEMPERATURES}
    for _fold_name, fd in fold_tta_summary.items():
        tr = fd.get("temp_results", {})
        for t in cfg.TEMPERATURES:
            tv = tr.get(str(float(t)))
            if tv is not None:
                temp_scores[float(t)].append(float(tv))

    mean_temp_scores = {t: (sum(v) / len(v)) for t, v in temp_scores.items() if len(v) > 0}
    if len(mean_temp_scores) > 0:
        best_temp_global = max(mean_temp_scores.items(), key=lambda kv: kv[1])[0]
    else:
        best_temp_global = float(cfg.TEMPERATURES[0])

    # save summary
    fold_tta_summary["_global"] = {
        "best_temp_mean_miou": float(best_temp_global),
        "mean_miou_per_temp": {str(k): float(v) for k, v in sorted(mean_temp_scores.items(), key=lambda kv: kv[0])},
    }
    with open(os.path.join(output_dir, "tta_summary.json"), "w") as f:
        json.dump(fold_tta_summary, f, indent=2)

    # submission
    print("\n" + "=" * 60)
    print("SUBMISSION PHASE")
    print("=" * 60)

    weight_paths = collect_fold_weights(output_dir, cfg.N_FOLDS)
    test_image_dir = os.path.join(cfg.TEST_DIR, "image")
    test_depth_dir = os.path.join(cfg.TEST_DIR, "depth")
    os.makedirs("tmp", exist_ok=True)

    submission_npy_path = "tmp/submission.npy"
    # NOTE: np.save will create "tmp/submission.npy" when given "tmp/submission"
    make_submission_npy(
        weight_paths=weight_paths,
        test_image_dir=test_image_dir,
        test_depth_dir=test_depth_dir,
        output_path=submission_npy_path,
        cfg=cfg,
        temperature=float(best_temp_global),
    )

    build_submission_zip("submission.zip", submission_npy_path)
    print("Done! submission.zip created.")


if __name__ == "__main__":
    main()
