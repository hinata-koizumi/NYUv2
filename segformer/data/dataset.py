import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List
import albumentations as A



from .augmentations import smart_crop, CopyPasteAugmentation, DepthDropout

class NYUDataset(Dataset):
    """
    Exp100 Final Version.
    - Enforces Nearest Neighbor for Depth/Valid masks.
    - Handles 4-channel input (RGB + Normalized Inv Depth).
    """

    def __init__(
        self,
        image_paths: np.ndarray,
        label_paths: Optional[np.ndarray],
        depth_paths: Optional[np.ndarray],
        cfg,
        transform: Optional[A.Compose] = None,
        color_transform: Optional[A.Compose] = None,
        enable_smart_crop: bool = False,
        is_train: Optional[bool] = None,
    ):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.depth_paths = depth_paths
        self.cfg = cfg
        self.transform = transform
        self.color_transform = color_transform
        self.enable_smart_crop = enable_smart_crop
        self.is_train = (self.label_paths is not None) if is_train is None else bool(is_train)

        # Mutable dataset state
        self.current_epoch: int = 0

        # Sanity check counter
        self._sanity_remaining = int(getattr(cfg, "SANITY_CHECK_FIRST_N", 0))

        # RNG
        self._rng = np.random.default_rng(int(getattr(cfg, "SEED", 42)))

        # Fixed 4ch normalization/constants
        self._rgb_mean = torch.tensor(cfg.MEAN, dtype=torch.float32).view(3, 1, 1)
        self._rgb_std = torch.tensor(cfg.STD, dtype=torch.float32).view(3, 1, 1)
        self._inv_min = 1.0 / float(cfg.DEPTH_MAX)
        self._inv_max = 1.0 / float(cfg.DEPTH_MIN)
        self._inv_denom = float(self._inv_max - self._inv_min) or 1.0
        self._depth_denom = float(cfg.DEPTH_MAX - cfg.DEPTH_MIN) or 1.0

        # Augmentations
        self.copy_paste = CopyPasteAugmentation(cfg, image_paths, label_paths, depth_paths, self) if self.is_train else None
        self.depth_dropout = DepthDropout(cfg) if self.is_train else None

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_rgb(self, path: str) -> np.ndarray:
        """Loads an image from disk and converts BGR to RGB."""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        # SAFETY: Ensure RGB for pretrained models
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_label(self, path: Optional[str], shape_hw: Tuple[int, int]) -> np.ndarray:
        """Loads a label mask, handling missing paths by returning a zero mask."""
        if not path:
            return np.zeros(shape_hw, dtype=np.uint8)
        lbl = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if lbl is None:
            raise FileNotFoundError(f"Label not found: {path}")
        if lbl.ndim == 3:
            lbl = lbl[:, :, 0]
        return lbl

    def _load_depth_mm(self, path: Optional[str], shape_hw: Tuple[int, int], *, strict: bool) -> np.ndarray:
        """Loads depth map (in mm), handling missing paths."""
        if not path:
            return np.zeros(shape_hw, dtype=np.float32)
        d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if d is None:
            if strict:
                raise FileNotFoundError(f"Depth not found or unreadable: {path}")
            return np.zeros(shape_hw, dtype=np.float32)
        return d.astype(np.float32)

    def _norm_rgb(self, rgb: np.ndarray) -> torch.Tensor:
        """Normalizes RGB image using ImageNet stats."""
        img_f = rgb.astype(np.float32)
        if img_f.max() > 1.5:
            img_f = img_f / 255.0
        rgb_t = torch.from_numpy(img_f).permute(2, 0, 1).contiguous().float()
        return (rgb_t - self._rgb_mean) / self._rgb_std

    def _minmax01(self, x: np.ndarray) -> np.ndarray:
        """Clips and normalizes depth values to [0, 1] range."""
        y = (x - self._inv_min) / self._inv_denom
        return np.clip(y, 0.0, 1.0).astype(np.float32)

    def _make_input(self, rgb: np.ndarray, depth_m: np.ndarray, valid_mask: np.ndarray) -> torch.Tensor:
        rgb_t = self._norm_rgb(rgb)

        v_mask = (valid_mask > 0.5)
        m = v_mask & (depth_m > 0)

        inv = np.zeros_like(depth_m, dtype=np.float32)
        inv[m] = 1.0 / depth_m[m]

        inv_norm = np.zeros_like(inv, dtype=np.float32)
        inv_norm[m] = self._minmax01(inv[m])

        d_t = torch.from_numpy(inv_norm).unsqueeze(0).float()
        
        if getattr(self.cfg, "IN_CHANNELS", 4) == 3:
            return rgb_t
            
        return torch.cat([rgb_t, d_t], dim=0)

    def _resize_crop_to_target(self, img, lbl, depth_m, valid, out_h: int, out_w: int):
        if img.shape[0] == out_h and img.shape[1] == out_w:
            return img, lbl, depth_m, valid

        # Explicitly enforce NEAREST for Depth/Valid to avoid artifacts
        # FIX: Align depth interpolation with transforms (Linear) to prevent distribution shift
        # Revert to Nearest for Depth to avoid "Halo" artifacts during aggressive zoom.
        # Strict values are better than blurred edges for the teacher signal.
        img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        lbl = cv2.resize(lbl, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        
        # Exp101: Linear Interpolation for Depth
        depth_interp = cv2.INTER_LINEAR if getattr(self.cfg, "DEPTH_INTERPOLATION", "linear") == "linear" else cv2.INTER_NEAREST
        depth_m = cv2.resize(depth_m, (out_w, out_h), interpolation=depth_interp)
        valid = cv2.resize(valid, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        
        valid = (valid > 0).astype(np.float32)
        return img, lbl, depth_m, valid

    def _smart_crop(self, img, lbl, depth_m, valid, crop_h, crop_w):
        curr_epoch = int(getattr(self, "current_epoch", 0))
        focus_sofa_bed = curr_epoch > 5 and self._rng.random() < 0.05
        if focus_sofa_bed:
            target_ids, prob = [3, 4], 1.0
        else:
            target_ids, prob = list(self.cfg.SMALL_OBJ_IDS), float(self.cfg.SMART_CROP_PROB)

        base_h, base_w = int(crop_h), int(crop_w)
        zoom = False
        if not focus_sofa_bed:
            zoom_prob = float(getattr(self.cfg, "SMART_CROP_ZOOM_PROB", 0.0))
            if zoom_prob > 0.0:
                zoom_only_small = bool(getattr(self.cfg, "SMART_CROP_ZOOM_ONLY_SMALL", True))
                small_ids = list(self.cfg.SMALL_OBJ_IDS)
                has_small = bool(small_ids) and np.isin(lbl, small_ids).any()
                if (not zoom_only_small) or has_small:
                    zmin, zmax = getattr(self.cfg, "SMART_CROP_ZOOM_RANGE", (1.0, 1.0))
                    zmin = max(1e-3, min(1.0, float(zmin)))
                    zmax = max(zmin, min(1.0, float(zmax)))
                    if self._rng.random() < zoom_prob:
                        scale = float(self._rng.uniform(zmin, zmax))
                        crop_h = max(1, int(round(base_h * scale)))
                        crop_w = max(1, int(round(base_w * scale)))
                        zoom = (crop_h != base_h) or (crop_w != base_w)

        img, lbl, depth_m, valid = smart_crop(
            img, lbl, depth_m, valid, crop_h, crop_w,
            target_ids=target_ids,
            prob=prob,
            rng=self._rng,
        )
        if zoom:
            img, lbl, depth_m, valid = self._resize_crop_to_target(img, lbl, depth_m, valid, base_h, base_w)
        return img, lbl, depth_m, valid

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict, Optional[torch.Tensor], Optional[torch.Tensor]]:
        has_labels = self.label_paths is not None
        is_train = bool(self.is_train)

        img_path = str(self.image_paths[idx])
        img = self._load_rgb(img_path)
        h_orig, w_orig = img.shape[:2]

        lbl = self._load_label(None if not has_labels else self.label_paths[idx], (h_orig, w_orig))

        strict_depth = bool(getattr(self.cfg, "STRICT_DEPTH_FOR_TRAIN", False)) and is_train
        raw_depth_mm = self._load_depth_mm(
            None if self.depth_paths is None else self.depth_paths[idx],
            (h_orig, w_orig),
            strict=strict_depth,
        )

        valid_mask = (raw_depth_mm > 0).astype(np.float32)
        depth_m = np.clip(raw_depth_mm / 1000.0, self.cfg.DEPTH_MIN, self.cfg.DEPTH_MAX)

        meta = {
            "file_id": os.path.basename(img_path),
            "orig_h": int(h_orig),
            "orig_w": int(w_orig),
            "scale": 1.0,
            "h": int(h_orig),
            "w": int(w_orig),
            "pad_h": 0,
            "pad_w": 0,
        }

        if is_train and self.copy_paste:
            img, lbl, depth_m, valid_mask = self.copy_paste.apply(img, lbl, depth_m, valid_mask, self._rng)

        # Geometric transforms
        if self.transform is not None:
            # Note: 'depth' is now treated as 'image' (Linear) in transforms.py
            aug = self.transform(image=img, mask=lbl, depth=depth_m, depth_valid=valid_mask)
            img = aug["image"]
            lbl = aug["mask"]
            depth_m = aug["depth"]
            valid_mask = aug["depth_valid"]

            # FIX: Explicitly treat 255 (Ignore Index from transforms) as 0 (Invalid) for Valid Mask
            if self.cfg.IGNORE_INDEX > 0:
                valid_mask[valid_mask == self.cfg.IGNORE_INDEX] = 0.0

            valid_mask = (valid_mask > 0).astype(np.float32)
            depth_m = np.clip(depth_m, self.cfg.DEPTH_MIN, self.cfg.DEPTH_MAX)

            h_new, w_new = img.shape[:2]
            meta["h"] = int(h_new)
            meta["w"] = int(w_new)
            try:
                base_h = int(getattr(self.cfg, "RESIZE_HEIGHT"))
                base_w = int(getattr(self.cfg, "RESIZE_WIDTH"))
                meta["pad_h"] = max(0, int(h_new) - base_h)
                meta["pad_w"] = max(0, int(w_new) - base_w)
            except Exception:
                meta["pad_h"] = 0
                meta["pad_w"] = 0

        # Smart crop (train only)
        if self.enable_smart_crop and is_train and (self.cfg.CROP_SIZE is not None):
            ch, cw = self.cfg.CROP_SIZE
            if img.shape[0] > ch or img.shape[1] > cw:
                img, lbl, depth_m, valid_mask = self._smart_crop(img, lbl, depth_m, valid_mask, ch, cw)
                meta["h"] = int(img.shape[0])
                meta["w"] = int(img.shape[1])
                meta["pad_h"] = 0
                meta["pad_w"] = 0

        if is_train and self.depth_dropout:
            depth_m, valid_mask = self.depth_dropout.apply(depth_m, valid_mask, self._rng)

        if (self.color_transform is not None) and is_train:
            img = self.color_transform(image=img)["image"]

        if self._sanity_remaining > 0:
            self._sanity_remaining -= 1
            if has_labels:
                m_lbl = (lbl != self.cfg.IGNORE_INDEX)
                if np.any(m_lbl):
                    mn, mx = int(np.min(lbl[m_lbl])), int(np.max(lbl[m_lbl]))
                    if mn < 0 or mx >= int(self.cfg.NUM_CLASSES):
                        raise ValueError(f"Label value out of range: {mn}-{mx}")

        depth_target = None
        depth_valid = None

        if is_train and bool(getattr(self.cfg, "USE_DEPTH_AUX", False)) and float(getattr(self.cfg, "DEPTH_LOSS_LAMBDA", 0.0)) > 0.0:
            # Prepare Aux Target (Normalized Inverse Depth)
            
            # Recalculate for target
            v_mask_t = (valid_mask > 0.5)
            m_t = v_mask_t & (depth_m > 0)
            
            inv_t = np.zeros_like(depth_m, dtype=np.float32)
            inv_t[m_t] = 1.0 / depth_m[m_t]
            
            inv_norm_t = np.zeros_like(inv_t, dtype=np.float32)
            inv_norm_t[m_t] = self._minmax01(inv_t[m_t])
            
            depth_target = torch.from_numpy(inv_norm_t).unsqueeze(0).float() # (1, H, W)
            depth_valid = torch.from_numpy(valid_mask).unsqueeze(0).float()  # (1, H, W)

        x = self._make_input(img, depth_m, valid_mask)
        y = torch.from_numpy(lbl).long()

        if depth_target is not None:
            return x, y, meta, depth_target, depth_valid
        return x, y, meta