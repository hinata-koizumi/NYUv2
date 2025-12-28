import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List, Dict
import albumentations as A

class NYUDataset(Dataset):
    """
    Returns RAW data:
        - rgb: (H, W, 3) uint8 or float32 (depending on transform)
        - depth: (H, W) float32 (meters)
        - label: (H, W) uint8 or int64
        - meta: Dict with 'orig_h', 'orig_w', 'pad_h', 'pad_w', 'scale', 'file_id'
        
    Note: No Normalization or 4ch stacking here. That is handled by Adapters.
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
        adapter=None,
    ):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.depth_paths = depth_paths
        self.cfg = cfg
        self.transform = transform
        self.color_transform = color_transform
        self.enable_smart_crop = enable_smart_crop
        self.adapter = adapter

        
        # Sanity check counter
        self._sanity_remaining = int(getattr(cfg, "SANITY_CHECK_FIRST_N", 0))

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
        
    def _smart_crop_wrapper(self, img, lbl, depth_m, valid, crop_h, crop_w):
        return smart_crop(
             img, lbl, depth_m, valid, crop_h, crop_w, 
             self.cfg.SMALL_OBJ_IDS, self.cfg.SMART_CROP_PROB
        )

    def __getitem__(self, idx: int):
        img_path = str(self.image_paths[idx])
        img = self._load_rgb(img_path)
        h_orig, w_orig = img.shape[:2]
        
        # Load Label
        lbl = None
        if self.label_paths is not None:
            lbl = self._load_label(str(self.label_paths[idx]), (h_orig, w_orig))
        else:
            lbl = np.zeros((h_orig, w_orig), dtype=np.uint8)

        # Load Depth
        strict_depth = bool(getattr(self.cfg, "STRICT_DEPTH_FOR_TRAIN", False)) and (self.label_paths is not None)
        raw_depth_mm = self._load_depth_mm(
            None if self.depth_paths is None else self.depth_paths[idx],
            (h_orig, w_orig),
            strict=strict_depth,
        )
        
        # Preprocess Depth (mm -> m)
        valid_mask = (raw_depth_mm > 0).astype(np.float32)
        depth_m = np.clip(raw_depth_mm / 1000.0, self.cfg.DEPTH_MIN, self.cfg.DEPTH_MAX)

        # Apply Transforms
        # Note: 'valid_mask' and 'depth_m' are passed to albumentations
        # Albumentations expects 'mask' for label, 'image' for RGB.
        # We use 'additional_targets' for depth and depth_valid.
        
        # Initial meta
        meta = {
            "file_id": os.path.basename(img_path),
            "orig_h": int(h_orig),
            "orig_w": int(w_orig),
            "pad_h": 0,
            "pad_w": 0,
            "scale": 1.0, 
        }

        if self.transform is not None:
            # Albumentations will return 'depth_valid' as a mask (0 or 255/1 depending on processing)
            # We must be careful about interpolation of masks.
            
            aug = self.transform(image=img, mask=lbl, depth=depth_m, depth_valid=valid_mask)
            img = aug["image"]
            lbl = aug["mask"]
            depth_m = aug["depth"]
            valid_mask = aug["depth_valid"]
            
            # Post-transform fixups
            # valid_mask might be interpolated if resized? 
            # Ideally depth_valid is passed as 'mask' type to use nearest neighbor.
            valid_mask = ((valid_mask > 0.5) & (valid_mask < 1.5)).astype(np.float32)
            depth_m = np.clip(depth_m, self.cfg.DEPTH_MIN, self.cfg.DEPTH_MAX)
            
            # Update meta with new size
            h_new, w_new = img.shape[:2]
            
            # Check for padding (assuming Resize -> Pad workflow for Valid/Test)
            if self.cfg.RESIZE_HEIGHT and self.cfg.RESIZE_WIDTH:
                 # Check if we are larger than resize target (meaning we padded)
                 if h_new >= self.cfg.RESIZE_HEIGHT and w_new >= self.cfg.RESIZE_WIDTH:
                     meta["pad_h"] = int(h_new - self.cfg.RESIZE_HEIGHT)
                     meta["pad_w"] = int(w_new - self.cfg.RESIZE_WIDTH)

        # Smart Crop (Train Only)
        if (
            self.enable_smart_crop
            and (self.label_paths is not None)
            and (self.cfg.CROP_SIZE is not None)
        ):
            ch, cw = self.cfg.CROP_SIZE
            if img.shape[0] > ch or img.shape[1] > cw:
                img, lbl, depth_m, valid_mask = self._smart_crop_wrapper(
                    img, lbl, depth_m, valid_mask, ch, cw
                )

        # Color Aug (RGB Only, Train Only)
        if (self.color_transform is not None) and (self.label_paths is not None):
             img = self.color_transform(image=img)["image"]
             
        # Sanity Check
        if self._sanity_remaining > 0:
            self._sanity_remaining -= 1
            if self.label_paths is not None:
                m_lbl = (lbl != self.cfg.IGNORE_INDEX)
                if np.any(m_lbl):
                    mn, mx = int(np.min(lbl[m_lbl])), int(np.max(lbl[m_lbl]))
                    if mn < 0 or mx >= int(self.cfg.NUM_CLASSES):
                         raise ValueError(f"Label value out of range: {mn}-{mx}")

        # Return Raw or Adapted
        if self.adapter is not None:
             x, y = self.adapter(img, depth_m, valid_mask, lbl)
             return x, y, meta
             
        return img, depth_m, valid_mask, lbl, meta

import random
def smart_crop(image, label, depth, valid, crop_h, crop_w, target_ids, prob):
    """
    Stolen from base_model_093_5.py
    """
    h, w = label.shape[:2]
    max_y = h - crop_h
    max_x = w - crop_w
    
    if max_y < 0 or max_x < 0:
        top = max(0, (h - crop_h) // 2)
        left = max(0, (w - crop_w) // 2)
        sh = min(h, crop_h)
        sw = min(w, crop_w)
        return (
            image[top:top+sh, left:left+sw],
            label[top:top+sh, left:left+sw],
            depth[top:top+sh, left:left+sw],
            valid[top:top+sh, left:left+sw]
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
