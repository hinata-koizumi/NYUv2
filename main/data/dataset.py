import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List
import albumentations as A


def smart_crop(image, label, depth, valid, crop_h, crop_w, target_ids, prob, rng: Optional[np.random.Generator] = None):
    """
    Crop (crop_h, crop_w).
    With probability=prob, try to ensure the crop contains at least one pixel from target_ids.
    Fallback: random crop. If image is smaller than crop, center-crop the available region.
    """
    rng = rng or np.random.default_rng()

    h, w = label.shape[:2]
    max_y = h - crop_h
    max_x = w - crop_w

    # If smaller than crop size: center crop within available bounds
    if max_y < 0 or max_x < 0:
        top = max(0, (h - crop_h) // 2)
        left = max(0, (w - crop_w) // 2)
        sh = min(h, crop_h)
        sw = min(w, crop_w)
        return (
            image[top : top + sh, left : left + sw],
            label[top : top + sh, left : left + sw],
            depth[top : top + sh, left : left + sw],
            valid[top : top + sh, left : left + sw],
        )

    top = left = None

    if (rng.random() < prob) and target_ids:
        mask = np.isin(label, target_ids)
        if mask.any():
            ys, xs = np.where(mask)
            k = int(rng.integers(0, len(ys)))
            cy, cx = int(ys[k]), int(xs[k])

            min_t = max(0, cy - crop_h + 1)
            max_t = min(max_y, cy)
            min_l = max(0, cx - crop_w + 1)
            max_l = min(max_x, cx)

            if min_t <= max_t and min_l <= max_l:
                top = int(rng.integers(min_t, max_t + 1))
                left = int(rng.integers(min_l, max_l + 1))

    if top is None:
        top = int(rng.integers(0, max_y + 1))
        left = int(rng.integers(0, max_x + 1))

    return (
        image[top : top + crop_h, left : left + crop_w],
        label[top : top + crop_h, left : left + crop_w],
        depth[top : top + crop_h, left : left + crop_w],
        valid[top : top + crop_h, left : left + crop_w],
    )


class NYUDataset(Dataset):
    """
    Returns model-ready tensors:
        - x: (4, H, W) float32 (normalized RGB + normalized inverse depth)
        - y: (H, W) int64 label (zeros for test)
        - meta: Dict with 'orig_h', 'orig_w', 'scale', 'file_id', 'h', 'w' (post-transform)
        - depth_target/depth_valid (optional) for depth auxiliary loss
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

        # RNG (single source; avoids np/random mixing)
        self._rng = np.random.default_rng(int(getattr(cfg, "SEED", 42)))

        # Fixed 4ch normalization/constants
        self._rgb_mean = torch.tensor(cfg.MEAN, dtype=torch.float32).view(3, 1, 1)
        self._rgb_std = torch.tensor(cfg.STD, dtype=torch.float32).view(3, 1, 1)
        self._inv_min = 1.0 / float(cfg.DEPTH_MAX)
        self._inv_max = 1.0 / float(cfg.DEPTH_MIN)
        self._inv_denom = float(self._inv_max - self._inv_min) or 1.0
        self._depth_denom = float(cfg.DEPTH_MAX - cfg.DEPTH_MIN) or 1.0

        # Copy-paste augmentation cache (train only)
        self._copy_paste_enable = bool(getattr(cfg, "COPY_PASTE_ENABLE", False)) and self.is_train
        self._copy_paste_prob = float(getattr(cfg, "COPY_PASTE_PROB", 0.0))
        self._copy_paste_max_objs = int(getattr(cfg, "COPY_PASTE_MAX_OBJS", 1))
        self._copy_paste_bg_ids = tuple(getattr(cfg, "COPY_PASTE_BG_IDS", ()))
        self._copy_paste_bg_min_cover = float(getattr(cfg, "COPY_PASTE_BG_MIN_COVER", 0.5))
        self._copy_paste_max_tries = int(getattr(cfg, "COPY_PASTE_MAX_TRIES", 20))
        self._copy_paste_db: List[dict] = []
        if self._copy_paste_enable and self._copy_paste_prob > 0.0:
            self._build_copy_paste_db()
            if len(self._copy_paste_db) == 0:
                self._copy_paste_enable = False

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_rgb(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_label(self, path: Optional[str], shape_hw: Tuple[int, int]) -> np.ndarray:
        if not path:
            return np.zeros(shape_hw, dtype=np.uint8)
        lbl = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if lbl is None:
            raise FileNotFoundError(f"Label not found: {path}")
        if lbl.ndim == 3:
            lbl = lbl[:, :, 0]
        return lbl

    def _load_depth_mm(self, path: Optional[str], shape_hw: Tuple[int, int], *, strict: bool) -> np.ndarray:
        if not path:
            return np.zeros(shape_hw, dtype=np.float32)
        d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if d is None:
            if strict:
                raise FileNotFoundError(f"Depth not found or unreadable: {path}")
            return np.zeros(shape_hw, dtype=np.float32)
        return d.astype(np.float32)

    def _norm_rgb(self, rgb: np.ndarray) -> torch.Tensor:
        """
        rgb: (H, W, 3) uint8 or float32 in [0,255] or [0,1]
        returns: (3, H, W) float32 normalized by cfg.MEAN/STD
        """
        img_f = rgb.astype(np.float32)
        if img_f.max() > 1.5:
            img_f = img_f / 255.0
        rgb_t = torch.from_numpy(img_f).permute(2, 0, 1).contiguous().float()
        return (rgb_t - self._rgb_mean) / self._rgb_std

    def _minmax01(self, x: np.ndarray) -> np.ndarray:
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
        return torch.cat([rgb_t, d_t], dim=0)

    def _resize_crop_to_target(self, img, lbl, depth_m, valid, out_h: int, out_w: int):
        if img.shape[0] == out_h and img.shape[1] == out_w:
            return img, lbl, depth_m, valid

        img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        lbl = cv2.resize(lbl, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        depth_m = cv2.resize(depth_m, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        valid = cv2.resize(valid, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        valid = (valid > 0).astype(np.float32)
        return img, lbl, depth_m, valid

    def _build_copy_paste_db(self) -> None:
        obj_ids = list(getattr(self.cfg, "COPY_PASTE_OBJ_IDS", self.cfg.SMALL_OBJ_IDS))
        if not obj_ids or self.label_paths is None:
            return

        min_area = int(getattr(self.cfg, "COPY_PASTE_MIN_AREA", 0))
        max_area = int(getattr(self.cfg, "COPY_PASTE_MAX_AREA", 0))
        max_area_ratio = float(getattr(self.cfg, "COPY_PASTE_MAX_AREA_RATIO", 0.0))
        max_total = int(getattr(self.cfg, "COPY_PASTE_MAX_OBJS_TOTAL", 0))
        strict_depth = bool(getattr(self.cfg, "STRICT_DEPTH_FOR_TRAIN", False))

        for i in range(len(self.image_paths)):
            img_path = str(self.image_paths[i])
            lbl_path = None if self.label_paths is None else str(self.label_paths[i])
            dep_path = None if self.depth_paths is None else str(self.depth_paths[i])

            img = self._load_rgb(img_path)
            h, w = img.shape[:2]
            lbl = self._load_label(lbl_path, (h, w))

            raw_depth_mm = self._load_depth_mm(dep_path, (h, w), strict=strict_depth)
            valid_mask = (raw_depth_mm > 0).astype(np.float32)
            depth_m = np.clip(raw_depth_mm / 1000.0, self.cfg.DEPTH_MIN, self.cfg.DEPTH_MAX)

            max_area_pix = None
            if max_area_ratio > 0.0:
                max_area_pix = int(max_area_ratio * h * w)
            if max_area > 0:
                max_area_pix = max_area if max_area_pix is None else min(max_area_pix, max_area)

            for cls_id in obj_ids:
                mask = (lbl == cls_id)
                if not mask.any():
                    continue
                num, comp_map, stats, _centroids = cv2.connectedComponentsWithStats(
                    mask.astype(np.uint8), connectivity=8
                )
                for comp_id in range(1, num):
                    area = int(stats[comp_id, cv2.CC_STAT_AREA])
                    if min_area and area < min_area:
                        continue
                    if max_area_pix is not None and area > max_area_pix:
                        continue

                    left = int(stats[comp_id, cv2.CC_STAT_LEFT])
                    top = int(stats[comp_id, cv2.CC_STAT_TOP])
                    bw = int(stats[comp_id, cv2.CC_STAT_WIDTH])
                    bh = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
                    if bw <= 0 or bh <= 0:
                        continue

                    comp_mask = (comp_map[top : top + bh, left : left + bw] == comp_id)
                    if not comp_mask.any():
                        continue

                    rgb_patch = img[top : top + bh, left : left + bw].copy()
                    depth_patch = depth_m[top : top + bh, left : left + bw].copy()
                    valid_patch = valid_mask[top : top + bh, left : left + bw].copy()

                    self._copy_paste_db.append(
                        {
                            "rgb": rgb_patch,
                            "depth": depth_patch,
                            "valid": valid_patch,
                            "mask": comp_mask.astype(np.bool_),
                            "label": int(cls_id),
                        }
                    )
                    if max_total and len(self._copy_paste_db) >= max_total:
                        return

    def _apply_copy_paste(self, img, lbl, depth_m, valid_mask):
        if not self._copy_paste_enable or not self._copy_paste_db:
            return img, lbl, depth_m, valid_mask
        if self._rng.random() >= self._copy_paste_prob:
            return img, lbl, depth_m, valid_mask

        max_objs = max(1, self._copy_paste_max_objs)
        num_objs = int(self._rng.integers(1, max_objs + 1))
        h, w = lbl.shape[:2]

        for _ in range(num_objs):
            obj = self._copy_paste_db[int(self._rng.integers(0, len(self._copy_paste_db)))]
            mask = obj["mask"]
            ph, pw = mask.shape[:2]
            if ph >= h or pw >= w:
                continue

            placed = False
            for _try in range(self._copy_paste_max_tries):
                top = int(self._rng.integers(0, h - ph + 1))
                left = int(self._rng.integers(0, w - pw + 1))
                if self._copy_paste_bg_ids:
                    target_lbl = lbl[top : top + ph, left : left + pw]
                    target_vals = target_lbl[mask]
                    if target_vals.size == 0:
                        continue
                    valid_sel = (target_vals != self.cfg.IGNORE_INDEX)
                    if not np.any(valid_sel):
                        continue
                    in_bg = np.isin(target_vals[valid_sel], self._copy_paste_bg_ids)
                    if float(np.mean(in_bg)) < self._copy_paste_bg_min_cover:
                        continue
                placed = True
                break

            if not placed:
                continue

            region_img = img[top : top + ph, left : left + pw]
            region_lbl = lbl[top : top + ph, left : left + pw]
            region_depth = depth_m[top : top + ph, left : left + pw]
            region_valid = valid_mask[top : top + ph, left : left + pw]

            region_img[mask] = obj["rgb"][mask]
            region_lbl[mask] = obj["label"]
            region_depth[mask] = obj["depth"][mask]
            region_valid[mask] = obj["valid"][mask]

        return img, lbl, depth_m, valid_mask

    def _apply_depth_dropout(self, depth_m, valid_mask):
        p_drop = float(getattr(self.cfg, "DEPTH_CHANNEL_DROPOUT_PROB", 0.0))
        if p_drop > 0.0 and self._rng.random() < p_drop:
            return (
                np.zeros_like(depth_m, dtype=np.float32),
                np.zeros_like(valid_mask, dtype=np.float32),
            )

        p_coarse = float(getattr(self.cfg, "DEPTH_COARSE_DROPOUT_PROB", 0.0))
        if p_coarse > 0.0 and self._rng.random() < p_coarse:
            depth_m, valid_mask = self._apply_depth_coarse_dropout(depth_m, valid_mask)

        return depth_m, valid_mask

    def _apply_depth_coarse_dropout(self, depth_m, valid_mask):
        max_holes = int(getattr(self.cfg, "DEPTH_COARSE_DROPOUT_MAX_HOLES", 0))
        if max_holes <= 0:
            return depth_m, valid_mask

        h, w = depth_m.shape[:2]
        min_frac = float(getattr(self.cfg, "DEPTH_COARSE_DROPOUT_MIN_FRAC", 0.0))
        max_frac = float(getattr(self.cfg, "DEPTH_COARSE_DROPOUT_MAX_FRAC", 0.0))
        if max_frac <= 0.0:
            return depth_m, valid_mask

        min_frac = max(0.0, min(1.0, min_frac))
        max_frac = max(min_frac, min(1.0, max_frac))
        min_h = max(1, int(round(h * min_frac)))
        max_h = max(min_h, int(round(h * max_frac)))
        min_w = max(1, int(round(w * min_frac)))
        max_w = max(min_w, int(round(w * max_frac)))

        num_holes = int(self._rng.integers(1, max_holes + 1))
        for _ in range(num_holes):
            hole_h = int(self._rng.integers(min_h, max_h + 1))
            hole_w = int(self._rng.integers(min_w, max_w + 1))
            if hole_h >= h or hole_w >= w:
                continue
            top = int(self._rng.integers(0, h - hole_h + 1))
            left = int(self._rng.integers(0, w - hole_w + 1))
            depth_m[top : top + hole_h, left : left + hole_w] = 0.0
            valid_mask[top : top + hole_h, left : left + hole_w] = 0.0

        return depth_m, valid_mask

    def _smart_crop(self, img, lbl, depth_m, valid, crop_h, crop_w):
        # 5% chance: confusion focus on Sofa(3)/Bed(4) after epoch 5
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

    def __getitem__(self, idx: int):
        has_labels = self.label_paths is not None
        is_train = bool(self.is_train)

        img_path = str(self.image_paths[idx])
        img = self._load_rgb(img_path)
        h_orig, w_orig = img.shape[:2]

        # Label (NYUv2 labels are already 0-indexed + ignore (255))
        lbl = self._load_label(None if not has_labels else self.label_paths[idx], (h_orig, w_orig))

        # Depth
        strict_depth = bool(getattr(self.cfg, "STRICT_DEPTH_FOR_TRAIN", False)) and is_train
        raw_depth_mm = self._load_depth_mm(
            None if self.depth_paths is None else self.depth_paths[idx],
            (h_orig, w_orig),
            strict=strict_depth,
        )

        # Depth preprocess (mm -> m)
        valid_mask = (raw_depth_mm > 0).astype(np.float32)
        depth_m = np.clip(raw_depth_mm / 1000.0, self.cfg.DEPTH_MIN, self.cfg.DEPTH_MAX)

        # Meta (keep it minimal + reliable)
        meta = {
            "file_id": os.path.basename(img_path),
            "orig_h": int(h_orig),
            "orig_w": int(w_orig),
            "scale": 1.0,
            "h": int(h_orig),
            "w": int(w_orig),
            # padding at right/bottom only (top_left). Used by validation/inference unpad logic.
            "pad_h": 0,
            "pad_w": 0,
        }

        # Copy-paste (train only, before geometric transforms)
        # CHECK: CopyPaste runs BEFORE geometric transforms and SmartCrop.
        # CHECK: CopyPaste implementation (_apply_copy_paste) handles RGB(3ch) + Depth(1ch) = 4ch correctly.
        if is_train and self._copy_paste_enable:
            img, lbl, depth_m, valid_mask = self._apply_copy_paste(img, lbl, depth_m, valid_mask)

        # Geometric transforms (RGB/Label/Depth/Valid)
        if self.transform is not None:
            aug = self.transform(image=img, mask=lbl, depth=depth_m, depth_valid=valid_mask)
            img = aug["image"]
            lbl = aug["mask"]
            depth_m = aug["depth"]
            valid_mask = aug["depth_valid"]

            # Robust binarization for {0,1} or {0,255} masks
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
                # crop removes padding (if any) and changes spatial dims
                meta["h"] = int(img.shape[0])
                meta["w"] = int(img.shape[1])
                meta["pad_h"] = 0
                meta["pad_w"] = 0

        # Depth dropout (train only, after crop)
        if is_train:
            depth_m, valid_mask = self._apply_depth_dropout(depth_m, valid_mask)

        # Color aug (RGB only, train only)
        if (self.color_transform is not None) and is_train:
            img = self.color_transform(image=img)["image"]

        # Sanity check (first N samples)
        if self._sanity_remaining > 0:
            self._sanity_remaining -= 1
            if has_labels:
                m_lbl = (lbl != self.cfg.IGNORE_INDEX)
                if np.any(m_lbl):
                    mn, mx = int(np.min(lbl[m_lbl])), int(np.max(lbl[m_lbl]))
                    if mn < 0 or mx >= int(self.cfg.NUM_CLASSES):
                        raise ValueError(f"Label value out of range: {mn}-{mx}")

        x = self._make_input(img, depth_m, valid_mask)
        y = torch.from_numpy(lbl).long()

        # Depth aux target (optional)
        depth_target = None
        depth_valid_t = None
        if bool(getattr(self.cfg, "USE_DEPTH_AUX", False)) and float(getattr(self.cfg, "DEPTH_LOSS_LAMBDA", 0.0)) > 0.0:
            depth_target_np = ((depth_m - float(self.cfg.DEPTH_MIN)) / self._depth_denom).astype(np.float32)
            depth_valid_np = (valid_mask > 0).astype(np.float32)
            depth_target_np *= depth_valid_np

            depth_target = torch.from_numpy(depth_target_np).unsqueeze(0).float()
            depth_valid_t = torch.from_numpy(depth_valid_np).unsqueeze(0).float()

        if depth_target is not None:
            return x, y, meta, depth_target, depth_valid_t
        return x, y, meta
