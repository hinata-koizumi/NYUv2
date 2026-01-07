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


def dynamic_resize_train(img, lbl, depth, valid, scale: float):
    """
    Resize all inputs by scale factor.
    """
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Image: Linear
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Label: Nearest
    lbl = cv2.resize(lbl, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    # Depth: Linear (Match Transforms valid pipeline)
    depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Valid: Nearest (Strict 0/1)
    valid = cv2.resize(valid, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    return img, lbl, depth, valid


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

        # Copy-paste augmentation setup
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
        # SAFETY: Ensure RGB for pretrained models
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

        # Explicitly enforce NEAREST for Valid masks.
        # Use LINEAR for Depth to match transforms pipeline (treats depth as image)
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
                    comp_id_val = int(cls_id)
                    # Books Specific Filtering (Exp-B)
                    # Reduce noise by filtering very small or very large book fragments
                    books_id = int(getattr(self.cfg, "CLASS_ID_BOOKS", 1))
                    enable_exp_b = bool(getattr(self.cfg, "ENABLE_BOOKS_IMP", True))
                    
                    if enable_exp_b and comp_id_val == books_id:
                        if area < 50:  # stricter min area for books
                            continue
                        # Optional: max area for books?
                        # if area > 5000: continue 

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
            
            # Books Random Scaling (Exp-B)
            # Resize the patch to be smaller (simulate books further away or smaller books)
            books_id = int(getattr(self.cfg, "CLASS_ID_BOOKS", 1))
            enable_exp_b = bool(getattr(self.cfg, "ENABLE_BOOKS_IMP", True))

            if enable_exp_b and obj["label"] == books_id:
                 # Scale 0.3 to 0.8
                 s_book = float(self._rng.uniform(0.3, 0.8))
                 
                 # Resize patch components
                 # obj["rgb"] is (H, W, 3)
                 rgb_p = cv2.resize(obj["rgb"], None, fx=s_book, fy=s_book, interpolation=cv2.INTER_LINEAR)
                 # Masks: Nearest
                 mask_p = cv2.resize(obj["mask"].astype(np.uint8), None, fx=s_book, fy=s_book, interpolation=cv2.INTER_NEAREST).astype(bool)
                 depth_p = cv2.resize(obj["depth"], None, fx=s_book, fy=s_book, interpolation=cv2.INTER_NEAREST)
                 valid_p = cv2.resize(obj["valid"], None, fx=s_book, fy=s_book, interpolation=cv2.INTER_NEAREST)
                 
                 # Update patch dims
                 ph, pw = mask_p.shape[:2]
                 
                 # If resize made it too small/large or invalid, safer to skip or check
                 if ph < 1 or pw < 1:
                     continue
                 
                 # Override for this placement attempt (we don't modify DB in place ideally, 
                 # but here we extracted variables. accessing obj[...] is DB access.)
                 # We use local vars for pasting
                 p_rgb, p_mask, p_depth, p_valid = rgb_p, mask_p, depth_p, valid_p
            else:
                 p_rgb, p_mask, p_depth, p_valid = obj["rgb"], obj["mask"], obj["depth"], obj["valid"]

            for _try in range(self._copy_paste_max_tries):
                if ph >= h or pw >= w:
                    break # Patch too big for image

                top = int(self._rng.integers(0, h - ph + 1))
                left = int(self._rng.integers(0, w - pw + 1))
                if self._copy_paste_bg_ids:
                    target_lbl = lbl[top : top + ph, left : left + pw]
                    target_vals = target_lbl[p_mask] # Use scaled mask
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

            region_img[p_mask] = p_rgb[p_mask]
            region_lbl[p_mask] = obj["label"]
            region_depth[p_mask] = p_depth[p_mask]
            region_valid[p_mask] = p_valid[p_mask]

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
        curr_epoch = int(getattr(self, "current_epoch", 0))
        focus_sofa_bed = curr_epoch > 5 and self._rng.random() < 0.05
        if focus_sofa_bed:
            target_ids, prob = [3, 4], 1.0
        else:
            target_ids, prob = list(self.cfg.SMALL_OBJ_IDS), float(self.cfg.SMART_CROP_PROB)
            
            # Table Bonus Logic (Exp-A)
            # Check if table (ID 9) exists in label, boost probability if so.
            table_id = int(getattr(self.cfg, "CLASS_ID_TABLE", 9))
            bonus = float(getattr(self.cfg, "SMART_CROP_TABLE_BONUS_PROB", 0.0))
            if bonus > 0.0 and (table_id in target_ids):
                # We need to know if table is actually in the image to apply bonus effectively?
                # smart_crop function checks intersection with target_ids.
                # Here we just boost 'prob' if table is potentially interesting.
                # But 'prob' is "probability to TRY smart crop".
                # If we want to prioritize table, we should ensure smart crop triggers.
                # Let's perform a lightweight check if we can.
                if np.any(lbl == table_id):
                     prob = min(1.0, prob + bonus)

        # Dynamic Resize Logic usually happens before this if we are resizing the whole image.
        # But `_smart_crop` method definition: takes `img, lbl...`.
        # The caller `__getitem__` invokes `_smart_crop`.
        
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

        if is_train and self._copy_paste_enable:
            img, lbl, depth_m, valid_mask = self._apply_copy_paste(img, lbl, depth_m, valid_mask)
        
        # --- Multi-scale Training (Exp-A) ---
        # Dynamic Resize: scale ~ U[0.75, 1.0]
        # Then we either pad (if too small) or crop (via smart crop).
        # We assume original images are large enough (approx 480x640) vs CROP_SIZE (576x768).
        # Wait, NYUv2 avg size is 480x640. CROP_SIZE is larger (576x768)?
        # Ah, config RESIZE_HEIGHT=720, WIDTH=960. 
        # So we were resizing UP to 720x960, then cropping 576x768.
        # So inputs to __getitem__ are original size (~480x640).
        # We MUST resize to at least target scale.
        
        if is_train:
            # Dynamic Random Scale for Resize
            # Base target: 720x960
            base_h_tgt = int(getattr(self.cfg, "RESIZE_HEIGHT", 720))
            base_w_tgt = int(getattr(self.cfg, "RESIZE_WIDTH", 960))
            
            # Sample scale factor
            # We want "visual scale" of 0.75 to 1.0.
            # Meaning: The objects should look smaller (0.75) or normal (1.0).
            # To make objects look smaller, we render the scene at a smaller resolution relative to the crop?
            # Or we resize the image to be smaller?
            # If we resize image to 0.75x (540x720), and crop 576x768, we might need padding.
            
            s_min = float(getattr(self.cfg, "DYNAMIC_RESIZE_MIN_SCALE", 0.75))
            s_max = 1.0
            scale = float(self._rng.uniform(s_min, s_max))
            
            # Calculate target dimensions
            target_h = int(base_h_tgt * scale)
            target_w = int(base_w_tgt * scale)
            
            # Resize everything to this target size
            # Note: We are resizing from ORIG (480x640) to TARGET (e.g. 540x720 or 720x960).
            # This is technically "Pre-Resize".
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            lbl = cv2.resize(lbl, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            depth_m = cv2.resize(depth_m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            valid_mask = cv2.resize(valid_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            valid_mask = (valid_mask > 0).astype(np.float32)

            meta["scale"] = scale
            meta["h"] = target_h
            meta["w"] = target_w
            
        else:
            # Validation / Test: Fixed Resize if not handled by transforms
            # In current setup, val transforms HAVE A.Resize.
            # So we pass original image to transform.
            pass

        # Geometric transforms
        if self.transform is not None:
            # Validation transform includes Resize.
            # Train transform NO LONGER includes Resize (we removed it).
            # So for Train, we pass the dynamically resized image.
            # For Valid, we pass original (or pre-processed) image, and A.Resize handles it.
            
            # Note: 'depth' is now treated as 'mask' in transforms.py, so it uses NEAREST.
            aug = self.transform(image=img, mask=lbl, depth=depth_m, depth_valid=valid_mask)
            img = aug["image"]
            lbl = aug["mask"]
            depth_m = aug["depth"]
            valid_mask = aug["depth_valid"]

            # FIX: Explicitly treat 255 (Ignore Index from transforms) as 0 (Invalid) for Valid Mask
            # Albumentations fills borders with one value. For Label it's 255 (OK), 
            # but for Valid Mask we want 0.
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
            # If image matches crop size (rare with random resize), we might skip?
            # SmartCrop handles "smaller than crop" by center cropping/padding logic if implemented,
            # OR typically we assume image >= crop.
            # If dynamic resize made it smaller than crop (e.g. 0.75 * 720 = 540 < 576),
            # smart_crop needs to handle it (pad).
            # existing smart_crop logic in dataset.py: 
            # "If smaller than crop size: center crop within available bounds" -> It returns smaller image!
            # We need to Ensure Output is CROP_SIZE.
            # Let's check smart_crop implementation... 
            # It returns min(h, crop_h). 
            # So we need to Pad if result is small.
            
            img, lbl, depth_m, valid_mask = self._smart_crop(img, lbl, depth_m, valid_mask, ch, cw)
            
            # Auto-Pad if result is smaller than crop
            h_c, w_c = img.shape[:2]
            if h_c < ch or w_c < cw:
                pad_h = max(0, ch - h_c)
                pad_w = max(0, cw - w_c)
                # Pad Right/Bottom
                img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
                lbl = cv2.copyMakeBorder(lbl, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.cfg.IGNORE_INDEX)
                depth_m = cv2.copyMakeBorder(depth_m, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0.0)
                valid_mask = cv2.copyMakeBorder(valid_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0.0)

            # Update meta?
            meta["h"] = int(img.shape[0])
            meta["w"] = int(img.shape[1])
            meta["pad_h"] = 0
            meta["pad_w"] = 0

        if is_train:
            depth_m, valid_mask = self._apply_depth_dropout(depth_m, valid_mask)

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
            # Create a fresh mask/target from the already augmented depth_m
            # Note: _make_input already computed inv_norm for the 4th channel, but we need it as a target.
            
            # Recalculate for target (similar to _make_input logic)
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