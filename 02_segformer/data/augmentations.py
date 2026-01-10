
import numpy as np
import cv2
import torch
from typing import Optional, List, Tuple

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


class CopyPasteAugmentation:
    def __init__(self, cfg, image_paths, label_paths, depth_paths, loader_impl):
        self.cfg = cfg
        self.enable = bool(getattr(cfg, "COPY_PASTE_ENABLE", False))
        self.prob = float(getattr(cfg, "COPY_PASTE_PROB", 0.0))
        self.max_objs = int(getattr(cfg, "COPY_PASTE_MAX_OBJS", 1))
        self.bg_ids = tuple(getattr(cfg, "COPY_PASTE_BG_IDS", ()))
        self.bg_min_cover = float(getattr(cfg, "COPY_PASTE_BG_MIN_COVER", 0.5))
        self.max_tries = int(getattr(cfg, "COPY_PASTE_MAX_TRIES", 20))
        self.db: List[dict] = []
        self._rng = np.random.default_rng(int(getattr(cfg, "SEED", 42)))
        self.loader_impl = loader_impl
        
        if self.enable and self.prob > 0.0:
            self._build_copy_paste_db(image_paths, label_paths, depth_paths)
            if len(self.db) == 0:
                self.enable = False

    def _build_copy_paste_db(self, image_paths, label_paths, depth_paths) -> None:
        obj_ids = list(getattr(self.cfg, "COPY_PASTE_OBJ_IDS", self.cfg.SMALL_OBJ_IDS))
        if not obj_ids or label_paths is None:
            return

        min_area = int(getattr(self.cfg, "COPY_PASTE_MIN_AREA", 0))
        max_area = int(getattr(self.cfg, "COPY_PASTE_MAX_AREA", 0))
        max_area_ratio = float(getattr(self.cfg, "COPY_PASTE_MAX_AREA_RATIO", 0.0))
        max_total = int(getattr(self.cfg, "COPY_PASTE_MAX_OBJS_TOTAL", 0))
        strict_depth = bool(getattr(self.cfg, "STRICT_DEPTH_FOR_TRAIN", False))

        for i in range(len(image_paths)):
            img_path = str(image_paths[i])
            lbl_path = None if label_paths is None else str(label_paths[i])
            dep_path = None if depth_paths is None else str(depth_paths[i])

            img = self.loader_impl._load_rgb(img_path)
            h, w = img.shape[:2]
            lbl = self.loader_impl._load_label(lbl_path, (h, w))

            raw_depth_mm = self.loader_impl._load_depth_mm(dep_path, (h, w), strict=strict_depth)
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

                    self.db.append(
                        {
                            "rgb": rgb_patch,
                            "depth": depth_patch,
                            "valid": valid_patch,
                            "mask": comp_mask.astype(np.bool_),
                            "label": int(cls_id),
                        }
                    )
                    if max_total and len(self.db) >= max_total:
                        return

    def apply(self, img, lbl, depth_m, valid_mask, rng):
        if not self.enable or not self.db:
            return img, lbl, depth_m, valid_mask
        if rng.random() >= self.prob:
            return img, lbl, depth_m, valid_mask

        max_objs = max(1, self.max_objs)
        num_objs = int(rng.integers(1, max_objs + 1))
        h, w = lbl.shape[:2]

        for _ in range(num_objs):
            obj = self.db[int(rng.integers(0, len(self.db)))]
            mask = obj["mask"]
            ph, pw = mask.shape[:2]
            if ph >= h or pw >= w:
                continue

            placed = False
            for _try in range(self.max_tries):
                top = int(rng.integers(0, h - ph + 1))
                left = int(rng.integers(0, w - pw + 1))
                if self.bg_ids:
                    target_lbl = lbl[top : top + ph, left : left + pw]
                    target_vals = target_lbl[mask]
                    if target_vals.size == 0:
                        continue
                    valid_sel = (target_vals != self.cfg.IGNORE_INDEX)
                    if not np.any(valid_sel):
                        continue
                    in_bg = np.isin(target_vals[valid_sel], self.bg_ids)
                    if float(np.mean(in_bg)) < self.bg_min_cover:
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


class DepthDropout:
    def __init__(self, cfg):
        self.cfg = cfg
        self.p_drop = float(getattr(cfg, "DEPTH_CHANNEL_DROPOUT_PROB", 0.0))
        self.p_coarse = float(getattr(cfg, "DEPTH_COARSE_DROPOUT_PROB", 0.0))

    def apply(self, depth_m, valid_mask, rng):
        if self.p_drop > 0.0 and rng.random() < self.p_drop:
            return (
                np.zeros_like(depth_m, dtype=np.float32),
                np.zeros_like(valid_mask, dtype=np.float32),
            )

        if self.p_coarse > 0.0 and rng.random() < self.p_coarse:
            depth_m, valid_mask = self._apply_depth_coarse_dropout(depth_m, valid_mask, rng)

        return depth_m, valid_mask

    def _apply_depth_coarse_dropout(self, depth_m, valid_mask, rng):
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

        num_holes = int(rng.integers(1, max_holes + 1))
        for _ in range(num_holes):
            hole_h = int(rng.integers(min_h, max_h + 1))
            hole_w = int(rng.integers(min_w, max_w + 1))
            if hole_h >= h or hole_w >= w:
                continue
            top = int(rng.integers(0, h - hole_h + 1))
            left = int(rng.integers(0, w - hole_w + 1))
            depth_m[top : top + hole_h, left : left + hole_w] = 0.0
            valid_mask[top : top + hole_h, left : left + hole_w] = 0.0

        return depth_m, valid_mask
