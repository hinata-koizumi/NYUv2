import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from configs import default as config
except ImportError:
    import configs.default as config


class ModelCDataset(Dataset):
    def __init__(
        self,
        image_paths,
        label_paths=None,
        depth_paths=None,
        is_train=True,
        ids=None,
    ):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.depth_paths = depth_paths
        self.is_train = is_train and (label_paths is not None)
        self.ids = ids

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).float()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).float()

        self.depth_min = float(config.DEPTH_MIN)
        self.depth_max = float(config.DEPTH_MAX)
        self.depth_range = float(self.depth_max - self.depth_min) or 1.0

        self.use_hard_mask = bool(getattr(config, "USE_HARD_MASK", False))
        self.hard_mask_dir = getattr(config, "HARD_MASK_DIR", None)

    def __len__(self):
        return len(self.image_paths)

    def _load_files(self, idx):
        img = cv2.imread(self.image_paths[idx])
        if img is None:
            raise FileNotFoundError(f"Image not found: {self.image_paths[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lbl = None
        if self.label_paths is not None:
            lbl = cv2.imread(self.label_paths[idx], cv2.IMREAD_UNCHANGED)
            if lbl is None:
                raise FileNotFoundError(f"Label not found: {self.label_paths[idx]}")
            if lbl.ndim == 3:
                lbl = lbl[:, :, 0]

        depth = None
        if self.depth_paths is not None:
            depth = cv2.imread(self.depth_paths[idx], cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise FileNotFoundError(f"Depth not found: {self.depth_paths[idx]}")
            depth = depth.astype(np.float32) / 1000.0

        return img, lbl, depth

    def _load_hard_mask(self, idx):
        if not self.use_hard_mask or not self.hard_mask_dir or self.ids is None:
            return None
        fid = str(self.ids[idx])
        mask_path = os.path.join(self.hard_mask_dir, f"{fid}.npz")
        if not os.path.exists(mask_path):
            return None
        obj = np.load(mask_path)
        hard_books = obj.get("hard_books")
        hard_table = obj.get("hard_table")
        if hard_books is None or hard_table is None:
            return None
        return hard_books.astype(bool), hard_table.astype(bool)

    def _random_scale(self, img, lbl, depth):
        scale = np.random.uniform(config.TRAIN_SCALE_MIN, config.TRAIN_SCALE_MAX)
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if lbl is not None:
            lbl = cv2.resize(lbl, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        if depth is not None:
            depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return img, lbl, depth

    def _pad_to_size(self, img, lbl, depth, out_h, out_w):
        h, w = img.shape[:2]
        pad_h = max(0, out_h - h)
        pad_w = max(0, out_w - w)
        if pad_h == 0 and pad_w == 0:
            return img, lbl, depth
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        if lbl is not None:
            lbl = cv2.copyMakeBorder(lbl, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)
        if depth is not None:
            depth = cv2.copyMakeBorder(depth, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        return img, lbl, depth

    def _depth_in_range(self, depth, dmin, dmax):
        return (depth >= dmin) & (depth <= dmax)

    def _sample_from_mask(self, mask):
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return None
        k = np.random.randint(0, len(ys))
        return int(ys[k]), int(xs[k])

    def _sample_books_center(self, lbl, depth):
        mask = (lbl == config.CLASS_ID_BOOKS) & self._depth_in_range(
            depth, *config.BOOKS_DEPTH_RANGE
        )
        return self._sample_from_mask(mask)

    def _sample_table_center(self, lbl, depth):
        mask_near = (lbl == config.CLASS_ID_TABLE) & self._depth_in_range(
            depth, *config.TABLE_DEPTH_RANGE_NEAR
        )
        mask_far = (lbl == config.CLASS_ID_TABLE) & self._depth_in_range(
            depth, *config.TABLE_DEPTH_RANGE_FAR
        )

        if np.random.rand() < config.TABLE_NEAR_PROB and mask_near.any():
            return self._sample_from_mask(mask_near)
        if mask_far.any():
            return self._sample_from_mask(mask_far)
        if mask_near.any():
            return self._sample_from_mask(mask_near)
        return None

    def _sample_target_center(self, lbl, depth):
        pick_books = np.random.rand() < config.TARGETED_BOOKS_PROB
        tries = int(getattr(config, "TARGETED_SAMPLE_TRIES", 2))
        for _ in range(max(1, tries)):
            if pick_books:
                center = self._sample_books_center(lbl, depth)
            else:
                center = self._sample_table_center(lbl, depth)
            if center is not None:
                return center
            pick_books = not pick_books
        return None

    def _sample_hard_center(self, hard_books, hard_table):
        pick_books = np.random.rand() < float(getattr(config, "HARD_MASK_BOOKS_PROB", 0.5))
        tries = int(getattr(config, "HARD_SAMPLE_TRIES", 2))
        for _ in range(max(1, tries)):
            mask = hard_books if pick_books else hard_table
            center = self._sample_from_mask(mask)
            if center is not None:
                return center
            pick_books = not pick_books
        return None

    def _smart_crop(self, img, lbl, depth, crop_h, crop_w, hard_masks=None):
        h, w = img.shape[:2]
        if h <= crop_h or w <= crop_w:
            return 0, 0

        center = None
        if self.is_train and (np.random.rand() < config.TARGETED_CROP_PROB):
            use_hard = (
                hard_masks is not None
                and np.random.rand() < float(getattr(config, "HARD_CROP_PROB", 0.0))
            )
            if use_hard:
                center = self._sample_hard_center(hard_masks[0], hard_masks[1])
            if center is None:
                center = self._sample_target_center(lbl, depth)

        if center is None:
            top = np.random.randint(0, h - crop_h + 1)
            left = np.random.randint(0, w - crop_w + 1)
            return top, left

        cy, cx = center
        top = int(np.clip(cy - crop_h // 2, 0, h - crop_h))
        left = int(np.clip(cx - crop_w // 2, 0, w - crop_w))
        return top, left

    def _hflip(self, img, lbl, depth):
        if np.random.rand() > config.HFLIP_PROB:
            return img, lbl, depth
        img = np.ascontiguousarray(img[:, ::-1])
        if lbl is not None:
            lbl = np.ascontiguousarray(lbl[:, ::-1])
        if depth is not None:
            depth = np.ascontiguousarray(depth[:, ::-1])
        return img, lbl, depth

    def _augment_depth(self, depth):
        if not self.is_train or depth is None:
            return depth
        depth = depth.copy()
        valid = depth > 0

        if config.DEPTH_SCALE_JITTER > 0:
            scale = 1.0 + np.random.uniform(-config.DEPTH_SCALE_JITTER, config.DEPTH_SCALE_JITTER)
            depth[valid] = depth[valid] * scale

        if config.DEPTH_NOISE_STD > 0:
            noise = np.random.normal(0.0, config.DEPTH_NOISE_STD, depth.shape).astype(np.float32)
            depth[valid] = depth[valid] + noise[valid]

        if config.DEPTH_DROPOUT_PROB > 0:
            drop_mask = np.random.rand(*depth.shape) < config.DEPTH_DROPOUT_PROB
            depth[drop_mask] = 0

        if config.DEPTH_DROPOUT_BLOCKS > 0:
            h, w = depth.shape[:2]
            for _ in range(config.DEPTH_DROPOUT_BLOCKS):
                if np.random.rand() < 0.5:
                    continue
                bh = np.random.randint(config.DEPTH_DROPOUT_BLOCK_SIZE[0], config.DEPTH_DROPOUT_BLOCK_SIZE[1] + 1)
                bw = np.random.randint(config.DEPTH_DROPOUT_BLOCK_SIZE[0], config.DEPTH_DROPOUT_BLOCK_SIZE[1] + 1)
                if bh >= h or bw >= w:
                    continue
                y0 = np.random.randint(0, h - bh + 1)
                x0 = np.random.randint(0, w - bw + 1)
                depth[y0:y0 + bh, x0:x0 + bw] = 0

        if config.DEPTH_QUANT_STEP > 0:
            depth[valid] = np.round(depth[valid] / config.DEPTH_QUANT_STEP) * config.DEPTH_QUANT_STEP

        valid = depth > 0
        depth[valid] = np.clip(depth[valid], self.depth_min, self.depth_max)
        return depth

    def _compute_depth_features(self, depth):
        valid = depth > 0
        depth = depth.copy()
        depth[valid] = np.clip(depth[valid], self.depth_min, self.depth_max)

        depth_norm = np.zeros_like(depth, dtype=np.float32)
        depth_norm[valid] = (depth[valid] - self.depth_min) / self.depth_range

        dy, dx = np.gradient(depth_norm.astype(np.float32))
        dx = np.clip(dx * config.DEPTH_GRAD_SCALE, -1.0, 1.0)
        dy = np.clip(dy * config.DEPTH_GRAD_SCALE, -1.0, 1.0)

        curv = cv2.Laplacian(depth_norm, cv2.CV_32F, ksize=3)
        curv = np.abs(curv) * config.DEPTH_CURV_SCALE
        curv = np.clip(curv, 0.0, 1.0)

        depth_norm[~valid] = 0
        dx[~valid] = 0
        dy[~valid] = 0
        curv[~valid] = 0

        planar_target = (curv < config.PLANAR_CURV_THRESH) & valid
        planar_target = planar_target.astype(np.float32)
        planar_valid = valid.astype(np.float32)

        return depth_norm, dx, dy, curv, planar_target, planar_valid

    def __getitem__(self, idx):
        img, lbl, depth = self._load_files(idx)
        hard_masks = self._load_hard_mask(idx)

        if self.is_train:
            img, lbl, depth = self._random_scale(img, lbl, depth)
            img, lbl, depth = self._pad_to_size(img, lbl, depth, *config.TRAIN_SIZE)

            top, left = self._smart_crop(img, lbl, depth, *config.TRAIN_SIZE, hard_masks=hard_masks)
            img = img[top:top + config.TRAIN_SIZE[0], left:left + config.TRAIN_SIZE[1]]
            if lbl is not None:
                lbl = lbl[top:top + config.TRAIN_SIZE[0], left:left + config.TRAIN_SIZE[1]]
            if depth is not None:
                depth = depth[top:top + config.TRAIN_SIZE[0], left:left + config.TRAIN_SIZE[1]]

            img, lbl, depth = self._hflip(img, lbl, depth)
        else:
            if img.shape[:2] != tuple(config.TRAIN_SIZE):
                img = cv2.resize(img, (config.TRAIN_SIZE[1], config.TRAIN_SIZE[0]), interpolation=cv2.INTER_LINEAR)
                if lbl is not None:
                    lbl = cv2.resize(lbl, (config.TRAIN_SIZE[1], config.TRAIN_SIZE[0]), interpolation=cv2.INTER_NEAREST)
                if depth is not None:
                    depth = cv2.resize(depth, (config.TRAIN_SIZE[1], config.TRAIN_SIZE[0]), interpolation=cv2.INTER_LINEAR)

        depth = self._augment_depth(depth)
        depth_norm, dx, dy, curv, planar_target, planar_valid = self._compute_depth_features(depth)

        img_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        img_t = (img_t - self.mean) / self.std

        depth_t = torch.from_numpy(depth_norm).unsqueeze(0)
        dx_t = torch.from_numpy(dx).unsqueeze(0)
        dy_t = torch.from_numpy(dy).unsqueeze(0)
        feats = [img_t, depth_t, dx_t, dy_t]
        if config.USE_CURVATURE:
            feats.append(torch.from_numpy(curv).unsqueeze(0))
        x = torch.cat(feats, dim=0).contiguous()

        if lbl is None:
            lbl = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        y = torch.from_numpy(lbl).long()

        if config.PLANAR_HEAD_ENABLE:
            planar_target_t = torch.from_numpy(planar_target).unsqueeze(0)
            planar_valid_t = torch.from_numpy(planar_valid).unsqueeze(0)
        else:
            planar_target_t = None
            planar_valid_t = None

        id_val = self.ids[idx] if self.ids is not None else str(idx)
        if config.PLANAR_HEAD_ENABLE:
            return x, y, id_val, planar_target_t, planar_valid_t
        return x, y, id_val
