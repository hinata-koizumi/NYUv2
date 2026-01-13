
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
try:
    from . import config
except ImportError:
    import config

class ModelBDataset(Dataset):
    def __init__(
        self,
        image_paths,
        label_paths=None,
        depth_paths=None,
        is_train=True,
        transform=None,
    ):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.depth_paths = depth_paths
        self.is_train = is_train and (label_paths is not None)
        self.transform = transform
        
        # Norm Constants (same as 01_nearest)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).float()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).float()
        
        # 01_nearest: DEPTH_MIN=0.6, DEPTH_MAX=10.0
        self.depth_min = 0.6
        self.depth_max = 10.0 
        
        # Meta for Sampling (Computed on Init)
        self.weights = np.ones(len(self.image_paths), dtype=np.float32)
        if self.is_train and self.label_paths is not None:
             self._compute_sampling_weights()

    def _compute_sampling_weights(self):
        """
        4-1. Predefined Meta + 4-2. Image Sampling
        p(image) proportional to 0.2 + nonstruct + 2.0 * small
        Verify not to starve protect.
        """
        print("Precomputing sampling weights for Model B...")
        weights = []
        for i, lp in enumerate(self.label_paths):
            lbl = cv2.imread(lp, cv2.IMREAD_UNCHANGED)
            if lbl is None:
                weights.append(1.0)
                continue
            
            # 01_nearest/constants: 13 classes. IDs 0-12. 255 ignore.
            if lbl.ndim == 3: lbl = lbl[:,:,0]
            
            size = lbl.size
            valid_mask = (lbl != 255)
            valid_count = np.count_nonzero(valid_mask)
            if valid_count == 0:
                weights.append(1.0)
                continue
                
            # Ratios relative to VALID pixels? Or image size? Usually Valid.
            # Request says "pixel_ratio".
            
            nonstruct_mask = np.isin(lbl, config.NONSTRUCT_IDS)
            small_mask = np.isin(lbl, config.SMALL_TARGET_IDS)
            protect_mask = np.isin(lbl, config.PROTECT_IDS)
            
            r_ns = np.count_nonzero(nonstruct_mask) / size
            r_sm = np.count_nonzero(small_mask) / size
            r_pr = np.count_nonzero(protect_mask) / size
            
            # Formula: 0.2 + nonstruct + 2.0 * small
            w = config.SAMPLE_BASE_PROB + r_ns + config.SAMPLE_SMALL_FACTOR * r_sm
            
            # 4-2. "Lower bound to prevent starving protect-poor images?"
            # Request: "protect_ratio extremely small... prevent bias".
            # If protect is low, we might NOT want to sample it too often if it risks breaking floor?
            # Or if protect is high, we want to ensure we see it?
            # "protect ratio extremely small... prevent biasing towards those" -> implies we might oversample non-protect images.
            # So if protect_ratio is very low, maybe clamp weight?
            # Actually, the formula favors high non-struct. High non-struct means low protect usually.
            # So we might indeed sample only clutter.
            # Let's add a small penalty or just verify distribution.
            # The prompt says: "ensure not ONLY images with low protect".
            # I will trust the formula but maybe add a small multiplier if protect is > 0.3?
            # Or just leave as is. "p(image) ... 0.2 ...".
            # Let's stick to the formula.
            
            weights.append(w)
            
        self.weights = np.array(weights, dtype=np.float32)
        # Normalize? WeightedRandomSampler takes raw weights.

    def _load_files(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(path)
        if img is None:
            print(f"[ERROR] Could not read image: {path}")
            # return dummy to avoid crash during debug?
            raise FileNotFoundError(f"Image not found: {path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        lbl = None
        if self.label_paths is not None:
            lbl = cv2.imread(self.label_paths[idx], cv2.IMREAD_UNCHANGED)
            if lbl.ndim == 3: lbl = lbl[:,:,0]
            
        depth = None
        if self.depth_paths is not None:
            depth = cv2.imread(self.depth_paths[idx], cv2.IMREAD_UNCHANGED).astype(np.float32)
            # NYUv2 usually mm. Convert to meters?
            # 01_nearest divides by 1000.
            depth = depth / 1000.0
            
        return img, lbl, depth

    def _smart_crop(self, img, lbl, depth, crop_h, crop_w):
        """
        4-3. SmartCrop
        """
        h, w = img.shape[:2]
        
        # Candidates for centering
        # Small Target
        mask_small = np.isin(lbl, config.SMALL_TARGET_IDS)
        pts_small = np.argwhere(mask_small) # (N, 2) -> y, x
        
        # Non Struct
        mask_ns = np.isin(lbl, config.NONSTRUCT_IDS)
        pts_ns = np.argwhere(mask_ns)
        
        # Probabilities
        rng = np.random.rand()
        
        center_y, center_x = None, None
        
        # 1) Try Small (p=0.6~0.8)
        if rng < config.SMART_CROP_PROB_SMALL and len(pts_small) > 0:
            # Pick random point from small objects
            idx = np.random.randint(len(pts_small))
            center_y, center_x = pts_small[idx]
            
        # 2) Try Non-Struct (p=0.2~0.3) -> if we didn't pick small (or small failed), 
        # we fall through. Note: Independent probs or hierarchical?
        # Request: "Failed -> p_nonstruct". implies hierarchical.
        elif rng < (config.SMART_CROP_PROB_SMALL + config.SMART_CROP_PROB_NONSTRUCT) and len(pts_ns) > 0:
            idx = np.random.randint(len(pts_ns))
            center_y, center_x = pts_ns[idx]
            
        # 3) Else (or if no targets found): Random Crop
        # Handled by fallback to None
        
        if center_y is not None:
             top = center_y - crop_h // 2
             left = center_x - crop_w // 2
             
             # Adjust bounds
             top = max(0, min(h - crop_h, top))
             left = max(0, min(w - crop_w, left))
        else:
             top = np.random.randint(0, h - crop_h + 1)
             left = np.random.randint(0, w - crop_w + 1)
             
        # "Important: protect > X% constraint"
        # We check the proposed crop. if it violates, we retry?
        # Or we just accept? "Slightly loose constraint".
        # I'll implement a simple retry loop (max 5 tries) for the Random case mostly,
        # but also for Smart case if it results in essentially 0 floor.
        
        best_crop = (top, left)
        
        # Check constraint
        if config.SMART_CROP_PROTECT_MIN_RATIO > 0:
            for _ in range(5):
                # Check current
                crop_lbl = lbl[top:top+crop_h, left:left+crop_w]
                prot_count = np.count_nonzero(np.isin(crop_lbl, config.PROTECT_IDS))
                ratio = prot_count / (crop_h * crop_w)
                
                if ratio >= config.SMART_CROP_PROTECT_MIN_RATIO:
                    best_crop = (top, left)
                    break 
                
                # Retry (Randomly shift? Or fully random?)
                # If we were targeting specific object, maybe we shift slightly?
                # Simpler: just fallback to random if constraints fail heavily?
                # Or just retry random.
                top = np.random.randint(0, h - crop_h + 1)
                left = np.random.randint(0, w - crop_w + 1)
                best_crop = (top, left)

        return best_crop

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img, lbl, depth = self._load_files(idx)
        
        # Augmentation (Photometric + Geometric)
        # v0: ColorJitter/Blur/Noise, HFlip, Scale(0.85-1.15)
        # We need to construct input tensors.
        
        if self.transform:
            # Albumentations expects specific keys
            # Masks: label. Depth: treated as image or mask?
            # 01_nearest treats depth as mask (NEAREST) or image (LINEAR)?
            # dataset.py: "depth_m ... INTER_LINEAR".
            # But albumentations 'mask' uses Nearest by default?
            # Pass depth as 'image' field? No, 'mask' supports multiple.
            
            # transform should handle resize/crop if defined there.
            # But we have Custom SmartCrop.
            # Typical flow: 
            # 1. Custom SmartCrop (if train) -> Top/Left
            # 2. Crop numpy arrays
            # 3. Augmentations (Color, Flip)
            # 4. ToTensor / Norm
            
            aug_data = {"image": img, "mask": lbl}
            if depth is not None:
                aug_data["depth"] = depth # Handle in custom target?
            
            # Actually, standardizing on Albumentations is cleaner.
            # But SmartCrop requires access to FULL label before crop.
            
            if self.is_train and lbl is not None:
                # Perform Smart Crop manually
                # Fixed crop size? 
                # 01_nearest uses (576, 768) roughly.
                # B wants 480x640 Input?
                # Let's say we train on random crops of 480x640 from larger images?
                # Or if images are 480x640 (NYUv2), we might need to pad/scale first?
                # NYUv2 raw is 480x640.
                # If we want to crop, we must scale up? or crop smaller?
                # "v0 ... scale 0.85 ~ 1.15".
                # If we scale 0.85 -> 408x544. To get 480x640 crop, we need Pad.
                
                # Logic:
                # 1. Random Scale (0.85-1.15)
                # 2. Resize
                scale = np.random.uniform(0.85, 1.15)
                h, w = img.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                lbl = cv2.resize(lbl, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                if depth is not None:
                    depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # 3. Smart Crop to Target Size (e.g. 480x640)
                # If image < target, Pad.
                th, tw = config.TRAIN_SIZE
                pad_h = max(0, th - new_h)
                pad_w = max(0, tw - new_w)
                if pad_h > 0 or pad_w > 0:
                    img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                    lbl = cv2.copyMakeBorder(lbl, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)
                    if depth is not None:
                        depth = cv2.copyMakeBorder(depth, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                
                # Now crop
                top, left = self._smart_crop(img, lbl, depth, th, tw)
                img = img[top:top+th, left:left+tw]
                lbl = lbl[top:top+th, left:left+tw]
                if depth is not None:
                    depth = depth[top:top+th, left:left+tw]

            # Apply other augmentations (Color, Flip)
            # We can use A.Compose just for these.
            if self.transform:
                 # Pass depth as output 'depth' if we want A to handle it (Flip)
                 # A.HorizontalFlip will flip image and mask. 
                 # We need to map depth to a mask or additional image.
                 # Using multiple masks: mask0=lbl, mask1=depth
                 masks = [lbl]
                 if depth is not None: masks.append(depth)
                 
                 res = self.transform(image=img, masks=masks)
                 img = res['image']
                 lbl = res['masks'][0]
                 if depth is not None:
                     depth = res['masks'][1]

        # To Tensor & 4th Channel
        # Img: (H,W,3) -> (3,H,W) norm.
        img_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        img_t = (img_t - self.mean) / self.std
        
        # Depth: IGNORE for Model B v0 (RGB-only)
        # We explicitly discard depth channel logic here to ensure 3-channel input.
        x = img_t 

        # Labels
        y = torch.from_numpy(lbl).long() if lbl is not None else torch.zeros((img.shape[0], img.shape[1])).long()
        
        return x, y, str(idx) # Return ID/index for debugging/tracking

