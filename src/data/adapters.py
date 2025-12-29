import torch
import numpy as np

class BaseAdapter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mean = torch.tensor(cfg.MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(cfg.STD, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, rgb, depth, valid, label=None):
        raise NotImplementedError

class RGBDAdapter(BaseAdapter):
    def __call__(self, rgb, depth, valid, label=None):
        """
        Args:
            rgb: (H, W, 3) uint8 or float32 (RGB)
            depth: (H, W) float32 (meters)
            valid: (H, W) float32 (0 or 1)
            label: (H, W) uint8 (optional)
        Returns:
            x: (4, H, W) Tensor (normalized RGB + normalized inv-depth)
            y: (H, W) Tensor (long) or None
        """
        # Normalize RGB
        img_f = rgb.astype(np.float32) / 255.0
        rgb_t = torch.from_numpy(img_f).permute(2, 0, 1).float()
        rgb_t = (rgb_t - self.mean) / self.std

        # Process Depth (Inv Depth)
        inv = np.zeros_like(depth, dtype=np.float32)
        m = (valid > 0.5) & (depth > 0)
        inv[m] = 1.0 / depth[m]

        min_inv = 1.0 / self.cfg.DEPTH_MAX
        max_inv = 1.0 / self.cfg.DEPTH_MIN
        inv_norm = np.zeros_like(inv, dtype=np.float32)
        inv_norm[m] = (inv[m] - min_inv) / (max_inv - min_inv)
        inv_norm = np.clip(inv_norm, 0.0, 1.0)
        
        d_t = torch.from_numpy(inv_norm).unsqueeze(0).float()
        
        # Concat
        x = torch.cat([rgb_t, d_t], dim=0)

        y = None
        if label is not None:
            y = torch.from_numpy(label).long()
            
        return x, y

class RGBD6chAdapter(BaseAdapter):
    def __call__(self, rgb, depth, valid, label=None):
        """
        Args:
            rgb: (H, W, 3) uint8 or float32 (RGB)
            depth: (H, W) float32 (meters)
            valid: (H, W) float32 (0 or 1)
            label: (H, W) uint8 (optional)
        Returns:
            x: (6, H, W) Tensor
                0-2: RGB (ImageNet Norm)
                3:   Inverse Depth (Linear Norm [0, 1])
                4:   Log Depth (Linear Norm [0, 1])
                5:   Valid Mask (0.0 or 1.0)
            y: (H, W) Tensor (long) or None
        """
        # Normalize RGB
        img_f = rgb.astype(np.float32) / 255.0
        rgb_t = torch.from_numpy(img_f).permute(2, 0, 1).float()
        rgb_t = (rgb_t - self.mean) / self.std

        # Process Depth
        # Strict Normalization Rules (Fixed Range 0.6 - 10.0)
        d_min = self.cfg.DEPTH_MIN # 0.6
        d_max = self.cfg.DEPTH_MAX # 10.0
        
        # Clip Depth
        d_clipped = np.clip(depth, d_min, d_max)
        
        # Valid Mask (Ensure binary)
        # Assuming input valid is float 0.0 or 1.0 or interpolated
        # Threshold just in case
        v_mask = (valid > 0.5).astype(np.float32)

        # 1. Inverse Depth
        # inv = 1/d. Range: [1/10, 1/0.6] -> [0.1, 1.666]
        inv = 1.0 / d_clipped
        inv_min = 1.0 / d_max
        inv_max = 1.0 / d_min
        inv_norm = (inv - inv_min) / (inv_max - inv_min)
        inv_norm = np.clip(inv_norm, 0.0, 1.0)
        
        # 2. Log Depth
        # log_d. Range: [log(0.6), log(10)]
        log_d = np.log(d_clipped)
        log_min = np.log(d_min)
        log_max = np.log(d_max)
        log_norm = (log_d - log_min) / (log_max - log_min)
        log_norm = np.clip(log_norm, 0.0, 1.0)
        
        # Apply Mask to Depth Channels (Force invalid to 0)
        inv_norm = inv_norm * v_mask
        log_norm = log_norm * v_mask
        
        # Stack 3 depth channels: Inv, Log, Mask
        # Shapes: (H, W) -> (1, H, W)
        d_stack = np.stack([inv_norm, log_norm, v_mask], axis=0)
        d_t = torch.from_numpy(d_stack).float()
        
        # Concat RGB + Depth Stack
        x = torch.cat([rgb_t, d_t], dim=0)

        y = None
        if label is not None:
            y = torch.from_numpy(label).long()
            
        return x, y

def get_adapter(cfg):
    if cfg.INPUT_MODE == "rgbd_4ch":
        return RGBDAdapter(cfg)
    elif cfg.INPUT_MODE == "rgbd_6ch":
        return RGBD6chAdapter(cfg)
    else:
        raise NotImplementedError(f"Adapter for {cfg.INPUT_MODE} not implemented")
