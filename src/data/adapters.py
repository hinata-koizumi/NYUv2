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

def get_adapter(cfg):
    if cfg.INPUT_MODE == "rgbd_4ch":
        return RGBDAdapter(cfg)
    else:
        raise NotImplementedError(f"Adapter for {cfg.INPUT_MODE} not implemented")
