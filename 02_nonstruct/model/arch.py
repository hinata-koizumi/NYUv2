
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import importlib

# Reuse FPNDecoder from 01_nearest
# We use importlib to handle the numeric module name
n01 = importlib.import_module("01_nearest.model.meta_arch")
FPNDecoder = n01.FPNDecoder

class ConvNeXtBaseFPN3Ch(nn.Module):
    """
    Model B v0 (RGB-only).
    - Encoder: ConvNeXt Base (ImageNet pretrained).
    - Input: 3ch (RGB) - No depth.
    - Decoder: FPN (from 01_nearest).
    - Output: segmentation logits.
    """

    def __init__(
        self,
        num_classes: int,
        fpn_channels: int = 256,
        pretrained: bool = True,
    ):
        super().__init__()
        
        # Encoder: 3ch input
        self.encoder = timm.create_model(
            "convnext_base",
            pretrained=bool(pretrained),
            features_only=True,
            in_chans=3,
            out_indices=(0, 1, 2, 3),
        )
        enc_ch = list(self.encoder.feature_info.channels())
        
        # Decoder: FPN
        self.decoder = FPNDecoder(in_channels=[int(c) for c in enc_ch], fpn_channels=int(fpn_channels))
        
        # Segmentation Head
        self.seg_head = nn.Conv2d(int(fpn_channels), int(num_classes), kernel_size=1)
        
        # No Aux heads for Model B v0

    def forward(self, x: torch.Tensor):
        # x: [B, 3, H, W]
        feats = self.encoder(x)  # list[Tensor] length=4
        dec = self.decoder(feats)  # (B, fpn_channels, H/4, W/4)

        seg_logits = self.seg_head(dec)
        seg_logits = F.interpolate(seg_logits, size=x.shape[2:], mode="bilinear", align_corners=False)

        return seg_logits
