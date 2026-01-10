import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import logging
from .segformer import SegFormer

logger = logging.getLogger(__name__)


class FPNDecoder(nn.Module):
    """
    Lightweight in-repo FPN decoder.

    Inputs: list[Tensor] of 4 feature maps at strides {4,8,16,32} (low->high resolution).
    Output: fused feature map at stride=4.
    """

    def __init__(self, in_channels: list[int], fpn_channels: int = 256):
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError(f"FPNDecoder expects 4 feature maps (got {len(in_channels)})")

        self.fpn_channels = int(fpn_channels)

        # 1x1 lateral projections
        self.lateral = nn.ModuleList([nn.Conv2d(int(c), self.fpn_channels, kernel_size=1) for c in in_channels])
        # 3x3 smoothing after top-down merge
        self.smooth = nn.ModuleList(
            [nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1) for _ in in_channels]
        )

        # Fuse pyramid levels by upsampling to the highest resolution (p2) and concatenating
        self.fuse = nn.Sequential(
            nn.Conv2d(self.fpn_channels * 4, self.fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.fpn_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        if len(feats) != 4:
            raise ValueError(f"Expected 4 feature maps, got {len(feats)}")

        # feats: [c2,c3,c4,c5]  (stride 4/8/16/32)
        c2, c3, c4, c5 = feats

        p5 = self.lateral[3](c5)
        p4 = self.lateral[2](c4) + F.interpolate(p5, size=c4.shape[2:], mode="nearest")
        p3 = self.lateral[1](c3) + F.interpolate(p4, size=c3.shape[2:], mode="nearest")
        p2 = self.lateral[0](c2) + F.interpolate(p3, size=c2.shape[2:], mode="nearest")

        p2 = self.smooth[0](p2)
        p3 = self.smooth[1](p3)
        p4 = self.smooth[2](p4)
        p5 = self.smooth[3](p5)

        # Upsample all to p2 resolution and fuse
        p3u = F.interpolate(p3, size=p2.shape[2:], mode="nearest")
        p4u = F.interpolate(p4, size=p2.shape[2:], mode="nearest")
        p5u = F.interpolate(p5, size=p2.shape[2:], mode="nearest")
        fused = self.fuse(torch.cat([p2, p3u, p4u, p5u], dim=1))
        return fused


class ConvNeXtBaseFPN4Ch(nn.Module):
    """
    Exp100 Final Model.
    - Encoder: ConvNeXt Base (ImageNet pretrained) with Zero-Init 4th channel.
    - Input: 4ch (RGB + InverseDepth)
    - Decoder: FPN
    - Output: segmentation logits
    """

    def __init__(
        self,
        num_classes: int,
        use_depth_aux: bool = False,
        fpn_channels: int = 256,
        *,
        pretrained: bool = True,
    ):
        super().__init__()
        self.use_depth_aux = bool(use_depth_aux)

        # timm handles in_chans != 3 with pretrained weights by adapting the stem weights.
        # features_only returns a list of stage feature maps.
        self.encoder = timm.create_model(
            "convnext_base",
            pretrained=bool(pretrained),
            features_only=True,
            in_chans=4,
            out_indices=(0, 1, 2, 3),
        )
        enc_ch = list(self.encoder.feature_info.channels())
        if len(enc_ch) != 4:
            raise ValueError(f"Unexpected ConvNeXt features_only channels: {enc_ch}")

        self.decoder = FPNDecoder(in_channels=[int(c) for c in enc_ch], fpn_channels=int(fpn_channels))

        # --- Zero Init for 4th Channel (Inverse Depth) ---
        # Instead of Mean Init (which pollutes initial features), we use Zero Init.
        # This allows the model to start training as a pure RGB model (ImageNet behavior)
        # and gradually learn to utilize depth information.
        self._init_stem_zero()

        self.seg_head = nn.Conv2d(int(fpn_channels), int(num_classes), kernel_size=1)

        if self.use_depth_aux:
            self.depth_head = nn.Conv2d(int(fpn_channels), 1, kernel_size=3, padding=1)

    def _init_stem_zero(self):
        """
        Robustly finds the first Conv2d layer in the encoder (Stem)
        and initializes the 4th channel weights to ZERO.
        """
        target_conv = None
        
        # Method: Recursively search for the first Conv2d with in_channels=4
        # This works regardless of timm version or model structure wrappers.
        for name, module in self.encoder.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.in_channels == 4:
                    target_conv = module
                    # logger.info(f"Found stem Conv2d at: {name}")
                    break
        
        if target_conv is not None:
            with torch.no_grad():
                # weight shape: [out_ch, 4, k, k]
                if target_conv.weight.shape[1] == 4:
                    target_conv.weight[:, 3:4, :, :].zero_()
                    # print("ConvNeXt stem: 4th channel initialized to ZERO.") # 必要ならログ出力
                else:
                    pass # Should not happen given the check above
        else:
            print("[WARN] Could not find stem Conv2d (in=4) to initialize 4th channel.")

    def forward(self, x: torch.Tensor):
        # x: [B, 4, H, W]
        feats = self.encoder(x)  # list[Tensor] length=4
        dec = self.decoder(feats)  # (B, fpn_channels, H/4, W/4)

        seg_logits = self.seg_head(dec)
        seg_logits = F.interpolate(seg_logits, size=x.shape[2:], mode="bilinear", align_corners=False)

        if not self.use_depth_aux:
            return seg_logits

        depth_pred = self.depth_head(dec)
        depth_pred = F.interpolate(depth_pred, size=x.shape[2:], mode="bilinear", align_corners=False)

        return seg_logits, depth_pred


def build_model(cfg) -> nn.Module:
    use_depth_aux = bool(getattr(cfg, "USE_DEPTH_AUX", False)) and float(getattr(cfg, "DEPTH_LOSS_LAMBDA", 0.0)) > 0.0
    # Training default: ImageNet pretrained encoder with Zero Init logic applied in __init__.
    # Exp101 Default: SegFormer (MiT-B3)
    return SegFormer(
        num_classes=int(cfg.NUM_CLASSES),
        phi='b3', 
        in_channels=int(getattr(cfg, "IN_CHANNELS", 4)),
        pretrained=True
    )
    # Original ConvNeXt logic preserved below for reference if needed
    # return ConvNeXtBaseFPN4Ch(num_classes=int(cfg.NUM_CLASSES), use_depth_aux=use_depth_aux, pretrained=True)


class SegFPN(ConvNeXtBaseFPN4Ch):
    """
    Backward-compatible alias for ConvNeXtBaseFPN4Ch.
    Used by existing checkpoints or submit scripts.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 4,
        use_depth_aux: bool = True,
        *,
        pretrained: bool = True,
    ):
        if int(in_channels) != 4:
            raise ValueError(f"SegFPN is fixed to in_channels=4 (got {in_channels})")
        super().__init__(num_classes=int(num_classes), use_depth_aux=bool(use_depth_aux), pretrained=bool(pretrained))