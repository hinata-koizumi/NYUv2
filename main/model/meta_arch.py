import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


def _init_4ch_stem(stem_conv: nn.Conv2d) -> None:
    """
    stem_conv.weight shape == [out_ch, 4, k, k] を想定し、
    4ch目を mean(RGB) で初期化する（既存の3ch ImageNet重みを活かす）。
    """
    if not isinstance(stem_conv, nn.Conv2d):
        return
    if int(stem_conv.weight.shape[1]) != 4:
        return
    with torch.no_grad():
        rgb_w = stem_conv.weight[:, :3, :, :]
        mean_w = rgb_w.mean(dim=1)  # [out, k, k]
        stem_conv.weight[:, 3, :, :] = mean_w


class FPNConvNeXt4Ch(nn.Module):
    """
    Fixed: ResNet50 + FPN, input=4ch, output=seg logits.
    Optional: depth auxiliary head (returns (seg_logits, depth_pred)).
    """
    def __init__(self, num_classes: int, use_depth_aux: bool = False):
        super().__init__()
        self.use_depth_aux = bool(use_depth_aux)

        self.net = smp.FPN(
            # NOTE: SMP 0.3.3 + tu-convnext_* has a 0-channel out_channels entry which breaks FPN decoder.
            # This repo is "fixed" (no architecture options); use a stable encoder.
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=4,
            encoder_depth=5,
            classes=num_classes,
        )

        # Initialize 4th channel from RGB weights (when encoder_weights is used).
        enc = self.net.encoder
        if hasattr(enc, "conv1") and isinstance(enc.conv1, nn.Conv2d):
            _init_4ch_stem(enc.conv1)
        elif hasattr(enc, "model") and hasattr(enc.model, "stem_0"):
            # Fallback for some timm-style encoders
            _init_4ch_stem(enc.model.stem_0)

        if self.use_depth_aux:
            dec_ch = int(self.net.decoder.out_channels)
            self.depth_head = nn.Conv2d(dec_ch, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        # x: [B, 4, H, W]
        if not self.use_depth_aux:
            logits = self.net(x)
            return F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)

        feats = self.net.encoder(x)
        dec = self.net.decoder(feats)

        seg_logits = self.net.segmentation_head(dec)
        seg_logits = F.interpolate(seg_logits, size=x.shape[2:], mode="bilinear", align_corners=False)

        depth_pred = self.depth_head(dec)
        depth_pred = F.interpolate(depth_pred, size=x.shape[2:], mode="bilinear", align_corners=False)

        return seg_logits, depth_pred


def build_model(cfg) -> nn.Module:
    use_depth_aux = bool(getattr(cfg, "USE_DEPTH_AUX", False)) and float(getattr(cfg, "DEPTH_LOSS_LAMBDA", 0.0)) > 0.0
    return FPNConvNeXt4Ch(num_classes=int(cfg.NUM_CLASSES), use_depth_aux=use_depth_aux)


class SegFPN(FPNConvNeXt4Ch):
    """
    Backward-compatible name used by `main/cli.py` submit path.

    This repository is fixed to RGB+InvDepth 4ch input. We keep the older
    constructor signature but enforce in_channels==4.
    """

    def __init__(self, num_classes: int, in_channels: int = 4, use_depth_aux: bool = True):
        if int(in_channels) != 4:
            raise ValueError(f"SegFPN is fixed to in_channels=4 (got {in_channels})")
        super().__init__(num_classes=int(num_classes), use_depth_aux=bool(use_depth_aux))
