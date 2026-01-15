import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import importlib

n01 = importlib.import_module("01_nearest.model.meta_arch")
FPNDecoder = n01.FPNDecoder


class ContextPyramid(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
            )
            for ps in pool_sizes
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_sizes) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2:]
        feats = [x]
        for block in self.proj:
            pooled = block(x)
            up = F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=False)
            feats.append(up)
        return self.fuse(torch.cat(feats, dim=1))


class ConvNeXtBaseFPNContext(nn.Module):
    """
    Model C v0.
    - Encoder: ConvNeXt Base (ImageNet pretrained).
    - Input: 6/7ch (RGB + Depth + Depth Gradients (+ Curvature)).
    - Decoder: FPN + Context Pyramid Head.
    - Optional Planar Head.
    """

    def __init__(
        self,
        num_classes,
        in_chans=7,
        fpn_channels=256,
        context_channels=256,
        pretrained=True,
        planar_head=False,
    ):
        super().__init__()
        self.planar_head_enabled = bool(planar_head)

        self.encoder = timm.create_model(
            "convnext_base",
            pretrained=bool(pretrained),
            features_only=True,
            in_chans=int(in_chans),
            out_indices=(0, 1, 2, 3),
        )
        enc_ch = list(self.encoder.feature_info.channels())

        self.decoder = FPNDecoder(in_channels=[int(c) for c in enc_ch], fpn_channels=int(fpn_channels))
        self.context = ContextPyramid(int(fpn_channels), int(context_channels))

        self.seg_head = nn.Conv2d(int(context_channels), int(num_classes), kernel_size=1)

        if self.planar_head_enabled:
            self.planar_head = nn.Conv2d(int(context_channels), 1, kernel_size=1)

    def forward(self, x):
        feats = self.encoder(x)
        dec = self.decoder(feats)
        ctx = self.context(dec)

        seg_logits = self.seg_head(ctx)
        seg_logits = F.interpolate(seg_logits, size=x.shape[2:], mode="bilinear", align_corners=False)

        if self.planar_head_enabled:
            planar_logits = self.planar_head(ctx)
            planar_logits = F.interpolate(planar_logits, size=x.shape[2:], mode="bilinear", align_corners=False)
            return seg_logits, planar_logits
        return seg_logits
