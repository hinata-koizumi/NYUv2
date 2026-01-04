
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, head_dim=32, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, head_dim, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = nn.Identity() # Placeholder for DropPath
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x_reshaped = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], H, W)
        x = x + self.drop_path(self.mlp(x_reshaped).flatten(2).transpose(1, 2))
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.depths = depths
        self.embed_dims = embed_dims

        # Patch Embeddings
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # Encoder Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], head_dim=embed_dims[0] // num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], head_dim=embed_dims[1] // num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], head_dim=embed_dims[2] // num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], head_dim=embed_dims[3] // num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

    def _init_weights_custom(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            nn.init.normal_(m.weight, std=math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self):
        self.apply(self._init_weights_custom)

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # Stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x_out = self.norm1(x).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x_out)

        # Stage 2
        x, H, W = self.patch_embed2(x_out)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x_out = self.norm2(x).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x_out)

        # Stage 3
        x, H, W = self.patch_embed3(x_out)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x_out = self.norm3(x).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x_out)

        # Stage 4
        x, H, W = self.patch_embed4(x_out)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x_out = self.norm4(x).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x_out)

        return outs

    def forward(self, x):
        return self.forward_features(x)

class SegFormerHead(nn.Module):
    def __init__(self, in_channels, embedding_dim=768, num_classes=13):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = Mlp(c4_in_channels, c4_in_channels, embedding_dim)
        self.linear_c3 = Mlp(c3_in_channels, c3_in_channels, embedding_dim)
        self.linear_c2 = Mlp(c2_in_channels, c2_in_channels, embedding_dim)
        self.linear_c1 = Mlp(c1_in_channels, c1_in_channels, embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim*4, embedding_dim, kernel_size=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        self.cls = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, features):
        c1, c2, c3, c4 = features

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,3,1).permute(0,3,1,2)
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,3,1).permute(0,3,1,2)
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,3,1).permute(0,3,1,2)
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,3,1).permute(0,3,1,2)
        
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.cls(x)
        return x

class SegFormer(nn.Module):
    def __init__(self, num_classes=13, phi='b3', in_channels=4, pretrained=True):
        super().__init__()
        self.in_channels = in_channels
        
        # MiT-B3 Config
        if phi == 'b3':
            embed_dims = [64, 128, 320, 512]
            depths = [3, 4, 18, 3]
            num_heads = [1, 2, 5, 8]  # head_dim = [64, 64, 64, 64]
            sr_ratios = [8, 4, 2, 1]
            drop_path_rate = 0.1
        else:
            raise NotImplementedError(f"SegFormer phi={phi} not implemented.")

        self.backbone = MixVisionTransformer(
            in_chans=3, # Initialize as 3ch to load ImageNet weights properly later if needed
            embed_dims=embed_dims,
            num_heads=num_heads,
            depths=depths,
            sr_ratios=sr_ratios,
            drop_path_rate=drop_path_rate
        )
        # Initialize weights (generic)
        self.backbone.init_weights()

        # Handle 4th channel zero init
        # PatchEmbed1 is the entry point.
        # It's a Conv2d(3, 64, kernel=7, stride=4, padding=3)
        # We need to extend it to 4 channels.
        old_proj = self.backbone.patch_embed1.proj
        new_proj = nn.Conv2d(in_channels, embed_dims[0], kernel_size=7, stride=4, padding=3)
        
        # Copy weights from RGB
        with torch.no_grad():
            new_proj.weight[:, :3] = old_proj.weight
            if in_channels > 3:
                new_proj.weight[:, 3:].zero_() # Zero Init
            new_proj.bias = old_proj.bias
        
        self.backbone.patch_embed1.proj = new_proj
        
        self.head = SegFormerHead(in_channels=embed_dims, num_classes=num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return logits

def segformer_mit_b3(num_classes, in_channels=4, pretrained=True):
    return SegFormer(num_classes=num_classes, phi='b3', in_channels=in_channels, pretrained=pretrained)
