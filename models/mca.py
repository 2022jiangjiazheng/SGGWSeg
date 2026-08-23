# Project repository: https://github.com/2022jiangjiazheng

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath


"""Multi-scale Context Aggregation module (MCA)."""

class IRB(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, ksize=3, act_layer=nn.GELU,
                 extra_act=False, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.act = act_layer()
        self.conv = nn.Conv2d(hidden_features, hidden_features, kernel_size=ksize, padding=ksize // 2, stride=1,
                              groups=hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.conv(x)
        x = self.fc2(x)
        return x


class PoolingAttention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 pooled_sizes=[1, 2, 3, 6], q_pooled_size=-1, **kwargs):

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.num_elements = np.array([t * t for t in pooled_sizes]).sum()
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Conv2d(dim, dim, 1, 1, 0, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pooled_sizes = pooled_sizes
        self.pools = nn.ModuleList()
        self.eps = 0.001

        self.norm = nn.LayerNorm(dim)

        self.q_pooled_size = q_pooled_size
        self.upsample = nn.Upsample(scale_factor=q_pooled_size) if q_pooled_size > 1 else nn.Identity()

    def forward(self, x, d_convs=None):
        B, C, H, W = x.shape
        H, W = int(H), int(W)

        if d_convs is None:
            d_convs = [None] * len(self.pooled_sizes)

        if self.q_pooled_size > -1:
            q_pooled_size = self.q_pooled_size
            q_pooled_size = (q_pooled_size, round(W * float(q_pooled_size) / H + self.eps)) if W >= H else (
                round(H * float(q_pooled_size) / W + self.eps), q_pooled_size)
            q = F.adaptive_avg_pool2d(x, q_pooled_size)
            _, _, H1, W1 = q.shape
            q = self.q(q).reshape(B, self.num_heads, C // self.num_heads, -1).permute(0, 1, 3,
                                                                                      2).contiguous()  # B, N, C
        else:
            q = self.q(x).reshape(B, self.num_heads, C // self.num_heads, -1).permute(0, 1, 3,
                                                                                      2).contiguous()  # B, N, C

        pools = []
        for (pooled_size, l) in zip(self.pooled_sizes, d_convs):
            pooled_size = (pooled_size, round(W * pooled_size / H + self.eps)) if W >= H else (
                round(H * pooled_size / W + self.eps), pooled_size)
            pool = F.adaptive_avg_pool2d(x, pooled_size)
            if l is not None:
                pool = pool + l(pool)  # fix backward bug in higher torch versions when training
            pools.append(pool.view(B, C, -1))

        pools = torch.cat(pools, dim=2)
        pools = self.norm(pools.permute(0, 2, 1))

        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B Head N N
        attn = attn.softmax(dim=-1)
        x = (attn @ v)  # B Head N C
        x = x.transpose(1, 2).reshape(B, -1, C)

        # post conv seem not good: 83.4, id 9677
        # if self.q_conv is not None:
        #    qpe = self.q_conv(q.view(B, -1, C).transpose(1, 2).reshape(B, C, H1, W1))
        #    x = qpe.view(B, C, -1).transpose(1,2) + x

        x = self.proj(x)

        if self.q_pooled_size > -1:
            x = x.transpose(1, 2).reshape(B, C, H1, W1)
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        else:
            x = x.transpose(1, 2).reshape(B, C, H, W)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_cfg=None, ls=True, pooled_sizes=[12, 16, 20, 24],
                 q_pooled_size=1, q_conv=False, extra_act=False):
        super().__init__()
        # self.norm1 = nn.LayerNorm(dim)
        self.norm1 = nn.GroupNorm(1, dim)
        # _, self.norm1 = build_norm_layer(norm_cfg, dim)
        self.attn = PoolingAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, pooled_sizes=pooled_sizes, q_pooled_size=q_pooled_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # self.norm2 = nn.LayerNorm(dim)
        self.norm2 = nn.GroupNorm(1, dim)
        # _, self.norm2 = build_norm_layer(norm_cfg, dim)
        self.mlp = IRB(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop, ksize=3,
                       extra_act=extra_act)

        self.cpe = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.ls = ls  # layer scale
        if self.ls:
            layer_scale_init_value = 1e-6
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, d_convs=None):
        x = self.cpe(x) + x

        if self.ls:
            x = x + self.drop_path(self.layer_scale_1[None, :, None, None] * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2[None, :, None, None] * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

# Basic components used by MCA.

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class ConvModule(nn.Module):
    """Conv-SyncBN-ReLU block matching the original decoder behavior."""

    def __init__(self, in_ch, out_ch, k, s=1, p=0, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=bias)
        self.bn = nn.SyncBatchNorm(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))



class MCAModule(nn.Module):
    """Aggregate multi-scale decoder features and feed them back residually."""

    def __init__(self, embed_dim, feat_proj_dim, align_corners=False):
        super().__init__()
        self.align_corners = align_corners



        # Downsampling paths used before pixel unshuffle.
        self.conv_downsample_2 = ConvModule(embed_dim, embed_dim * 2, k=3, s=2, p=1)
        self.conv_downsample_4 = ConvModule(embed_dim, embed_dim * 4, k=5, s=4, p=1)

        # Channel projections for the three aligned scales.
        self.reduce_channels = nn.ModuleList([
            ConvModule(embed_dim * 4, embed_dim, k=1),  # x_4
            ConvModule(embed_dim * 8, embed_dim, k=1),  # x_2_unshuffle
            ConvModule(embed_dim * 16, embed_dim, k=1)  # x_unshuffle
        ])

        # Core attention block.
        q_pooled_size = 32
        base_pooled_sizes = [11, 8, 6, 4]  # basic ratio corresponding q_pooled_size=16
        pooled_sizes = [round(temp * q_pooled_size / 16. + 0.001) for temp in base_pooled_sizes]
        self.mca_inner = Block(embed_dim * 3, num_heads=8, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                 drop=0., attn_drop=0.,
                                 drop_path=0., pooled_sizes=pooled_sizes,
                                 q_pooled_size=q_pooled_size)

        # Local shortcut and global image-level context paths.
        self.short_path = ConvModule(embed_dim, embed_dim, k=1)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(embed_dim, embed_dim, k=1)
        )
        self.proj_out = ConvModule(embed_dim * 3, feat_proj_dim, k=1)

        # Final projection after concatenating all context features.
        self.cat = ConvModule(embed_dim * 5 + feat_proj_dim * 3, embed_dim, k=1)

    def forward(self, x, c2, c3, c4):


        _, _, H, W = x.shape

        # 1. Local and image-level context.
        out_base = [
            self.short_path(x),
            F.interpolate(self.image_pool(x), size=(H, W), mode='bilinear', align_corners=self.align_corners)
        ]

        # 2. Multi-scale pixel unshuffle.
        x_2 = self.conv_downsample_2(x)
        x_4 = self.conv_downsample_4(x)
        x_2_un = F.pixel_unshuffle(x_2, 2)
        x_un = F.pixel_unshuffle(x, 4)

        # 3. Multi-scale aggregation and attention.
        m_x = torch.cat([
            self.reduce_channels[2](x_un),
            self.reduce_channels[1](x_2_un),
            self.reduce_channels[0](x_4)
        ], dim=1)

        v_out = self.mca_inner(m_x)
        v_out_up = F.interpolate(v_out, size=(H, W), mode='bilinear', align_corners=self.align_corners)

        # 4. Residual feedback to the decoder features.
        v_feedback = self.proj_out(v_out_up)
        c2, c3, c4 = c2 + v_feedback, c3 + v_feedback, c4 + v_feedback

        # 5. Concatenate and project the aggregated features.
        final_list = out_base + [v_out_up, c2, c3, c4]
        return self.cat(torch.cat(final_list, dim=1))
