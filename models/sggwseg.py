# Project repository: https://github.com/2022jiangjiazheng
"""SGGWSeg network architecture and Lightning training hooks.

The network uses a SegMAN-S encoder, multi-scale feature fusion, MCA context
aggregation, SGEM skeleton guidance, and separate segmentation/skeleton heads.
"""

import math
from pathlib import Path
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from models.mca import MCAModule, ConvModule
from models.sgem import SGEM
from models.segman_encoder import SegMANEncoder_s


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

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
    


class ResizeLike(nn.Module):
    """Resize a tensor to match a reference feature map."""

    def __init__(self, mode='bilinear', align_corners=False):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x, ref):
        return F.interpolate(
            x,
            size=ref.shape[2:],
            mode=self.mode,
            align_corners=self.align_corners
        ).contiguous()

class SGGWSegBase(pl.LightningModule):
    """Core SGGWSeg architecture with shared Lightning train/eval behavior."""

    def __init__(self, hparams, metric, in_channel=3, num_classes=2, drop_rate=0.2, backbone='small'):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.metric = metric
        self.in_channel = in_channel
        self.n_channels_of_input = in_channel
        self.num_classes = num_classes
        self.drop = nn.Dropout2d(drop_rate)
        # Lightning 2.x epoch hooks no longer receive step outputs.
        self._train_step_outputs = []
        self._validation_step_outputs = []
        self._test_step_outputs = []

        # ------------------------------------------------------------------
        # SegMAN-S encoder
        # ------------------------------------------------------------------
        if backbone == 'small':
            self.backbone = SegMANEncoder_s(image_size=512)
            self.backbone_dim = [64, 144, 288, 512]
            self.embed_dim = 144
            self.feat_proj_dim = 288

        # Load encoder weights. Relative paths are resolved from project_dir.
        default_pretrained_path = Path(__file__).resolve().parents[1] / "pretrained" / "SegMAN_Encoder_s.pth.tar"
        pretrained_path = Path(self.hparams.get("pretrained_path", default_pretrained_path)).expanduser()
        if not pretrained_path.is_absolute():
            pretrained_path = Path(self.hparams.get("project_dir", ".")) / pretrained_path
        if not pretrained_path.is_file():
            raise FileNotFoundError(f"SegMAN encoder weights not found: {pretrained_path}")
        save_model = torch.load(pretrained_path, map_location="cpu")

        # SegMAN checkpoints may wrap parameters in ``state_dict_ema``.
        if 'state_dict_ema' in save_model:
            pretrained_dict = save_model['state_dict_ema']
        else:
            pretrained_dict = save_model

        model_dict = self.backbone.state_dict()

        # Keep matching parameters and resize relative-position bias tables
        # when the checkpoint and configured image sizes differ.
        state_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if v.shape != model_dict[k].shape:
                    if "rpb" in k:
                        print(f"Interpolating mismatched RPB {k}: {v.shape} -> {model_dict[k].shape}")
                        # [heads, h, w] -> [1, heads, h, w] for interpolation.
                        v_resized = F.interpolate(
                            v.unsqueeze(0),
                            size=model_dict[k].shape[1:],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                        state_dict[k] = v_resized
                    else:
                        print(f"Warning: skipped mismatched parameter {k}: {v.shape} vs {model_dict[k].shape}")
                else:
                    state_dict[k] = v

        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        print(f"--- Loaded SegMAN pretrained weights: {pretrained_path} ---")

        # ------------------------------------------------------------------
        # Multi-scale decoder feature projection and attention
        # ------------------------------------------------------------------
        self.fuse4 = nn.Conv2d(self.backbone_dim[3], self.feat_proj_dim, kernel_size=1)  # 512->288
        self.fuse3 = nn.Conv2d(self.backbone_dim[2], self.feat_proj_dim, kernel_size=1)  # 288->288
        self.fuse2 = nn.Conv2d(self.backbone_dim[1], self.feat_proj_dim, kernel_size=1)  # 144->288
        
        q_pooled_size = 32
        base_pooled_sizes = [11, 8, 6, 4]  # basic ratio corresponding q_pooled_size=16
        pooled_sizes = [round(temp * q_pooled_size / 16. + 0.001) for temp in base_pooled_sizes]
        self.fuse_block4 = nn.ModuleList([
            Block(self.feat_proj_dim, num_heads=8, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                 drop=0., attn_drop=0.,
                                 drop_path=0., pooled_sizes=pooled_sizes,
                                 q_pooled_size=q_pooled_size)
            for _ in range(3)
        ])
        self.fuse_block3 = nn.ModuleList([
            Block(self.feat_proj_dim, num_heads=8, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                 drop=0., attn_drop=0.,
                                 drop_path=0., pooled_sizes=pooled_sizes,
                                 q_pooled_size=q_pooled_size)
            for _ in range(3)
        ])
        self.fuse_block2 = nn.ModuleList([
            Block(self.feat_proj_dim, num_heads=8, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                 drop=0., attn_drop=0.,
                                 drop_path=0., pooled_sizes=pooled_sizes,
                                 q_pooled_size=q_pooled_size)
            for _ in range(3)
        ])

        # ASPP bottleneck for the skeleton branch.
        self.bottleneck = ASPP(self.backbone_dim[3], [2, 4, 8, 16], self.feat_proj_dim)
        self.up = ResizeLike()

        # Semantic multi-scale fusion.
        self.linear_fuse = ConvModule(
            in_ch=self.feat_proj_dim * 3,
            out_ch=self.embed_dim,
            k=1,
            bias=False
        )

        # Skeleton-guidance multi-scale fusion.
        self.skl_linear_fuse = ConvModule(
            in_ch=self.feat_proj_dim * 4,
            out_ch=self.embed_dim,
            k=1,
            bias=False
        )

        self.mca = MCAModule(embed_dim=self.embed_dim, feat_proj_dim=self.feat_proj_dim)
        self.sgem = SGEM(semantic_dim=self.embed_dim, skeleton_dim=self.embed_dim, out_dim=self.embed_dim)

        # Segmentation and skeleton prediction heads.
        self.mask_head = nn.Conv2d(self.embed_dim, self.num_classes, 1)
        self.skl_head1 = nn.Conv2d(self.embed_dim, 1, 1)
        self.skl_head2 = nn.Conv2d(1, 1, kernel_size=15, stride=1, padding=7)

    def forward(self, x):
        """Return ``[segmentation_logits, skeleton_logits]``."""
        input_size = x.shape[2:]

        # 1. Extract four encoder stages.
        encoder_features = self.backbone(x)
        x1 = encoder_features[1]  # [B, 64, H/4, W/4]
        x2 = encoder_features[3]  # [B, 144, H/8, W/8]
        x2 = self.drop(x2)
        x3 = encoder_features[5]  # [B, 288, H/16, W/16]
        x3 = self.drop(x3)
        x4 = encoder_features[7]  # [B, 512, H/32, W/32]
        x4 = self.drop(x4)

        # 2. Preserve the skeleton-branch inputs before channel projection.
        skl_x2 = x2
        skl_x3 = x3
        skl_x4 = x4
        skl_bottleneck = self.bottleneck(skl_x4)

        # 3. Project all decoder features to feat_proj_dim channels.
        x2 = self.fuse2(x2)
        x3 = self.fuse3(x3)
        x4 = self.fuse4(x4)
        skl_x2 = self.fuse2(skl_x2)
        skl_x3 = self.fuse3(skl_x3)
        skl_x4 = self.fuse4(skl_x4)

        # 4. Align spatial sizes at the x2 resolution.
        x3 = self.up(x3, x2)
        x4 = self.up(x4, x2)
        skl_bottleneck = self.up(skl_bottleneck, skl_x2)
        skl_x3 = self.up(skl_x3, skl_x2)
        skl_x4 = self.up(skl_x4, skl_x2)

        # 5. Refine semantic features independently at each scale.
        for blk in self.fuse_block4:
            x4 = blk(x4)
        for blk in self.fuse_block3:
            x3 = blk(x3)
        for blk in self.fuse_block2:
            x2 = blk(x2)

        # 6. Fuse semantic and skeleton-guidance features.
        c234 = self.linear_fuse(torch.cat([x2, x3, x4], dim=1))
        skl_c234 = self.skl_linear_fuse(torch.cat([skl_x2, skl_x3, skl_x4, skl_bottleneck], dim=1))

        # 7. MCA context aggregation followed by SGEM guidance.
        mca_out = self.mca(c234, x2, x3, x4)
        mca_out = F.interpolate(
            input=mca_out,
            size=x1.shape[2:],
            mode='bilinear',
            align_corners=False)
        skl_c234 = F.interpolate(
            input=skl_c234,
            size=x1.shape[2:],
            mode='bilinear',
            align_corners=False)
        fused_features = self.sgem(mca_out, skl_c234)

        # 8. Produce full-resolution segmentation and skeleton logits.
        mask_out = self.mask_head(fused_features)
        mask_out = F.interpolate(
            input=mask_out,
            size=input_size,
            mode='bilinear',
            align_corners=False)
        
        skel_out = self.skl_head2(self.skl_head1(skl_c234))
        skel_out = F.interpolate(
            input=skel_out,
            size=input_size,
            mode='bilinear',
            align_corners=False)
    
        return [mask_out, skel_out]

    def adapt_mask(self, y):
        """Convert ``[B, 1, H, W]`` labels to the loss-compatible dtype."""
        mask_type = torch.float32 if self.num_classes == 1 else torch.long
        y = y.squeeze(1)
        y = y.type(mask_type)
        return y

    def give_prediction_for_batch(self, batch):
        """Run inference for one dataloader batch with finite-value checks."""
        x, y, _, _ = batch

        # Safety check
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)) or \
                torch.any(torch.isnan(y)) or torch.any(torch.isinf(y)):
            print(f"invalid input detected: x {x}, y {y}", file=sys.stderr)

        y_hat, y_skel = self.forward(x)
        
        # Safety check
        if torch.any(torch.isnan(y_hat)) or torch.any(torch.isinf(y_hat)):
            print(f"invalid output detected: y_hat {y_hat}", file=sys.stderr)

        return [y_hat, y_skel]

    def calc_loss(self, y_hat, y):
        return NotImplemented, NotImplemented

    def make_batch_dictionary(self, loss, metric, name_of_loss):
        return NotImplemented

    def log_metric(self, outputs, train_or_val_or_test):
        pass

    @staticmethod
    def _detach_output(output):
        """Cache detached tensors required for epoch-level statistics."""
        return {
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in output.items()
        }

    def training_step(self, batch, batch_idx):
        x, y, x_name, y_names = batch
        assert x.shape[1] == self.in_channel, \
            f'Network has been defined with {self.in_channel} input channels, ' \
            f'but loaded images have {x.shape[1]} channels. Please check that ' \
            'the images are loaded correctly.'
        y_hat = self.give_prediction_for_batch(batch)
        train_loss, metric = self.calc_loss(y_hat, y)

        self.log('train_loss', train_loss, on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=x.shape[0], sync_dist=True)
        output = self.make_batch_dictionary(train_loss, metric, "loss")
        self._train_step_outputs.append(self._detach_output(output))
        return output

    def on_train_epoch_end(self):
        if not self._train_step_outputs:
            return
        avg_loss = torch.stack([x['loss'] for x in self._train_step_outputs]).mean()
        if self.trainer.is_global_zero:
            self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        self.log_metric(self._train_step_outputs, "Train")
        self._train_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x, y, x_name, y_names = batch
        y_hat = self.give_prediction_for_batch(batch)
        val_loss, metric = self.calc_loss(y_hat, y)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=x.shape[0], sync_dist=True)
        output = self.make_batch_dictionary(val_loss, metric, "val_loss")
        self._validation_step_outputs.append(self._detach_output(output))
        return output

    def on_validation_epoch_end(self):
        if not self._validation_step_outputs:
            return
        avg_loss = torch.stack([x['val_loss'] for x in self._validation_step_outputs]).mean()
        if self.trainer.is_global_zero:
            self.logger.experiment.add_scalar("Loss/Val", avg_loss, self.current_epoch)
        self.log_metric(self._validation_step_outputs, "Val")
        self.log('avg_loss_validation', avg_loss, sync_dist=True)
        self._validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y, x_name, y_names = batch
        y_hat = self.give_prediction_for_batch(batch)
        test_loss, metric = self.calc_loss(y_hat, y)
        self.log('test_loss', test_loss, on_step=False, on_epoch=True,
                 batch_size=x.shape[0], sync_dist=True)
        output = self.make_batch_dictionary(test_loss, metric, "test_loss")
        self._test_step_outputs.append(self._detach_output(output))
        return output

    def on_test_epoch_end(self):
        if not self._test_step_outputs:
            return
        avg_loss = torch.stack([x['test_loss'] for x in self._test_step_outputs]).mean()
        if self.trainer.is_global_zero:
            self.logger.experiment.add_scalar("Loss/Test", avg_loss, self.current_epoch)
        self.log_metric(self._test_step_outputs, "Test")
        self._test_step_outputs.clear()

    def configure_optimizers(self):
        """Configure AdamW with linear warmup and cosine learning-rate decay."""
        max_lr = self.hparams.max_lr
        min_lr = self.hparams.base_lr
        total_steps = 120000
        warmup_ratio = 0.02

        warmup_steps = int(warmup_ratio * total_steps)

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=max_lr,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))

            if step >= total_steps:
                return min_lr / max_lr

            progress = (step - warmup_steps) / float(total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

            return (min_lr / max_lr) + (1 - min_lr / max_lr) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda
        )
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "step",
        }
        return [optimizer], [scheduler_dict]
    

    @staticmethod
    def add_model_specific_args(parent_parser):
        return NotImplemented


# if __name__ == "__main__":
#     tensor = torch.randn((2, 3, 512, 512)).cuda()
#     net = Field(hparams=None, metric=None, in_channel=3, num_classes=2).cuda()
#     # Print trainable parameter names.
#     # for name, module in net.named_modules():
#     #     print(f'Layer Name: {name}')
#     outputs = net(tensor)
#     for output in outputs:
#         print(output.size())


    # with SummaryWriter(logdir="network") as w:
    #     w.add_graph(net, tensor)
    #
    #     w.close()
