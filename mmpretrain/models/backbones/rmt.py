from copy import deepcopy
from typing import Sequence
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_

from mmpretrain.registry import MODELS
from ..utils import (ConditionalPositionEncoding, build_norm_layer, to_2tuple,
                     SwiGLUFFNFused)
from .base_backbone import BaseBackbone


def bidirectional_explicit_decay(slen, gamma):
    coordinates = torch.arange(slen)
    mask = coordinates[:, None] - coordinates[None, :]
    mask = torch.abs(mask)
    decay = gamma**mask
    return decay


def two_dimensional_explicit_decay(height, width, gamma):
    arrayH = torch.arange(height)
    arrayW = torch.arange(width)
    coordinates = torch.cartesian_prod(arrayH, arrayW)
    diff = torch.abs(coordinates[:, None] - coordinates[None, :])
    mask = diff.sum(dim=-1)
    decay = gamma**mask
    return decay


class RetentiveSelfAttention(nn.Module):
    """Retentive Self-Attention Block for RMT.
    
    Args:
        resolution (int | tuple): The expected input shape.
        embed_dim (int): The feature dimension (for key and queue vectors).
            Defaults to 768.
        value_dim (int): The feature dimension for value vector.
            Defaults to 768.
        num_heads (int): Parallel attention heads.
            Defaults to 8.
        attn_drop (float): The drop out rate for attention output weights.
            Defaults to 0.
        proj_drop (float): Probability of an element to be zeroed after
            the feed forward layer. Defaults to 0.
    """

    def __init__(self,
                 resolution,
                 embed_dim=768,
                 value_dim=768,
                 num_heads=8,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.resolution = resolution
        self.embed_dim = embed_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.head_dim = self.value_dim // self.num_heads
        self.key_dim = self.embed_dim // self.num_heads
        self.scaling = self.key_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, value_dim, bias=True)
        self.proj = nn.Linear(value_dim, embed_dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def resa(self, qr, kr, decay):
        qk_mat = torch.matmul(
            qr, kr.transpose(-1, -2)
        )  # (bsz, num_heads, num_tokens, key_dim) * (bsz, num_heads, key_dim, num_tokens)
        qk_mat = nn.functional.softmax(qk_mat, dim=-1)
        qk_mat = qk_mat * decay.to(qk_mat.device)
        return qk_mat

    def regular(self, qr, kr, vr, decay):
        Attn = self.resa(qr, kr, decay)
        Attn = self.attn_drop(Attn)
        output = torch.matmul(Attn, vr)
        output = output.transpose(1, 2)
        return output

    def decomposition(self, qr, kr, vr, decayH, decayW):
        Hpatch, Wpatch = self.resolution[0], self.resolution[1]
        bsz, num_heads, num_tokens, key_dim = qr.size()

        # vertical
        qh = qr.view(bsz, num_heads, Wpatch, Hpatch,
                     key_dim)  # num_tokens -> Wpatch, H_patch
        kh = kr.view(bsz, num_heads, Wpatch, Hpatch, key_dim)
        AttnH = self.resa(qh, kh,
                          decayH)  # (bsz, num_heads, Wpatch, Hpatch, Hpatch)
        AttnH = self.attn_drop(AttnH)

        # horizontal
        qw = qr.view(bsz, num_heads, Hpatch, Wpatch, key_dim)
        kw = kr.view(bsz, num_heads, Hpatch, Wpatch, key_dim)
        AttnW = self.resa(qw, kw,
                          decayW)  # (bsz, num_heads, Hpatch, Wpatch, Wpatch)
        AttnW = self.attn_drop(AttnW)

        vr = vr.view(bsz, num_heads, Hpatch, Wpatch, self.head_dim)
        output = torch.matmul(
            AttnW, vr
        )  # (bsz, num_heads, Hpatch, Wpatch, Wpatch) * (bsz, num_heads, Hpatch, Wpatch, value_dim)
        output = torch.matmul(
            AttnH, output.transpose(-2, -3)
        )  # (bsz, num_heads, Wpatch, Hpatch, Hpatch) * (bsz, num_heads, Wpatch, Hpatch, value_dim)
        output = output.view(bsz, num_heads, num_tokens, self.head_dim)
        output = output.transpose(1, 2)
        return output

    def forward(self, x, decay, decomposition=False):
        batch_size, pH, pW, embed_dim = x.size()
        assert pH == self.resolution[0] and pW == self.resolution[1], \
                        f'Input shape does not fit the input resolution. (H, W): {pH, pW}, self.input_resolution: {self.resolution}'
        num_tokens = pH * pW
        x = x.view(batch_size, num_tokens,
                   embed_dim)  # bsz, num_tokens, embed_dim

        qr = self.q_proj(x)
        kr = self.k_proj(x)
        vr = self.v_proj(x)

        kr *= self.scaling
        qr = qr.view(batch_size, num_tokens, self.num_heads,
                     self.key_dim).transpose(1, 2)
        kr = kr.view(batch_size, num_tokens, self.num_heads,
                     self.key_dim).transpose(1, 2)
        vr = vr.view(batch_size, num_tokens, self.num_heads,
                     self.head_dim).transpose(1, 2)

        if decomposition:
            decayH, decayW = decay
            output = self.decomposition(qr, kr, vr, decayH, decayW)
        else:
            output = self.regular(qr, kr, vr, decay)

        output = output.reshape(batch_size, num_tokens,
                                self.num_heads * self.head_dim)
        output = output.view(batch_size, pH, pW, -1)
        output = self.proj(output)
        output = self.proj_drop(output)

        return output


class RMTBlock(BaseModule):
    """Implements one RMTBlock in RMT
    
    Args:
        resolution (int | tuple): The expected input shape
        embed_dim (int): The feature dimension
        value_dim (int): The feature dimension for value vector.
        num_heads (int): Parallel attention heads
        mlp_ratio (float): Ratio of hidden dimension of FFNs to the input dimension
        gamma (float): The coefficient used in the weight decay matrix
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 0.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        decomposition (bool): Determine whether the ReSA is decomposed or not.
            Defaults to False.
        dwcv_kernel (int): Kernel size of the depthwise convolution module.
            Defaults to 3.
        ffn_type (str): Select the type of ffn layers. Defaults to 'origin'.
        act_cfg (dict): The activation config for FFNs.
            Defaults to ''dict(type='GELU')''.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 resolution,
                 embed_dim,
                 value_dim,
                 num_heads,
                 mlp_ratio,
                 gamma,
                 layer_scale_init_value=0.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 decomposition=False,
                 dwcv_kernel=3,
                 ffn_type='origin',
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(RMTBlock, self).__init__(init_cfg=init_cfg)
        self.resolution = resolution
        self.embed_dim = embed_dim

        self.dwcv = DepthwiseSeparableConvModule(
            embed_dim, embed_dim, kernel_size=dwcv_kernel, stride=1, padding=1)

        self.ln1 = build_norm_layer(norm_cfg, self.embed_dim)

        self.resa = RetentiveSelfAttention(
            resolution=resolution,
            embed_dim=embed_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate)

        self.ln2 = build_norm_layer(norm_cfg, self.embed_dim)

        if ffn_type == 'origin':
            self.ffn = FFN(
                embed_dims=embed_dim,
                feedforward_channels=embed_dim * mlp_ratio,
                num_fcs=num_fcs,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                act_cfg=act_cfg,
                layer_scale_init_value=layer_scale_init_value)
        elif ffn_type == 'swiglu_fused':
            self.ffn = SwiGLUFFNFused(
                embed_dims=embed_dim,
                feedforward_channels=embed_dim * mlp_ratio,
                layer_scale_init_value=layer_scale_init_value)
        else:
            raise NotImplementedError

        self.decomposition = decomposition
        if decomposition:
            self.decay = (bidirectional_explicit_decay(self.resolution[0],
                                                       gamma),
                          bidirectional_explicit_decay(self.resolution[1],
                                                       gamma))
        else:
            self.decay = two_dimensional_explicit_decay(
                self.resolution[0], self.resolution[1], gamma)

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def init_weights(self):
        super(RMTBlock, self).init_weights()
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        x = x + self.dwcv(x)  # B, embed_dim, pH, pW
        x = x.permute(0, 2, 3, 1)  # B, pH, pW, embed_dim
        x = x + self.resa(self.ln1(x), self.decay, self.decomposition)
        x = x + self.ffn(self.ln2(x))
        x = x.permute(0, 3, 1, 2)  # B, embed_dim, pH, pW

        return x


@MODELS.register_module()
class RMT(BaseBackbone):
    """Retention Network Meets Vision Transformer.

    A PyTorch implement of : `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Args:
        arch (str | dict): Swin Transformer architecture. If use string, choose
            from 'tiny', 'small', 'base' and 'large'. If use dict, it should
            have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **depths** (List[int]): The number of blocks in each stage.
            - **num_heads** (List[int]): The number of heads in attention
              modules of each stage.
            - **mlp_ratios** (List[float]): The expansion ratio of mlp
              features in each stage.

            Defaults to 'tiny'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 4.
        in_channels (int): The num of input channels. Defaults to 3.
        value_dim (int): The feature dimension for value vector.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        gamma (float): The coefficient used in the weight decay matrix.
            Defaults to 0.5.
        decomposition (tuple): Determine whether the ReSA is decomposed or not.
            Defaults to False.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 0.
        kernels (tuple): Kernel size of the convolution module preceeding the RMT block.
            Defaults to (3, 3, 3, 3).
        strides (tuple): Stride size of the convolution module preceeding the RMT block.
            Defaults to (2, 2, 2, 2).
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        cpe_cfg (dict): Configs of conditional position encoding. 
            Defaults to an empty dict.
        conv_cfgs (dict): Configs of convolution module preceeding the RMT block.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    arch_zoo = {
        **dict.fromkeys(
            ['t', 'tiny'], {
                'embed_dims': 96,
                'depths': [2, 2, 6, 2],
                'num_heads': [3, 6, 12, 24],
                'mlp_ratios': [4, 4, 4, 4]
            }),
        **dict.fromkeys(
            ['s', 'small'], {
                'embed_dims': 96,
                'depths': [2, 2, 18, 2],
                'num_heads': [3, 6, 12, 24],
                'mlp_ratios': [4, 4, 8, 8]
            }),
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 128,
                'depths': [2, 2, 18, 2],
                'num_heads': [4, 8, 16, 32],
                'mlp_ratios': [8, 8, 8, 8]
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 192,
                'depths': [2, 2, 18, 2],
                'num_heads': [6, 12, 24, 48],
                'mlp_ratios': [8, 8, 8, 8]
            }),
    }

    def __init__(self,
                 arch='tiny',
                 img_size=224,
                 patch_size=4,
                 in_channels=3,
                 value_dim=768,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN'),
                 gamma=[0.5, 0.25, 0.125, 0.5**3],
                 decomposition=[False, False, False, True],
                 final_norm=True,
                 frozen_stages=-1,
                 layer_scale_init_value=0.,
                 kernels=[3, 3, 3, 3],
                 strides=[2, 2, 2, 2],
                 patch_cfg=dict(),
                 cpe_cfg=dict(),
                 conv_cfgs=dict(),
                 layer_cfgs=dict(),
                 init_cfg=None):
        super(RMT, self).__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'depths', 'num_heads', 'mlp_ratios'
            }
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch need a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.num_heads = self.arch_settings['num_heads']
        self.mlp_ratios = self.arch_settings['mlp_ratios']
        self.num_layers = len(self.depths)

        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            norm_cfg=dict(type='LN'),
        )

        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        patch_resolution = self.patch_embed.init_out_size

        _cpe_cfg = dict(
            in_channels=self.embed_dims,
            embed_dims=self.embed_dims,
            stride=1,
        )
        _cpe_cfg.update(cpe_cfg)
        self.cpe = ConditionalPositionEncoding(**_cpe_cfg)

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]

        count_dpr = 0

        self.stages = ModuleList()

        for i, (dpth, nh, mr, gmm, dec, krnls, strd) in enumerate(
                zip(self.depths, self.num_heads, self.mlp_ratios, gamma,
                    decomposition, kernels, strides)):

            if isinstance(conv_cfgs, Sequence):
                conv_cfg = conv_cfgs[i]
            else:
                conv_cfg = deepcopy(conv_cfgs)

            _conv_cfg = dict(
                in_channels=self.embed_dims,
                out_channels=self.embed_dims,
                kernel_size=krnls,
                stride=strd,
                padding=0,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU'),
                **conv_cfg)

            conv = ConvModule(**_conv_cfg)

            self.stages.append(conv)

            h_out = (patch_resolution[0] + 2 * conv.padding -
                     conv.dilation[0] *
                     (conv.kernel_size[0] - 1) - 1) // conv.stride[0] + 1
            w_out = (patch_resolution[1] + 2 * conv.padding -
                     conv.dilation[1] *
                     (conv.kernel_size[1] - 1) - 1) // conv.stride[1] + 1

            patch_resolution = (h_out, w_out)

            if isinstance(layer_cfgs, Sequence):
                layer_cfg = layer_cfgs[i]
            else:
                layer_cfg = deepcopy(layer_cfgs)

            for _ in range(dpth):
                _layer_cfg = dict(
                    resolution=patch_resolution,
                    embed_dim=self.embed_dims,
                    value_dim=value_dim,
                    num_heads=nh,
                    mlp_ratio=mr,
                    gamma=gmm,
                    layer_scale_init_value=layer_scale_init_value,
                    drop_rate=drop_rate,
                    attn_drop_rate=drop_rate,
                    drop_path_rate=dpr[count_dpr],
                    decomposition=dec,
                    ffn_type='origin',
                    act_cfg=dict(type='GELU'),
                    norm_cfg=dict(type='LN'),
                    **layer_cfg)

                stage = RMTBlock(**_layer_cfg)

                count_dpr += 1

                self.stages.append(stage)

        self.final_norm = final_norm
        if final_norm:
            self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)
        else:
            self.ln1 = nn.Identity()

        self.frozen_stages = frozen_stages
        if self.frozen_stages >= 0:
            self._freeze_stages()

    def init_weights(self):
        super(RMT, self).init_weights()

        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            return

    def _freeze_stages(self):
        # freeze patch embedding
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        # set dropout to eval mode
        self.drop_after_pos.eval()
        # freeze layers
        for i in range(1, self.frozen_stages + 1):
            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        if self.frozen_stages == len(self.stages) // 2:
            if self.final_norm:
                self.ln1.eval()
                for param in self.ln1.parameters():
                    param.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)  # B, pH * pW, embed_dim
        x = self.cpe(x, patch_resolution)  # B, pH * pW, embed_dim
        x = self.drop_after_pos(x)  # B, pH * pW, embed_dim
        x = x.reshape(B, -1, patch_resolution[0],
                      patch_resolution[1])  # B, embed_dim, pH, pW

        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)

            if i == len(self.stages) - 1 and self.final_norm:
                x = x.permute(0, 2, 3, 1)  # B, pH, pW, embed_dim
                x = self.ln1(x)
                outs.append(x.permute(0, 3, 1, 2))  # B, embed_dim, pH, pW
        return tuple(outs)
