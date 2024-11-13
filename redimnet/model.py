# MIT License
# 
# Copyright (c) 2024 ID R&D, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import torch
import functools
import numpy as np
import torch.nn as nn
from typing import List
from torch import Tensor
import torch.nn.functional as F
from collections import OrderedDict
from typing import Iterable, Optional

import redimnet.layers.features as features
import redimnet.layers.features_tf as features_tf
import redimnet.layers.poolings as pooling_layers
from redimnet.layers.layernorm import LayerNorm
from redimnet.layers.resblocks import ResBasicBlock
from redimnet.layers.convnext import ConvNeXtLikeBlock
from redimnet.layers.attention import TransformerEncoderLayer
from redimnet.layers.redim_structural import to1d, to2d, to1d_tfopt, to2d_tfopt, weigth1d

#------------------------------------------
#              Main blocks
#------------------------------------------

class ConvBlock2d(nn.Module):
    def __init__(self, c, f, block_type="convnext_like", Gdiv=1):
        super().__init__()
        if block_type == "convnext_like":
            self.conv_block = ConvNeXtLikeBlock(c, dim=2, kernel_sizes=[(3,3)], 
                                                Gdiv=Gdiv, padding='same', activation='gelu')
        elif block_type == "convnext_like_relu":
            self.conv_block = ConvNeXtLikeBlock(c, dim=2, kernel_sizes=[(3,3)], 
                                                Gdiv=Gdiv, padding='same', activation='relu')
        elif block_type == "basic_resnet":
            self.conv_block = ResBasicBlock(c, c, f, stride=1, se_channels=min(64,max(c,32)), Gdiv=Gdiv, use_fwSE=False)
        elif block_type == "basic_resnet_fwse":
            self.conv_block = ResBasicBlock(c, c, f, stride=1, se_channels=min(64,max(c,32)), Gdiv=Gdiv, use_fwSE=True)
        else:
            raise NotImplemented()

    def forward(self, x):
        return self.conv_block(x)

#------------------------------------------
#                1D block
#------------------------------------------

class PosEncConv(nn.Module):
    def __init__(self, C, ks, groups=None):
        super().__init__()
        assert ks % 2 == 1
        self.conv = nn.Conv1d(C,C,ks,
                              padding=ks//2,
                              groups=C if groups is None else groups)
        self.norm = LayerNorm(C, eps=1e-6, data_format="channels_first")
        
    def forward(self,x):        
        return x + self.norm(self.conv(x))

class TimeContextBlock1d(nn.Module):
    def __init__(self, 
        C, 
        hC,
        pos_ker_sz = 59,
        block_type = 'att',
        red_dim_conv = None,
        exp_dim_conv = None
    ):
        super().__init__()
        assert pos_ker_sz 
        
        self.red_dim_conv = nn.Sequential(
            nn.Conv1d(C,hC,1),
            LayerNorm(hC, eps=1e-6, data_format="channels_first")
        )
        if block_type == 'fc':
            self.tcm = nn.Sequential(
                nn.Conv1d(hC,hC*2,1),
                LayerNorm(hC*2, eps=1e-6, 
                          data_format="channels_first"),
                nn.GELU(),
                nn.Conv1d(hC*2,hC,1)
            )
        elif block_type == 'conv':
            # Just large kernel size conv like in convformer
            self.tcm = nn.Sequential(*[ConvNeXtLikeBlock(
                hC, dim=1, kernel_sizes=[7, 15, 31], Gdiv=1, padding='same'
            ) for i in range(4)])
        elif block_type == 'att':
            # Basic Transformer self-attention encoder block
            self.tcm = nn.Sequential(
                PosEncConv(hC, ks=pos_ker_sz, groups=hC),
                TransformerEncoderLayer(
                    n_state=hC, 
                    n_mlp=hC*2, 
                    n_head=4
                )
            )
        elif block_type == 'conv+att':
            # Basic Transformer self-attention encoder block
            self.tcm = nn.Sequential(
                ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[7], Gdiv=1, padding='same'),
                ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[19], Gdiv=1, padding='same'),
                ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[31], Gdiv=1, padding='same'),
                ConvNeXtLikeBlock(hC, dim=1, kernel_sizes=[59], Gdiv=1, padding='same'),
                TransformerEncoderLayer(
                    n_state=hC, 
                    n_mlp=hC, 
                    n_head=4
                )
            )
        else:
            raise NotImplemented()
            
        self.exp_dim_conv = nn.Conv1d(hC,C,1)
        
    def forward(self,x):
        skip = x
        x = self.red_dim_conv(x)
        x = self.tcm(x)
        x = self.exp_dim_conv(x)
        return skip + x

# Custom wrapper to allow overwriting
def custom_partial(func, **fixed_kwargs):
    def wrapper(**kwargs):
        # Merge fixed kwargs with new kwargs
        merged_kwargs = {**fixed_kwargs, **kwargs}
        return func(**merged_kwargs)
    return wrapper

#------------------------------------------
#                 ReDimNet
#------------------------------------------
class ReDimNet(nn.Module):
    def __init__(self,
        F = 72,
        C = 12,
        block_1d_type = "att",
        block_2d_type = "convnext_like",
        stages_setup = [
            # stride, num_blocks, conv_exp, kernel_size, layer_ext, att_block_red
            (1,2,1,[(3,3)],None), # 16
            (2,3,1,[(3,3)],None), # 32 
            (3,4,1,[(3,3)],8), # 64, (72*12 // 8) = 108 - channels in attention block
            (2,5,1,[(3,3)],8), # 128
            (1,5,1,[(7,1)],8), # 128 # TDNN - time context
            (2,3,1,[(3,3)],8), # 256
        ],
        group_divisor = 1,
        out_channels = 512,
        feat_agg_dropout = 0.0,
        #-----------------------
        #     Subnet stuff
        #-----------------------
        return_2d_output = False,
        return_all_outputs = False,
        offset_fm_weights = 0,
        is_subnet = False,
    ):
        super().__init__()
        self.F = F
        self.C = C

        self.block_1d_type = block_1d_type
        self.block_2d_type = block_2d_type

        self.stages_setup = stages_setup

        self.feat_agg_dropout = feat_agg_dropout
        self.return_2d_output = return_2d_output
        
        # Subnet stuff
        self.is_subnet = is_subnet
        self.offset_fm_weights = offset_fm_weights
        self.return_all_outputs = return_all_outputs
        
        self.build(F,C,stages_setup,group_divisor,out_channels,offset_fm_weights,is_subnet)
        
    def build(self,F,C,stages_setup,group_divisor,out_channels,offset_fm_weights,is_subnet):
        self.F = F
        self.C = C
        
        c = C
        f = F
        s = 1
        self.num_stages = len(stages_setup)
        
        if not is_subnet:
            self.stem = nn.Sequential(
                nn.Conv2d(1, int(c), kernel_size=3, stride=1, padding='same'),
                LayerNorm(int(c), eps=1e-6, data_format="channels_first"),
                to1d()
            )
        else:
            self.stem = nn.Sequential(
                weigth1d(N=offset_fm_weights,C=F*C),
                to2d(f=F,c=C),
                nn.Conv2d(C, C, kernel_size=3, stride=1, padding='same'),
                LayerNorm(C, eps=1e-6, data_format="channels_first"),
                to1d()
            )

        Block1d = functools.partial(TimeContextBlock1d,block_type=self.block_1d_type)
        Block2d = functools.partial(ConvBlock2d,block_type=self.block_2d_type)
        
        self.stages_cfs = []
        for stage_ind, (stride, num_blocks, conv_exp, kernel_sizes, att_block_red) in enumerate(stages_setup):
            assert stride in [1,2,3]

            # Pool frequencies & expand channels if needed
            num_feats_to_weight = offset_fm_weights+stage_ind+1
            layers = [
                weigth1d(N=num_feats_to_weight, C=F*C if num_feats_to_weight>1 else 1, 
                         requires_grad=num_feats_to_weight>1),
                to2d(f=f, c=c),
                
                nn.Conv2d(int(c), int(stride*c*conv_exp), 
                          kernel_size=(stride,1), 
                          stride=(stride,1),
                          padding=0, groups=1),
            ]
            
            self.stages_cfs.append((c,f))

            c = stride * c
            assert f % stride == 0
            f = f // stride
        
            for block_ind in range(num_blocks):
                # ConvBlock2d(f, c, block_type="convnext_like", Gdiv=1)
                layers.append(Block2d(c=int(c*conv_exp), f=f, Gdiv=group_divisor))
            
            if conv_exp != 1:
                # Squeeze back channels to align with ReDimNet c+f reshaping:
                _group_divisor = group_divisor
                # if c // group_divisor == 0:
                    # _group_divisor = c
                layers.append(nn.Sequential(
                    nn.Conv2d(int(c*conv_exp), c, kernel_size=(3,3), stride=1, padding='same', 
                              groups=c // _group_divisor if _group_divisor is not None else 1),
                    nn.BatchNorm2d(c, eps=1e-6,),
                    nn.ReLU() if (('relu' in self.block_1d_type) and ('relu' in self.block_2d_type)) else nn.GELU(),
                    nn.Conv2d(c, c, 1)
                ))

            layers.append(to1d())
            
            if att_block_red is not None:
                layers.append(Block1d(C*F,hC=(C*F)//att_block_red))
                
            setattr(self,f'stage{stage_ind}',nn.Sequential(*layers))


        num_feats_to_weight_fin = offset_fm_weights+len(stages_setup)+1
        self.fin_wght1d = weigth1d(N=num_feats_to_weight_fin, C=F*C, 
                         requires_grad=num_feats_to_weight>1)
        
        if out_channels is not None:
            self.mfa = nn.Sequential(
                nn.Conv1d(self.F * self.C, out_channels, kernel_size=1, padding='same'),
                # LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
                nn.BatchNorm1d(out_channels, affine=True)
            )
        else:
            self.mfa = nn.Identity()

        if self.return_2d_output:
            self.fin_to2d = to2d(f=f,c=c)
        else:
            self.fin_to2d = nn.Identity()
    
    def run_stage(self,prev_outs_1d, stage_ind):
        stage = getattr(self,f'stage{stage_ind}')
        x = stage(prev_outs_1d)
        return x
        
    def forward(self,inp):
        if not self.is_subnet:
            x = self.stem(inp)
            # outputs_1d = [self.to1d(x)]
            outputs_1d = [x]
        else:
            assert isinstance(inp,list)
            outputs_1d = list(inp)
            x = self.stem(inp)
            # outputs_1d.append(self.to1d(x))
            outputs_1d.append(x)
            
        for stage_ind in range(self.num_stages):
            # outputs_1d.append(self.run_stage(outputs_1d,stage_ind))
            outputs_1d.append(F.dropout(self.run_stage(outputs_1d,stage_ind), 
                                        p=self.feat_agg_dropout, training=self.training))
        # x = self.weigth1d(outputs_1d,-1)
        x = self.fin_wght1d(outputs_1d)
        outputs_1d.append(x)
        x = self.mfa(self.fin_to2d(x))

        if self.return_all_outputs:
            return x, outputs_1d
        else:
            return x
    
class ReDimNetWrap(nn.Module):
    def __init__(self,
        F = 72,     
        C = 16,
        block_1d_type = "att",
        block_2d_type = "convnext_like",
        # Default setup: M version:
        stages_setup = [
            # stride, num_blocks, kernel_size, layer_ext, drop_path_prob, att_block_red
            (1,2,1,[(3,3)],12),
            (2,2,1,[(3,3)],12), 
            (1,3,1,[(3,3)],12),
            (2,4,1,[(3,3)],8),
            (1,4,1,[(3,3)],8),
            (2,4,1,[(3,3)],4),
        ],
        group_divisor = 4,
        out_channels = None,
        #-------------------------
        embed_dim=192,
        num_classes=None,
        class_dropout=0.0,
        feat_agg_dropout=0.0,
        head_activation=None,
        hop_length=160,
        pooling_func='ASTP',
        feat_type='pt',
        global_context_att=False,
        emb_bn=False,
        #-------------------------
        spec_params = dict(
            do_spec_aug=False,
            freq_mask_width = (0, 6), 
            time_mask_width = (0, 8),
        ),
        #-------------------------
        return_all_outputs = False,
        tf_optimized_arch = False,
        # offset_fm_weights = 0,
        # is_subnet = False
    ):
        super().__init__()

        self.return_all_outputs = return_all_outputs
        self.tf_optimized_arch = tf_optimized_arch

        if tf_optimized_arch:
            _ReDimNet = ReDimNetTFOpt
        else:
            _ReDimNet = ReDimNet
        
        self.backbone = _ReDimNet(
            F,C,
            block_1d_type,
            block_2d_type,
            stages_setup,
            group_divisor,
            out_channels,
            feat_agg_dropout = feat_agg_dropout,
            return_2d_output = False,
            return_all_outputs = return_all_outputs,
            offset_fm_weights = 0,
            is_subnet = False
        )
        if feat_type in ['pt','pt_mel']:
            self.spec = features.MelBanks(n_mels=F,hop_length=hop_length,**spec_params)
        elif feat_type in ['tf','tf_mel']:
            self.spec = features_tf.TFMelBanks(n_mels=F,hop_length=hop_length,**spec_params)
        elif feat_type == 'tf_spec':
            self.spec = features_tf.TFSpectrogram(**spec_params)
        
        if out_channels is None:
            out_channels = C*F
        self.pool = getattr(pooling_layers, pooling_func)(
            in_dim=out_channels, global_context_att=global_context_att)
        
        self.pool_out_dim = self.pool.get_out_dim()
        self.bn = nn.BatchNorm1d(self.pool_out_dim)
        self.linear = nn.Linear(self.pool_out_dim, embed_dim)
        self.emb_bn = emb_bn
        if emb_bn:  # better in SSL for SV
            self.bn2 = nn.BatchNorm1d(embed_dim)
        else:
            self.bn2 = None

        if num_classes is not None:
            self.cls_head = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Dropout(p=class_dropout, inplace=False),
                nn.Linear(embed_dim, num_classes),
                eval(head_activation) if head_activation is not None else nn.Identity()
            )
        else:
            self.cls_head = None
            
    def forward(self,x):
        x = self.spec(x)
        if self.tf_optimized_arch:
            x = x.permute(0,2,1)
            
        if x.ndim == 3:
            x = x.unsqueeze(1)
        
        if self.return_all_outputs:
            out, all_outs_1d = self.backbone(x)
        else:
            out = self.backbone(x)
            
        out = self.bn(self.pool(out))
        out = self.linear(out)
        
        if self.bn2 is not None:
            out = self.bn2(out)
            
        if self.cls_head is not None:
            out = self.cls_head(out)
            
        if self.return_all_outputs:
            return out, all_outs_1d
        else:
            return out