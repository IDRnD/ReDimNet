import sys
import math
import torch
import functools
import torchaudio
import torch.nn as nn
from typing import List
from torch import Tensor
import torch.nn.functional as F
from collections import OrderedDict
from typing import Iterable, Optional

#-------------------------------------------------------------
#          Code for input features (mel-spectrogram)
#-------------------------------------------------------------

class NormalizeAudio(nn.Module):
    def __init__(self,eps:float=1e-6):
        super().__init__()
        self.eps = eps
        
    def forward(self,x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return (x - x.mean(dim=2, keepdims=True)) /\
                     (x.std(dim=2, keepdims=True, unbiased=False) + self.eps)

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = F.pad(x, (1, 0), 'reflect')
        return F.conv1d(x, self.flipped_filter).squeeze(1)

class FbankAug(nn.Module):
    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10), freq_start_bin=0):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        self.freq_start_bin = freq_start_bin
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(self.freq_start_bin, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
            
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):    
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x

class MelBanks(nn.Module):
    def __init__(self, 
        sample_rate=16000, 
        n_fft=512, 
        win_length=400, 
        hop_length=160,
        f_min = 20, 
        f_max = 7600, 
        n_mels=80, 
        do_spec_aug=False,
        norm_signal=False,
        do_preemph=True,
        freq_start_bin = 0,
        freq_mask_width = (0, 8), 
        time_mask_width = (0, 10),
    ):
        super(MelBanks, self).__init__()
        self.torchfbank = torch.nn.Sequential(
            NormalizeAudio() if norm_signal else nn.Identity(),            
            PreEmphasis() if do_preemph else nn.Identity(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft,
                                                 win_length=win_length,hop_length=hop_length, 
                                                 f_min = f_min, f_max = f_max, n_mels=n_mels, 
                                                 window_fn=torch.hamming_window),
            )
        if do_spec_aug:
            self.specaug = FbankAug(
                freq_start_bin=freq_start_bin,
                freq_mask_width=freq_mask_width,
                time_mask_width=time_mask_width) # Spec augmentation
        else:
            self.specaug = nn.Identity()


    def forward(self, x):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfbank(x)+1e-6
                x = x.log()   
                x = x - torch.mean(x, dim=-1, keepdim=True)
                if self.training:
                    x = self.specaug(x)
        return x

class LayerNorm(nn.Module): # ⚡
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, T, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, T).
    """
    def __init__(self, C, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(C))
        self.bias = nn.Parameter(torch.zeros(C))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.C = (C, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.C, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            
            w = self.weight
            b = self.bias
            for _ in range(x.ndim-2):
                w = w.unsqueeze(-1)
                b = b.unsqueeze(-1)
            x = w * x + b # ⚡
            return x
        
    def extra_repr(self) -> str:
        return ", ".join([f"{k}={v}" for k,v in {"C" : self.C, "data_format" : self.data_format, "eps" : self.eps}.items()])
    
#-------------------------------------------------------------
# Copy multi-head attention module from hugginface wav2vec2
#-------------------------------------------------------------

# Copied from https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/models/wav2vec2/modeling_wav2vec2.py
# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->Wav2Vec2
class MultiHeadAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self,hidden_states: torch.Tensor) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)
        return attn_output
    
#-------------------------------------------------------------
# Import activations from hugginface
#-------------------------------------------------------------
# Copied from https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/activations.py

def gelu(x):
    """This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class NewGELUActivation(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class GELUActivation(nn.Module):
    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = F.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)

class ClippedGELUActivation(nn.Module):
    def __init__(self, min: float, max: float):
        if min > max:
            raise ValueError(f"min should be < max (got min: {min}, max: {max})")

        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: Tensor) -> Tensor:
        return torch.clip(gelu(x), self.min, self.max)

# Copied from https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/activations.py
class FastGELUActivation(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))

# Copied from https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/activations.py
class QuickGELUActivation(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(1.702 * input)

# Copied from https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/activations.py
class SiLUActivation(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.silu(input)
    
class MishActivation(nn.Module):
    def __init__(self):
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.9.0"):
            self.act = self._mish_python
        else:
            self.act = F.mish

    def _mish_python(self, input: Tensor) -> Tensor:
        return input * torch.tanh(F.softplus(input))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)

class LinearActivation(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input

class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)
    
ACT2CLS = {
    "gelu": GELUActivation,
    "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
    "gelu_fast": FastGELUActivation,
    "gelu_new": NewGELUActivation,
    "gelu_python": (GELUActivation, {"use_gelu_python": True}),
    "linear": LinearActivation,
    "mish": MishActivation,
    "quick_gelu": QuickGELUActivation,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": SiLUActivation,
    "swish": SiLUActivation,
    "tanh": nn.Tanh,
}
ACT2FN = ClassInstantier(ACT2CLS)

#-------------------------------------------------------------
#  Copy FeedForward & Encoder from hugginface wav2vec2
#-------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, 
            hidden_size:int,
            intermediate_size:int,
            hidden_act:str='gelu_new',
            activation_dropout:float=0.0,
            hidden_dropout:float=0.0
        ):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(activation_dropout)
        self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = ACT2FN[hidden_act]
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, 
            n_state: int, 
            n_mlp : int, 
            n_head: int,
            channel_last: bool = False,
            act: str = 'gelu_new',
            act_do:float = 0.0,
            att_do:float = 0.0,
            hid_do:float = 0.0,
            ln_eps:float=1e-6
        ):
        
        hidden_size=n_state
        num_attention_heads=n_head
        intermediate_size=n_mlp
        hidden_act=act
        activation_dropout=act_do
        attention_dropout=att_do
        hidden_dropout=hid_do
        layer_norm_eps=ln_eps
        
        super().__init__()
        self.channel_last = channel_last
        self.attention = MultiHeadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_dropout
        )
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            hidden_act=hidden_act,
            intermediate_size=intermediate_size,
            activation_dropout=activation_dropout,
            hidden_dropout=hidden_dropout
        )
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states):
        if not self.channel_last:
            hidden_states = hidden_states.permute(0, 2, 1)
        attn_residual = hidden_states
        hidden_states = self.attention(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = hidden_states
        if not self.channel_last:
            outputs = outputs.permute(0, 2, 1)
        return outputs

#------------------------------------------
#       ResNet basice block (+fwSE)
#------------------------------------------

class fwSEBlock(nn.Module):
    """
    Squeeze-and-Excitation block
    link: https://arxiv.org/pdf/1709.01507.pdf
    PyTorch implementation
    """
    def __init__(self, num_freq, num_feats=64):
        super(fwSEBlock, self).__init__()
        self.squeeze = nn.Linear(num_freq, num_feats)
        self.exitation = nn.Linear(num_feats, num_freq)

        self.activation = nn.ReLU()  # Assuming ReLU, modify as needed

    def forward(self, inputs):
        # [bs, C, F, T]
        x = torch.mean(inputs, dim=[1, 3])
        x = self.squeeze(x)
        x = self.activation(x)
        x = self.exitation(x)
        x = torch.sigmoid(x)
        # Reshape and apply excitation
        x = x[:,None,:,None]
        x = inputs * x
        return x

class ResBasicBlock(nn.Module):
    def __init__(self, inc, outc, num_freq, stride=1, se_channels=64, Gdiv=4, use_fwSE=False):
        super().__init__()
        # if inc//Gdiv==0:
        #     Gdiv = inc
        self.conv1 = nn.Conv2d(inc, inc if Gdiv is not None else outc, kernel_size=3, stride=stride, padding=1, bias=False, 
                               groups=inc//Gdiv if Gdiv is not None else 1)
        if Gdiv is not None:
            self.conv1pw = nn.Conv2d(inc, outc, 1)
        else:
            self.conv1pw = nn.Identity()
            
        self.bn1 = nn.BatchNorm2d(outc)
        self.conv2 = nn.Conv2d(outc, outc, kernel_size=3, padding=1, bias=False, groups=outc//Gdiv if Gdiv is not None else 1)
        
        if Gdiv is not None:
            self.conv2pw = nn.Conv2d(outc, outc, 1)
        else:
            self.conv2pw = nn.Identity()
            
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu = nn.ReLU(inplace=True)
        
        if use_fwSE:
            self.se = fwSEBlock(num_freq, se_channels)
        else:
            self.se = nn.Identity()
            
        if outc != inc:
            self.downsample = nn.Sequential(
                nn.Conv2d(inc, outc, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outc),
            )
        else:
            self.downsample = nn.Identity()
        
    def forward(self, x):
        residual = x

        out = self.conv1pw(self.conv1(x))
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2pw(self.conv2(out))
        out = self.bn2(out)
        out = self.se(out)

        out += self.downsample(residual)
        out = self.relu(out)
        # print(out.size())
        return out

#------------------------------------------
#           ConvNeXtV2 block
#------------------------------------------

MaxPoolNd = {
    1 : nn.MaxPool1d,
    2 : nn.MaxPool2d
}

ConvNd = {
    1 : nn.Conv1d,
    2 : nn.Conv2d
}

BatchNormNd = {
    1 : nn.BatchNorm1d,
    2 : nn.BatchNorm2d
}

# https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
# https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
class ConvNeXtLikeBlock(nn.Module):
    def __init__(self, C, dim=2, kernel_sizes=[(3,3),], Gdiv=1, padding='same'):
        super().__init__()
        # if C//Gdiv==0:
        #     Gdiv = C
        self.dwconvs = nn.ModuleList(modules=[
            ConvNd[dim](C, C, kernel_size=ks, 
                                padding=padding, groups=C//Gdiv if Gdiv is not None else 1) for ks in kernel_sizes
        ])
        self.norm = BatchNormNd[dim](C * len(kernel_sizes))
        self.act = nn.GELU()
        self.pwconv1 = ConvNd[dim](C * len(kernel_sizes), C, 1) # pointwise/1x1 convs, implemented with linear layers


    def forward(self, x):
        skip = x
        x = torch.cat([dwconv(x) for dwconv in self.dwconvs],dim=1)
        x = self.act(self.norm(x))
        x = self.pwconv1(x)
        x = skip + x
        return x

#------------------------------------------
#              Main blocks
#------------------------------------------

#------------------------------------------
#                2D block
#------------------------------------------

class ConvBlock2d(nn.Module):
    def __init__(self, c, f, block_type="convnext_like", Gdiv=1):
        super().__init__()
        if block_type == "convnext_like":
            self.conv_block = ConvNeXtLikeBlock(c, dim=2, kernel_sizes=[(3,3)], Gdiv=Gdiv, padding='same')
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
        self.conv = nn.Conv1d(C,C,ks,padding=ks//2,groups=C if groups is None else groups)
        self.norm = LayerNorm(C, eps=1e-6, data_format="channels_first")
        
    def forward(self,x):        
        return x + self.norm(self.conv(x))

class GRU(nn.Module):
    def __init__(self,*args,**kwargs):
        super(GRU, self).__init__()
        self.gru = nn.GRU(*args,**kwargs)
        
    def forward(self, x):
        # x : (bs,C,T) 
        return self.gru(x.permute((0,2,1)))[0].permute((0,2,1))

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
        elif block_type == 'gru':
            # Just GRU
            self.tcm = nn.Sequential(
                GRU(
                    input_size=hC, hidden_size=hC,
                    num_layers=1, bias=True, batch_first=False, 
                    dropout=0.0, bidirectional=True
                ), nn.Conv1d(2*hC, hC, 1)
            )
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

#------------------------------------------
#           ReDimNet misc blocks
#------------------------------------------

class to1d(nn.Module):
    def forward(self,x):
        size = x.size()
        bs,c,f,t = tuple(size)
        return x.permute((0,2,1,3)).reshape((bs,c*f,t))

class to2d(nn.Module):
    def __init__(self, f, c):
        super().__init__()
        self.f = f
        self.c = c

    def forward(self,x):
        size = x.size()
        bs,cf,t = tuple(size)
        out = x.reshape((bs,self.f,self.c,t)).permute((0,2,1,3))
        # print(f"to2d : {out.size()}")
        return out

    def extra_repr(self) -> str:
        return f"f={self.f},c={self.c}"

class weigth1d(nn.Module):
    def __init__(self, N, C):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1,N,C,1),requires_grad=True)

    def forward(self, xs):
        xs = torch.cat([t.unsqueeze(1) for t in xs],dim=1)
        w = F.softmax(self.w,dim=1)
        x = (w*xs).sum(dim=1)
        # print(f"weigth1d : {x.size()}")
        return x

    def extra_repr(self) -> str:
        return f"w={tuple(self.w.size())}"

#------------------------------------------
#                 ReDimNet
#------------------------------------------
class ReDimNet(nn.Module):
    def __init__(self,
        F = 72,     
        C = 16,
        block_1d_type = "conv+att",
        block_2d_type = "basic_resnet",
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
    ):
        super().__init__()
        self.F = F
        self.C = C

        self.block_1d_type = block_1d_type
        self.block_2d_type = block_2d_type

        self.stages_setup = stages_setup
        
        # Subnet stuff
        self.build(F,C,stages_setup,group_divisor,out_channels)
        
    def build(self,F,C,stages_setup,group_divisor,out_channels):
        self.F = F
        self.C = C
        
        c = C
        f = F
        s = 1
        self.num_stages = len(stages_setup)

        self.inputs_weights = torch.nn.ParameterList([
            nn.Parameter(torch.ones(1,1,1,1),requires_grad=False)]+[
            nn.Parameter(torch.zeros(1,num_inputs+1,C*F,1),requires_grad=True) for num_inputs in 
                range(1,len(stages_setup)+1)])
        
        self.stem = nn.Sequential(
            nn.Conv2d(1, int(c), kernel_size=3, stride=1, padding='same'),
            LayerNorm(int(c), eps=1e-6, data_format="channels_first")
        )


        Block1d = functools.partial(TimeContextBlock1d,block_type=self.block_1d_type)
        Block2d = functools.partial(ConvBlock2d,block_type=self.block_2d_type)
        
        self.stages_cfs = []
        for stage_ind, (stride, num_blocks, conv_exp, kernel_sizes, att_block_red) in enumerate(stages_setup):
            assert stride in [1,2,3]

            # Pool frequencies & expand channels if needed
            layers = [nn.Conv2d(int(c), int(stride*c*conv_exp), kernel_size=(stride,1), stride=(stride,1),
                            padding=0, groups=1), ]
            
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
                    nn.GELU(),
                    nn.Conv2d(c, c, 1)
                ))

            layers.append(to1d())
            
            if att_block_red is not None:
                layers.append(Block1d(C*F,hC=(C*F)//att_block_red))
                
            setattr(self,f'stage{stage_ind}',nn.Sequential(*layers))
        
        if out_channels is not None:
            self.mfa = nn.Sequential(
                nn.Conv1d(self.F * self.C, out_channels, kernel_size=1, padding='same'),
                # LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
                nn.BatchNorm1d(out_channels, affine=True)
            )
        else:
            self.mfa = nn.Identity()

    def to1d(self,x):
        size = x.size()
        bs,c,f,t = tuple(size)
        return x.permute((0,2,1,3)).reshape((bs,c*f,t))

    def to2d(self,x,c,f):
        size = x.size()
        bs,cf,t = tuple(size)
        return x.reshape((bs,f,c,t)).permute((0,2,1,3))
    
    def weigth1d(self,outs_1d,i):
        xs = torch.cat([t.unsqueeze(1) for t in outs_1d],dim=1)
        # xs.size() = (bs, len*(outs_1d), f*c, t)
        w = F.softmax(self.inputs_weights[i],dim=1)
        x = (w*xs).sum(dim=1)
        return x
    
    def run_stage(self,prev_outs_1d, stage_ind):
        stage = getattr(self,f'stage{stage_ind}')
        c, f = self.stages_cfs[stage_ind]
        
        x = self.weigth1d(prev_outs_1d, stage_ind)
        x = self.to2d(x,c,f)
        x = stage(x)
        return x
        
    def forward(self,inp):
        x = self.stem(inp)
        outputs_1d = [self.to1d(x)]
        for stage_ind in range(self.num_stages):
            outputs_1d.append(self.run_stage(outputs_1d,stage_ind))
        x = self.weigth1d(outputs_1d,-1)
        x = self.mfa(x)
        return x

#------------------------------------------
#             Pooling layers:
#------------------------------------------

class ASTP(nn.Module):
    """ Attentive statistics pooling: Channel- and context-dependent
        statistics pooling, first used in ECAPA_TDNN.
    """

    def __init__(self,
                 in_dim,
                 bottleneck_dim=128,
                 global_context_att=False,
                 **kwargs):
        super(ASTP, self).__init__()
        self.in_dim = in_dim
        self.global_context_att = global_context_att

        # Use Conv1d with stride == 1 rather than Linear, then we don't
        # need to transpose inputs.
        if global_context_att:
            self.linear1 = nn.Conv1d(
                in_dim * 3, bottleneck_dim,
                kernel_size=1)  # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(
                in_dim, bottleneck_dim,
                kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim,
                                 kernel_size=1)  # equals V and k in the paper

    def forward(self, x):
        """
        x: a 3-dimensional tensor in tdnn-based architecture (B,F,T)
            or a 4-dimensional tensor in resnet architecture (B,C,F,T)
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        assert len(x.shape) == 3

        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(
                torch.var(x, dim=-1, keepdim=True) + 1e-7).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here! ReLU may be hard to converge.
        alpha = torch.tanh(
            self.linear1(x_in))  # alpha = F.relu(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(var.clamp(min=1e-7))
        return torch.cat([mean, std], dim=1)

    def get_out_dim(self):
        self.out_dim = 2 * self.in_dim
        return self.out_dim

#------------------------------------------
#          Aggregated final model
#------------------------------------------

class ReDimNetWrap(nn.Module):
    def __init__(self,
        F = 72,     
        C = 16,
        block_1d_type = "conv+att",
        block_2d_type = "basic_resnet",
        # Default setup: M version:
        stages_setup = [
            # stride, num_blocks, kernel_sizes, layer_ext, att_block_red
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
        hop_length=160,
        pooling_func='ASTP',
        global_context_att=False,
        emb_bn=False,
        #-------------------------
        feat_type = 'pt',
        spec_params = dict(
            do_spec_aug=False,
            freq_mask_width = (0, 6), 
            time_mask_width = (0, 8),
        ),
    ):
        
        super().__init__()
        self.backbone = ReDimNet(
            F,C,
            block_1d_type,
            block_2d_type,
            stages_setup,
            group_divisor,
            out_channels
        )

        if feat_type in ['pt','pt_mel']:
            self.spec = MelBanks(n_mels=F,hop_length=hop_length,**spec_params)
        else:
            raise NotImplementedError()
            
        if out_channels is None:
            out_channels = C*F
        self.pool = getattr(sys.modules[__name__], pooling_func)(
            in_dim=out_channels, global_context_att=global_context_att)
        
        self.pool_out_dim = self.pool.get_out_dim()
        self.bn = nn.BatchNorm1d(self.pool_out_dim)
        self.linear = nn.Linear(self.pool_out_dim, embed_dim)
        self.emb_bn = emb_bn
        if emb_bn:  # better in SSL for SV
            self.bn2 = nn.BatchNorm1d(embed_dim)
        else:
            self.bn2 = nn.Identity()
            
    def forward(self,x):
        x = self.spec(x).unsqueeze(1)
        out = self.backbone(x)
        out = self.bn(self.pool(out))
        out = self.linear(out)
        if self.emb_bn:
            out = self.bn2(out)
        return out
