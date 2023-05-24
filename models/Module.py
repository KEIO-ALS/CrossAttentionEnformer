import torch
from torch import nn, einsum
from torch.nn import functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import math


def _exists(val):
    return val is not None

def _default(val, d):
    return val if _exists(val) else d

# Transformer
def get_positional_features_exponential(positions, features, seq_len, min_half_life = 3.):
    max_range = math.log(seq_len) / math.log(2.)
    half_life = 2 ** torch.linspace(min_half_life, max_range, features, device = positions.device)
    half_life = half_life[None, ...]
    positions = positions.abs()[..., None]
    return torch.exp(-math.log(2.) / half_life * positions)

def get_positional_features_central_mask(positions, features, seq_len):
    center_widths = 2 ** torch.arange(1, features + 1, device = positions.device).float()
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).float()

def gamma_pdf(x, concentration, rate):
    log_unnormalized_prob = torch.xlogy(concentration - 1., x) - rate * x
    log_normalization = (torch.lgamma(concentration) - concentration * torch.log(rate))
    return torch.exp(log_unnormalized_prob - log_normalization)

def get_positional_features_gamma(positions, features, seq_len, stddev = None, start_mean = None, eps = 1e-8):
    if not _exists(stddev):
        stddev = seq_len / (2 * features)

    if not _exists(start_mean):
        start_mean = seq_len / features

    mean = torch.linspace(start_mean, seq_len, features, device = positions.device)
    mean = mean[None, ...]
    concentration = (mean / stddev) ** 2
    rate = mean / stddev ** 2
    probabilities = gamma_pdf(positions.float().abs()[..., None], concentration, rate)
    probabilities = probabilities + eps
    outputs = probabilities / torch.amax(probabilities, dim = -1, keepdim = True)
    return outputs

def get_positional_embed(seq_len, feature_size, device):
    distances = torch.arange(-seq_len + 1, seq_len, device = device)

    feature_functions = [
        get_positional_features_exponential,
        get_positional_features_central_mask,
        get_positional_features_gamma
    ]

    num_components = len(feature_functions) * 2

    num_basis_per_class = feature_size // num_components

    embeddings = []
    for fn in feature_functions:
        embeddings.append(fn(distances, num_basis_per_class, seq_len))

    embeddings = torch.cat(embeddings, dim = -1)
    embeddings = torch.cat((embeddings, torch.sign(distances)[..., None] * embeddings), dim = -1)
    return embeddings

def _relative_shift(x, h):
    x = rearrange(x, '(b h) n d -> b h n d', h = h)
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim = -1)
    _, h, t1, t2 = x.shape
    x = x.reshape(-1, h, t2, t1)
    x = x[:, :, 1:, :]
    x = x.reshape(-1, h, t1, t2 - 1)
    return rearrange(x[..., :((t2 + 1) // 2)], "b h n d -> (b h) n d", h=h)

class Attention(nn.Module):
    def __init__(
            self,
            q_dim,
            kv_dim=None,
            num_head=8,
            head_dim=64,
            dropout=.05,
            pos_dropout=.01,
    ) -> None:
        super().__init__()

        inner_dim = head_dim*num_head
        self.inner_dim = inner_dim
        kv_dim = _default(kv_dim, q_dim)

        self.scale = head_dim**-0.5
        self.num_head = num_head

        self.Wq = nn.Linear(q_dim, inner_dim, bias=False)
        self.Wk = nn.Linear(kv_dim, inner_dim, bias=False)
        self.Wv = nn.Linear(kv_dim, inner_dim, bias=False)

        self.content_bias = nn.Parameter(torch.randn(num_head, 1, head_dim))

        self.pos_bias = nn.Parameter(torch.randn(num_head, 1, head_dim))
        self.pos_dropout = nn.Dropout(pos_dropout)
        self.Wp = nn.Linear(kv_dim, inner_dim)

        self.dropout = nn.Dropout(dropout)
        self.Wout = nn.Linear(inner_dim, q_dim)

    def forward(self, x, **kwargs):
        kv = kwargs.get('kv', None)
        mask = kwargs.get('mask', None)

        h = self.num_head

        _, kv_n, kv_hd = _default(kv, x).shape

        q, k, v = self.Wq(x), self.Wk(_default(kv, x)), self.Wv(_default(kv, x))

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        q *= self.scale

        qk = einsum('b i d, b j d -> b i j', q+self.content_bias, k)
        
        # Enformer's Position Encoding
        pos = get_positional_embed(kv_n, kv_hd, qk.device)
        pos = self.Wp(self.pos_dropout(pos))
        pos = rearrange(pos, 'n (h d) -> h n d', h = h)
        pos = einsum('H i d, h j d -> H i j', q+self.pos_bias, pos)
        pos = _relative_shift(pos, h)
        qk += pos

        if _exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(qk.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            qk.masked_fill_(~mask, max_neg_value)

        attn = F.softmax(qk, dim=-1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.Wout(out)

class PreNorm(nn.Module):
    def __init__(self, fn, q_dim, kv_dim = None):
        super().__init__()
        self.fn = fn
        self.q_norm = nn.LayerNorm(q_dim)
        self.kv_norm = nn.LayerNorm(kv_dim) if _exists(kv_dim) else None

    def forward(self, x, **kwargs):
        x = self.q_norm(x)
        if _exists(self.kv_norm):
            kv = kwargs['kv']
            kv_normed = self.kv_norm(kv)
            kwargs.update(kv=kv_normed)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = .05):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, dim*mult),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim*mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff(x)

def TransformerBlock(input_dim, num_head, head_dim, mult, dropout):
    return nn.Sequential(
        PreNorm(Attention(
            q_dim=input_dim,
            num_head=num_head,
            head_dim=head_dim,
            dropout=dropout,
        ), q_dim=input_dim),
        PreNorm(FeedForward(
            dim=num_head*head_dim,
            mult=mult,
            dropout=dropout,
        ), q_dim=input_dim)
    )

# Convolution
class Residual(nn.Module):
    def __init__(self, module:nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return x + self.module(x, *args, **kwargs)
    
def ConvBlock(input_dim, output_dim = None, kernel_size = 1):
    return nn.Sequential(
        nn.BatchNorm1d(input_dim),
        nn.GELU(),
        nn.Conv1d(
            in_channels=input_dim,
            out_channels=_default(output_dim, input_dim),
            kernel_size=kernel_size,
            padding=kernel_size//2,
        )
    )

def Stem(channels):
    return nn.Sequential(
        Rearrange('b n d -> b d n'),
        nn.Conv1d(4, channels//2, 15, padding="same"),
        Residual(ConvBlock(channels//2)),
        AttentionPool(channels//2, pool_size=2)
    )

def _exponential_linspace_int(start, end, num, divisible_by=1, reverse=False):
    def _round(x):
        return int(round(x/divisible_by)*divisible_by)
    base = math.exp(math.log(end/start) / (num))
    result = [_round(start*base**i) for i in range(num+1)]
    if reverse:
        result.reverse()
        return result
    return result

def ConvTower(channels, num_conv):
    filter_list = _exponential_linspace_int(
        start=channels//2,
        end=channels,
        num=num_conv,
        divisible_by=2,
    )
    conv_layers = []
    for input_dim, output_dim in zip(filter_list[:-1], filter_list[1:]):
        conv_layers.append(nn.Sequential(
            ConvBlock(input_dim, output_dim, kernel_size=5),
            Residual(ConvBlock(output_dim, output_dim, 1)),
            AttentionPool(output_dim, pool_size=2)
        ))
    return nn.Sequential(*conv_layers)

class AttentionPool(nn.Module):
    def __init__(self, input_dim, pool_size = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = pool_size)
        self.to_attn_logits = nn.Parameter(torch.eye(input_dim))

    def forward(self, x):
        _, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = x[:,:,:-remainder]

        attn_logits = einsum('b d n, d e -> b e n', x, self.to_attn_logits)
        x = self.pool_fn(x)
        logits = self.pool_fn(attn_logits)

        attn = logits.softmax(dim = -1)
        return (x * attn).sum(dim = -1)

# Head  
class Pointwise(nn.Module):
    def __init__(self, trim, channels, dropout) -> None:
        super().__init__()
        self.trim = trim

        self.conv_block = nn.Sequential(
            nn.BatchNorm1d(896),
            nn.GELU(),
            nn.Linear(channels, channels*2, bias=False),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.trim>0:
            x = x[..., self.trim:-self.trim, :]
        x = self.conv_block(x)
        x = self.dropout(x)
        return F.gelu(x)
    
class EnformerOutputHead(nn.Module):
    def __init__(self, channels, output_dim) -> None:
        super().__init__()

        self.conv = nn.Linear(channels*2, output_dim, bias=False)
    
    def forward(self, x):
        x = self.conv(x)
        return F.softplus(x)

# Perceiver
class CrossAttentionBlock(nn.Module):
    def __init__(self, latent_dim, input_dim, num_head=8, dropout=.4):
        super().__init__()

        self.cross_attn = PreNorm(Attention(
            q_dim=latent_dim,
            kv_dim=input_dim,
            num_head=num_head,
            head_dim=latent_dim//num_head,
            dropout=dropout,
        ), q_dim=latent_dim)

        self.ff = PreNorm(FeedForward(
            dim=latent_dim,
            mult=4,
            dropout=dropout,
        ), q_dim=latent_dim)
    
    def forward(self, x, kv):
        x = self.cross_attn(x, kv=kv)
        return self.ff(x)