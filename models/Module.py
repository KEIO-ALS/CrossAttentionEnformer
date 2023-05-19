import torch
from torch import nn, einsum
from torch.nn import functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce, Rearrange


def _exists(val):
    return val is not None

def _default(val, d):
    return val if _exists(val) else d



class Attention(nn.Module):
    def __init__(
            self,
            q_dim,
            kv_dim=None,
            num_head=8,
            head_dim=64,
            dropout=0.,
    ) -> None:
        super().__init__()

        inner_dim = head_dim*num_head
        kv_dim = _default(kv_dim, q_dim)

        self.scale = head_dim**-0.5
        self.num_head = num_head

        self.Wq = nn.Linear(q_dim, inner_dim, bias=False)
        self.Wk = nn.Linear(kv_dim, inner_dim, bias=False)
        self.Wv = nn.Linear(kv_dim, inner_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.Wout = nn.Linear(inner_dim, q_dim)

    def forward(self, x: torch.Tensor, kv=None, mask=None):
        h = self.num_head

        q, k, v = self.Wq(x), self.Wk(_default(kv, x)), self.Wv(_default(kv, x))

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        qk = einsum('b i d, b j d -> b i j', q, k) * self.scale

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
    

class Residual(nn.Module):
    def __init__(self, module:nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return x + self.module(x, *args, **kwargs)
    
class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x
    
def ConvBlock(input_dim, output_dim = None, kernel_size = 1):
    return nn.Sequential(
        nn.BatchNorm1d(input_dim),
        GELU(),
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
        nn.Conv1d(4, channels//2, 15, padding=7),
        Residual(ConvBlock(channels//2)),
        AttentionPool(channels//2, pool_size=2)
    )

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