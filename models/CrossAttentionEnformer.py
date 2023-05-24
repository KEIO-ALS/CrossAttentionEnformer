import torch
from torch import nn
from models.Module import CrossAttentionBlock, TransformerBlock, Pointwise, EnformerOutputHead
from einops import repeat, rearrange
from einops.layers.torch import Rearrange


class CrossAttentionEnformer(nn.Module):
    def __init__(self, param, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        token_dim = param["token_dim"]
        num_latents = param["num_latents"]
        channels = param["channels"]
        num_attn = param["num_attn"]
        dropout_attn = param["dropout_attn"]
        dropout_output = param["dropout_output"]

        self.latents = nn.Parameter(torch.randn(num_latents, channels))

        self.token_emb = nn.Sequential(
            nn.Embedding(4, token_dim),
            nn.Dropout(dropout_attn)
        )
        self.pos_emb = nn.Embedding(131072, token_dim)

        self.encode = CrossAttentionBlock(channels, input_dim=token_dim)

        self.transformer_tower = nn.Sequential(
            *[TransformerBlock(channels, 8, 192, 4, dropout_attn) for _ in range(num_attn)]
        )

        self.decode_human = nn.Sequential(
            Pointwise(64, channels, dropout_output),
            EnformerOutputHead(channels, 5313),
        )

        self.decode_mouse = nn.Sequential(
            Pointwise(64, channels, dropout_output),
            EnformerOutputHead(channels, 1643),
        )
    
    def forward(self, x, organisms):
        b, _, _ = x.shape
        latent_array = repeat(self.latents, "n d -> b n d", b=b)

        x = self.token_emb(torch.argmax(x, dim=-1))
        pos = self.pos_emb(torch.arange(x.shape[1], device=x.device))
        pos = rearrange(pos, "n d -> () n d")
        x += pos
        
        x = self.encode(latent_array, x)
        x = self.transformer_tower(x)
        if organisms == "human":
            x = self.decode_human(x)
        elif organisms == "mouse":
            x = self.decode_mouse(x)
        return x