import torch
from torch import nn
from models.Module import Stem, ConvTower, TransformerBlock, Pointwise, EnformerOutputHead

from einops.layers.torch import Rearrange


class Enformer(nn.Module):
    def __init__(self, param, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        channels = param["channels"]
        num_conv = param["num_conv"]
        num_attn = param["num_attn"]
        dropout_attn = param["dropout_attn"]
        dropout_output = param["dropout_output"]

        self.encode = nn.Sequential(
            Stem(channels),
            ConvTower(channels, num_conv),
            Rearrange("b d n -> b n d"),
        )

        self.transformer_tower = nn.Sequential(
            *[TransformerBlock(channels, 8, channels//8, 4, dropout_attn) for _ in range(num_attn)]
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
        x = self.encode(x)
        x = self.transformer_tower(x)
        if organisms == "human":
            x = self.decode_human(x)
        elif organisms == "mouse":
            x = self.decode_mouse(x)
        return x