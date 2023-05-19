from torch import nn
from models.Module import Stem


class Enformer(nn.Module):
    def __init__(self, param, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        channels = param["channels"]

        self.stem = Stem(channels)
    
    def forward(self, x):
        x = self.stem(x)

        return x