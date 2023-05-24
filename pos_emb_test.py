import torch
import numpy as np

from models.Module import get_positional_embed

seq_len = 100
feature_size = 1024
device = torch.device("cpu")


x = np.arange(seq_len)
y = get_positional_embed(seq_len, feature_size, device)

print(y.shape)