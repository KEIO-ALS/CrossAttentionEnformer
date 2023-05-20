import torch

from models.Enformer import Enformer
from data.dataset_utils import load_basenji2
from config import get_config


trainloader, testloader, validloader = load_basenji2()


model = Enformer(get_config("models", "Enformer", "param"))

for data in trainloader:
    x, y, o = data
    print(x.shape, o)
    out = model(x, o)
    print(out.shape)
    break