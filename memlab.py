import torch

from pytorch_memlab import MemReporter
from models.model_utils import get_models

input = torch.Tensor(1, 131072, 4)

for model, config in get_models():
    print(config["name"], "model")
    reporter = MemReporter(model)
    print(reporter.report())

    out = model(input, "human")
    out.mean().backward()
    reporter.report()