import torch_audiomentations
import torch
from torch import Tensor, nn


class ColorNoise(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.AddColoredNoise(*args, **kwargs)
        self.prob = 0

    def __call__(self, data: Tensor):
        return self._aug(data.unsqueeze(1)).squeeze(1) if self.prob < torch.rand(1) else data
