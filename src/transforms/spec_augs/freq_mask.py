import torch
import torchaudio.transforms
from torch import Tensor, nn


class FreqMasking(nn.Module):
    def __init__(self, prob, freq_mask_param):
        super().__init__()
        self._aug = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.prob = prob

    def __call__(self, spectrogram: Tensor):
        return self._aug(spectrogram) if torch.rand(1) < self.prob else spectrogram
