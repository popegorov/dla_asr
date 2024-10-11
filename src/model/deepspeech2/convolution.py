import torch.nn as nn
from torch import Tensor

class Convolution(nn.Module):
    def __init__(
        self, in_channels: int = 1, hid_channels: int = 32, out_channels: int = 96):
        super(Convolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hid_channels,
                kernel_size=(41, 11),
                stride=(2, 2),
                padding=(20, 5),
                bias=False,
            ),
            nn.BatchNorm2d(hid_channels),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(
                hid_channels,
                hid_channels,
                kernel_size=(21, 11),
                stride=(2, 1),
                padding=(10, 5),
                bias=False,
            ),
            nn.BatchNorm2d(hid_channels),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(
                hid_channels,
                out_channels,
                kernel_size=(21, 11),
                stride=(2, 1),
                padding=(10, 5),
                bias=False,
            ),
        )

    def forward(self, spectrogram: Tensor, **batch):
        """
        inputs: torch.FloatTensor (batch, dimension, time)
        input_lengths: torch.IntTensor (batch)
        """
        outputs = self.conv(spectrogram.unsqueeze(1))
        batch_size, channels, dimension, seq_lengths = outputs.size()
        outputs = outputs.view(batch_size, channels * dimension, seq_lengths)

        return outputs

    def get_ouput_size(self, n_feats):
        input_size = n_feats
        input_size = (input_size - 1) // 2 + 1
        input_size = (input_size - 1) // 2 + 1
        input_size = (input_size - 1) // 2 + 1
        return input_size * self.out_channels

    def get_seq_len(self, seq_lengths: Tensor):
        """
        Calculate convolutional neural network receptive formula
        Args:
            module (torch.nn.Module): module of CNN
            seq_lengths (torch.IntTensor): The actual length of each sequence in the batch
        Returns: seq_lengths
            - **seq_lengths**: Sequence length of output from the module
        """

        for module_ in self.conv:
            if isinstance(module_, nn.Conv2d):
                numerator = (
                    seq_lengths
                    + 2 * module_.padding[1]
                    - module_.dilation[1] * (module_.kernel_size[1] - 1)
                    - 1
                )
                seq_lengths = numerator.float() / float(module_.stride[1])
                seq_lengths = seq_lengths.int() + 1

        return seq_lengths.int()
