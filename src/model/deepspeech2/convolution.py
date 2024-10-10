# MIT License
#
# Copyright (c) 2021 Soohwan Kim.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

# class MaskCNN(nn.Module):
#     """
#     Masking Convolutional Neural Network
#     Adds padding to the output of the module based on the given lengths.
#     This is to ensure that the results of the model do not change when batch sizes change during inference.
#     Input needs to be in the shape of (batch_size, channel, hidden_dim, seq_len)
#     Refer to https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py
#     Copyright (c) 2017 Sean Naren
#     MIT License
#     Args:
#         sequential (torch.nn): sequential list of convolution layer
#     Inputs: inputs, seq_lengths
#         - **inputs** (torch.FloatTensor): The input of size BxCxHxT
#         - **seq_lengths** (torch.IntTensor): The actual length of each sequence in the batch
#     Returns: output, seq_lengths
#         - **output**: Masked output from the sequential
#         - **seq_lengths**: Sequence length of output from the sequential
#     """
#     def __init__(self, sequential: nn.Sequential) -> None:
#         super(MaskCNN, self).__init__()
#         self.sequential = sequential

#     def forward(self, inputs: Tensor, seq_lengths: Tensor) -> Tuple[Tensor, Tensor]:
#         output = None

#         for module in self.sequential:
#             output = module(inputs)
#             mask = torch.BoolTensor(output.size()).fill_(0)

#             if output.is_cuda:
#                 mask = mask.cuda()

#             seq_lengths = self._get_sequence_lengths(module, seq_lengths)

#             for idx, length in enumerate(seq_lengths):
#                 length = length.item()

#                 if (mask[idx].size(2) - length) > 0:
#                     mask[idx].narrow(dim=2, start=length, length=mask[idx].size(2) - length).fill_(1)

#             output = output.masked_fill(mask, 0)
#             inputs = output

#         return output, seq_lengths

#     def _get_sequence_lengths(self, module: nn.Module, seq_lengths: Tensor) -> Tensor:
#         """
#         Calculate convolutional neural network receptive formula
#         Args:
#             module (torch.nn.Module): module of CNN
#             seq_lengths (torch.IntTensor): The actual length of each sequence in the batch
#         Returns: seq_lengths
#             - **seq_lengths**: Sequence length of output from the module
#         """
#         if isinstance(module, nn.Conv2d):
#             numerator = seq_lengths + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1
#             seq_lengths = numerator.float() / float(module.stride[1])
#             seq_lengths = seq_lengths.int() + 1

#         elif isinstance(module, nn.MaxPool2d):
#             seq_lengths >>= 1

#         return seq_lengths.int()


class DeepSpeech2Extractor(nn.Module):
    def __init__(
        self, in_channels: int = 1, hid_channels: int = 32, out_channels: int = 96
    ) -> None:
        super(DeepSpeech2Extractor, self).__init__()
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

    def forward(self, spectrogram: Tensor, **batch) -> Tuple[Tensor, Tensor]:
        """
        inputs: torch.FloatTensor (batch, dimension, time)
        input_lengths: torch.IntTensor (batch)
        """
        outputs = self.conv(spectrogram.unsqueeze(1))
        batch_size, channels, dimension, seq_lengths = outputs.size()
        outputs = outputs.view(batch_size, channels * dimension, seq_lengths)

        return outputs

    def get_output_dim(self, n_feats):
        input_size = n_feats
        input_size = (input_size - 1) // 2 + 1
        input_size = (input_size - 1) // 2 + 1
        input_size = (input_size - 1) // 2 + 1
        return input_size * self.out_channels

    def _get_sequence_lengths(self, seq_lengths: Tensor) -> Tensor:
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
