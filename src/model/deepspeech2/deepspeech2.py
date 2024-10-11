import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.model.deepspeech2.convolution import Convolution
from src.model.deepspeech2.modules import Linear
from src.model.deepspeech2.rnn import RNNBlock


class DeepSpeech2(nn.Module):
    def __init__(
        self,
        n_tokens: int,
        n_feats: int,
        rnn_type="gru",
        num_rnn_layers: int = 5,
        rnn_hidden_dim: int = 512,
        dropout_p: float = 0.1,
        bidirectional: bool = True,
        device: torch.device = "cuda",
    ):
        super(DeepSpeech2, self).__init__()
        self.device = device
        self.conv = Convolution()

        self.num_rnn_layers = num_rnn_layers

        self.rnn_layers = nn.Sequential()
        for i in range(self.num_rnn_layers):
            input_size = self.conv.get_ouput_size(n_feats) if not i else rnn_hidden_dim
            self.rnn_layers.add_module(
                f"RNN {i}",
                RNNBlock(
                    input_size=input_size,
                    hidden_state_dim=rnn_hidden_dim,
                    rnn_type=rnn_type,
                    bidirectional=bidirectional,
                    dropout_p=dropout_p,
                ),
            )

        self.fc = Linear(rnn_hidden_dim, n_tokens, bias=False)

    def forward(self, spectrogram: Tensor, spectrogram_length: Tensor, **batch):
        outputs = self.conv(spectrogram)  # BxDxT
        output_lengths = self.conv.get_seq_len(spectrogram_length)
        outputs = outputs.permute(0, 2, 1).contiguous()  # BxTxD

        for rnn_layer in self.rnn_layers:
            outputs = rnn_layer(outputs, output_lengths)

        outputs = self.fc(outputs).log_softmax(dim=-1)
        return {
            "log_probs": F.log_softmax(outputs, dim=-1),
            "log_probs_length": output_lengths,
        }
