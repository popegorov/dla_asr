import torch.nn as nn
from torch import Tensor


class RNNBlock(nn.Module):
    rnns = {
        "lstm": nn.LSTM,
        "gru": nn.GRU,
        "rnn": nn.RNN,
    }

    def __init__(
        self,
        input_size: int,
        hidden_state_dim: int = 512,
        rnn_type: str = "gru",
        bidirectional: bool = True,
        dropout_p: float = 0.1,
    ):
        super(RNNBlock, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.batch_norm = nn.BatchNorm1d(input_size)
        rnn_cell = self.rnns[rnn_type]
        self.bidirectional = bidirectional
        self.rnn = rnn_cell(
            input_size=input_size,
            hidden_size=hidden_state_dim,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=bidirectional,
        )

    def forward(self, spectrogram: Tensor, spectrogram_length: Tensor, **batch):
        x = self.batch_norm(spectrogram.transpose(1, 2))

        x = x.transpose(1, 2)
        x = nn.utils.rnn.pack_padded_sequence(
            x, spectrogram_length.cpu(), enforce_sorted=False, batch_first=True
        )
        x, _ = self.rnn(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return (
            outputs[..., : self.rnn.hidden_size] + outputs[..., self.rnn.hidden_size :]
            if self.bidirectional
            else outputs
        )
