import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.batch_norm = nn.BatchNorm1d(in_features)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.batch_norm(x.transpose(1, 2))
        return self.linear(x.transpose(1, 2))
