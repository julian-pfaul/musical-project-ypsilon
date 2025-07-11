import torch
import torch.nn as nn

from .layer_normalization import *

class ResidualConnection(nn.Module):
    def __init__(self, dropout):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
