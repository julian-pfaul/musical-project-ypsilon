import torch
import torch.nn as nn
import mamba_ssm

import numpy as np

from .feed_forward_layer import *
from .residual_connection import *

class Decoder(nn.Module):
    def __init__(self, context_features, d_model, d_ff, num_layers, dropout):
        super().__init__()

        self.num_layers = num_layers

        self.unifier = nn.Bilinear(context_features, d_model, d_model)

        self.feed_forward_layers = nn.ModuleList([FeedForwardLayer(d_model, d_ff, dropout) for _ in range(self.num_layers)])
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(self.num_layers)])
        self.mambas = nn.ModuleList([mamba_ssm.Mamba(d_model, 16, 4, 2) for _ in range(self.num_layers)])

    def forward(self, x, context):
        x = self.unifier(context, x)

        for layer in range(self.num_layers):
            x = self.residual_connections[layer](x, self.mambas[layer])
            x = self.residual_connections[layer](x, self.feed_forward_layers[layer])

        return x
