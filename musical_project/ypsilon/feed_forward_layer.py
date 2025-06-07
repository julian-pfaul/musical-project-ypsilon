import torch
import torch.nn as nn

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()

        self.proj1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.proj0 = nn.Linear(d_ff, d_model)

        for layer in [self.proj1, self.proj0]:
            nn.init.xavier_uniform_(layer.weight)

            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.proj0(self.dropout(torch.relu(self.proj1(x))))
