import torch
import torch.nn as nn
import mamba_ssm

import numpy as np

TWO_PI = 2 * np.pi
PITCH_MAX = 128.0

class Encoder(nn.Module):
    def __init__(self, num_features, d_model, dropout):
        super().__init__()

        self.num_features = num_features
        self.d_model = d_model

        self.dropout = nn.Dropout(dropout)

        mul_term = TWO_PI * torch.arange(0, self.d_model // 2)
        self.register_buffer("mul_term", mul_term)

        self.mambas = nn.ModuleList([mamba_ssm.Mamba(d_model, 16, 4, 2) for _ in range(self.num_features)])
        self.t = mamba_ssm.Mamba(d_model, 16, 4, 2)
        self.pool = nn.MaxPool1d(self.num_features)

    def forward(self, ipt):
        assert ipt.dim() == 3
        assert ipt.size(2) == self.num_features

        # ipt : (B, L, C)

        x = ipt.unsqueeze(dim=-1).expand(-1, -1, -1, self.d_model // 2)
        x = x * self.mul_term.cuda().requires_grad_(False)
        x = x.permute((0, 1, 3, 2))

        proj_x = torch.zeros(ipt.size(0), ipt.size(1), self.d_model, self.num_features).to(ipt.device)
        proj_x[:, :, 0::2, :] = torch.sin(x)
        proj_x[:, :, 1::2, :] = torch.cos(x)

        out = torch.dstack([self.mambas[feature](proj_x[:, :, :, feature]) for feature in range(self.num_features)])

        return self.pool(self.dropout(out)) # ipt : (B, L, d_model)










