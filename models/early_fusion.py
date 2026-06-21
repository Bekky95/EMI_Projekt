import torch.nn as nn
import torch


class EarlyFusion(nn.Module):
    def __init__(self, d_t=384, d_i=512, hidden=256, n_classes=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_t + d_i, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, t, i):
        x = torch.cat([t, i], dim=-1)
        return self.net(x)