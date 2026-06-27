import torch.nn as nn
import torch

class EarlyFusion(nn.Module):
    def __init__(self, d_t=768, d_i=128, n_classes=8, hidden=128):
        super().__init__()

        self.norm_a = nn.LayerNorm(d_t)
        self.norm_v = nn.LayerNorm(d_i)
        self.head = nn.Sequential(
            nn.Linear(d_t + d_i, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, h_t, h_i):
        z = torch.cat([self.norm_a(h_t), self.norm_v(h_i)], dim=-1)
        return self.head(z)


