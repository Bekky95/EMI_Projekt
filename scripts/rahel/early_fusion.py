import torch.nn as nn


class EarlyFusion(nn.Module):
    def __init__(self, d_a=768, d_v=128, n_classes=8, hidden=128):
        super().__init__()
        self.norm_a = nn.LayerNorm(d_a)
        self.norm_v = nn.LayerNorm(d_v)
        self.head = nn.Sequential(
        nn.Linear(d_a + d_v, hidden),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, n_classes),
    )

    def forward(self, h_a, h_v):
        z = torch.cat([self.norm_a(h_a), self.norm_v(h_v)], dim=-1)
        return self.head(z)