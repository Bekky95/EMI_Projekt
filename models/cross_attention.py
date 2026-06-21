import torch.nn as nn
import torch


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_t=384, d_i=512, d=256, n_classes=3):
        super().__init__()

        self.t_proj = nn.Linear(d_t, d)
        self.i_proj = nn.Linear(d_i, d)

        self.attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=4,
            batch_first=True
        )

        self.classifier = nn.Linear(d, n_classes)

    def forward(self, t, i):
        t = self.t_proj(t).unsqueeze(1)
        i = self.i_proj(i).unsqueeze(1)

        x, _ = self.attn(t, i, i)

        x = x.squeeze(1)
        return self.classifier(x)