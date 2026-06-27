import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

class CrossAttention(nn.Module):
    def __init__(self, d):
        super().__init__()

         # getrennte Projektionen
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)

        # Skalierung
        self.scale = d ** 0.5

    def forward(self, x_query, x_kv):
        """
        x_query: (B, n_A, d)
        x_kv   : (B, n_B, d)
        """

        # Queries aus Modalitaet A
        Q = self.W_q(x_query)   # (B, n_A, d)

        # Keys + Values aus Modalitaet B
        K = self.W_k(x_kv)      # (B, n_B, d)
        V = self.W_v(x_kv)      # (B, n_B, d)

        # Attention Scores
        scores = Q @ K.transpose(-2, -1) / self.scale
        # (B, n_A, n_B)

        # Normalisierung ueber alle Keys
        attn = F.softmax(scores, dim=-1)

        # gewichtete Kombination der Values
        out = attn @ V
        # (B, n_A, d)

        return out, attn

    # ChatGPT Lösung zu Aufgabe
    # class CrossAttentionFusion(nn.Module):
    #     def __init__(self, d_t=384, d_i=512, d=256, n_classes=3):
    #         super().__init__()
    #
    #         self.t_proj = nn.Linear(d_t, d)
    #         self.i_proj = nn.Linear(d_i, d)
    #
    #         self.attn = nn.MultiheadAttention(embed_dim=d, num_heads=4, batch_first=True)
    #
    #         self.classifier = nn.Sequential(
    #             nn.Linear(d, n_classes)
    #         )
    #
    #     def forward(self, t, i):
    #         t = self.t_proj(t).unsqueeze(1)  # (B,1,d)
    #         i = self.i_proj(i).unsqueeze(1)  # (B,1,d)
    #
    #         x, _ = self.attn(t, i, i)  # text queries image
    #
    #         x = x.squeeze(1)
    #         return self.classifier(x)

"""
example code from Kaggle
"""
class CrossAttentionFusion(nn.Module):
    """
    Simple concat fusion — more stable than cross-attention on small datasets.
    Concatenates CLIP image + RoBERTa text features, projects to fusion_dim.
    """
    def __init__(self, img_dim, txt_dim, fusion_dim, **kwargs):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(img_dim + txt_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

    def forward(self, img_feat, txt_feat):
        combined = torch.cat([img_feat, txt_feat], dim=-1)  # (B, 512+768)
        return self.proj(combined)                           # (B, fusion_dim)

print("Concat Fusion ready ✅")