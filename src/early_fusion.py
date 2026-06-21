import torch
import torch.nn as nn

from src.encoders import ImageEncoder, TextEncoder


class EarlyFusionModel(nn.Module):
    def __init__(self, dim=512, num_classes=3):
        super().__init__()
        self.img_enc = ImageEncoder(dim)
        self.txt_enc = TextEncoder(dim)

        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    # def __init__(self, d_a=768, d_v=128, n_classes=8, hidden=128):
    #     super().__init__()
    #     self.norm_a = nn.LayerNorm(d_a)
    #     self.norm_v = nn.LayerNorm(d_v)
    #     self.head = nn.Sequential(
    #         nn.Linear(d_a + d_v, hidden),
    #         nn.ReLU(),
    #         nn.Dropout(0.3),
    #         nn.Linear(hidden, n_classes),
    #     )

    def forward(self, image, input_ids, attention_mask):
        img_feat = self.img_enc(image)
        txt_feat = self.txt_enc(input_ids, attention_mask)

        fused = torch.cat([img_feat, txt_feat], dim=-1)
        return self.classifier(fused)

    # def forward(self, h_a, h_v):
    #     z = torch.cat([self.norm_a(h_a), self.norm_v(h_v)], dim=-1)
    #     return self.head(z)
