import torch
import torch.nn as nn

from src.encoders import ImageEncoder, TextEncoder


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key_value):
        attn_out, _ = self.attn(query, key_value, key_value)
        return self.norm(query + attn_out)


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=512, num_classes=3):
        super().__init__()

        self.img_enc = ImageEncoder(dim)
        self.txt_enc = TextEncoder(dim)

        self.txt_to_img = CrossAttention(dim)
        self.img_to_txt = CrossAttention(dim)

        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):

        img = self.img_enc(image).unsqueeze(1)  # (B,1,D)
        txt = self.txt_enc(input_ids, attention_mask).unsqueeze(1)

        # bidirektionale cross attention
        img_fused = self.txt_to_img(img, txt)
        txt_fused = self.img_to_txt(txt, img)

        fused = torch.cat([
            img_fused.squeeze(1),
            txt_fused.squeeze(1)
        ], dim=-1)

        return self.classifier(fused)

# class CrossAttention(nn.Module):
#     def __init__(self, d):
#         super().__init__()
#          # getrennte Projektionen
#         self.W_q = nn.Linear(d, d)
#         self.W_k = nn.Linear(d, d)
#         self.W_v = nn.Linear(d, d)
#
#         # Skalierung
#         self.scale = d ** 0.5
#
#     def forward(self, x_query, x_kv):
#         """
#         x_query: (B, n_A, d)
#         x_kv   : (B, n_B, d)
#         """
#
#         # Queries aus Modalitaet A
#         Q = self.W_q(x_query)   # (B, n_A, d)
#
#         # Keys + Values aus Modalitaet B
#         K = self.W_k(x_kv)      # (B, n_B, d)
#         V = self.W_v(x_kv)      # (B, n_B, d)
#
#         # Attention Scores
#         scores = Q @ K.transpose(-2, -1) / self.scale
#         # (B, n_A, n_B)
#
#         # Normalisierung ueber alle Keys
#         attn = F.softmax(scores, dim=-1)
#
#         # gewichtete Kombination der Values
#         out = attn @ V
#         # (B, n_A, d)
#
#         return out, attn
#
# B, d = 4, 16
#
# text  = torch.randn(B, 6, d)   # n_A = 6
# audio = torch.randn(B, 8, d)   # n_B = 8
#
#
# # Test
# ca = CrossAttention(d)
# out, A = ca(text, audio)
# print('output:', out.shape, '  <- erwartet (4, 6, 16)')
# print('attn  :', A.shape,   '  <- erwartet (4, 6, 8)  Text fragt Audio')