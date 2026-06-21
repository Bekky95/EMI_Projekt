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