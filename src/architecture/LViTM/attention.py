import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x, key_padding_mask=None):

        out, _ = self.attn(
            x, x, x,
            key_padding_mask=key_padding_mask
        )

        return out