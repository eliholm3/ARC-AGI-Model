import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    

# 2. Adding Positional Embeddings
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        x: (B, S, D)
        Returns x + sinusoidal positional encoding of shape (1, S, D)
        """
        B, S, D = x.size()
        pe = self.build_sinusoidal_pe(S, D, x.device)
        return x + pe

    def build_sinusoidal_pe(self, seq_len, dim, device):
        """
        seq_len: dynamic token count (patches)
        dim: embedding dimension
        """
        pe = torch.zeros(seq_len, dim, device=device)

        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)     # even indices
        pe[:, 1::2] = torch.cos(position * div_term)     # odd indices

        return pe.unsqueeze(0)  # (1, seq_len, dim)