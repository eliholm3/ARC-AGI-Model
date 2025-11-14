import torch
import torch.nn as nn
from attention import MultiHeadAttention
from preprocessing import PatchEmbedding, PositionalEncoding


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        # Attention mechanism
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        # Extract patterns in a more rich latent space
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), # increase dimensionality
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim) # return to embedding dimension
        )
        # Normalize (subtract mean and divide by standard deviation)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Apply attention to normalized input
        x = x + self.attn(self.norm1(x))
        # Normalize and run through MLP
        x = x + self.mlp(self.norm2(x))
        return x
    

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=10, embed_dim=768, num_heads=8, depth=6, mlp_dim=1024):
        super().__init__()
        # Initial embedding (input_dim -> embed_dim)
        # ! DIMENSIONS NEED TO BE CHANGED
        self.patch_embedding = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        # Matrix positioning
        self.pos_encoding = PositionalEncoding(embed_dim, (img_size // patch_size) ** 2)
        # Sequential transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)
        ])

    def forward(self, x):
        x = self.patch_embedding(x) # embed patch
        x = self.pos_encoding(x) # positional encoding
        for block in self.transformer_blocks: # run through sequence of vision transformer blocks
            x = block(x)
        return x # (B, num_patches, embed_dim)
