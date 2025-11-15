import torch
import torch.nn as nn
from src.architecture.ViT.attention import MultiHeadAttention
from src.architecture.ViT.preprocessing import PatchEmbedding, PositionalEncoding    

class VisionTransformer(nn.Module):
    def __init__(
            self, 
            img_size=30, 
            patch_size=1, 
            embed_dim=128, 
            num_heads=4, 
            depth=6, 
            mlp_dim=256,
            in_channels=1
    ):
        super().__init__()

        ########################
        #   Patch Embeddings   #
        ########################
        
        self.patch_embedding = PatchEmbedding(
            img_size=img_size, 
            patch_size=patch_size, 
            in_channels=in_channels, 
            embed_dim=embed_dim
        )

        ########################
        #    Context Token     #
        ########################

        self.c_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        #############################
        #    Positional Encoding    #
        #############################

        self.pos_encoding = PositionalEncoding(embed_dim)

        self.dropout = nn.Dropout(0.1) # customizeable?

        ####################################
        #    Transformer Encoder Blocks    #
        ####################################

        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim) 
            for _ in range(depth)
        ])

        #########################
        #    Final LayerNorm    #
        #########################

        self.fLayerNorm = nn.LayerNorm(embed_dim)

    def forward(
            self, 
            x, 
            mask
    ):
        #################################
        #    B = batch size             #
        #    N = number of patchs       #
        #    D = embedding dimension    #
        #################################

        B = x.size(0) 

        x = self.patch_embedding(x) # (B, N, D)

        # Flatten mask 
        if mask is not None:
            mask = mask.reshape(x.size(0), -1)  # (B, H, W) -> (B, N)

        # Prepend context token mask
        if mask is not None:
            c_mask = torch.zeros((mask.size(0), 1), dtype=torch.bool, device=mask.device)
            mask = torch.cat([c_mask, mask], dim=1)  # (B, 1+N)
            mask = mask.to(torch.bool)

        # Convert mask to padding mask
        # mask: (B, H, W) or None
        if mask is not None:
            # Convert spatial mask (B,H,W) → token mask (B,N)
            mask_flat = mask.flatten(1)      # (B, H*W)
            key_padding_mask = ~mask_flat    # invert boolean mask
        else:
            key_padding_mask = None



        # Prepend context
        c = self.c_token.expand(B, -1, -1)
        x = torch.cat([c, x], dim=1)

        x = self.pos_encoding(x) 
        x = self.dropout(x)

        for block in self.transformer_blocks: # run through sequence of vision transformer blocks
            x = block(x, key_padding_mask)
        
        x = self.fLayerNorm(x)

        return x # (B, 1+N, D)
    
    def forward_grid(
            self, 
            x, 
            mask=None
    ):
        tokens = self.forward(x, mask)
        return tokens[:, 0] # context embedding

class TransformerEncoderBlock(nn.Module):
    def __init__(
            self, 
            embed_dim, 
            num_heads, 
            mlp_dim
    ):
        super().__init__()
        
        ###########################
        #   Attention Mechanism   #
        ###########################

        self.attn = MultiHeadAttention(embed_dim, num_heads)

        ###########
        #   MLP   #
        ###########

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), # increase dimensionality
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim) # return to embedding dimension
        )
        
        ###################
        #   Layer Norms   #
        ###################

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):

        # Apply attention to normalized input
        x = x + self.attn(self.norm1(x), key_padding_mask=mask)
        # Normalize and run through MLP
        x = x + self.mlp(self.norm2(x))
        return x
    

class ConditionalTransformerEncoderBlock(nn.Module):
    """
    Encodes test input grid using shared VisionTransformer 

    Conditioned on aggregate context vector C from example pairs
    """

    def __init__(
            self, 
            vit_encoder: nn.Module
    ):
        super().__init__()
        self.vit = vit_encoder
        self.embed_dim = vit_encoder.c_token.size(-1)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
            self, 
            I_test, 
            mask_test, 
            C
    ):
        ###############################
        #   B = batch size            #
        #   N = sequence length       #
        #   D = embedding dimension   #
        #   H = height                #
        #   W = width                 #
        ###############################
        
        B = I_test.size(0)

        ####################
        #   Flatten Mask   #
        ####################

        if mask_test is not None:
            mask_test = mask_test.reshape(B, -1)  # (B, H*W=N)
        
        # Project C -> (B, 1, D) (one per batch)
        C_proj = self.c_proj(C).unsqueeze(1)

        #############
        #   Embed   #
        #############

        patches = self.vit.patch_embedding(I_test)  # (B, N, D)

        ###################
        #   Build Input   #
        ###################

        tokens = torch.cat([C_proj, patches], dim=1)  # (B, 1+N, D)

        ##################
        #   Build Mask   #
        ##################

        if mask_test is not None:
            c_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask_test.device)  # (B, 1)
            key_padding_mask = torch.cat([c_mask, ~mask_test], dim=1)  # (B, 1+N)
        else:
            key_padding_mask = None

        ###########################
        #   Positional Encoding   #
        #   and Dropout           #
        ###########################

        tokens = self.vit.pos_encoding(tokens)
        tokens = self.vit.dropout(tokens)

        ####################################
        #   Inference Transformer Blocks   #
        ####################################

        for block in self.vit.transformer_blocks:
            tokens = block(tokens, key_padding_mask)

        #####################
        #   Normalization   #
        #####################

        tokens = self.vit.fLayerNorm(tokens)

        return tokens  # (B, 1+N, D)