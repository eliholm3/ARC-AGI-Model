import torch
import torch.nn as nn
from src.architecture.ViT.attention import MultiHeadAttention
from src.architecture.ViT.preprocessing import PatchEmbedding, PositionalEncoding   
import os
from dotenv import load_dotenv 

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

        load_dotenv()
        self.DEBUGGING = os.getenv("DEBUGGING")

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
            mask=None
    ):
        #################################
        #    B = batch size             #
        #    N = number of patchs       #
        #    D = embedding dimension    #
        #################################

        B = x.size(0)

        # Patch embedding: (B, C, H, W) -> (B, N, D)
        x = self.patch_embedding(x)  # (B, N, D)
        B_, N, D = x.shape

        ##############################
        #   Build key_padding_mask   #
        ##############################
        # We expect `mask` to be a "valid" mask:
        #   - shape (B, H, W) or (B, N)
        #   - True where the token is VALID (not padding)

        if mask is not None:
            # If spatial (B, H, W), flatten to (B, N)
            if mask.dim() == 3:
                mask = mask.view(B_, -1)      # (B, N)
            else:
                mask = mask.view(B_, -1)      # (B, N)

            # Sanity check: mask length must match number of patches
            if mask.size(1) != N:
                raise ValueError(
                    f"VisionTransformer: mask length {mask.size(1)} "
                    f"does not match number of patches {N}"
                )

            # Prepend a valid mask entry for the context token
            c_mask = torch.ones(B_, 1, dtype=torch.bool, device=mask.device)  # context is always valid
            keep_mask = torch.cat([c_mask, mask.to(torch.bool)], dim=1)       # (B, 1+N), True = keep

            # PyTorch attention expects key_padding_mask: True = PAD
            key_padding_mask = ~keep_mask                                     # (B, 1+N)
        else:
            key_padding_mask = None

        ########################
        #   Prepend context    #
        ########################

        c = self.c_token.expand(B_, -1, -1)   # (B, 1, D)
        x = torch.cat([c, x], dim=1)          # (B, 1+N, D)

        x = self.pos_encoding(x)
        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x, key_padding_mask)

        x = self.fLayerNorm(x)

        return x  # (B, 1+N, D)

    
    def forward_grid(
            self, 
            x, 
            mask=None
    ):
        
        # print("\n[ViT.forward_grid] x input:", x.shape)
        # if mask is not None:
        #     if self.DEBUGGING:
        #         print("[ViT.forward_grid] mask shape:", mask.shape)

        tokens = self.forward(x, mask)
        ctx = tokens[:,0]
        # if self.DEBUGGING:
        #     print("[ViT.forward_grid] ctx mean/std:", ctx.mean().item(), ctx.std().item())

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