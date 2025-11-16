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
        
        print("\n[ViT.forward_grid] x input:", x.shape)
        if mask is not None:
            if self.DEBUGGING:
                print("[ViT.forward_grid] mask shape:", mask.shape)

        tokens = self.forward(x, mask)
        ctx = tokens[:,0]
        if self.DEBUGGING:
            print("[ViT.forward_grid] ctx mean/std:", ctx.mean().item(), ctx.std().item())

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

import torch
import torch.nn as nn
from src.architecture.LViTM.attention import MultiHeadAttention


class LViTMBlock(nn.Module):
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

        #################
        #   Normalize   #
        #################

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
            self, 
            x,  # (B, L, D)
            key_padding_mask=None  # (B, L)
    ):
        # Apply attention to normalized input
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        # Normalize and run through MLP
        x = x + self.mlp(self.norm2(x))
        return x
    

class LargeVisionTransformerModel(nn.Module):
    def __init__(
            self, 
            embed_dim,
            num_heads,
            mlp_dim,
            depth,
            num_proposals,
            z_dim
    ):
        super().__init__()

        #################
        #   Variables   #
        #################

        self.embed_dim = embed_dim
        self.num_proposals = num_proposals
        self.z_dim = z_dim

        #######################
        #   Proposal Tokens   #
        #######################

        self.proposal_tokens = nn.Parameter(
            torch.randn(1, num_proposals, embed_dim)
        )

        # No positional encoding
        self.pos_encoding = None

        ##########################
        #   Transformer Blocks   #
        ##########################

        self.blocks = nn.ModuleList(
            [LViTMBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)]
        )

        ##################################
        #   Project to Latent Proposal   #
        ##################################

        self.proposal_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, z_dim)
        )

    def forward(
            self, 
            C,
            test_tokens,
            key_padding_mask=None
    ):
        ###############################
        #   B = num batches           #
        #   S = sequence length       #
        #   D = embedding dimension   #
        #   T = num proposal tokens   #
        ###############################

        B, S, D = test_tokens.shape
        T = self.num_proposals

        ##############################
        #   Prepare Special Tokens   #
        ##############################
        
        # Expand context to a single token
        C_token = C.unsqueeze(1)

        # Place proposal tokens in each batch
        proposal_tok = self.proposal_tokens.expand(B, self.num_proposals, self.embed_dim)

        ###################
        #   Build Input   #
        ###################

        x = torch.cat([C_token, proposal_tok, test_tokens], dim=1)  # (B, 1+T+S, D)

        ##################
        #   Build Mask   #
        ##################

        if key_padding_mask is not None:
            # context token padding: (B, 1)
            c_pad = torch.zeros(B, 1, dtype=torch.bool, device=key_padding_mask.device)
            # proposal tokens padding: (B, T)
            p_pad = torch.zeros(B, T, dtype=torch.bool, device=key_padding_mask.device)
            # full mask: (B, 1 + T + S)
            full_mask = torch.cat([c_pad, p_pad, key_padding_mask], dim=1)
        else:
            full_mask = None

        
        # Optional positional encoding here

        ####################
        #   LViTM Blocks   #
        ####################

        for block in self.blocks:
            x = block(x, key_padding_mask=full_mask)
        
        ##########################
        #   Retrieve Proposals   #
        ##########################

        proposal_outs = x[:, 1:1+T, :]  # (B, 1+T+S, D) -> (B, T, D)

        ###############################
        #   Project to Latent Space   #
        ###############################

        Z = self.proposal_head(proposal_outs)

        return Z
    

"""
Inference-time sequential reasoning loop:

grid = I_test_clone  # current grid state

for step in range(T_steps):
    # 1. Re-encode current grid with conditional encoder
    test_tokens, key_padding_mask = cond_encoder(grid, C, mask_grid)

    # 2. Get proposals for this state
    Z = lvittm(C, test_tokens, key_padding_mask)  # (B, T, z_dim)

    # 3. Choose a z (e.g. best by critic, or z_0)
    z = select_proposal(Z, grid, critic, examples, C)  # shape (B, z_dim)

    # 4. Apply it with executor
    grid = executor(grid, z)  # new grid state

# final grid is your prediction
O_final = grid

"""

import torch
import torch.nn as nn

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
    def __init__(self, embed_dim, seq_len):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim))  # Adjusted for [CLS] token

    def forward(self, x):
        return x + self.pos_embed

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

import torch.nn as nn
from src.architecture.executor.FiLM import FiLM

class CNNBlock(nn.Module):
    def __init__(
            self,
            channels: int,
            z_dim: int | None = None
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            channels,  # in
            channels,  # out
            kernel_size=3, 
            padding=1
        )
        self.norm = nn.GroupNorm(channels, channels)  # normalize each channel with respect to itself
        self.activation = nn.GELU()

        self.film = FiLM(channels, z_dim) if z_dim is not None else None

    def forward(
            self,
            x,
            z=None
    ):
        # Feature extraction
        x = self.norm(self.conv(x))

        # Proposed feature modulation
        if self.film is not None:
            x = self.film(x, z)

        return self.activation(x)
import torch
import torch.nn as nn
from src.architecture.executor.CNNBlock import CNNBlock
from src.architecture.ViT.body import TransformerEncoderBlock
import os
from dotenv import load_dotenv

# Hybrid ViT and CNN
class Executor(nn.Module):
    """
    Applies a latent transformation z to an input grid
    """

    def __init__(
            self,
            embed_dim,
            num_heads,
            mlp_dim,
            depth,
            z_dim,
            hidden_channels=64,
            num_classes=10  # ARC colors
    ):
        super().__init__()

        ############################
        #   CNN Feature Enricher   #
        ############################

        self.enricher = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, padding=1),
            nn.GELU()
        )

        ######################################
        #   CNN Proposal Feature Detection   #
        ######################################

        self.cnn_blocks = nn.ModuleList([
            CNNBlock(hidden_channels, z_dim=z_dim)
            for _ in range(2)
        ])

        ##################
        #   Tokenizers   #
        ##################

        self.to_embedding = nn.Linear(hidden_channels, embed_dim)

        # Interpret proposal 
        self.z_token = nn.Linear(z_dim, embed_dim)

        ##################
        #   ViT Layers   #
        ##################

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim)
            for _ in range(depth)
        ])

        #######################
        #   CNN Discretizer   #
        #######################

        self.discretizer = nn.Sequential(
            nn.Conv2d(  # detect features in token
                embed_dim, 
                hidden_channels, 
                kernel_size=3, 
                padding=1
            ),
            nn.GELU(),
            nn.Conv2d(  # convert token features to a classification
                hidden_channels, 
                num_classes, 
                kernel_size=1
            )
        )

    def forward(
            self, 
            grid, 
            z
    ):
        ###########################
        #   grid = (B, 1, H, W)   #
        #   z = (B, z_dim)        #
        ###############################
        #   B = batch size            #    
        #   D = embedding dimension   #
        #   H = height                #
        #   W = width                 #
        ###############################

        B, _, H, W = grid.shape

        #############################################
        #   Enricher + Proposed Feature Modulator   #
        #############################################

        x = self.enricher(grid)

        for block in self.cnn_blocks:
            x = block(x, z)

        ################
        #   Tokenize   #
        ################

        # (B, C, H, W) -> (B, H*W, C)
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H*W, -1)

        tokens = self.to_embedding(x_flat)  # (B, S, D)

        # Add proposal z token
        z_token = self.z_token(z).unsqueeze(1)  # (B, 1, D) one for each batch
        tokens = torch.cat([z_token, tokens], dim=1)  # (B, 1+S, D)

        ################################
        #   ViT for Global Reasoning   #
        ################################

        for block in self.blocks:
            tokens = block(tokens, None)

        ###################
        #   Un-tokenize   #
        ###################

        # Remove z token
        x_tokens = tokens[:, 1:, :]  # (B, S, D)

        # Reshape to (B, D, H, W)
        x_feats = x_tokens.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        ##################
        #   Discretize   #
        ##################

        # Compute on the embedding dimension
        logits = self.discretizer(x_feats)  # (B, num_classes, H, W)

        print("\n[Executor] logits mean/std:", logits.mean().item(), logits.std().item())

        return logits

import torch
import torch.nn as nn

# Feature-wise modulation
class FiLM(nn.Module): 
    def __init__(
            self,
            feature_dim,
            z_dim
    ):
        super().__init__()
        self.to_gamma = nn.Linear(z_dim, feature_dim)  # scale factor
        self.to_beta = nn.Linear(z_dim, feature_dim)  # shift factor

    def forward(
            self,
            x: torch.Tensor,
            z: torch.Tensor
    ):
        ########################
        #   x = (B, C, H, W)   #
        #   z = (B, z_dim)     #
        ########################
        #   B = batch size     #       
        #   C = channels       #
        #   H = height         #
        #   W = width          #
        ########################

        ############################
        #   Compute Coefficients   #
        ############################

        # Expand across input
        gamma = self.to_gamma(z).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.to_beta(z).unsqueeze(-1).unsqueeze(-1)
        
        ###############################
        #   Apply Feture Modulation   #
        ###############################

        return x * (1 + gamma) + beta

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
import torch.nn as nn
from src.architecture.executor.FiLM import FiLM

class CNNBlock(nn.Module):
    def __init__(
            self,
            channels: int,
            z_dim: int | None = None
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            channels,  # in
            channels,  # out
            kernel_size=3, 
            padding=1
        )
        self.norm = nn.GroupNorm(channels, channels)  # normalize each channel with respect to itself
        self.activation = nn.GELU()

        self.film = FiLM(channels, z_dim) if z_dim is not None else None

    def forward(
            self,
            x,
            z=None
    ):
        # Feature extraction
        x = self.norm(self.conv(x))

        # Proposed feature modulation
        if self.film is not None:
            x = self.film(x, z)

        return self.activation(x)

import torch
import torch.nn as nn
from src.architecture.executor.CNNBlock import CNNBlock
from src.architecture.ViT.body import TransformerEncoderBlock
import os
from dotenv import load_dotenv

# Hybrid ViT and CNN
class Executor(nn.Module):
    """
    Applies a latent transformation z to an input grid
    """

    def __init__(
            self,
            embed_dim,
            num_heads,
            mlp_dim,
            depth,
            z_dim,
            hidden_channels=64,
            num_classes=10  # ARC colors
    ):
        super().__init__()

        ############################
        #   CNN Feature Enricher   #
        ############################

        self.enricher = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, padding=1),
            nn.GELU()
        )

        ######################################
        #   CNN Proposal Feature Detection   #
        ######################################

        self.cnn_blocks = nn.ModuleList([
            CNNBlock(hidden_channels, z_dim=z_dim)
            for _ in range(2)
        ])

        ##################
        #   Tokenizers   #
        ##################

        self.to_embedding = nn.Linear(hidden_channels, embed_dim)

        # Interpret proposal 
        self.z_token = nn.Linear(z_dim, embed_dim)

        ##################
        #   ViT Layers   #
        ##################

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim)
            for _ in range(depth)
        ])

        #######################
        #   CNN Discretizer   #
        #######################

        self.discretizer = nn.Sequential(
            nn.Conv2d(  # detect features in token
                embed_dim, 
                hidden_channels, 
                kernel_size=3, 
                padding=1
            ),
            nn.GELU(),
            nn.Conv2d(  # convert token features to a classification
                hidden_channels, 
                num_classes, 
                kernel_size=1
            )
        )

    def forward(
            self, 
            grid, 
            z
    ):
        ###########################
        #   grid = (B, 1, H, W)   #
        #   z = (B, z_dim)        #
        ###############################
        #   B = batch size            #    
        #   D = embedding dimension   #
        #   H = height                #
        #   W = width                 #
        ###############################

        B, _, H, W = grid.shape

        #############################################
        #   Enricher + Proposed Feature Modulator   #
        #############################################

        x = self.enricher(grid)

        for block in self.cnn_blocks:
            x = block(x, z)

        ################
        #   Tokenize   #
        ################

        # (B, C, H, W) -> (B, H*W, C)
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H*W, -1)

        tokens = self.to_embedding(x_flat)  # (B, S, D)

        # Add proposal z token
        z_token = self.z_token(z).unsqueeze(1)  # (B, 1, D) one for each batch
        tokens = torch.cat([z_token, tokens], dim=1)  # (B, 1+S, D)

        ################################
        #   ViT for Global Reasoning   #
        ################################

        for block in self.blocks:
            tokens = block(tokens, None)

        ###################
        #   Un-tokenize   #
        ###################

        # Remove z token
        x_tokens = tokens[:, 1:, :]  # (B, S, D)

        # Reshape to (B, D, H, W)
        x_feats = x_tokens.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        ##################
        #   Discretize   #
        ##################

        # Compute on the embedding dimension
        logits = self.discretizer(x_feats)  # (B, num_classes, H, W)

        print("\n[Executor] logits mean/std:", logits.mean().item(), logits.std().item())

        return logits

    import torch
import torch.nn as nn

# Feature-wise modulation
class FiLM(nn.Module): 
    def __init__(
            self,
            feature_dim,
            z_dim
    ):
        super().__init__()
        self.to_gamma = nn.Linear(z_dim, feature_dim)  # scale factor
        self.to_beta = nn.Linear(z_dim, feature_dim)  # shift factor

    def forward(
            self,
            x: torch.Tensor,
            z: torch.Tensor
    ):
        ########################
        #   x = (B, C, H, W)   #
        #   z = (B, z_dim)     #
        ########################
        #   B = batch size     #       
        #   C = channels       #
        #   H = height         #
        #   W = width          #
        ########################

        ############################
        #   Compute Coefficients   #
        ############################

        # Expand across input
        gamma = self.to_gamma(z).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.to_beta(z).unsqueeze(-1).unsqueeze(-1)
        
        ###############################
        #   Apply Feture Modulation   #
        ###############################

        return x * (1 + gamma) + beta

import torch
import torch.nn as nn


class AdversarialVisionTransformer(nn.Module):

    def __init__(
            self,
            vit_encoder: nn.Module,
            z_dim: int | None = None,   # proposal latent dim
            c_dim: int | None = None,   # context dim
            hidden_dim: int = 256
    ):
        super().__init__()
        self.vit = vit_encoder
        self.z_dim = z_dim
        self.c_dim = c_dim

        # ViT output embedding dimension
        embed_dim = self.vit.c_token.size(-1)   # typically 128

        # === PROJECTIONS ===
        # Project z and C into ViT embedding dimension
        self.z_proj = nn.Linear(z_dim, embed_dim) if z_dim is not None else None
        self.c_proj = nn.Linear(c_dim, embed_dim) if c_dim is not None else None

        # Final concatenated feature dimension
        # h (128) + z_proj (128) + c_proj (128) = 128 * N_parts
        in_dim = embed_dim
        if z_dim is not None:
            in_dim += embed_dim
        if c_dim is not None:
            in_dim += embed_dim

        # === CRITIC HEAD ===
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )


    ###########################################################################
    # FORWARD
    ###########################################################################

    def forward(
            self,
            I_in: torch.Tensor,         # (B, C_in, H, W)
            O_pred: torch.Tensor,       # (B, C_out, H, W) or (B, T, C_out, H, W)
            mask_in: torch.Tensor | None = None,
            mask_out: torch.Tensor | None = None,
            z: torch.Tensor | None = None,
            C: torch.Tensor | None = None
    ) -> torch.Tensor:

        B, C_in, H, W = I_in.shape

        #######################################################################
        # MULTI-PROPOSAL BRANCH
        #######################################################################
        if O_pred.dim() == 5:
            B, T, C_out, H, W = O_pred.shape

            # Expand I and flatten
            I_exp = I_in.unsqueeze(1).expand(B, T, C_in, H, W)
            I_flat = I_exp.reshape(B*T, C_in, H, W)

            # Flatten O_pred
            O_flat = O_pred.reshape(B*T, C_out, H, W)

            # Convert logits → single channel (argmax) if needed
            if O_flat.size(1) > 1:
                O_flat = torch.argmax(O_flat, dim=1, keepdim=True).float()

            # Combine masks if present
            if mask_in is not None or mask_out is not None:
                mask_in  = mask_in  if mask_in  is not None else torch.zeros(B, H, W, dtype=torch.bool, device=O_pred.device)
                mask_out = mask_out if mask_out is not None else torch.zeros(B, H, W, dtype=torch.bool, device=O_pred.device)
                mask = torch.logical_or(mask_in, mask_out)
                mask = mask.unsqueeze(1).expand(B, T, H, W).reshape(B*T, H, W)
            else:
                mask = None

            # Concatenate input + predicted output grid
            x = torch.cat([I_flat, O_flat], dim=1)  # (B*T, 2, H, W)

            # Encode each proposal pair with ViT
            h_flat = self.vit.forward_grid(x, mask=mask)  # (B*T, embed_dim)
            h = h_flat.reshape(B, T, -1)                  # (B, T, embed_dim)

            # ===== Assemble features =====
            feats = [h]

            if z is not None and self.z_proj is not None:
                if z.dim() == 2:
                    z = z.unsqueeze(1).expand(B, T, -1)
                z_proj = self.z_proj(z)
                feats.append(z_proj)


            if C is not None and self.c_proj is not None:
                C_exp = C.unsqueeze(1).expand(B, T, -1)
                C_proj = self.c_proj(C_exp)               # (B, T, embed_dim)
                feats.append(C_proj)

            # Concatenate feature parts
            feat = torch.cat(feats, dim=-1)               # (B, T, in_dim)
            # Flatten (B,T,*) → (B*T,*)
            feat = feat.reshape(B*T, -1)

            # Run MLP
            scores = self.mlp(feat).squeeze(-1)   # (B*T,)

            # Reshape back to (B, T)
            scores = scores.view(B, T)

            return scores



        #######################################################################
        # SINGLE-PROPOSAL BRANCH
        #######################################################################
        B, C_out, H, W = O_pred.shape

        # Convert O_pred logits → continuous class channel
        if O_pred.size(1) > 1:
            classes = torch.arange(O_pred.size(1), device=O_pred.device).view(1, -1, 1, 1)
            probs = O_pred.softmax(dim=1)
            O_pred = (probs * classes).sum(dim=1, keepdim=True)

        # Combine masks
        if mask_in is not None or mask_out is not None:
            mask_in  = mask_in  if mask_in  is not None else torch.zeros(B, H, W, dtype=torch.bool, device=O_pred.device)
            mask_out = mask_out if mask_out is not None else torch.zeros(B, H, W, dtype=torch.bool, device=O_pred.device)
            mask = torch.logical_or(mask_in, mask_out)
        else:
            mask = None

        # Grid input to ViT
        x = torch.cat([I_in, O_pred], dim=1)              # (B, 2, H, W)
        h = self.vit.forward_grid(x, mask=mask)           # (B, embed_dim)

        # ===== Assemble features =====
        feats = [h]
        if z is not None and self.z_proj is not None:
            feats.append(self.z_proj(z))
        if C is not None and self.c_proj is not None:
            feats.append(self.c_proj(C))

        feat = torch.cat(feats, dim=-1)                   # (B, in_dim)

        return self.mlp(feat).squeeze(-1)                 # (B)

import os
import sys
import json
from pathlib import Path

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Load all JSONs (recursive)
# ----------------------------
def load_jsons_from_folder(dir_path):
    """
    Read every .json file under dir_path (recursively) and return a dict
    keyed by the file's relative path (without the .json extension).
    """
    root = Path(dir_path).expanduser().resolve()
    files = sorted(p for p in root.rglob("*.json") if p.is_file())

    if not files:
        raise FileNotFoundError(f"No .json files found under: {root}")

    data = {}
    for p in files:
        key = str(p.relative_to(root).with_suffix(""))  # e.g. "subdir/file"
        try:
            with p.open("r", encoding="utf-8") as fh:
                data[key] = json.load(fh)
        except Exception as e:
            print(f"Failed to read {p}: {e}")

    if not data:
        raise FileNotFoundError(f"Unable to load any .json files under: {root}")

    return data


# ----------------------------
# Preprocess helpers
# ----------------------------
def _add_one_to_all_values_in_place(data):
    """
    Adds +1 to every scalar value in each input/output grid across all samples.
    Done BEFORE padding so pad_value=0 remains 0.
    """
    for sample in data.values():
        for split in ["train", "test"]:
            for pairs in sample.get(split, []):
                # input grid
                r = 0
                while r < len(pairs["input"]):
                    c = 0
                    row = pairs["input"][r]
                    while c < len(row):
                        row[c] = row[c] + 1
                        c += 1
                    r += 1
                # output grid
                r = 0
                while r < len(pairs["output"]):
                    c = 0
                    row = pairs["output"][r]
                    while c < len(row):
                        row[c] = row[c] + 1
                        c += 1
                    r += 1


def get_metrics(data):
    metric_dict = {
        "max_train_len": 0,
        "max_test_len": 0,
        "max_train_input_height": 0,
        "max_test_input_height": 0,
        "max_train_output_height": 0,
        "max_test_output_height": 0,
        "max_train_input_width": 0,
        "max_test_input_width": 0,
        "max_train_output_width": 0,
        "max_test_output_width": 0
    }

    for sample in data.values():
        if (len(sample['train']) > metric_dict['max_train_len']):
            metric_dict['max_train_len'] = len(sample['train'])
        if (len(sample['test']) > metric_dict['max_test_len']):
            metric_dict['max_test_len'] = len(sample['test'])
        for pairs in sample['train']:
            if (len(pairs['input']) > metric_dict['max_train_input_height']):
                metric_dict['max_train_input_height'] = len(pairs['input'])
            if (len(pairs['output']) > metric_dict['max_train_output_height']):
                metric_dict['max_train_output_height'] = len(pairs['output'])
            for inp in pairs['input']:
                if (len(inp) > metric_dict['max_train_input_width']):
                    metric_dict['max_train_input_width'] = len(inp)
            for output in pairs['output']:
                if (len(output) > metric_dict['max_train_output_width']):
                    metric_dict['max_train_output_width'] = len(output)
        for pairs in sample['test']:
            if (len(pairs['input']) > metric_dict['max_test_input_height']):
                metric_dict['max_test_input_height'] = len(pairs['input'])
            if (len(pairs['output']) > metric_dict['max_test_output_height']):
                metric_dict['max_test_output_height'] = len(pairs['output'])
            for inp in pairs['input']:
                if (len(inp) > metric_dict['max_test_input_width']):
                    metric_dict['max_test_input_width'] = len(inp)
            for output in pairs['output']:
                if (len(output) > metric_dict['max_test_output_width']):
                    metric_dict['max_test_output_width'] = len(output)
    return metric_dict


def pad_data(data, metric_dict=None, pad_value=0):
    """
    Pads the ENTIRE dataset so that:
      • all TRAIN pairs are square-padded to the same dataset-wide size, and
      • all TEST  pairs are square-padded to the same dataset-wide size.    # CHANGED
    If metric_dict is None, it will be computed from the data.               # NEW
    """
    # ----- compute global (dataset-wide) sizes -----                         # NEW
    if metric_dict is None:
        metric_dict = get_metrics(data)

    max_train_size = max(
        metric_dict["max_train_input_height"],
        metric_dict["max_train_input_width"],
        metric_dict["max_train_output_height"],
        metric_dict["max_train_output_width"]
    )
    max_test_size = max(
        metric_dict["max_test_input_height"],
        metric_dict["max_test_input_width"],
        metric_dict["max_test_output_height"],
        metric_dict["max_test_output_width"]
    )

    # ----- pad EVERY sample to the global split sizes -----                  # CHANGED
    for sample in data.values():
        # TRAIN -> global train size
        for pairs in sample.get('train', []):
            # input
            while len(pairs['input']) < max_train_size:
                pairs['input'].append([pad_value] * max_train_size)
            for inp in pairs['input']:
                while len(inp) < max_train_size:
                    inp.append(pad_value)
            # output
            while len(pairs['output']) < max_train_size:
                pairs['output'].append([pad_value] * max_train_size)
            for outp in pairs['output']:
                while len(outp) < max_train_size:
                    outp.append(pad_value)

        # TEST -> global test size
        for pairs in sample.get('test', []):
            # input
            while len(pairs['input']) < max_test_size:
                pairs['input'].append([pad_value] * max_test_size)
            for inp in pairs['input']:
                while len(inp) < max_test_size:
                    inp.append(pad_value)
            # output
            while len(pairs['output']) < max_test_size:
                pairs['output'].append([pad_value] * max_test_size)
            for outp in pairs['output']:
                while len(outp) < max_test_size:
                    outp.append(pad_value)

    return data


def _infer_original_size_from_padded(grid, pad_value=0):
    h = 0
    w = 0
    r = 0
    while r < len(grid):
        row = grid[r]
        any_nonpad = False
        last_nonpad = -1
        c = 0
        while c < len(row):
            if row[c] != pad_value:
                any_nonpad = True
                last_nonpad = c
            c += 1
        if any_nonpad:
            if (r + 1) > h:
                h = r + 1
            if (last_nonpad + 1) > w:
                w = last_nonpad + 1
        r += 1
    return (h, w)


def build_sample_level_dataset(data, pad_value=0):
    """
    Build a list of per-sample records.
    NEW: also stores per-pair masks: 1 where value != pad_value, else 0.
    """
    dataset = []
    for sample_name, sample in data.items():
        # containers
        train_pairs = []
        test_pairs = []

        # track original (unpadded) sizes per split
        train_max_h = 0
        train_max_w = 0
        test_max_h = 0
        test_max_w = 0

        # ----- TRAIN -----
        idx = 0
        for pairs in sample['train']:
            inp_grid = pairs['input']
            out_grid = pairs['output']

            # original sizes (prefer stored, else infer)
            if ('orig_input_size' in pairs):
                in_h, in_w = pairs['orig_input_size']
            else:
                in_h, in_w = _infer_original_size_from_padded(inp_grid, pad_value)
            if ('orig_output_size' in pairs):
                out_h, out_w = pairs['orig_output_size']
            else:
                out_h, out_w = _infer_original_size_from_padded(out_grid, pad_value)

            # update split-wide original size (max over inputs/outputs)
            if in_h > train_max_h: train_max_h = in_h
            if out_h > train_max_h: train_max_h = out_h
            if in_w > train_max_w: train_max_w = in_w
            if out_w > train_max_w: train_max_w = out_w

            # tensors
            inp_tensor = torch.tensor(inp_grid).long()
            out_tensor = torch.tensor(out_grid).long()

            # NEW: masks (1 for non-pad, 0 for pad)
            inp_mask = (inp_tensor != pad_value).long()
            out_mask = (out_tensor != pad_value).long()

            # store pair
            train_pairs.append({
                "input": inp_tensor,
                "output": out_tensor,
                "input_mask": inp_mask,
                "output_mask": out_mask
            })
            idx += 1

        # ----- TEST -----
        idx = 0
        for pairs in sample['test']:
            inp_grid = pairs['input']
            out_grid = pairs['output']

            if ('orig_input_size' in pairs):
                in_h, in_w = pairs['orig_input_size']
            else:
                in_h, in_w = _infer_original_size_from_padded(inp_grid, pad_value)
            if ('orig_output_size' in pairs):
                out_h, out_w = pairs['orig_output_size']
            else:
                out_h, out_w = _infer_original_size_from_padded(out_grid, pad_value)

            if in_h > test_max_h: test_max_h = in_h
            if out_h > test_max_h: test_max_h = out_h
            if in_w > test_max_w: test_max_w = in_w
            if out_w > test_max_w: test_max_w = out_w

            inp_tensor = torch.tensor(inp_grid).long()
            out_tensor = torch.tensor(out_grid).long()

            # NEW: masks (1 for non-pad, 0 for pad)
            inp_mask = (inp_tensor != pad_value).long()
            out_mask = (out_tensor != pad_value).long()

            test_pairs.append({
                "input": inp_tensor,
                "output": out_tensor,
                "input_mask": inp_mask,
                "output_mask": out_mask
            })
            idx += 1

        # assemble sample-level record
        item = {
            "id": str(sample_name),
            "train_pairs": train_pairs,
            "test_pairs": test_pairs,
            "train_original_size": (train_max_h, train_max_w),
            "test_original_size": (test_max_h, test_max_w)
        }
        dataset.append(item)

    return dataset


# ----------------------------
# Torch dataset
# ----------------------------
class ARCSampleDataset(Dataset):
    def __init__(self, sample_list):
        self.data = sample_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # stack per-sample pairs into tensors
        train_inputs = torch.stack([p["input"] for p in sample["train_pairs"]])      # [num_train, H, W]
        train_outputs = torch.stack([p["output"] for p in sample["train_pairs"]])    # [num_train, H, W]
        test_inputs = torch.stack([p["input"] for p in sample["test_pairs"]])        # [num_test, H, W]
        test_outputs = torch.stack([p["output"] for p in sample["test_pairs"]])      # [num_test, H, W]

        # masks
        train_input_masks = torch.stack([p["input_mask"] for p in sample["train_pairs"]])
        train_output_masks = torch.stack([p["output_mask"] for p in sample["train_pairs"]])
        test_input_masks  = torch.stack([p["input_mask"] for p in sample["test_pairs"]])
        test_output_masks = torch.stack([p["output_mask"] for p in sample["test_pairs"]])

        return {
            "id": sample["id"],
            "train_inputs": train_inputs,
            "train_outputs": train_outputs,
            "test_inputs": test_inputs,
            "test_outputs": test_outputs,
            "train_input_masks": train_input_masks,
            "train_output_masks": train_output_masks,
            "test_input_masks": test_input_masks,
            "test_output_masks": test_output_masks,
            "train_original_size": torch.tensor(sample["train_original_size"], dtype=torch.long),
            "test_original_size": torch.tensor(sample["test_original_size"], dtype=torch.long)
        }


def arc_collate_fn_bs1(batch):
    # batch size is guaranteed to be 1; return the single dict unchanged
    return batch[0]


# ----------------------------
# NEW: Small pretty-printer for grids (cropped)
# ----------------------------
def _pretty_grid(tensor, max_rows=6, max_cols=10):  # NEW
    arr = tensor.tolist()
    lines = []
    r = 0
    while r < min(len(arr), max_rows):
        row = arr[r]
        row_disp = row[:max_cols]
        row_txt = str(row_disp) + (" ... " if len(row) > max_cols else "")
        lines.append(row_txt)
        r += 1
    if len(arr) > max_rows:
        lines.append("...")
    return "\n".join(lines)


# ----------------------------
# NEW: Data module wrapper
# ----------------------------
class ARCDataModule:
    """
    Simple wrapper to produce a DataLoader from your folder.
    Usage:
        dm = ARCDataModule("~/path/to/training").prepare()
        loader = dm.get_loader()
        for batch in loader: ...
    """
    def __init__(
        self,
        dir_path,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        pad_value=0,
    ):
        self.dir_path = Path(dir_path).expanduser().resolve()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.pad_value = pad_value

        self.dataset = None
        self._loader = None
        self.metrics = None  # NEW

    def prepare(self):
        # load + preprocess
        data = load_jsons_from_folder(self.dir_path)
        _add_one_to_all_values_in_place(data)

        # compute dataset-wide metrics + pad globally                        # CHANGED
        self.metrics = get_metrics(data)                                     # NEW
        padded = pad_data(data, metric_dict=self.metrics, pad_value=self.pad_value)

        sample_list = build_sample_level_dataset(padded, pad_value=self.pad_value)

        # build dataset + loader
        self.dataset = ARCSampleDataset(sample_list=sample_list)
        self._loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=arc_collate_fn_bs1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return self  # allow chaining

    def get_loader(self):
        if self._loader is None:
            self.prepare()
        return self._loader

    # convenience so the module itself is iterable
    def __iter__(self):
        return iter(self.get_loader())

    def __len__(self):
        return len(self.dataset) if self.dataset is not None else 0


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # Point to your local folder named "training"
    folder_path = Path("~/ARC-AGI-Model/src/data_pipeline/ARC_data/data/training")

    data_module = ARCDataModule(
        dir_path=folder_path,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        pad_value=0,
    ).prepare()

    arc_loader = data_module.get_loader()

    # Expected global sizes (dataset-wide)                                    # NEW
    M = data_module.metrics
    GLOBAL_TRAIN_SIZE = max(
        M["max_train_input_height"], M["max_train_input_width"],
        M["max_train_output_height"], M["max_train_output_width"]
    )
    GLOBAL_TEST_SIZE = max(
        M["max_test_input_height"], M["max_test_input_width"],
        M["max_test_output_height"], M["max_test_output_width"]
    )
    print("=== DATASET-WIDE PAD SIZES ===")
    print(f"TRAIN -> {GLOBAL_TRAIN_SIZE}x{GLOBAL_TRAIN_SIZE}")
    print(f"TEST  -> {GLOBAL_TEST_SIZE}x{GLOBAL_TEST_SIZE}")

    # Print up to 10 concise, readable examples
    printed = 0
    for batch in arc_loader:
        num_train = int(batch["train_inputs"].shape[0])
        num_test  = int(batch["test_inputs"].shape[0])

        # original (max over pairs before padding, per sample)
        train_orig_h, train_orig_w = map(int, batch["train_original_size"].tolist())
        test_orig_h,  test_orig_w  = map(int, batch["test_original_size"].tolist())

        # padded sizes (actual tensor shapes)
        train_in_h, train_in_w   = batch["train_inputs"].shape[1], batch["train_inputs"].shape[2]
        train_out_h, train_out_w = batch["train_outputs"].shape[1], batch["train_outputs"].shape[2]
        test_in_h,  test_in_w    = batch["test_inputs"].shape[1], batch["test_inputs"].shape[2]
        test_out_h, test_out_w   = batch["test_outputs"].shape[1], batch["test_outputs"].shape[2]

        # Validate against global expectations
        train_ok = (train_in_h == GLOBAL_TRAIN_SIZE == train_out_h) and (train_in_w == GLOBAL_TRAIN_SIZE == train_out_w)
        test_ok  = (test_in_h  == GLOBAL_TEST_SIZE  == test_out_h)  and (test_in_w  == GLOBAL_TEST_SIZE  == test_out_w)

        print(f"\n=== SUMMARY (sample {printed+1}) ===")
        print(f"id: {batch['id']}")
        print(f"#train: {num_train} | #test: {num_test}")
        print(f"Train original max: ({train_orig_h}, {train_orig_w})")
        print(f"Test  original max: ({test_orig_h}, {test_orig_w})")
        print(f"Padded sizes — train_in: ({train_in_h}, {train_in_w}), "
              f"train_out: ({train_out_h}, {train_out_w}), "
              f"test_in: ({test_in_h}, {test_in_w}), "
              f"test_out: ({test_out_h}, {test_out_w})")
        print(f"Matches global TRAIN size? {train_ok} | Matches global TEST size? {test_ok}")

        if num_train > 0:
            print("\n--- Example TRAIN pair [0] (cropped) ---")
            print("input:\n"  + _pretty_grid(batch["train_inputs"][0], 6, 10))
            print("output:\n" + _pretty_grid(batch["train_outputs"][0], 6, 10))
        if num_test > 0:
            print("\n--- Example TEST pair [0] (cropped) ---")
            print("input:\n"  + _pretty_grid(batch["test_inputs"][0], 6, 10))
            print("output:\n" + _pretty_grid(batch["test_outputs"][0], 6, 10))

        printed += 1
        if printed >= 10:
            break

    print("\nDataLoader type:", type(arc_loader))

    from pathlib import Path
from src.data_pipeline.utils import load_jsons_from_folder, _add_one_to_all_values_in_place, pad_data, build_sample_level_dataset, arc_collate_fn_bs1
from src.data_pipeline.dataset import ARCSampleDataset
from torch.utils.data import DataLoader


class ARCDataModule:
    """
    Simple wrapper to produce a DataLoader from your folder.
    Usage:
        dm = ARCDataModule("~/path/to/training").prepare()
        loader = dm.get_loader()
        for batch in loader: ...
    """
    def __init__(
        self,
        dir_path,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        pad_value=0,
    ):
        self.dir_path = Path(dir_path).expanduser().resolve()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.pad_value = pad_value

        self.dataset = None
        self._loader = None

    def prepare(self):
        # load + preprocess
        data = load_jsons_from_folder(self.dir_path)
        _add_one_to_all_values_in_place(data)

        # pad each sample independently (metric_dict unused)
        padded = pad_data(data, metric_dict=None, pad_value=self.pad_value)
        sample_list = build_sample_level_dataset(padded, pad_value=self.pad_value)

        # build dataset + loader
        self.dataset = ARCSampleDataset(sample_list=sample_list)
        self._loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=arc_collate_fn_bs1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return self  # allow chaining

    def get_loader(self):
        if self._loader is None:
            self.prepare()
        return self._loader

    # convenience so the module itself is iterable
    def __iter__(self):
        return iter(self.get_loader())

    def __len__(self):
        return len(self.dataset) if self.dataset is not None else 0

    import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    A nn.Linear wrapper
    """

    def __init__(
            
    )

import torch
import torch.nn as nn

from src.architecture.LViTM.body import LargeVisionTransformerModel
from src.architecture.executor.executor import Executor
from src.architecture.adViT.critic import AdversarialVisionTransformer
from src.architecture.context_encoding.conditional_encoder import ConditionalTestInputEncoder


class HybridExecuteController(nn.Module):

    def __init__(
            self,
            lvitm: LargeVisionTransformerModel,
            executor: Executor, 
            cond_encoder: ConditionalTestInputEncoder,
            critic: AdversarialVisionTransformer | None = None
    ):
        ###############################
        #   B = batch size            #    
        #   D = token embedding dim   #
        #   P = num proposals         #
        #   H = height                #
        #   W = width                 #
        ###############################

        super().__init__()
        self.lvitm = lvitm
        self.executor = executor
        self.cond_encoder = cond_encoder
        self.critic = critic

    ################################
    #   Parallel Mode (Training)   #
    ################################

    def apply_parallel(
        self,
        I_test: torch.Tensor,  # (B, 1, H, W)
        mask_test: torch.Tensor,  # (B, H, W)
        C: torch.Tensor,  # (B, D)
        examples=None
    ):
        B, _, H, W = I_test.shape

        # Encode test input with context
        tokens, key_padding_mask = self.cond_encoder(I_test, mask_test, C)

        #######################
        #   Reasoning Model   #
        #######################

        # Compute proposals
        Z = self.lvitm(C, tokens, key_padding_mask)

        B, P, z_dim = Z.shape

        ################
        #   Executor   #
        ################

        # Flatten input for executor
        grid_expansion = I_test.unsqueeze(1).expand(B, P, 1, H, W).reshape(B*P, 1, H, W)
        z_flat = Z.reshape(B*P, z_dim)

        # Execute proposals
        out_flat = self.executor(grid_expansion, z_flat)
        num_classes = out_flat.size(1)
        outputs = out_flat.view(B, P, num_classes, H, W)

        ##############
        #   Critic   #
        ##############

        scores = None
        best_idx = None
        if self.critic is not None and examples is not None:
            scores = self.critic(
                I_in=I_test,            # (B, 1, H, W)
                O_pred=outputs,         # (B, P, num_classes, H, W)
                mask_in=mask_test,      # (B, H, W)
                mask_out=mask_test,     # assume same valid region as input
                z=Z,                    # (B, P, z_dim)
                C=C                     # (B, D)
            )     
            best_idx = scores.argmax(dim=1)  # (B,)

        return outputs, scores, best_idx
    
    ###################################
    #   Sequential Mode (Inference)   #
    ###################################

    @torch.no_grad()
    def apply_sequential(
        self,
        init_grid: torch.Tensor,  # (B, 1, H, W)
        init_mask: torch.Tensor,  # (B, H, W)
        C,                        # (B, D)
        examples=None,
        num_steps=3
    ):
        grid = init_grid
        mask = init_mask
        history: list[torch.Tensor] = [grid.clone()]

        #########################
        #   Iterate Proposals   #
        #########################

        for _ in range(num_steps):
            B, _, H, W = grid.shape

            # Encode test input with context
            tokens, key_padding_mask = self.cond_encoder(grid, mask, C)

            #######################
            #   Reasoning Model   #
            #######################

            # Compute proposal
            Z = self.lvitm(C, tokens, key_padding_mask)  # (B, P, z_dim)
            B, P, z_dim = Z.shape

            ################
            #   Executor   #
            ################

            # Flatten input for executor
            grid_rep = grid.unsqueeze(1).expand(B, P, 1, H, W).reshape(B * P, 1, H, W)
            z_flat = Z.reshape(B * P, z_dim)

            # Execute proposals
            out_flat = self.executor(grid_rep, z_flat)  # (B*T, num_classes, H, W)
            num_classes = out_flat.size(1)
            outputs = out_flat.view(B, P, num_classes, H, W)

            ##############
            #   Critic   #
            ##############

            # Choose proposal
            if self.critic is not None:
                scores = self.critic(
                    I_in=grid,
                    O_pred=outputs,
                    mask_in=mask,
                    mask_out=mask,
                    z=Z,
                    C=C
                )  # (B, P)
                best_idx = scores.argmax(dim=1)  # (B,)
            else:
                # Or take first proposal
                best_idx = torch.zeros(B, dtype=torch.long, device=grid.device)

            # Gather best output per batch
            idx = best_idx.view(B, 1, 1, 1, 1).expand(B, 1, num_classes, H, W)
            best_out_logits = outputs.gather(dim=1, index=idx).squeeze(1)  # (B, C_out, H, W)

            # Discretize logits
            best_out_grid = best_out_logits.argmax(dim=1, keepdim=True)    # (B, 1, H, W)

            grid = best_out_grid
            # mask stays the same spatially
            history.append(grid.clone())

        final_grid_logits = best_out_logits  
        return final_grid_logits, history

        ############################################################
    #   PPO Rollout + (stub) Update for Phase 4                #
    ############################################################
    def ppo_rollout_and_update(
        self,
        init_grid: torch.Tensor,       # (B, 1, H, W)
        init_mask: torch.Tensor,       # (B, H, W)
        C: torch.Tensor,               # (B, D)
        ppo_refiner,                   # PPORefiner object (currently unused in stub)
        num_steps: int,
        gamma: float,
    ):
        """
        Minimal implementation to satisfy Phase 4:

        - Encodes the test grid with cond_encoder
        - Runs LViTM once to get latent proposals z
        - Picks the first proposal
        - Runs Executor to get final logits
        - Returns logits and a dummy PPO stats dict

        This does NOT yet perform real PPO; it is a safe stub
        that keeps all dimensions consistent and lets training run.
        """

        # 1. Encode test grid into tokens
        #    ConditionalTestInputEncoder.forward: (I_test, mask_I) -> (tokens, key_padding_mask)
        test_tokens, key_padding_mask = self.cond_encoder(
            I_test=init_grid,
            mask_test=init_mask,
            C=C
        )


        # 2. Get latent proposals z from LViTM
        #    LargeVisionTransformerModel.forward: (tokens, key_padding_mask, C) -> Z
        Z = self.lvitm(
            C=C,
            test_tokens=test_tokens,
            key_padding_mask=key_padding_mask
        )
        # typically (B, T, z_dim) or (B, z_dim)

        # 3. Choose a single z per sample (e.g., first proposal)
        if Z.dim() == 3:  # (B, T, z_dim)
            Z0 = Z[:, 0, :]           # (B, z_dim)
        else:             # already (B, z_dim)
            Z0 = Z

        # 4. Run Executor to get logits over ARC colors
        #    Executor.forward is expected to look like:
        #       forward(I_test, Z, key_padding_mask=None)
        logits = self.executor(init_grid, Z0)
        # (B, num_classes, H, W)

        # 5. Package PPO stats (stubbed out for now)
        ppo_stats = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
        }

        # Phase 4 expects: (final_logits, ppo_stats)
        return logits, ppo_stats
import torch
import torch.nn as nn
from typing import Dict, Any
import os
from dotenv import load_dotenv

class ARCGenerator(nn.Module):
    """
    For each sample:
      - Uses ALL training example pairs (I_i, O_i) to compute a context vector C.
      - Then, for EACH test input in that sample:
          * encodes the test input conditioned on C,
          * uses LViTM to propose latent transformation vectors Z,
          * chooses a single z (here: the first proposal),
          * runs the Executor to produce logits for that test.
    """

    def __init__(
        self,
        example_pair_encoder: nn.Module,
        aggregator: nn.Module,
        cond_encoder: nn.Module,
        lvitm: nn.Module,
        executor: nn.Module,
    ):
        super().__init__()
        self.example_pair_encoder = example_pair_encoder
        self.aggregator = aggregator
        self.cond_encoder = cond_encoder
        self.lvitm = lvitm
        self.executor = executor

    def forward(
        self,
        train_inputs: torch.Tensor,        # (K_train, H, W)   or (B, K_train, H, W)
        train_outputs: torch.Tensor,       # (K_train, H, W)   or (B, K_train, H, W)
        train_input_masks: torch.Tensor,   # (K_train, H, W)   or (B, K_train, H, W)
        train_output_masks: torch.Tensor,  # (K_train, H, W)   or (B, K_train, H, W)
        test_inputs: torch.Tensor,         # (K_test, H, W)    or (B, K_test, H, W)
        test_input_masks: torch.Tensor,    # (K_test, H, W)    or (B, K_test, H, W)
    ) -> Dict[str, Any]:

        ##################################
        #   Normalize Batch Dimensions   #
        ##################################

        load_dotenv()
        DEBUGGING = os.getenv("DEBUGGING")

        # Ensure batch dimension exists
        if train_inputs.dim() == 3:
            train_inputs       = train_inputs.unsqueeze(0)
            train_outputs      = train_outputs.unsqueeze(0)
            train_input_masks  = train_input_masks.unsqueeze(0)
            train_output_masks = train_output_masks.unsqueeze(0)

        if test_inputs.dim() == 3:
            test_inputs       = test_inputs.unsqueeze(0)
            test_input_masks  = test_input_masks.unsqueeze(0)


        B, K_train, H, W = train_inputs.shape
        _, K_test, H_t, W_t = test_inputs.shape
        assert H == H_t and W == W_t, "Train and test grids must share padded size per sample."

        # Ensure masks are boolean
        train_input_masks = train_input_masks.bool()
        train_output_masks = train_output_masks.bool()
        test_input_masks = test_input_masks.bool()

        if DEBUGGING:  # toggle debug
            print("\n[ARCGenerator] train_inputs:", train_inputs.shape)
            print("[ARCGenerator] test_inputs:", test_inputs.shape)
            print("[ARCGenerator] train_input_masks:", train_input_masks.shape)
            print("[ARCGenerator] test_input_masks:", test_input_masks.shape)


        #########################################
        #   Encode ALL training example pairs   #
        #########################################

        # For each training pair (I_i, O_i), we get a context embedding h_i.
        h_list = []

        for k in range(K_train):
            # Shapes: (B, 1, H, W)
            I_k = train_inputs[:, k].unsqueeze(1).float()
            O_k = train_outputs[:, k].unsqueeze(1).float()

            mask_I_k = train_input_masks[:, k]   # (B, H, W)
            mask_O_k = train_output_masks[:, k]  # (B, H, W)

            h_k = self.example_pair_encoder(
                I_i=I_k,
                O_i=O_k,
                mask_I=mask_I_k,
                mask_O=mask_O_k
            )  # (B, D)
            h_list.append(h_k)

        # Stack: (B, K_train, D)
        h = torch.stack(h_list, dim=1)

        # Optional: pair_mask could be used if we ever have invalid pairs.
        pair_mask = None  # (B, K_train) if needed

        ##############################
        #   Aggregate to context C   #
        ##############################

        # Single context vector per sample, shared by all test pairs.
        C = self.aggregator(h, mask=pair_mask)  # (B, D)

        if DEBUGGING:
            print("[ARCGenerator] C mean/std:", C.mean().item(), C.std().item())


        # ----------------------------------
        #   Loop over ALL test inputs
        # ----------------------------------
        all_logits = []
        all_Z = []
        all_z_chosen = []

        for j in range(K_test):
            # Test grid j: (B, 1, H, W)
            I_test_j = test_inputs[:, j].unsqueeze(1).float()
            mask_test_j = test_input_masks[:, j]   # (B, H, W)

            # Encode test input with context C
            test_tokens_j, key_padding_mask_j = self.cond_encoder(
                I_test_j,      # (B,1,H,W)
                mask_test_j,   # (B,H,W)
                C              # (B,D)
            )
            # test_tokens_j: (B, S, D)
            # key_padding_mask_j: (B, S) or None

            # LViTM proposes latent transformation vectors for this test input
            Z_j = self.lvitm(
                C=C,
                test_tokens=test_tokens_j,
                key_padding_mask=key_padding_mask_j
            )  # (B, P, z_dim)

            Bz, P, z_dim = Z_j.shape
            assert Bz == B, "Batch size mismatch between context and proposals."

            # Choose a single proposal z for this test input
            # Minimal baseline: pick first proposal
            z_chosen_j = Z_j[:, 0, :]  # (B, z_dim)

            # Executor predicts the output grid logits
            logits_j = self.executor(
                grid=I_test_j,   # (B,1,H,W)
                z=z_chosen_j     # (B,z_dim)
            )  # (B, num_classes, H, W)

            all_logits.append(logits_j)          # each: (B, C_out, H, W)
            all_Z.append(Z_j)                    # each: (B, P, z_dim)
            all_z_chosen.append(z_chosen_j)      # each: (B, z_dim)

        # -----------------------------------------
        #   Stack results across ALL test inputs
        # -----------------------------------------
        # logits: (B, K_test, num_classes, H, W)
        logits = torch.stack(all_logits, dim=1)

        # Sanitize logits to avoid NaNs/Infs propagating into losses
        logits = torch.nan_to_num(
            logits,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )

        # Z_all: (B, K_test, P, z_dim)
        Z_all = torch.stack(all_Z, dim=1)

        # z_chosen: (B, K_test, z_dim)
        z_chosen = torch.stack(all_z_chosen, dim=1)

        if DEBUGGING:
            print("[ARCGenerator] logits mean/std:", logits.mean().item(), logits.std().item())


        return {
            "logits": logits,         # (B, K_test, C_out, H, W)
            "Z_all": Z_all,           # (B, K_test, P, z_dim)
            "z_chosen": z_chosen,     # (B, K_test, z_dim)
            "C": C                    # (B, D)
        }

"""
Usage:

batch = next(iter(arc_loader))

out = arc_generator(
    train_inputs=batch["train_inputs"],         # (K_train, H, W)
    train_outputs=batch["train_outputs"],       # (K_train, H, W)
    train_input_masks=batch["train_input_masks"],
    train_output_masks=batch["train_output_masks"],
    test_inputs=batch["test_inputs"],           # (K_test, H, W)
    test_input_masks=batch["test_input_masks"],
)

logits = out["logits"]  # (1, K_test, num_classes, H, W)


Loss:

import torch.nn.functional as F

# target: (1, K_test, H, W) → (1 * K_test, H, W)
target = batch["test_outputs"].unsqueeze(0) if batch["test_outputs"].dim() == 3 else batch["test_outputs"]
B, K_test, H, W = target.shape
target_flat = target.view(B * K_test, H, W)

# logits: (1, K_test, C_out, H, W) → (1 * K_test, C_out, H, W)
logits_flat = logits.view(B * K_test, logits.size(2), H, W)

loss = F.cross_entropy(logits_flat, target_flat)
loss.backward()
optimizer.step()

"""

import torch
import matplotlib.pyplot as plt
import numpy as np

from src.training.evaluate_final import (
    build_generator,
    build_critic,
    build_ppo,
    compute_context_C
)

from src.inference.execution_controller import HybridExecuteController
from src.data_pipeline.dataloader import ARCDataModule

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================
# Utility: Convert integer grid → RGB for visualization
# ===========================================================
ARC_COLORS = [
    (0, 0, 0),        # 0 black (padding)
    (0, 0, 255),      # 1 blue
    (0, 255, 0),      # 2 green
    (255, 0, 0),      # 3 red
    (255, 255, 0),    # 4 yellow
    (255, 165, 0),    # 5 orange
    (255, 0, 255),    # 6 magenta
    (0, 255, 255),    # 7 cyan
    (128, 0, 128),    # 8 purple
    (165, 42, 42),    # 9 brown
    (255, 255, 255)   # 10 white (your num_classes=11)
]


def grid_to_rgb(grid):
    """
    grid: (H, W) integer tensor
    returns: (H, W, 3) uint8 numpy array
    """
    H, W = grid.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            rgb[i, j] = ARC_COLORS[grid[i, j]]
    return rgb



# ===========================================================
# MAIN INFERENCE FUNCTION
# ===========================================================
def run_inference_on_sample(batch, generator, controller, actor, valuer):
    """
    Runs:
      - baseline generator prediction
      - sequential PPO-enhanced prediction
      - visualization
    """

    # Move to device
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(DEVICE)

    train_inputs       = batch["train_inputs"]
    train_outputs      = batch["train_outputs"]
    train_input_masks  = batch["train_input_masks"]
    train_output_masks = batch["train_output_masks"]

    test_inputs        = batch["test_inputs"]
    test_outputs       = batch["test_outputs"]
    test_input_masks   = batch["test_input_masks"]

    # Compute context C from example pairs
    C = compute_context_C(
        example_encoder=generator.example_pair_encoder,
        aggregator=generator.aggregator,
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        train_input_masks=train_input_masks,
        train_output_masks=train_output_masks
    )

    # =======================================================
    # (A) BASELINE GENERATOR PREDICTION (Phase1–3)
    # =======================================================
    gen_out = generator(
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        train_input_masks=train_input_masks,
        train_output_masks=train_output_masks,
        test_inputs=test_inputs,
        test_input_masks=test_input_masks
    )
    baseline_logits = gen_out["logits"]  # (1,K_test,C,H,W)
    baseline_pred = baseline_logits.argmax(dim=2).squeeze(0)  # (K_test,H,W)

    # =======================================================
    # (B) PPO-ENHANCED SEQUENTIAL EXECUTION CONTROLLER
    # =======================================================
    if test_inputs.dim() == 3:
        test_inputs = test_inputs.unsqueeze(0)
        test_input_masks = test_input_masks.unsqueeze(0)
        test_outputs = test_outputs.unsqueeze(0)

    init_grid = test_inputs[:, 0].unsqueeze(1).float()  # (1,1,H,W)
    init_mask = test_input_masks[:, 0]                  # (1,H,W)
    target = test_outputs[:, 0]                         # (1,H,W)

    final_logits, history = controller.apply_sequential(
        init_grid=init_grid,
        init_mask=init_mask,
        C=C,
        examples=None,
        num_steps=3
    )

    ppo_pred = final_logits.argmax(dim=1)[0]  # (H,W)

    # =======================================================
    # Visualization
    # =======================================================
    orig_h, orig_w = batch["test_original_size"].tolist()
    baseline_vis = baseline_pred[0, :orig_h, :orig_w].cpu()
    ppo_vis = ppo_pred[:orig_h, :orig_w].cpu()
    target_vis = target[0, :orig_h, :orig_w].cpu()
    test_in_vis = test_inputs[0, 0, :orig_h, :orig_w].cpu()

    # Plot
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].set_title("Test Input")
    axs[0].imshow(grid_to_rgb(test_in_vis))

    axs[1].set_title("Baseline Prediction")
    axs[1].imshow(grid_to_rgb(baseline_vis))

    axs[2].set_title("PPO Final Prediction")
    axs[2].imshow(grid_to_rgb(ppo_vis))

    axs[3].set_title("Ground Truth Output")
    axs[3].imshow(grid_to_rgb(target_vis))

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    return baseline_vis, ppo_vis, target_vis



# ===========================================================
# ENTRYPOINT
# ===========================================================
if __name__ == "__main__":
    print("Loading models...")

    # Build everything
    generator, example_encoder, aggregator, cond_encoder, lvitm, executor = build_generator()
    critic = build_critic()
    actor, valuer, ppo_refiner = build_ppo(z_dim=64)

    # Load checkpoints
    generator.load_state_dict(torch.load("checkpoints/generator_phase3_adv.pt", map_location=DEVICE), strict=False)
    critic.load_state_dict(torch.load("checkpoints/critic_phase3_adv.pt", map_location=DEVICE), strict=False)
    actor.load_state_dict(torch.load("checkpoints/ppo_actor_phase4.pt", map_location=DEVICE), strict=False)
    valuer.load_state_dict(torch.load("checkpoints/ppo_valuer_phase4.pt", map_location=DEVICE), strict=False)

    # Execution Controller
    controller = HybridExecuteController(
        lvitm=lvitm,
        executor=executor,
        cond_encoder=cond_encoder,
        critic=critic
    ).to(DEVICE)

    # Load dataset (small subset)
    print("Loading dataset...")
    data_module = ARCDataModule(
        dir_path="./src/data_pipeline/ARC_data/data/training",
        batch_size=1,
        shuffle=False,
        pad_value=0
    ).prepare()

    # Limit to 3 samples for testing
    data_module.dataset.data = data_module.dataset.data[:3]
    loader = data_module.get_loader()

    # ===========================================================
    # Run inference on 3 tasks
    # ===========================================================
    for i, batch in enumerate(loader):
        print(f"\n=========== SAMPLE {i} ===========")
        baseline, ppo, target = run_inference_on_sample(
            batch, generator, controller, actor, valuer
        )

    print("\nInference complete.")

import torch 
from torch import autograd, nn
from typing import Union, Literal, Dict

class WGAN_GP_Trainer:
    def __init__(
            self,
            generator: nn.Module,
            critic: nn.Module,
            gen_optim: torch.optim.Optimizer,
            crit_optim: torch.optim.Optimizer,
            latent_dim=128,
            grad_pen_weight=10.0,
            num_critic_updates=3,
            device: Union[Literal['cuda', 'cpu'], None] = None
            ) -> None:
        self.generator = generator
        self.critic = critic
        self.gen_optim = gen_optim
        self.crit_optim = crit_optim
        self.latent_dim = latent_dim
        self.grad_pen_weight = grad_pen_weight
        self.num_critic_updates = num_critic_updates
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Move to device
        self.generator.to(self.device)
        self.critic.to(self.device)

    def gradient_penalty(
            self, 
            real_images: torch.Tensor, # (B, C, H, W)
            fake_images: torch.Tensor # (B, C, H, W)
            ) -> torch.Tensor: # Scalar
        
        batch_size = real_images.size(0)

        # Each image int he batch gets a random number between 0 & 1 and stored in vector alpha
        # Uniform(0,1) per WGAN-GP paper
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)

        # Calculate set of interpolated images
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff
        interpolated.requires_grad_(True)

        # Critic scores each interpolated image
        interpolated_pred = self.critic(interpolated) # (B, ) or (B,1)
        # Squeeze if critic returns shape (B,1)
        if interpolated_pred.dim() > 1:
            interpolated_pred = interpolated_pred.view(-1)

        # Gradients of prediction calculated with respect to input image
        grads = autograd.grad(
            outputs=interpolated_pred,
            inputs=interpolated,
            grad_outputs=torch.ones_like(
                interpolated_pred, 
                device=self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0] # (B, C, H, W)

        grads = grads.view(batch_size, -1)
        # Calculate L2 norm of gradient vector
        grad_norm = grads.norm(2, dim=1)
        # Return average squared distance between the L2 nrom and I
        gp = ((grad_norm - 1.0) ** 2).mean()
        return gp
    
    
    def train_step(self, real_images: torch.Tensor) -> Dict[str, torch.Tensor]:

        # Enable training mode
        self.generator.train()
        self.critic.train()

        # Move to device 
        real_images = real_images.to(self.device)
        batch_size = real_images.size(0)

        # tracking variables
        crit_loss_tracker = None
        crit_wass_tracker = None
        crit_grad_pen_tracker = None
        gen_loss_tracker = None

        ##########################################################
        #                     Critic updates                     #
        ##########################################################

        for _ in range(self.num_critic_updates):
            # Sample latent vectors
            z = torch.randn(batch_size, self.latent_dim, device=self.device)

            # Generate fake images
            fake_images = self.generator(z)
            # Ensure same dtype as real images
            fake_images = fake_images.to(self.device)

            # Critic predictions
            real_pred = self.critic(real_images)
            fake_pred = self.critic(fake_images.detach())

            # flatten predictions
            if real_pred.dim() > 1:
                real_pred = real_pred.view(-1)
            if fake_pred.dim() > 1:
                fake_pred = fake_pred.view(-1)
            
            # Wasserstein critic loss
            crit_wass = fake_pred.mean() - real_pred.mean()

            # Gradient penalty
            crit_grad_pen = self.gradient_penalty(real_images, fake_images)

            # Critic loss = wieghted sum of penalties + wass
            crit_loss = crit_wass + self.grad_pen_weight * crit_grad_pen

            # Update critic
            self.crit_optim.zero_grad()
            crit_loss.backward()
            self.crit_optim.step()

            # Keep last values for logging
            crit_loss_tracker = crit_loss.item()
            crit_wass_tracker = crit_wass.item()
            crit_grad_pen_tracker = crit_grad_pen.item()
        
        ##########################################################
        #                   Generator update                     #
        ##########################################################

        # Sample latent space
        z = torch.randn(batch_size, self.latent_dim, device=self.device)

        # Generate fake images
        fake_images = self.generator(z)

        # Critic prediction
        fake_pred_for_gen = self.critic(fake_images)

        # Flatten
        if fake_pred_for_gen.dim() > 1:
            fake_pred_for_gen = fake_pred_for_gen.view(-1)

        # Generator loss = E[critic(fake)]
        gen_loss = -fake_pred_for_gen.mean()

        # Update generator
        self.gen_optim.zero_grad()
        gen_loss.backward()
        self.gen_optim.step()

        # Keep last values for logging
        gen_loss_tracker = gen_loss.item()

        return {
            "crit_loss": crit_loss_tracker,
            "crit_wass": crit_wass_tracker,
            "crit_grad_pen": crit_grad_pen_tracker,
            "gen_loss": gen_loss_tracker
        }

import os
import torch


###############################
#   Checkpoint Utilities      #
###############################

def ensure_dir(path: str):
    """
    Creates directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def save_checkpoint(
        obj,            # nn.Module or dict
        path: str
):
    """
    Saves a module or state dict to the specified path.
    """

    ensure_dir(os.path.dirname(path))
    torch.save(obj, path)
    print(f"[checkpoint] Saved: {path}")


def load_checkpoint(
        path: str,
        map_location=None,
        strict: bool = True
):
    """
    Loads a full checkpoint (.pt or .pth). If strict=False and it is a state_dict,
    caller can partially load it.
    """

    if not os.path.exists(path):
        print(f"[checkpoint] Not found: {path}")
        return None

    ckpt = torch.load(path, map_location=map_location)
    print(f"[checkpoint] Loaded: {path}")
    return ckpt

import torch

from src.training.checkpoints import load_checkpoint

from src.inference.generator import ARCGenerator
from src.training.ppo_actor import PPOActor
from src.training.ppo_value import PPOValuer
from src.training.ppo_refiner import PPORefiner

from src.architecture.ViT.body import VisionTransformer
from src.architecture.context_encoding.example_pair_encoder import ExamplePairEncoder
from src.architecture.context_encoding.example_pair_aggregator import ExamplePairAggregator
from src.architecture.context_encoding.conditional_encoder import ConditionalTestInputEncoder
from src.architecture.LViTM.body import LargeVisionTransformerModel
from src.architecture.executor.executor import Executor
from src.architecture.adViT.critic import AdversarialVisionTransformer
from src.data_pipeline.dataloader import ARCDataModule

from src.inference.execution_controller import HybridExecuteController


###############################
#   Device + Constants        #
###############################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 11


###############################
#   Generator Builder         #
###############################

def build_generator():
    vit_gen = VisionTransformer(
        img_size=30,
        patch_size=1,
        embed_dim=128,
        num_heads=4,
        depth=6,
        mlp_dim=256,
        in_channels=2
    ).to(DEVICE)

    example_encoder = ExamplePairEncoder(vit_gen).to(DEVICE)
    aggregator = ExamplePairAggregator(embed_dim=vit_gen.c_token.size(-1)).to(DEVICE)
    cond_encoder = ConditionalTestInputEncoder(vit_gen).to(DEVICE)

    lvitm = LargeVisionTransformerModel(
        embed_dim=vit_gen.c_token.size(-1),
        num_heads=4,
        mlp_dim=256,
        depth=8,
        num_proposals=4,
        z_dim=64
    ).to(DEVICE)

    executor = Executor(
        embed_dim=vit_gen.c_token.size(-1),
        num_heads=4,
        mlp_dim=256,
        depth=4,
        z_dim=64,
        hidden_channels=64,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    generator = ARCGenerator(
        example_pair_encoder=example_encoder,
        aggregator=aggregator,
        cond_encoder=cond_encoder,
        lvitm=lvitm,
        executor=executor
    ).to(DEVICE)

    return generator, example_encoder, aggregator, cond_encoder, lvitm, executor


###############################
#   Critic + PPO Builders     #
###############################

def build_critic():
    vit_critic = VisionTransformer(
        img_size=30,
        patch_size=1,
        embed_dim=128,
        num_heads=4,
        depth=6,
        mlp_dim=256,
        in_channels=1 + NUM_CLASSES
    ).to(DEVICE)

    critic = AdversarialVisionTransformer(
        vit_encoder=vit_critic,
        z_dim=None,
        c_dim=None,
        hidden_dim=256
    ).to(DEVICE)

    return critic


def build_ppo(z_dim=64):
    actor = PPOActor(z_dim=z_dim, embed_dim=256).to(DEVICE)
    valuer = PPOValuer(z_dim=z_dim, embed_dim=256).to(DEVICE)
    refiner = PPORefiner(actor=actor, value_fn=valuer, lr=1e-4)
    return actor, valuer, refiner


###############################
#   Context Computation       #
###############################

def compute_context_C(
        example_encoder,
        aggregator,
        train_inputs,
        train_outputs,
        train_input_masks,
        train_output_masks
):
    if train_inputs.dim() == 3:
        train_inputs = train_inputs.unsqueeze(0)
        train_outputs = train_outputs.unsqueeze(0)
        train_input_masks = train_input_masks.unsqueeze(0)
        train_output_masks = train_output_masks.unsqueeze(0)

    B, K_train, H, W = train_inputs.shape

    h_list = []

    for k in range(K_train):
        I_k = train_inputs[:, k].unsqueeze(1).float()
        O_k = train_outputs[:, k].unsqueeze(1).float()

        mask_I_k = train_input_masks[:, k]
        mask_O_k = train_output_masks[:, k]

        h_k = example_encoder(
            I_i=I_k,
            O_i=O_k,
            mask_I=mask_I_k,
            mask_O=mask_O_k
        )  # (B,D)
        h_list.append(h_k)

    h = torch.stack(h_list, dim=1)
    pair_mask = None

    C = aggregator(h, mask=pair_mask)
    return C


###############################
#   Evaluation Loop           #
###############################

@torch.no_grad()
def evaluate_final():
    """
    Evaluates:
        - baseline generator (direct logits)
        - sequential + PPO refinement (first test input)
    Prints exact match and pixel accuracy for both.
    """

    # Build models
    generator, example_encoder, aggregator, cond_encoder, lvitm, executor = build_generator()
    critic = build_critic()
    actor, valuer, ppo_refiner = build_ppo(z_dim=64)

    # Load checkpoints if they exist
    gen_p3 = load_checkpoint("checkpoints/generator_phase3_adv.pt", map_location=DEVICE)
    if gen_p3 is not None:
        generator.load_state_dict(gen_p3, strict=False)

    crit_p3 = load_checkpoint("checkpoints/critic_phase3_adv.pt", map_location=DEVICE)
    if crit_p3 is not None:
        critic.load_state_dict(crit_p3, strict=False)

    actor_p4 = load_checkpoint("checkpoints/ppo_actor_phase4.pt", map_location=DEVICE)
    if actor_p4 is not None:
        actor.load_state_dict(actor_p4, strict=False)

    val_p4 = load_checkpoint("checkpoints/ppo_valuer_phase4.pt", map_location=DEVICE)
    if val_p4 is not None:
        valuer.load_state_dict(val_p4, strict=False)

    controller = HybridExecuteController(
        lvitm=lvitm,
        executor=executor,
        cond_encoder=cond_encoder,
        critic=critic
    ).to(DEVICE)

    # Metrics
    total_exact_baseline = 0
    total_exact_ppo = 0
    total_tests = 0

    total_pix_baseline = 0
    total_pix_ppo = 0
    total_pixels = 0

    for batch in ARCDataModule:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(DEVICE)

        train_inputs       = batch["train_inputs"]
        train_outputs      = batch["train_outputs"]
        train_input_masks  = batch["train_input_masks"]
        train_output_masks = batch["train_output_masks"]

        test_inputs        = batch["test_inputs"]
        test_outputs       = batch["test_outputs"]
        test_input_masks   = batch["test_input_masks"]
        test_output_masks  = batch.get("test_output_masks", test_input_masks)

        # Baseline forward (all tests)
        gen_out = generator(
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            train_input_masks=train_input_masks,
            train_output_masks=train_output_masks,
            test_inputs=test_inputs,
            test_input_masks=test_input_masks,
        )

        logits = gen_out["logits"]  # (1,K_test,C_out,H,W)
        preds = logits.argmax(dim=2).squeeze(0)  # (K_test,H,W)
        targets = test_outputs                    # (K_test,H,W)

        K_test, H, W = targets.shape

        for j in range(K_test):
            pred_b = preds[j]
            target_b = targets[j]

            # If original sizes exist, crop; else assume full
            if "test_original_size" in batch:
                orig_h, orig_w = batch["test_original_size"].tolist()
                pred_b = pred_b[:orig_h, :orig_w]
                target_b = target_b[:orig_h, :orig_w]

            exact_b = (pred_b == target_b).all().item()
            total_exact_baseline += exact_b
            total_tests += 1

            total_pix_baseline += (pred_b == target_b).sum().item()
            total_pixels += target_b.numel()

        # PPO sequential evaluation on first test input only
        if test_inputs.dim() == 3:
            test_inputs = test_inputs.unsqueeze(0)
            test_input_masks = test_input_masks.unsqueeze(0)
            test_outputs = test_outputs.unsqueeze(0)

        B, K_test, H, W = test_inputs.shape
        init_grid = test_inputs[:, 0].unsqueeze(1).float()      # (B,1,H,W)
        init_mask = test_input_masks[:, 0]                       # (B,H,W)
        target_ppo = test_outputs[:, 0]                          # (B,H,W)

        C = compute_context_C(
            example_encoder,
            aggregator,
            train_inputs,
            train_outputs,
            train_input_masks,
            train_output_masks
        )

        final_logits, _ = controller.ppo_rollout_and_update(
            init_grid=init_grid,
            init_mask=init_mask,
            C=C,
            ppo_refiner=ppo_refiner,
            num_steps=3,
            gamma=0.99
        )

        pred_ppo = final_logits.argmax(dim=1)  # (B,H,W)

        pred_ppo_b = pred_ppo[0]
        target_ppo_b = target_ppo[0]

        if "test_original_size" in batch:
            orig_h, orig_w = batch["test_original_size"].tolist()
            pred_ppo_b = pred_ppo_b[:orig_h, :orig_w]
            target_ppo_b = target_ppo_b[:orig_h, :orig_w]

        exact_ppo = (pred_ppo_b == target_ppo_b).all().item()
        total_exact_ppo += exact_ppo

        total_pix_ppo += (pred_ppo_b == target_ppo_b).sum().item()

    exact_acc_baseline = total_exact_baseline / total_tests if total_tests > 0 else 0.0
    exact_acc_ppo = total_exact_ppo / total_tests if total_tests > 0 else 0.0

    pix_acc_baseline = total_pix_baseline / total_pixels if total_pixels > 0 else 0.0
    pix_acc_ppo = total_pix_ppo / total_pixels if total_pixels > 0 else 0.0

    print("\n===============================")
    print("        FINAL EVALUATION       ")
    print("===============================")
    print(f"Baseline Exact Match: {exact_acc_baseline * 100:.2f}%")
    print(f"PPO Exact Match:      {exact_acc_ppo * 100:.2f}%")
    print(f"Baseline Pixel Acc:   {pix_acc_baseline * 100:.2f}%")
    print(f"PPO Pixel Acc:        {pix_acc_ppo * 100:.2f}%")


if __name__ == "__main__":
    evaluate_final()
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

# Import your modules
from src.inference.generator import ARCGenerator
from src.architecture.context_encoding.example_pair_encoder import ExamplePairEncoder
from src.architecture.context_encoding.example_pair_aggregator import ExamplePairAggregator
from src.architecture.context_encoding.conditional_encoder import ConditionalTestInputEncoder
from src.architecture.LViTM.body import LargeVisionTransformerModel
from src.architecture.executor.executor import Executor
from src.architecture.ViT.body import VisionTransformer

# Training constants
LR = 1e-4
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 11


##############################################################
#   Build MODEL for PHASE 1 using TWO separate ViTs
##############################################################
def build_model():

    img_size   = 30
    patch_size = 1
    embed_dim  = 128
    num_heads  = 4
    depth_vit  = 6
    mlp_dim    = 256
    z_dim      = 64
    num_props  = 4

    # For (I,O) pairs
    vit_pair = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth_vit,
        mlp_dim=mlp_dim,
        in_channels=2
    ).to(DEVICE)

    # For test input
    vit_test = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth_vit,
        mlp_dim=mlp_dim,
        in_channels=1
    ).to(DEVICE)

    example_encoder = ExamplePairEncoder(vit_pair).to(DEVICE)
    aggregator = ExamplePairAggregator(embed_dim=vit_pair.c_token.size(-1)).to(DEVICE)
    cond_encoder = ConditionalTestInputEncoder(vit_test).to(DEVICE)

    lvitm = LargeVisionTransformerModel(
        embed_dim=vit_pair.c_token.size(-1),
        num_heads=4,
        mlp_dim=256,
        depth=8,
        num_proposals=num_props,
        z_dim=z_dim
    ).to(DEVICE)

    executor = Executor(
        embed_dim=vit_pair.c_token.size(-1),
        num_heads=4,
        mlp_dim=256,
        depth=4,
        z_dim=z_dim,
        hidden_channels=64,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    generator = ARCGenerator(
        example_pair_encoder=example_encoder,
        aggregator=aggregator,
        cond_encoder=cond_encoder,
        lvitm=lvitm,
        executor=executor
    ).to(DEVICE)

    return generator


##############################################################
#   PHASE 1 TRAINING LOOP
##############################################################
def train_phase1(arc_loader):

    generator = build_model()
    optimizer = Adam(generator.parameters(), lr=LR)
    generator.train()

    for epoch in tqdm(range(EPOCHS), desc="Epoch:"):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

        total_loss = 0.0
        count = 0

        for batch in tqdm(arc_loader, desc="Batch:"):

            ############################################################
            #   Move batch to device
            ############################################################
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(DEVICE)

            # === FIX 1: All masks must be boolean ===
            batch["train_input_masks"]  = batch["train_input_masks"].bool()
            batch["train_output_masks"] = batch["train_output_masks"].bool()
            batch["test_input_masks"]   = batch["test_input_masks"].bool()
            batch["test_output_masks"]  = batch["test_output_masks"].bool()

            # Extract tensors
            train_inputs       = batch["train_inputs"]
            train_outputs      = batch["train_outputs"]
            train_input_masks  = batch["train_input_masks"]
            train_output_masks = batch["train_output_masks"]

            test_inputs        = batch["test_inputs"]
            test_outputs       = batch["test_outputs"]
            test_input_masks   = batch["test_input_masks"]
            test_output_masks  = batch["test_output_masks"]

            ############################################################
            #   Forward through ARCGenerator
            ############################################################
            out = generator(
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                train_input_masks=train_input_masks,
                train_output_masks=train_output_masks,
                test_inputs=test_inputs,
                test_input_masks=test_input_masks,
            )

            logits = out["logits"]  # (B, K_test, C_out, H, W)

            # === FIX 3: sanitize logits to prevent NaN/Inf propagation ===
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

            B, K_test, C_out, H, W = logits.shape

            ############################################################
            #   Clean CE Loss (Masked)
            ############################################################
            logits_flat = logits.view(B * K_test, C_out, H, W)
            target_flat = test_outputs.view(B * K_test, H, W)

            PAD_TOKEN = -100
            targets = target_flat.clone()

            pad_mask = ~test_output_masks.view(B*K_test, H, W).bool()
            targets[pad_mask] = PAD_TOKEN

            per_pixel = F.cross_entropy(
                logits_flat,
                targets,
                ignore_index=PAD_TOKEN,
                reduction="none"
            )

            valid_mask = (targets != PAD_TOKEN).float()
            loss = (per_pixel * valid_mask).sum() / valid_mask.sum()

            ############################################################
            #   Backprop
            ############################################################
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

    return generator


##############################################################
#   Script Entry Point
##############################################################
# if __name__ == "__main__":
#     from src.data_pipeline.dataloader import ARCDataModule
#
#     model = train_phase1(ARCDataModule)
#     torch.save(model.state_dict(), "phase1_generator.pt")
#     print("Saved Phase 1 generator.")

import torch
from torch import autograd

from src.training.utils_debug import report_param_stats

from src.architecture.ViT.body import VisionTransformer
from src.architecture.adViT.critic import AdversarialVisionTransformer
from src.data_pipeline.dataloader import ARCDataModule


###############################
#   Hyperparameters           #
###############################

CRITIC_LR = 1e-4
CRITIC_EPOCHS = 5
LAMBDA_GP = 10.0
NUM_CLASSES = 10  # ARC colors
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###############################
#   Build Critic + ViT        #
###############################

def build_critic():
    from src.architecture.ViT.body import VisionTransformer
    from src.architecture.adViT.critic import AdversarialVisionTransformer

    img_size   = 30
    patch_size = 1
    embed_dim  = 128
    num_heads  = 4
    depth_vit  = 6
    mlp_dim    = 256

    # IMPORTANT: ALWAYS 2 CHANNELS
    #   ch1 = I_test  (1 channel)
    #   ch2 = O_real or O_fake (1 channel)
    vit_critic = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth_vit,
        mlp_dim=mlp_dim,
        in_channels=2       
    ).to(DEVICE)

    critic = AdversarialVisionTransformer(
        vit_encoder=vit_critic,
        z_dim=None,
        c_dim=None,
        hidden_dim=256
    ).to(DEVICE)

    return critic



###############################
#   Fake Output Generator     #
###############################

def make_fake_outputs(
        real_outputs,   # (K_test, H, W)
        mask            # (K_test, H, W)
):
    """
    Creates fake outputs by sampling random colors where mask == 1.
    """

    K_test, H, W = real_outputs.shape
    fake = torch.randint(
        low=0,
        high=NUM_CLASSES,
        size=(K_test, H, W),
        device=real_outputs.device
    )

    # Keep padding as 0; only randomize valid cells
    fake = torch.where(mask.bool(), fake, torch.zeros_like(fake))

    return fake


###############################
#   Gradient Penalty (GP)     #
###############################

def gradient_penalty(
        critic,
        I_real,    # (B, 1, H, W)
        O_real,    # (B, 1, H, W)
        O_fake,    # (B, 1, H, W)
        mask_in,   # (B, H, W)
        mask_out   # (B, H, W)
):
    """
    WGAN-GP gradient penalty on interpolated outputs.
    Only interpolates in the output space for simplicity.
    """

    B, _, H, W = O_real.shape

    # Interpolate between real and fake outputs
    epsilon = torch.rand(B, 1, 1, 1, device=O_real.device)
    O_interp = epsilon * O_real + (1.0 - epsilon) * O_fake

    # Enable gradient tracking
    O_interp.requires_grad_(True)

    # Input grid stays fixed (we could also interpolate I, but ARC rules are usually about O)
    I_interp = I_real

    # Critic score on interpolated pair
    scores = critic(
        I_in=I_interp,
        O_pred=O_interp,
        mask_in=mask_in,
        mask_out=mask_out,
        z=None,
        C=None
    )  # (B,)

    # Compute gradients of scores w.r.t O_interp
    grads = autograd.grad(
        outputs=scores.sum(),
        inputs=O_interp,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # (B, 1, H, W)

    grads = grads.view(B, -1)  # (B, H*W)
    grad_norm = grads.norm(2, dim=1)  # (B,)

    gp = ((grad_norm - 1.0) ** 2).mean()
    return gp


###############################
#   Critic Warmup Loop        #
###############################

def train_critic_phase2(critic, data_loader):

    critic.train()
    optimizer = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)

    for epoch in range(CRITIC_EPOCHS):
        total_loss = 0.0
        total_batches = 0

        print(f"\n=== Critic Warmup Epoch {epoch + 1}/{CRITIC_EPOCHS} ===")

        for batch in data_loader:

            ###############################
            #   Move tensors to device    #
            ###############################

            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(DEVICE)

            test_inputs = batch["test_inputs"]          # (K_test, H, W)
            test_outputs = batch["test_outputs"]        # (K_test, H, W)
            test_input_masks = batch["test_input_masks"]    # (K_test, H, W)
            test_output_masks = batch["test_output_masks"]  # (K_test, H, W)

            K_test, H, W = test_inputs.shape
            if K_test == 0:
                continue

            ################################
            #   Prepare real / fake pairs  #
            ################################

            # Real input/output (B = K_test)
            I_real = test_inputs.unsqueeze(1).float()       # (B, 1, H, W)
            O_real = test_outputs.unsqueeze(1).float()      # (B, 1, H, W)

            mask_in = test_input_masks.bool()               # (B, H, W)
            mask_out = test_output_masks.bool()             # (B, H, W)

            # Fake outputs
            O_fake_raw = make_fake_outputs(
                real_outputs=test_outputs,
                mask=test_output_masks
            )  # (B, H, W)
            O_fake = O_fake_raw.unsqueeze(1).float()        # (B, 1, H, W)

            ###############################
            #   Critic Scores             #
            ###############################

            # Real scores
            score_real = critic(
                I_in=I_real,
                O_pred=O_real,
                mask_in=mask_in,
                mask_out=mask_out,
                z=None,
                C=None
            )  # (B,)

            # Fake scores
            score_fake = critic(
                I_in=I_real,
                O_pred=O_fake,
                mask_in=mask_in,
                mask_out=mask_out,
                z=None,
                C=None
            )  # (B,)

            wasserstein = score_fake.mean() - score_real.mean()

            # Gradient penalty
            gp = gradient_penalty(
                critic=critic,
                I_real=I_real,
                O_real=O_real,
                O_fake=O_fake,
                mask_in=mask_in,
                mask_out=mask_out
            )

            loss = wasserstein + LAMBDA_GP * gp

            ###############################
            #   Optimize Critic           #
            ###############################

            optimizer.zero_grad()
            loss.backward()

            ###################################
            #   DEBUG: Check Critic Gradients
            ###################################
            for name, p in critic.named_parameters():
                if p.grad is None:
                    print(f"[PHASE2 WARNING] No grad for {name}")
                else:
                    if torch.isnan(p.grad).any():
                        print(f"[PHASE2 ERROR] NaN gradient detected in {name}")
                    if torch.all(p.grad == 0):
                        print(f"[PHASE2 WARNING] ZERO gradient in {name}")

            # Optional (expensive): print weight + grad stats
            # report_param_stats(critic, name="Phase2 Critic", max_layers=12)

            optimizer.step()


            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        print(f"Epoch {epoch + 1}: Critic Warmup Loss = {avg_loss:.4f}")

    return critic


#######################
#   Main Entrypoint   #
#######################

if __name__ == "__main__":
    critic = build_critic()
    critic = train_critic_phase2(critic, ARCDataModule)
    torch.save(critic.state_dict(), "critic_phase2_warmup.pt")
    print("\nSaved critic after Phase 2 warmup: critic_phase2_warmup.pt")

import torch
import torch.nn.functional as F
from torch import autograd

from src.inference.generator import ARCGenerator

from src.training.utils_debug import report_param_stats

from src.architecture.context_encoding.example_pair_encoder import ExamplePairEncoder
from src.architecture.context_encoding.example_pair_aggregator import ExamplePairAggregator
from src.architecture.context_encoding.conditional_encoder import ConditionalTestInputEncoder
from src.architecture.ViT.body import VisionTransformer
from src.architecture.LViTM.body import LargeVisionTransformerModel
from src.architecture.executor.executor import Executor
from src.architecture.adViT.critic import AdversarialVisionTransformer
from src.data_pipeline.dataloader import ARCDataModule



###############################
#   Hyperparameters           #
###############################

EPOCHS = 5

GEN_LR = 1e-4
CRITIC_LR = 1e-4

LAMBDA_GP = 10.0
LAMBDA_ADV = 0.1

N_CRITIC = 5  # critic steps per generator step

NUM_CLASSES = 11  # ARC colors
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###############################
#   Helper: grad toggle       #
###############################

def set_requires_grad(module, flag: bool):
    """
    Enables or disables gradients for a module.
    """
    for p in module.parameters():
        p.requires_grad = flag


###############################
#   Build Generator           #
###############################

def build_generator():
    from src.architecture.ViT.body import VisionTransformer

    img_size   = 30
    patch_size = 1
    embed_dim  = 128
    num_heads  = 4
    depth_vit  = 6
    mlp_dim    = 256
    z_dim      = 64
    num_props  = 4
    NUM_CLASSES = 11  

    # Two separate ViTs
    vit_pair = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth_vit,
        mlp_dim=mlp_dim,
        in_channels=2          # (I, O)
    ).to(DEVICE)

    vit_test = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth_vit,
        mlp_dim=mlp_dim,
        in_channels=1          # I_test
    ).to(DEVICE)

    example_encoder = ExamplePairEncoder(vit_pair).to(DEVICE)

    aggregator = ExamplePairAggregator(embed_dim=embed_dim).to(DEVICE)

    cond_encoder = ConditionalTestInputEncoder(vit_test).to(DEVICE)

    lvitm = LargeVisionTransformerModel(
        embed_dim=embed_dim,
        num_heads=4,
        mlp_dim=256,
        depth=8,
        num_proposals=num_props,
        z_dim=z_dim
    ).to(DEVICE)

    executor = Executor(
        embed_dim=embed_dim,
        num_heads=4,
        mlp_dim=256,
        depth=4,
        z_dim=z_dim,
        hidden_channels=64,
        num_classes=NUM_CLASSES  
    ).to(DEVICE)

    generator = ARCGenerator(
        example_pair_encoder=example_encoder,
        aggregator=aggregator,
        cond_encoder=cond_encoder,
        lvitm=lvitm,
        executor=executor
    ).to(DEVICE)

    return generator



###############################
#   Build Critic              #
###############################

def build_critic():
    from src.architecture.ViT.body import VisionTransformer
    from src.architecture.adViT.critic import AdversarialVisionTransformer

    img_size   = 30
    patch_size = 1
    embed_dim  = 128
    num_heads  = 4
    depth_vit  = 6
    mlp_dim    = 256

    # IMPORTANT: ALWAYS 2 CHANNELS
    #   ch1 = I_test  (1 channel)
    #   ch2 = O_real or O_fake (1 channel)
    vit_critic = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth_vit,
        mlp_dim=mlp_dim,
        in_channels=2       
    ).to(DEVICE)

    critic = AdversarialVisionTransformer(
        vit_encoder=vit_critic,
        z_dim=None,
        c_dim=None,
        hidden_dim=256
    ).to(DEVICE)

    return critic



###############################
#   One-hot encoder           #
###############################

def one_hot_from_int(
        grid,       # (B, H, W) int
        num_classes # scalar
):
    """
    Converts integer grid to one-hot channels: (B, num_classes, H, W)
    """

    B, H, W = grid.shape
    # (B, H, W, C)
    oh = F.one_hot(grid.long().clamp(min=0, max=num_classes - 1), num_classes=num_classes)
    oh = oh.permute(0, 3, 1, 2).float()  # (B, C, H, W)
    return oh


###############################
#   Gradient Penalty (GP)     #
###############################

def gradient_penalty(
        critic,
        I_real,    # (B,1,H,W)
        O_real,    # (B,1,H,W)
        O_fake,    # (B,1,H,W)
        mask_in,   # (B,H,W)
        mask_out   # (B,H,W)
):
    B, _, H, W = O_real.shape

    epsilon = torch.rand(B, 1, 1, 1, device=O_real.device)
    O_interp = epsilon * O_real + (1.0 - epsilon) * O_fake
    O_interp.requires_grad_(True)

    scores = critic(
        I_in=I_real,
        O_pred=O_interp,
        mask_in=mask_in,
        mask_out=mask_out,
        z=None,
        C=None
    )

    grads = autograd.grad(
        outputs=scores.sum(),
        inputs=O_interp,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    grads = grads.view(B, -1)
    grad_norm = grads.norm(2, dim=1)
    return ((grad_norm - 1.0)**2).mean()




###############################
#   Phase 3 Training Loop     #
###############################

def train_phase3_adversarial(
        generator,
        critic,
        data_loader
):
    """
    Joint adversarial training of generator + critic.
    Supervised CE + WGAN-GP adversarial loss.
    """

    # Optionally load checkpoints from Phase 1 and Phase 2
    try:
        generator.load_state_dict(torch.load("phase1_generator.pt", map_location=DEVICE))
        print("Loaded Phase 1 generator checkpoint.")
    except FileNotFoundError:
        print("Phase 1 generator checkpoint not found; training generator from scratch.")

    try:
        critic.load_state_dict(torch.load("critic_phase2_warmup.pt", map_location=DEVICE))
        print("Loaded Phase 2 critic checkpoint.")
    except FileNotFoundError:
        print("Phase 2 critic checkpoint not found; training critic from scratch.")

    gen_opt = torch.optim.Adam(generator.parameters(), lr=GEN_LR)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)

    for epoch in range(EPOCHS):
        print(f"\n=== Phase 3 Epoch {epoch + 1}/{EPOCHS} ===")
        total_gen_loss = 0.0
        total_critic_loss = 0.0
        n_gen_steps = 0
        n_critic_steps = 0

        for batch_idx, batch in enumerate(data_loader):

            ###############################
            #   Move batch to device      #
            ###############################

            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(DEVICE)

            train_inputs       = batch["train_inputs"]        # (K_train, H, W)
            train_outputs      = batch["train_outputs"]       # (K_train, H, W)
            train_input_masks  = batch["train_input_masks"]   # (K_train, H, W)
            train_output_masks = batch["train_output_masks"]  # (K_train, H, W)

            test_inputs        = batch["test_inputs"]         # (K_test, H, W)
            test_outputs       = batch["test_outputs"]        # (K_test, H, W)
            test_input_masks   = batch["test_input_masks"]    # (K_test, H, W)
            test_output_masks  = batch["test_output_masks"]   # (K_test, H, W)

            ###############################
            #   Forward: Generator        #
            ###############################

            # ARCGenerator handles B=1 internally
            gen_out = generator(
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                train_input_masks=train_input_masks,
                train_output_masks=train_output_masks,
                test_inputs=test_inputs,
                test_input_masks=test_input_masks,
            )

            logits = gen_out["logits"]  # (1, K_test, C_out, H, W)
            B, K_test, C_out, H, W = logits.shape
            assert B == 1, "Phase 3 currently assumes batch_size=1 per task."

            # -------------------------------
            #   Critic updates (N_CRITIC)    #
            # -------------------------------

            for _ in range(N_CRITIC):
                set_requires_grad(critic, True)
                set_requires_grad(generator, False)

                critic_opt.zero_grad()

                # Real outputs: single-channel integer grid
                targets = test_outputs.view(K_test, H, W)               # (K_test, H, W)
                O_real = targets.unsqueeze(1).float()                   # (K_test, 1, H, W)

                # Fake outputs: take argmax over class dimension
                logits_det = logits.detach().view(K_test, C_out, H, W)  # (K_test, C_out, H, W)
                O_fake = logits_det.argmax(dim=1, keepdim=True).float() # (K_test, 1, H, W)


                # Inputs
                I_real = test_inputs.view(K_test, H, W).unsqueeze(1).float()  # (K_test,1,H,W)

                # Masks
                mask_in = test_input_masks.view(K_test, H, W).bool()
                mask_out = test_output_masks.view(K_test, H, W).bool()

                score_real = critic(
                    I_in=I_real,
                    O_pred=O_real,
                    mask_in=mask_in,
                    mask_out=mask_out,
                    z=None,
                    C=None
                )

                score_fake = critic(
                    I_in=I_real,
                    O_pred=O_fake,       # FIXED
                    mask_in=mask_in,
                    mask_out=mask_out,
                    z=None,
                    C=None
                )


                wasserstein = score_fake.mean() - score_real.mean()
                gp = gradient_penalty(
                    critic=critic,
                    I_real=I_real,
                    O_real=O_real,
                    O_fake=O_fake,
                    mask_in=mask_in,
                    mask_out=mask_out
                )


                critic_loss = wasserstein + LAMBDA_GP * gp

                critic_loss.backward()

                for name, p in critic.named_parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any():
                            print(f"[PHASE3 CRITIC NaN GRAD] {name}")
                        if torch.all(p.grad == 0):
                            print(f"[PHASE3 CRITIC ZERO GRAD] {name}")

                critic_opt.step()


                total_critic_loss += critic_loss.item()
                n_critic_steps += 1

            ###############################
            #   Generator update          #
            ###############################

            set_requires_grad(critic, False)
            set_requires_grad(generator, True)

            gen_opt.zero_grad()

            # Supervised CE loss across all test pairs
            logits_flat = logits.view(B*K_test, C_out, H, W)
            targets_flat = test_outputs.view(B*K_test, H, W).long()

            PAD_TOKEN = -100
            targets = targets_flat.clone()
            pad_mask = ~test_output_masks.view(B*K_test, H, W).bool()
            targets[pad_mask] = PAD_TOKEN

            per_pixel = F.cross_entropy(
                logits_flat,
                targets,
                ignore_index=PAD_TOKEN,
                reduction="none"
            )

            valid = (targets != PAD_TOKEN).float()
            ce_loss = (per_pixel * valid).sum() / valid.sum()

            I_for_adv = test_inputs.view(K_test, H, W).unsqueeze(1).float()
            mask_in_adv = test_input_masks.view(K_test, H, W).bool()
            mask_out_adv = test_output_masks.view(K_test, H, W).bool()

            logits_for_adv = logits.view(K_test, C_out, H, W)
            O_fake_adv = logits_for_adv.argmax(dim=1, keepdim=True).float()

            fake_scores = critic(
                I_in=I_for_adv,
                O_pred=O_fake_adv,        
                mask_in=mask_in_adv,
                mask_out=mask_out_adv,
                z=None,
                C=None
            )


            gen_adv_loss = -fake_scores.mean()

            gen_loss = ce_loss + LAMBDA_ADV * gen_adv_loss

            gen_loss.backward()

            for name, p in generator.named_parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any():
                        print(f"[PHASE3 GEN NaN GRAD] {name}")
                    if torch.all(p.grad == 0):
                        print(f"[PHASE3 GEN ZERO GRAD] {name}")

            gen_opt.step()


            total_gen_loss += gen_loss.item()
            n_gen_steps += 1

        avg_gen = total_gen_loss / max(n_gen_steps, 1)
        avg_critic = total_critic_loss / max(n_critic_steps, 1)

        print(f"Epoch {epoch + 1}: Gen Loss = {avg_gen:.4f}, Critic Loss = {avg_critic:.4f}")

    return generator, critic


###############################
#   Main Entrypoint           #
###############################

if __name__ == "__main__":
    generator = build_generator()
    critic = build_critic()

    generator, critic = train_phase3_adversarial(
        generator=generator,
        critic=critic,
        data_loader=ARCDataModule
    )

    torch.save(generator.state_dict(), "generator_phase3_adv.pt")
    torch.save(critic.state_dict(), "critic_phase3_adv.pt")

    print("\nSaved Phase 3 generator and critic checkpoints.")
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal

from src.training.ppo_actor import PPOActor
from src.training.ppo_value import PPOValuer
from src.training.ppo_refiner import PPORefiner

from src.architecture.context_encoding.example_pair_encoder import ExamplePairEncoder
from src.architecture.context_encoding.example_pair_aggregator import ExamplePairAggregator
from src.architecture.context_encoding.conditional_encoder import ConditionalTestInputEncoder
from src.architecture.ViT.body import VisionTransformer
from src.architecture.LViTM.body import LargeVisionTransformerModel
from src.architecture.executor.executor import Executor
from src.architecture.adViT.critic import AdversarialVisionTransformer

from src.data_pipeline.dataloader import ARCDataModule
from src.inference.execution_controller import HybridExecuteController


############################################################
#              ** PPO HYPERPARAMETERS **
############################################################
PPO_EPOCHS = 4
PPO_STEPS = 8          # PPO rollout length
PPO_GAMMA = 0.99       # reward discount
PPO_LAMBDA = 0.95      # GAE lambda
PPO_CLIP = 0.2
PPO_LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
Z_DIM = 64
NUM_PROPOSALS = 4


############################################################
#            Build All Components Cleanly
############################################################
def build_generator_components():
    img_size = 30
    patch = 1
    embed_dim = 128
    heads = 4
    depth_vit = 6
    mlp_dim = 256

    vit_pair = VisionTransformer(
        img_size=img_size,
        patch_size=patch,
        embed_dim=embed_dim,
        num_heads=heads,
        depth=depth_vit,
        mlp_dim=mlp_dim,
        in_channels=2
    ).to(DEVICE)

    vit_test = VisionTransformer(
        img_size=img_size,
        patch_size=patch,
        embed_dim=embed_dim,
        num_heads=heads,
        depth=depth_vit,
        mlp_dim=mlp_dim,
        in_channels=1
    ).to(DEVICE)

    example_encoder = ExamplePairEncoder(vit_pair).to(DEVICE)
    aggregator = ExamplePairAggregator(embed_dim).to(DEVICE)
    cond_encoder = ConditionalTestInputEncoder(vit_test).to(DEVICE)

    lvitm = LargeVisionTransformerModel(
        embed_dim=embed_dim,
        num_heads=heads,
        mlp_dim=256,
        depth=8,
        num_proposals=NUM_PROPOSALS,
        z_dim=Z_DIM
    ).to(DEVICE)

    executor = Executor(
        embed_dim=embed_dim,
        num_heads=heads,
        mlp_dim=256,
        depth=4,
        z_dim=Z_DIM,
        hidden_channels=64,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    return example_encoder, aggregator, cond_encoder, lvitm, executor



def build_critic():
    vit = VisionTransformer(
        img_size=30,
        patch_size=1,
        embed_dim=128,
        num_heads=4,
        depth=6,
        mlp_dim=256,
        in_channels=2
    ).to(DEVICE)

    return AdversarialVisionTransformer(
        vit_encoder=vit,
        z_dim=None,
        c_dim=None,
        hidden_dim=256
    ).to(DEVICE)



############################################################
#         Compute Transformer Context C
############################################################
def compute_context(example_encoder, aggregator, train_inputs, train_outputs,
                    train_input_masks, train_output_masks):

    if train_inputs.dim() == 3:
        train_inputs = train_inputs.unsqueeze(0)
        train_outputs = train_outputs.unsqueeze(0)
        train_input_masks = train_input_masks.unsqueeze(0)
        train_output_masks = train_output_masks.unsqueeze(0)

    B, K, H, W = train_inputs.shape
    h_list = []

    for k in range(K):
        I_k = train_inputs[:, k].unsqueeze(1).float()
        O_k = train_outputs[:, k].unsqueeze(1).float()

        mI = train_input_masks[:, k]
        mO = train_output_masks[:, k]

        h_list.append(example_encoder(I_k, O_k, mI, mO))

    h = torch.stack(h_list, dim=1)
    return aggregator(h, mask=None)



############################################################
#              PPO Rollout + Update
############################################################
def ppo_rollout(controller, init_grid, init_mask, C, actor, valuer):
    """
    Run a PPO rollout:
       (1) sample z adjustments
       (2) calculate reward from critic
       (3) compute logprobs, advantages, returns
    """

    B, _, H, W = init_grid.shape

    # Storage
    states = []
    actions = []
    logprobs = []
    values = []
    rewards = []

    grid = init_grid.clone()

    for t in range(PPO_STEPS):

        # Encode for LVITM proposals
        tokens, padmask = controller.cond_encoder(grid, init_mask, C)
        Z = controller.lvitm(C, tokens, padmask)  # (B,T,z_dim)

        # Actor samples "action" = Δz shift
        mu, log_std = actor(Z)
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        action = dist.sample()           # (B,T,z_dim)
        logp = dist.log_prob(action).sum(dim=-1)  # (B,T)

        # Apply action (shift proposals)
        Z_refined = Z + 0.1 * action

        # Execute each proposal
        B,T,z_dim = Z_refined.shape
        grid_batch = grid.unsqueeze(1).expand(B,T,1,H,W).reshape(B*T,1,H,W)
        z_flat = Z_refined.reshape(B*T,z_dim)

        logits = controller.executor(grid_batch, z_flat)
        logits = logits.view(B,T,NUM_CLASSES,H,W)

        # Critic gives reward
        reward = controller.critic(
            I_in=grid,
            O_pred=logits,
            mask_in=init_mask,
            mask_out=init_mask,
            z=Z_refined,
            C=C
        )  # (B,T)

        # Value estimate
        value = valuer(Z)  # (B,T)

        # Store trajectory
        states.append(Z)
        actions.append(action)
        logprobs.append(logp)
        values.append(value)
        rewards.append(reward)

        # Select best output for next state
        best_idx = reward.argmax(dim=1)  # (B)
        idx = best_idx.view(B,1,1,1,1).expand(B,1,NUM_CLASSES,H,W)
        best_logits = logits.gather(dim=1, index=idx).squeeze(1)
        grid = best_logits.argmax(dim=1, keepdim=True).float()

    # Convert lists to tensors:
    states = torch.stack(states, dim=0)      # (T,B,P,D)
    actions = torch.stack(actions, dim=0)    # (T,B,P,D)
    logprobs = torch.stack(logprobs, dim=0)  # (T,B,P)
    values = torch.stack(values, dim=0)      # (T,B,P)
    rewards = torch.stack(rewards, dim=0)    # (T,B,P)

    return states, actions, logprobs, values, rewards



def compute_gae(values, rewards):
    """
    Generalized Advantage Estimation (GAE-Lambda)
    """
    T, B, P = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(B, P, device=DEVICE)

    for t in reversed(range(T)):
        delta = rewards[t] + PPO_GAMMA * (values[t+1] if t < T-1 else 0) - values[t]
        gae = delta + PPO_GAMMA * PPO_LAMBDA * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns



############################################################
#                   ** PHASE 4 MAIN **
############################################################
def train_phase4_ppo(data_loader):
    # Build models
    example_encoder, aggregator, cond_encoder, lvitm, executor = build_generator_components()
    critic = build_critic()

    controller = HybridExecuteController(
        lvitm=lvitm,
        executor=executor,
        cond_encoder=cond_encoder,
        critic=critic
    ).to(DEVICE)

    actor = PPOActor(z_dim=Z_DIM, embed_dim=256).to(DEVICE)
    valuer = PPOValuer(z_dim=Z_DIM, embed_dim=256).to(DEVICE)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(valuer.parameters()), lr=PPO_LR)

    # MAIN TRAIN LOOP
    for epoch in range(PPO_EPOCHS):
        print(f"\n==== PPO Epoch {epoch+1}/{PPO_EPOCHS} ====")

        for batch_idx, batch in enumerate(data_loader):
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(DEVICE)

            train_in = batch["train_inputs"]
            train_out = batch["train_outputs"]
            train_inm = batch["train_input_masks"]
            train_outm = batch["train_output_masks"]

            test_in = batch["test_inputs"]
            test_inm = batch["test_input_masks"]
            test_out = batch["test_outputs"]

            C = compute_context(example_encoder, aggregator,
                                train_in, train_out, train_inm, train_outm)

            if test_in.dim() == 3:
                test_in = test_in.unsqueeze(0)
                test_inm = test_inm.unsqueeze(0)
                test_out = test_out.unsqueeze(0)

            B, K, H, W = test_in.shape
            init_grid = test_in[:,0].unsqueeze(1).float()
            init_mask = test_inm[:,0]

            # === PPO Rollout ===
            states, actions, old_logp, values, rewards = ppo_rollout(
                controller, init_grid, init_mask, C, actor, valuer
            )

            advantages, returns = compute_gae(values, rewards)

            # === PPO UPDATE ===
            optimizer.zero_grad()

            T,B,P,D = states.shape
            states = states.detach()
            actions = actions.detach()
            advantages = advantages.detach()
            returns = returns.detach()
            old_logp = old_logp.detach()

            mu, log_std = actor(states)
            dist = Normal(mu, torch.exp(log_std))
            new_logp = dist.log_prob(actions).sum(dim=-1)

            ratio = torch.exp(new_logp - old_logp)
            clipped = torch.clamp(ratio, 1-PPO_CLIP, 1+PPO_CLIP)

            policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
            value_loss = F.mse_loss(valuer(states), returns)
            entropy = dist.entropy().mean()

            loss = policy_loss + 0.5*value_loss - 0.01*entropy
            loss.backward()
            optimizer.step()

            print(f"[PPO] Batch {batch_idx} | loss={loss.item():.4f} | policy={policy_loss.item():.4f}")

    return actor, valuer



if __name__ == "__main__":
    train_phase4_ppo(ARCDataModule)
import torch
import torch.nn as nn


class PPOActor(nn.Module):
    """
    Computes the mean and log_std for latent Z proposals.
    """

    def __init__(
            self, 
            z_dim, 
            embed_dim=256
    ):
        ###############################
        #   B = batch size            #    
        #   D = token embedding dim   #
        #   P = num proposals         #
        ###############################

        super().__init__()

        # Mean network
        self.mu_net = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, z_dim)
        )

        # Learnable log_std (per-dimension, shared over B and P)
        self.log_std = nn.Parameter(torch.zeros(z_dim))

    def forward(
            self,
            Z   # (B, P, z_dim)
    ):
        # Mean of Gaussian
        mu = self.mu_net(Z)  # (B, P, z_dim)

        # Expand log_std to match (B, P, z_dim)
        log_std = self.log_std.expand_as(mu)  # (B, P, z_dim)

        return mu, log_std
    import torch
import torch.nn.functional as F

from src.training.ppo_actor import PPOActor
from src.training.ppo_value import PPOValuer


class PPORefiner:
    """
    Performs PPO updates on latent Z proposals.
    """

    def __init__(
        self,
        actor: PPOActor,
        value_fn: PPOValuer,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        lr=1e-4,
    ):
        ###############################
        #   B = batch size            #    
        #   P = num proposals         #
        ###############################

        self.actor = actor
        self.value_fn = value_fn

        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.opt = torch.optim.Adam(
            list(actor.parameters()) + list(value_fn.parameters()),
            lr=lr
        )

    #######################
    #   Gaussian log-prob #
    #######################
    def _log_prob(self, actions, mu, log_std):
        # std
        std = torch.exp(log_std)  # (B, P, z_dim)

        # Gaussian log prob
        # sum over z_dim (the action dimensions)
        logp = -0.5 * (((actions - mu) / std) ** 2 + 2 * log_std + torch.log(2 * torch.pi))
        logp = logp.sum(dim=-1)  # (B, P)

        return logp

    #############################
    #   Refinement step (TTA)   #
    #############################
    @torch.no_grad()
    def refine(self, Z, steps=1, scale=0.1):
        """
        Test-time heuristic refinement:
        small shifts along actor mean.
        """
        Z_new = Z.clone()

        for _ in range(steps):
            mu, _ = self.actor(Z_new)
            Z_new = Z_new + scale * mu

        return Z_new

    ############################
    #        PPO UPDATE        #
    ############################
    def update(self, Z, actions, old_logp, returns, advantages):
        """
        Z:        (B, P, z_dim)
        actions:  (B, P, z_dim)
        old_logp: (B, P)
        returns:  (B, P)
        advantages: (B, P)
        """

        # New policy
        mu, log_std = self.actor(Z)  # (B, P, z_dim)

        # New log prob
        logp = self._log_prob(actions, mu, log_std)  # (B, P)

        # PPO ratio
        ratio = torch.exp(logp - old_logp)  # (B, P)

        # Clipped surrogate objective
        clipped_ratio = torch.clamp(
            ratio,
            1 - self.clip_ratio,
            1 + self.clip_ratio
        )
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages

        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        values = self.value_fn(Z)  # (B, P)
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus
        entropy = 0.5 * (log_std.exp() * torch.sqrt(torch.tensor(2 * torch.pi * torch.e))).mean()
        entropy_loss = -self.entropy_coef * entropy

        loss = policy_loss + self.value_coef * value_loss + entropy_loss

        # Update step
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }
import torch.nn as nn


class PPOValuer(nn.Module):
    """
    Value function for latent proposals Z
    """

    def __init__(
            self, 
            z_dim, 
            embed_dim=256
    ):
        ###############################
        #   B = batch size            #    
        #   P = num proposals         #
        ###############################

        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(
            self,
            Z   # (B, P, z_dim)
    ):
        *prefix, D = Z.shape
        Z_flat = Z.view(-1, D)

        # FIX: use the correct attribute
        v_flat = self.net(Z_flat).squeeze(-1)

        return v_flat.view(*prefix)
import argparse
import torch
from pathlib import Path

from src.training.checkpoints import save_checkpoint, load_checkpoint

###############################
#   Device                    #
###############################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###############################
#   Imports for each phase    #
###############################

# Phase 1
from src.training.phase1 import train_phase1          # def train_phase1(arc_loader) -> generator

# Phase 2
from src.training.phase2 import train_critic_phase2, build_critic
#   def build_critic() -> critic
#   def train_critic_phase2(critic, data_loader) -> critic

# Phase 3
from src.training.phase3 import train_phase3_adversarial, build_generator, build_critic as build_critic_phase3
#   def build_generator() -> generator
#   def build_critic_phase3() -> critic
#   def train_phase3_adversarial(generator, critic, data_loader) -> (generator, critic)

# Phase 4
from src.training.phase4 import train_phase4_ppo  # def train_phase4_ppo() -> None

# Data loader
from src.data_pipeline.dataloader import ARCDataModule

# Point to your local folder named "training"
folder_path = Path("./src/data_pipeline/ARC_data/data/training")

data_module = ARCDataModule(
    dir_path=folder_path,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    pad_value=0,
).prepare()

# ========================
# LIMIT DATASET FOR DEBUG
# ========================
LIMIT_SAMPLES = 1    # or 2

if LIMIT_SAMPLES is not None:
    data_module.dataset.data = data_module.dataset.data[:LIMIT_SAMPLES]

arc_loader = data_module.get_loader()


###############################
#   Orchestrator              #
###############################

def run_phase1():
    """
    Phase 1: supervised pretraining of generator.
    """
    print("\n===============================")
    print("   PHASE 1: Supervised Train   ")
    print("===============================")

    generator = train_phase1(arc_loader)

    # Save generator weights
    save_checkpoint(generator.state_dict(), "checkpoints/phase1_generator.pt")


def run_phase2():
    """
    Phase 2: critic warmup with WGAN-GP and random fake outputs.
    """
    print("\n===============================")
    print("   PHASE 2: Critic Warmup      ")
    print("===============================")

    critic = build_critic()

    critic = train_critic_phase2(critic, arc_loader)

    save_checkpoint(critic.state_dict(), "checkpoints/critic_phase2_warmup.pt")


def run_phase3():
    """
    Phase 3: joint adversarial training of generator + critic.
    Uses phase1 + phase2 checkpoints if available.
    """
    print("\n===============================")
    print("   PHASE 3: Adversarial Train  ")
    print("===============================")

    # Build fresh models
    generator = build_generator()
    critic = build_critic_phase3()

    # Optionally load phase 1 generator
    ckpt_gen_p1 = load_checkpoint("checkpoints/phase1_generator.pt", map_location=DEVICE)
    if ckpt_gen_p1 is not None:
        generator.load_state_dict(ckpt_gen_p1, strict=False)

    # Optionally load phase 2 critic
    ckpt_crit_p2 = load_checkpoint("checkpoints/critic_phase2_warmup.pt", map_location=DEVICE)
    if ckpt_crit_p2 is not None:
        critic.load_state_dict(ckpt_crit_p2, strict=False)

    generator, critic = train_phase3_adversarial(
        generator=generator,
        critic=critic,
        data_loader=arc_loader
    )

    save_checkpoint(generator.state_dict(), "checkpoints/generator_phase3_adv.pt")
    save_checkpoint(critic.state_dict(), "checkpoints/critic_phase3_adv.pt")


def run_phase4():
    """
    Phase 4: PPO refinement over latent proposals using HybridExecuteController.
    Uses phase3 checkpoints if available.
    """
    print("\n===============================")
    print("   PHASE 4: PPO Refinement     ")
    print("===============================")

    # train_phase4_ppo internally loads generator/critic if needed
    actor, value_fn = train_phase4_ppo(data_loader=arc_loader)

    # Save PPO modules
    torch.save(actor.state_dict(), "checkpoints/ppo_actor_phase4.pt")
    torch.save(value_fn.state_dict(), "checkpoints/ppo_valuer_phase4.pt")
    print("\nSaved PPO actor and valuer for Phase 4.")


###############################
#   Main Entrypoint           #
###############################

def main():
    parser = argparse.ArgumentParser(description="ARC-AGI-2 Training Orchestrator")

    parser.add_argument(
        "--start_phase",
        type=int,
        default=1,
        help="phase to start from (1-4)"
    )
    parser.add_argument(
        "--end_phase",
        type=int,
        default=4,
        help="phase to end at (1-4)"
    )

    args = parser.parse_args()

    run_phase1()

    for phase in range(args.start_phase, args.end_phase + 1):
        if phase == 1:
            run_phase1()
        elif phase == 2:
            run_phase2()
        elif phase == 3:
            run_phase3()
        elif phase == 4:
            run_phase4()
        else:
            print(f"[orchestrator] Unknown phase: {phase}")


if __name__ == "__main__":
    main()
import torch

def report_param_stats(model, name="", max_layers=10):
    print(f"\n========= PARAM REPORT: {name} =========")
    for i, (n, p) in enumerate(model.named_parameters()):
        if i >= max_layers:
            print("... (truncated)")
            break
        if p.grad is None:
            print(f"{n}: grad=None | weight mean={p.data.mean().item():.4f} std={p.data.std().item():.4f}")
        else:
            print(f"{n}: grad mean={p.grad.mean().item():.4f} | grad std={p.grad.std().item():.4f} | "
                  f"weight mean={p.data.mean().item():.4f} std={p.data.std().item():.4f}")
import torch
import torch.nn.functional as F

# Import your generator
from src.inference.generator import ARCGenerator

# Probably in the same file as ARCSampleDataset
from src.data_pipeline.dataloader import ARCDataModule


@torch.no_grad()
def validate_phase1(generator, val_loader, device):
    generator.eval()

    total_exact = 0
    total_tests = 0

    total_pixels = 0
    total_pixels_correct = 0

    for batch in val_loader:

        # Move tensors to device
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)

        K_test = batch["test_inputs"].shape[0]

        # Forward pass
        out = generator(
            train_inputs=batch["train_inputs"],
            train_outputs=batch["train_outputs"],
            train_input_masks=batch["train_input_masks"],
            train_output_masks=batch["train_output_masks"],
            test_inputs=batch["test_inputs"],
            test_input_masks=batch["test_input_masks"],
        )

        logits = out["logits"]  # (1, K_test, C_out, H, W)
        preds = logits.argmax(dim=2)  # (1, K_test, H, W)
        preds = preds.squeeze(0)      # (K_test, H, W)

        targets = batch["test_outputs"]  # (K_test, H, W)
        H, W = targets.shape[1], targets.shape[2]

        # For pixel accuracy
        for j in range(K_test):
            pred = preds[j]
            target = targets[j]

            # -------------------------------
            # Crop prediction back to original size
            # -------------------------------
            # Provided by your dataset
            orig_h, orig_w = batch["test_original_size"].tolist()
            pred_cropped = pred[:orig_h, :orig_w]
            target_cropped = target[:orig_h, :orig_w]

            # ---- Exact Match ----
            exact = (pred_cropped == target_cropped).all().item()
            total_exact += exact
            total_tests += 1

            # ---- Pixel Accuracy ----
            total_pixels += orig_h * orig_w
            total_pixels_correct += (pred_cropped == target_cropped).sum().item()

    exact_acc = total_exact / total_tests if total_tests > 0 else 0
    pixel_acc = total_pixels_correct / total_pixels if total_pixels > 0 else 0

    print("\n=== Validation Results ===")
    print(f"Exact Match Accuracy: {exact_acc * 100:.2f}%")
    print(f"Pixel Accuracy:       {pixel_acc * 100:.2f}%\n")

    return exact_acc, pixel_acc


# -------------------------------------------------
# Example usage (in a validation script)
# -------------------------------------------------
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your trained generator
    generator = ARCGenerator(...)  # build same as train script
    generator.load_state_dict(torch.load("phase1_generator.pt", map_location=DEVICE))
    generator.to(DEVICE)

    # Use same loader or a separate validation loader
    validate_phase1(generator, ARCDataModule, DEVICE)
# import torch
# from torch import nn, autograd


# class ARCWGAN_GP_Trainer:
#     """
#     Trains generator-critic relation
#     """

#     def __init__(
#         self,
#         example_pair_encoder: nn.Module,
#         aggregator: nn.Module,
#         cond_encoder: nn.Module,
#         lvittm: nn.Module,
#         executor: nn.Module,
#         critic: nn.Module,
#         gen_optim: torch.optim.Optimizer,
#         crit_optim: torch.optim.Optimizer,
#         grad_pen_weight: float = 10.0,
#         num_critic_updates: int = 3,
#         device: torch.device | None = None,
#         sup_loss_weight: float = 0.0,  # set >0 to add supervised loss
#     ):
#         self.example_pair_encoder = example_pair_encoder
#         self.aggregator = aggregator
#         self.cond_encoder = cond_encoder
#         self.lvittm = lvittm
#         self.executor = executor
#         self.critic = critic

#         self.gen_optim = gen_optim
#         self.crit_optim = crit_optim
#         self.grad_pen_weight = grad_pen_weight
#         self.num_critic_updates = num_critic_updates
#         self.sup_loss_weight = sup_loss_weight

#         self.device = device or (
#             torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#         )

#         # move modules to device
#         for m in [
#             self.example_pair_encoder,
#             self.aggregator,
#             self.cond_encoder,
#             self.lvittm,
#             self.executor,
#             self.critic,
#         ]:
#             m.to(self.device)

#         self.ce_loss = nn.CrossEntropyLoss(reduction="mean")

#     #################################
#     #   Context (C) from examples   #
#     #################################

#     def _compute_context(
#             self, 
#             train_inputs,  # (B, N_train, H, W)
#             train_outputs,  # (B, N_train, H, W)
#             train_orig_size  # (B, 2)
#     ):  # (B, D)
#         B, N, H, W = train_inputs.shape

#         # Flatten across examples
#         I_flat = train_inputs.view(B * N, 1, H, W).to(self.device)
#         O_flat = train_outputs.view(B * N, 1, H, W).to(self.device)

#         train_in_masks = batch["train_in_masks"].to(self.device)     # (B, N, H, W)
#         train_out_masks = batch["train_out_masks"].to(self.device)   # (B, N, H, W)

#         # Flatten masks
#         mask_I_flat = train_in_masks.view(B * N, H, W)
#         mask_O_flat = train_out_masks.view(B * N, H, W)


#         # Encode each pair to h_i
#         h_flat = self.example_pair_encoder(
#             I_flat, O_flat,
#             mask_I=masks,
#             mask_O=masks
#         )  # (B*N, D)

#         # Reshape to (B, N, D)
#         h = h_flat.view(B, N, -1)

#         # Aggregate to context C
#         C = self.aggregator(h)  # (B, D)
#         return C

#     # -------------------------
#     # Generator forward: I_test -> O_fake
#     # -------------------------
#     def _generator_forward(self, I_test, test_orig_size, C):
#         """
#         I_test:        (B, 1, H, W)
#         test_orig_size: (B, 2)
#         C:             (B, D)
#         returns: O_fake (B, num_classes, H, W), Z (B, T, z_dim)
#         """
#         B, _, H, W = I_test.shape
#         I_test = I_test.to(self.device)

#         # Build mask for test input
#         orig_sizes = test_orig_size.to("cpu")
#         masks = [generate_valid_mask(H, W, orig_sizes[b]) for b in range(B)]
#         mask_test = torch.stack(masks, dim=0).to(self.device)  # (B,H,W)

#         # Conditional encoding of test input
#         tokens, kpm = self.cond_encoder(I_test, mask_test, C)

#         # LViTM proposals
#         Z = self.lvittm(C, tokens, kpm)  # (B, T, z_dim)

#         # For now, just take first proposal z_0 per batch element
#         z0 = Z[:, 0, :]  # (B, z_dim)

#         # Executor applies z0
#         O_fake = self.executor(I_test, z0)  # (B, num_classes, H, W)
#         return O_fake, Z, mask_test

#     # -------------------------
#     # Gradient penalty
#     # -------------------------
#     def _gradient_penalty(self, I_test, real_out, fake_out, mask_test, C):
#         """
#         I_test:   (B, 1, H, W)
#         real_out: (B, C_out, H, W)
#         fake_out: (B, C_out, H, W)
#         mask_test: (B, H, W)
#         C:       (B, D)
#         """
#         B = real_out.size(0)
#         device = self.device

#         alpha = torch.rand(B, 1, 1, 1, device=device)
#         interpolated = real_out + alpha * (fake_out - real_out)
#         interpolated.requires_grad_(True)

#         # Critic score on interpolated outputs
#         scores = self.critic(
#             I_in=I_test,
#             O_pred=interpolated,
#             mask_in=mask_test,
#             mask_out=mask_test,
#             z=None,
#             C=C
#         )  # (B,)

#         # Compute gradient of scores wrt interpolated
#         grads = autograd.grad(
#             outputs=scores.sum(),
#             inputs=interpolated,
#             create_graph=True,
#             retain_graph=True,
#             only_inputs=True
#         )[0]  # (B, C_out, H, W)

#         grads = grads.view(B, -1)
#         grad_norm = grads.norm(2, dim=1)  # (B,)
#         gp = ((grad_norm - 1.0) ** 2).mean()
#         return gp

#     # -------------------------
#     # One training step on a batch
#     # -------------------------
#     def train_step(self, batch):
#         """
#         batch: dict from DataLoader, e.g.:
#           "train_inputs":        (B, N_train, H, W)
#           "train_outputs":       (B, N_train, H, W)
#           "test_inputs":         (B, N_test, H, W)
#           "test_outputs":        (B, N_test, H, W)
#           "train_original_size": (B, 2)
#           "test_original_size":  (B, 2)
#         """

#         # Put batch on device & prepare shapes
#         train_inputs = batch["train_inputs"].to(self.device)      # (B,N_t,H,W)
#         train_outputs = batch["train_outputs"].to(self.device)
#         test_inputs = batch["test_inputs"].to(self.device)        # (B,N_test,H,W)
#         test_outputs = batch["test_outputs"].to(self.device)
#         train_orig_size = batch["train_original_size"].to(self.device)  # (B,2)
#         test_orig_size = batch["test_original_size"].to(self.device)

#         B, N_test, H, W = test_inputs.shape
#         assert N_test == 1, "Current trainer assumes exactly 1 test pair per task."
#         I_test = test_inputs[:, 0].unsqueeze(1)   # (B,1,H,W)
#         O_real = test_outputs[:, 0]               # (B,H,W)
#         O_real = O_real.unsqueeze(1)              # (B,1,H,W) or adapt to num_classes

#         # -----------------------
#         # Compute context C once
#         # -----------------------
#         C = self._compute_context(train_inputs, train_outputs, train_orig_size)  # (B,D)

#         # -----------------------
#         # Critic updates
#         # -----------------------
#         self.critic.train()
#         self.example_pair_encoder.eval()
#         self.aggregator.eval()
#         self.cond_encoder.eval()
#         self.lvittm.eval()
#         self.executor.eval()

#         crit_loss_last = crit_wass_last = crit_gp_last = None

#         for _ in range(self.num_critic_updates):
#             # Fake output (no grad into generator here)
#             with torch.no_grad():
#                 O_fake, Z, mask_test = self._generator_forward(I_test, test_orig_size, C)
#             # Critic scores
#             real_scores = self.critic(
#                 I_in=I_test,
#                 O_pred=O_real,
#                 mask_in=mask_test,
#                 mask_out=mask_test,
#                 z=None,
#                 C=C
#             )  # (B,)
#             fake_scores = self.critic(
#                 I_in=I_test,
#                 O_pred=O_fake.detach(),
#                 mask_in=mask_test,
#                 mask_out=mask_test,
#                 z=None,
#                 C=C
#             )  # (B,)

#             crit_wass = fake_scores.mean() - real_scores.mean()
#             crit_gp = self._gradient_penalty(I_test, O_real, O_fake.detach(), mask_test, C)
#             crit_loss = crit_wass + self.grad_pen_weight * crit_gp

#             self.crit_optim.zero_grad()
#             crit_loss.backward()
#             self.crit_optim.step()

#             crit_loss_last = crit_loss.item()
#             crit_wass_last = crit_wass.item()
#             crit_gp_last = crit_gp.item()

#         # -----------------------
#         # Generator update
#         # -----------------------
#         self.critic.eval()
#         self.example_pair_encoder.train()
#         self.aggregator.train()
#         self.cond_encoder.train()
#         self.lvittm.train()
#         self.executor.train()

#         # Fresh fake output (with grad)
#         O_fake, Z, mask_test = self._generator_forward(I_test, test_orig_size, C)

#         # Adversarial loss
#         fake_scores_for_gen = self.critic(
#             I_in=I_test,
#             O_pred=O_fake,
#             mask_in=mask_test,
#             mask_out=mask_test,
#             z=None,
#             C=C
#         )  # (B,)
#         gen_adv_loss = -fake_scores_for_gen.mean()

#         # Optional supervised loss (e.g., cross-entropy over colors)
#         # assumes O_fake is (B, num_classes, H, W) and O_real is color indices (B,H,W)
#         gen_sup_loss = torch.tensor(0.0, device=self.device)
#         if self.sup_loss_weight > 0.0:
#             # Here you might need your true color labels, not binary grids
#             # Placeholder: treat O_real as indices already
#             target = O_real.squeeze(1).long()  # (B,H,W)
#             gen_sup_loss = self.ce_loss(O_fake, target)

#         gen_loss = gen_adv_loss + self.sup_loss_weight * gen_sup_loss

#         self.gen_optim.zero_grad()
#         gen_loss.backward()
#         self.gen_optim.step()

#         return {
#             "crit_loss": crit_loss_last,
#             "crit_wass": crit_wass_last,
#             "crit_grad_pen": crit_gp_last,
#             "gen_loss": gen_loss.item(),
#             "gen_adv_loss": gen_adv_loss.item(),
#             "gen_sup_loss": gen_sup_loss.item() if self.sup_loss_weight > 0 else 0.0,
#         }















