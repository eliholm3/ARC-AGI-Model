# src.architecture.adViT.critic

import torch
import torch.nn as nn


class AdversarialVisionTransformer(nn.Module):

    def __init__(
            self,
            vit_encoder: nn.Module,
            z_dim: int | None = None,  # proposal dimension
            c_dim: int | None = None,  # context dimension
            hidden_dim: int = 256
    ):
        super().__init__()
        self.vit = vit_encoder
        self.z_dim = z_dim
        self.c_dim = c_dim

        embed_dim = self.vit.c_token.size(-1)

        # Total feature dimension (one long vector)
        in_dim = embed_dim
        if z_dim is not None:
            in_dim += z_dim
        if c_dim is not None:
            in_dim += c_dim

        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(
            self,
            I_in: torch.Tensor,        # (B, C_in, H, W)
            O_pred: torch.Tensor,      # (B, C_out, H, W) or (B, T, C_out, H, W)
            mask_in: torch.Tensor | None = None,
            mask_out: torch.Tensor | None = None,
            z: torch.Tensor | None = None,
            C: torch.Tensor | None = None
    ) -> torch.Tensor:
        ###############################
        #   B = batch size            #    
        #   P = num proposals         #
        #   C_in = input channels     #
        #   C_out = output channels   #
        #   H = height                #
        #   W = width                 #
        ###############################

        B, C_in, H, W = I_in.shape

        #################################
        #   MULTI-PROPOSAL BRANCH       #
        #################################

        if O_pred.dim() == 5:
            B, T, C_out, H, W = O_pred.shape

            # Expand inputs
            I_exp = I_in.unsqueeze(1).expand(B, T, C_in, H, W)
            I_flat = I_exp.reshape(B*T, C_in, H, W)
            O_flat = O_pred.reshape(B*T, C_out, H, W)

            # Convert O_flat to 1 channel if needed
            if O_flat.size(1) > 1:
                O_flat = torch.argmax(O_flat, dim=1, keepdim=True).float()

            # Combine masks
            if mask_in is not None or mask_out is not None:
                if mask_in is None:
                    mask_in = torch.zeros(B, H, W, dtype=torch.bool, device=O_pred.device)
                if mask_out is None:
                    mask_out = torch.zeros(B, H, W, dtype=torch.bool, device=O_pred.device)
                mask = torch.logical_or(mask_in, mask_out)
                mask = mask.unsqueeze(1).expand(B, T, H, W).reshape(B*T, H, W)
            else:
                mask = None

            # Concatenate input + output
            x = torch.cat([I_flat, O_flat], dim=1)  # (B*T, 2, H, W)

            # Encode with ViT
            h_flat = self.vit.forward_grid(x, mask=mask)
            h = h_flat.view(B, T, -1)

            # Collect features
            feats = [h]

            if z is not None:
                if z.dim() == 2:
                    z = z.unsqueeze(1).expand(B, T, -1)
                feats.append(z)

            if C is not None:
                feats.append(C.unsqueeze(1).expand(B, T, -1))

            feat = torch.cat(feats, dim=-1)
            return self.mlp(feat).squeeze(-1)

        #################################
        #   SINGLE-PROPOSAL BRANCH      #
        #################################

        B, C_out, H, W = O_pred.shape

        # Convert O_pred to 1 channel if needed
        if O_pred.size(1) > 1:          # multi-channel class logits
            classes = torch.arange(O_pred.size(1), device=O_pred.device).view(1, -1, 1, 1)
            probs = O_pred.softmax(dim=1)
            O_pred = (probs * classes).sum(dim=1, keepdim=True)   # differentiable


        # Combine masks
        if mask_in is not None or mask_out is not None:
            if mask_in is None:
                mask_in = torch.zeros(B, H, W, dtype=torch.bool, device=O_pred.device)
            if mask_out is None:
                mask_out = torch.zeros(B, H, W, dtype=torch.bool, device=O_pred.device)
            mask = torch.logical_or(mask_in, mask_out)
        else:
            mask = None

        # Concatenate input + output
        x = torch.cat([I_in, O_pred], dim=1)  # (B, 2, H, W)

        # Encode with ViT
        h = self.vit.forward_grid(x, mask=mask)

        # Combine features for MLP
        feats = [h]
        if z is not None:
            feats.append(z)
        if C is not None:
            feats.append(C)
        feat = torch.cat(feats, dim=-1)

        return self.mlp(feat).squeeze(-1)

# src.architecture.context_encoding.example_pair_encoder

import torch
import torch.nn as nn


class ExamplePairEncoder(nn.Module):
    """
    Encods single example pair (I_i, O_i) into vector h_i
    """

    def __init__(
            self, 
            vit_pair: nn.Module
    ):
        super().__init__()
        self.vit = vit_pair # generic
        self.norm = nn.LayerNorm(self.vit.c_token.size(-1)) # normalize h_i

    def forward(
            self,
            I_i: torch.Tensor,
            O_i: torch.Tensor,
            mask_I: torch.Tensor,
            mask_O: torch.Tensor
    ) -> torch.Tensor:
        B, _, H, W = I_i.shape
        
        # Concatenate input-output as different channels
        x = torch.cat([I_i, O_i], dim=1)

        # Combine masks
        mask = torch.logical_or(mask_I, mask_O)
        key_padded_mask = ~mask

        # Pass through ViT for context embedding
        h_i = self.vit.forward_grid(x, mask=key_padded_mask)  # (B, embed_dim)

        # Normalize
        h_i = self.norm(h_i)

        return h_i

# src.architecture.context_encoding.example_pair_aggregator

import torch
import torch.nn as nn
import torch.nn.functional as F

class ExamplePairAggregator(nn.Module):
    """
    Aggregates the context vectors from k example pairs (I_i, O_i) into a single embedding
    """

    def __init__(
            self, 
            embed_dim: int,
            hidden_dim: int = 256
    ):
        super().__init__()

        # MLP to weigh context vectors
        self.score_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(), # [-1,1]
            nn.Linear(hidden_dim, 1)
        )

        # Normalize context vector
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
            self,
            h: torch.Tensor,  # all embeddings 
            mask: torch.Tensor | None = None  # where the tokens are valid
    ) -> torch.Tensor:
        ###############################
        #   B = batch size            #
        #   K = num example pairs     #
        #   D = embedding dimension   #
        ###############################

        B, K, D = h.shape

        # Infer scores
        scores = self.score_mlp(h)

        # Mask = where the tokens are
        if mask is not None:

            # Ensure boolean
            mask = mask.to(dtype=torch.bool)

            # Match to score shape
            mask_expanded = mask.unsqueeze(-1) # (B, K, 1)

            # Set where the mask is not to very negative
            scores = scores.masked_fill(~mask_expanded, float("-inf"))

        # Attention weights over k example pairs
        attn = F.softmax(scores, dim=1)

        # Weighted sum
        C = torch.sum(attn * h, dim=1)

        # Final normalization
        C = self.norm(C)

        return C

# src.architecture.context_encoding.conditional_encoder.py

import torch
import torch.nn as nn

class ConditionalTestInputEncoder(nn.Module):

    def __init__(
            self, 
            vit_test: nn.Module
    ):
        ###############################
        #   B = batch size            #    
        #   D = token embedding dim   #
        #   S = num tokens            #
        #   H = height                #
        #   W = width                 #
        ###############################

        super().__init__()
        self.vit = vit_test
        self.embed_dim = self.vit.c_token.size(-1)

        # Project context vector to embedding dim
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
            self, 
            I_test,  # (B, 1, H, W)
            mask_test,  # (B, H, W) or None
            C  # (B, D)
    ):

        B, _, H, W = I_test.shape

        ######################
        #   Encode with ViT  #
        ######################

        tokens = self.vit.patch_embedding(I_test)  # (B, S, D)

        #######################
        #   Build Test Mask   #
        #######################

        if mask_test is not None:
            flat_mask = mask_test.reshape(B, -1)  # (B, S)
            key_padding_mask = ~flat_mask         # True = pad
        else:
            key_padding_mask = None
        
        key_padding_mask = key_padding_mask.to(torch.bool)

        ##########################
        #   Add Context Vector   #
        ##########################

        C_token = self.c_proj(C).unsqueeze(1)  # (B,1,D)
        tokens = torch.cat([C_token, tokens], dim=1)  # (B,1+S,D)

        if key_padding_mask is not None:
            # Add context to mask
            c_pad = torch.zeros(B, 1, dtype=torch.bool, device=key_padding_mask.device)
            key_padding_mask = torch.cat([c_pad, key_padding_mask], dim=1)  # (B,1+S)

        #####################################
        #   Positional Encoding + Dropout   #
        #####################################

        tokens = self.vit.pos_encoding(tokens)
        tokens = self.vit.dropout(tokens)

        return tokens, key_padding_mask
    
# src.architecture.executor.attention.py

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
    
# src.architecture.executor.CNNBlock

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
    
# src.architecture.executor.executor

import torch
import torch.nn as nn
from src.architecture.executor.CNNBlock import CNNBlock
from src.architecture.ViT.body import TransformerEncoderBlock


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

        return logits
    
# src.architecture.executor.FiLM

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
    
# src.architecture.LViTM.attention
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
    
# src.architecture.LViTM.attention

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
    
# src.architecture.LViTM.body
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
    
# src.architecture.ViT.attention
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
    
# src.architecture.ViT.body
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
    
# src.architecture.ViT.preprocessing
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
    
# src.data_pipeline.dataset

import torch
from torch.utils.data import Dataset


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
    
# src.data_pipeline.dataloader

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
    
# src.data_pipeline.utils

import json
from pathlib import Path

import torch


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
    Pads each sample so that *both* train and test grids
    share the same square size per sample.

    metric_dict is ignored (kept for backward compatibility).
    """
    for sample in data.values():
        # -----------------------------------
        # 1. Compute per-sample global maxima
        #    across BOTH train and test
        # -----------------------------------
        max_input_height = 0
        max_input_width  = 0
        max_output_height = 0
        max_output_width  = 0

        # Look at TRAIN split
        for pairs in sample.get('train', []):
            # input
            if len(pairs['input']) > max_input_height:
                max_input_height = len(pairs['input'])
            for row in pairs['input']:
                if len(row) > max_input_width:
                    max_input_width = len(row)

            # output
            if len(pairs['output']) > max_output_height:
                max_output_height = len(pairs['output'])
            for row in pairs['output']:
                if len(row) > max_output_width:
                    max_output_width = len(row)

        # Look at TEST split
        for pairs in sample.get('test', []):
            # input
            if len(pairs['input']) > max_input_height:
                max_input_height = len(pairs['input'])
            for row in pairs['input']:
                if len(row) > max_input_width:
                    max_input_width = len(row)

            # output
            if len(pairs['output']) > max_output_height:
                max_output_height = len(pairs['output'])
            for row in pairs['output']:
                if len(row) > max_output_width:
                    max_output_width = len(row)

        # Single square size for this sample
        max_size = max(
            max_input_height,
            max_input_width,
            max_output_height,
            max_output_width,
        )

        # -----------------------------------
        # 2. Pad TRAIN grids to max_size
        # -----------------------------------
        for pairs in sample.get('train', []):
            # input
            while len(pairs['input']) < max_size:
                pairs['input'].append([pad_value] * max_size)
            for row in pairs['input']:
                while len(row) < max_size:
                    row.append(pad_value)

            # output
            while len(pairs['output']) < max_size:
                pairs['output'].append([pad_value] * max_size)
            for row in pairs['output']:
                while len(row) < max_size:
                    row.append(pad_value)

        # -----------------------------------
        # 3. Pad TEST grids to max_size
        # -----------------------------------
        for pairs in sample.get('test', []):
            # input
            while len(pairs['input']) < max_size:
                pairs['input'].append([pad_value] * max_size)
            for row in pairs['input']:
                while len(row) < max_size:
                    row.append(pad_value)

            # output
            while len(pairs['output']) < max_size:
                pairs['output'].append([pad_value] * max_size)
            for row in pairs['output']:
                while len(row) < max_size:
                    row.append(pad_value)

    return data



# def pad_data(data, metric_dict=None, pad_value=0):
    # """
    # Pads each sample independently to its own max square size.
    # metric_dict is ignored (kept for backward compatibility).
    # """
    # for sample in data.values():
    #     # ----- compute per-sample maxima for TRAIN -----
    #     max_train_input_height = 0
    #     max_train_input_width  = 0
    #     max_train_output_height = 0
    #     max_train_output_width  = 0

    #     for pairs in sample.get('train', []):
    #         if len(pairs['input'])  > max_train_input_height:  max_train_input_height  = len(pairs['input'])
    #         if len(pairs['output']) > max_train_output_height: max_train_output_height = len(pairs['output'])
    #         for inp in pairs['input']:
    #             if len(inp) > max_train_input_width:  max_train_input_width  = len(inp)
    #         for outp in pairs['output']:
    #             if len(outp) > max_train_output_width: max_train_output_width = len(outp)

    #     # ----- compute per-sample maxima for TEST -----
    #     max_test_input_height = 0
    #     max_test_input_width  = 0
    #     max_test_output_height = 0
    #     max_test_output_width  = 0

    #     for pairs in sample.get('test', []):
    #         if len(pairs['input'])  > max_test_input_height:  max_test_input_height  = len(pairs['input'])
    #         if len(pairs['output']) > max_test_output_height: max_test_output_height = len(pairs['output'])
    #         for inp in pairs['input']:
    #             if len(inp) > max_test_input_width:  max_test_input_width  = len(inp)
    #         for outp in pairs['output']:
    #             if len(outp) > max_test_output_width: max_test_output_width = len(outp)

    #     # ----- per-sample square sizes -----
    #     max_train_size = max(
    #         max_train_input_height,
    #         max_train_input_width,
    #         max_train_output_height,
    #         max_train_output_width
    #     )
    #     max_test_size = max(
    #         max_test_input_height,
    #         max_test_input_width,
    #         max_test_output_height,
    #         max_test_output_width
    #     )

    #     # ----- pad TRAIN for this sample -----
    #     for pairs in sample.get('train', []):
    #         # input
    #         while len(pairs['input']) < max_train_size:
    #             pairs['input'].append([pad_value] * max_train_size)
    #         for inp in pairs['input']:
    #             while len(inp) < max_train_size:
    #                 inp.append(pad_value)
    #         # output
    #         while len(pairs['output']) < max_train_size:
    #             pairs['output'].append([pad_value] * max_train_size)
    #         for outp in pairs['output']:
    #             while len(outp) < max_train_size:
    #                 outp.append(pad_value)

    #     # ----- pad TEST for this sample -----
    #     for pairs in sample.get('test', []):
    #         # input
    #         while len(pairs['input']) < max_test_size:
    #             pairs['input'].append([pad_value] * max_test_size)
    #         for inp in pairs['input']:
    #             while len(inp) < max_test_size:
    #                 inp.append(pad_value)
    #         # output
    #         while len(pairs['output']) < max_test_size:
    #             pairs['output'].append([pad_value] * max_test_size)
    #         for outp in pairs['output']:
    #             while len(outp) < max_test_size:
    #                 outp.append(pad_value)

    # return data


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

def arc_collate_fn_bs1(batch):
    # batch size is guaranteed to be 1; return the single dict unchanged
    return batch[0]

# src.data_pipeline.dataset_io

import os
import torch

def save_sample_level(sample_list, file_path):
    file_path = os.path.expanduser(file_path)
    folder = os.path.dirname(file_path)
    if folder and (not os.path.isdir(folder)):
        os.makedirs(folder, exist_ok=True)
    torch.save(sample_list, file_path)

def load_sample_level(file_path, map_location="cpu"):
    file_path = os.path.expanduser(file_path)
    obj = torch.load(file_path, map_location=map_location)
    return obj

# src.inference.generator

import torch
import torch.nn as nn
from typing import Dict, Any


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

        # If no batch dimension, assume B=1 and add it.
        if train_inputs.dim() == 3:
            train_inputs = train_inputs.unsqueeze(0)
            train_outputs = train_outputs.unsqueeze(0)
            train_input_masks = train_input_masks.unsqueeze(0)
            train_output_masks = train_output_masks.unsqueeze(0)

        if test_inputs.dim() == 3:
            test_inputs = test_inputs.unsqueeze(0)
            test_input_masks = test_input_masks.unsqueeze(0)

        B, K_train, H, W = train_inputs.shape
        _, K_test, H_t, W_t = test_inputs.shape
        assert H == H_t and W == W_t, "Train and test grids must share padded size per sample."

        # Ensure masks are boolean
        train_input_masks = train_input_masks.bool()
        train_output_masks = train_output_masks.bool()
        test_input_masks = test_input_masks.bool()

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

        # Z_all: (B, K_test, P, z_dim)
        Z_all = torch.stack(all_Z, dim=1)

        # z_chosen: (B, K_test, z_dim)
        z_chosen = torch.stack(all_z_chosen, dim=1)

        return {
            "logits": logits,         # (B, K_test, C_out, H, W)
            "Z_all": Z_all,           # (B, K_test, P, z_dim)
            "z_chosen": z_chosen,     # (B, K_test, z_dim)
            "C": C                    # (B, D)
        }

# src.inference.execution_controller

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

# src.training.checkpoints

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

# src.training.evaluate_final

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
NUM_CLASSES = 10


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

# src.training.ppo_actor

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

# src.training.ppo_refiner
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

# src.training.ppo_value
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
        B, T, D = Z.shape
        values = self.net(Z.view(B*T, D)).view(B, T)
        return values

# src.training.validation_phase1
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

# src.training.phase1
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

# Import your modules
from src.inference.generator import ARCGenerator

# Encoders & components
from src.architecture.context_encoding.example_pair_encoder import ExamplePairEncoder
from src.architecture.context_encoding.example_pair_aggregator import ExamplePairAggregator
from src.architecture.context_encoding.conditional_encoder import ConditionalTestInputEncoder
from src.architecture.LViTM.body import LargeVisionTransformerModel
from src.architecture.executor.executor import Executor
from src.architecture.ViT.body import VisionTransformer

# Training constants
LR = 1e-4
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 11


##############################################################
#   Build MODEL for PHASE 1 using TWO separate ViTs          #
##############################################################
def build_model():

    ##############################################################
    #   1. Vision Transformers                                   #
    #      vit_pair = (I, O) example pairs, 2 channels           #
    #      vit_test = I_test only, 1 channel                     #
    ##############################################################
    img_size   = 30
    patch_size = 1
    embed_dim  = 128
    num_heads  = 4
    depth_vit  = 6
    mlp_dim    = 256
    z_dim      = 64
    num_props  = 4

    # Example pair ViT: (I_i, O_i) → 2 channels
    vit_pair = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth_vit,
        mlp_dim=mlp_dim,
        in_channels=2
    ).to(DEVICE)

    # Test grid ViT: (I_test) → 1 channel
    vit_test = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth_vit,
        mlp_dim=mlp_dim,
        in_channels=1
    ).to(DEVICE)

    ##############################################################
    #   2. Context encoding components                           #
    ##############################################################
    example_encoder = ExamplePairEncoder(vit_pair).to(DEVICE)

    aggregator = ExamplePairAggregator(
        embed_dim=vit_pair.c_token.size(-1)
    ).to(DEVICE)

    cond_encoder = ConditionalTestInputEncoder(vit_test).to(DEVICE)

    ##############################################################
    #   3. LVITM (reasoning / proposal generation)               #
    ##############################################################
    lvitm = LargeVisionTransformerModel(
        embed_dim=vit_pair.c_token.size(-1),
        num_heads=4,
        mlp_dim=256,
        depth=8,
        num_proposals=num_props,
        z_dim=z_dim
    ).to(DEVICE)

    ##############################################################
    #   4. Executor                                              #
    ##############################################################
    executor = Executor(
        embed_dim=vit_pair.c_token.size(-1),
        num_heads=4,
        mlp_dim=256,
        depth=4,
        z_dim=z_dim,
        hidden_channels=64,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    ##############################################################
    #   5. Wrap everything into ARCGenerator                     #
    ##############################################################
    generator = ARCGenerator(
        example_pair_encoder=example_encoder,
        aggregator=aggregator,
        cond_encoder=cond_encoder,
        lvitm=lvitm,
        executor=executor
    ).to(DEVICE)

    return generator


##############################################################
#   Phase 1 Supervised Training                              #
##############################################################
def train_phase1(arc_loader):

    generator = build_model()
    optimizer = Adam(generator.parameters(), lr=LR)

    generator.train()

    for epoch in tqdm(range(EPOCHS), "Epoch:"):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

        total_loss = 0.0
        count = 0

        for batch in tqdm(arc_loader, "Batch:"):
            ############################################################
            #   Move batch to device                                  #
            ############################################################
            for k, v in tqdm(batch.items(), "Sample:"):
                if torch.is_tensor(v):
                    batch[k] = v.to(DEVICE)

            train_inputs       = batch["train_inputs"]
            train_outputs      = batch["train_outputs"]
            train_input_masks  = batch["train_input_masks"]
            train_output_masks = batch["train_output_masks"]

            test_inputs        = batch["test_inputs"]
            test_outputs       = batch["test_outputs"]
            test_input_masks   = batch["test_input_masks"]

            ############################################################
            #   Forward through ARCGenerator                           #
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
            B, K_test, C_out, H, W = logits.shape

            ############################################################
            #   CE loss: reshape logits & targets                     #
            ############################################################
            logits_flat = logits.view(B * K_test, C_out, H, W)
            target_flat = test_outputs.view(B * K_test, H, W)

            loss = F.cross_entropy(logits_flat, target_flat, ignore_index=0)

            ###########################
            #   Backprop & optimize   #
            ###########################

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

    return generator


##############################################################
#   Script Entry Point                                        #
# ##############################################################
# if __name__ == "__main__":
#     from src.data_pipeline.dataloader import ARCDataModule

#     model = train_phase1(ARCDataModule)
#     torch.save(model.state_dict(), "phase1_generator.pt")
#     print("Saved Phase 1 generator.")

# src.training.phase2
import torch
from torch import autograd

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

# src.training.phase3
import torch
import torch.nn.functional as F
from torch import autograd

from src.inference.generator import ARCGenerator

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

def gradient_penalty(critic, I_real, O_real, I_fake, O_fake, device):
    """
    Computes WGAN-GP gradient penalty for the critic.
    All inputs are (B,1,H,W).
    """

    # Interpolate input grid
    epsilon = torch.rand(I_real.size(0), 1, 1, 1, device=device)
    I_interp = epsilon * I_real + (1 - epsilon) * I_fake
    I_interp.requires_grad_(True)

    # Interpolate output grid
    epsilon2 = torch.rand(O_real.size(0), 1, 1, 1, device=device)
    O_interp = epsilon2 * O_real + (1 - epsilon2) * O_fake
    O_interp.requires_grad_(True)

    # Forward pass through critic
    scores = critic(I_interp, O_interp)  # (B,)

    # Compute grad w.r.t BOTH I and O
    grads = torch.autograd.grad(
        outputs=scores,
        inputs=[I_interp, O_interp],
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )

    # grads is a tuple: (grad_I, grad_O)
    grad_I, grad_O = grads

    # Combine the norms
    grad_norm = (
        grad_I.reshape(grad_I.size(0), -1).norm(2, dim=1) +
        grad_O.reshape(grad_O.size(0), -1).norm(2, dim=1)
    )

    gp = ((grad_norm - 1) ** 2).mean()
    return gp



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

                # Use detached logits for critic training
                logits_det = logits.detach().view(K_test, C_out, H, W)  # (B*K_test, C_out,H,W)

                # Real outputs: one-hot
                targets = test_outputs.view(K_test, H, W)               # (K_test,H,W)
                O_real = one_hot_from_int(targets, NUM_CLASSES)        # (K_test, C_out, H, W)

                # Inputs
                I_real = test_inputs.view(K_test, H, W).unsqueeze(1).float()  # (K_test,1,H,W)

                # Masks
                mask_in = test_input_masks.view(K_test, H, W).bool()
                mask_out = test_output_masks.view(K_test, H, W).bool()

                # Critic scores
                score_real = critic(
                    I_in=I_real,
                    O_pred=O_real,
                    mask_in=mask_in,
                    mask_out=mask_out,
                    z=None,
                    C=None
                )  # (K_test,)

                score_fake = critic(
                    I_in=I_real,
                    O_pred=logits_det,
                    mask_in=mask_in,
                    mask_out=mask_out,
                    z=None,
                    C=None
                )  # (K_test,)

                wasserstein = score_fake.mean() - score_real.mean()
                gp = gradient_penalty(
                    critic,
                    I_real,
                    O_real,
                    I_real,
                    logits_det,
                    DEVICE
                )

                critic_loss = wasserstein + LAMBDA_GP * gp

                critic_loss.backward()
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
            logits_flat = logits.view(B * K_test, C_out, H, W)         # (K_test,C,H,W)
            targets_flat = test_outputs.view(B * K_test, H, W).long()  # (K_test,H,W)

            ce_loss = F.cross_entropy(logits_flat, targets_flat)

            # Adversarial loss: - E[critic(I, O_fake)]
            logits_for_adv = logits.view(K_test, C_out, H, W)          # (K_test,C,H,W)

            I_for_adv = test_inputs.view(K_test, H, W).unsqueeze(1).float()
            mask_in_adv = test_input_masks.view(K_test, H, W).bool()
            mask_out_adv = test_output_masks.view(K_test, H, W).bool()

            fake_scores = critic(
                I_in=I_for_adv,
                O_pred=logits_for_adv,
                mask_in=mask_in_adv,
                mask_out=mask_out_adv,
                z=None,
                C=None
            )  # (K_test,)

            gen_adv_loss = -fake_scores.mean()

            gen_loss = ce_loss + LAMBDA_ADV * gen_adv_loss
            gen_loss.backward()
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

# src.training.phase4
import torch
import torch.nn.functional as F

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


###############################
#   Hyperparameters           #
###############################

PPO_EPOCHS = 3
PPO_STEPS = 3
PPO_GAMMA = 0.99

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10


###############################
#   Build Generator Modules   #
###############################

def build_generator_components():
    """
    Rebuilds generator components individually (not the ARCGenerator wrapper).
    Used to compute context C and supply modules to the controller.
    """

    ###########################
    #   Shared Hyperparams    #
    ###########################

    img_size   = 30
    patch_size = 1
    embed_dim  = 128
    num_heads  = 4
    depth_vit  = 6
    mlp_dim    = 256
    z_dim      = 64
    num_props  = 4

    ###############################
    #   Vision Transformers       #
    ###############################

    # For example pairs (I, O) -> 2 channels
    vit_pair = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth_vit,
        mlp_dim=mlp_dim,
        in_channels=2
    ).to(DEVICE)

    # For test input I_test -> 1 channel
    vit_test = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth_vit,
        mlp_dim=mlp_dim,
        in_channels=1
    ).to(DEVICE)

    ###########################################
    #   Context Encoders (Pair + Aggregator)  #
    ###########################################

    # Example pair encoder uses ViT with 2 input channels
    example_encoder = ExamplePairEncoder(vit_pair).to(DEVICE)

    # Aggregator uses the same embedding dimension as ViTs
    aggregator = ExamplePairAggregator(
        embed_dim=vit_pair.c_token.size(-1)
    ).to(DEVICE)

    ###########################################
    #   Conditional Test Input Encoder        #
    ###########################################

    # Conditional encoder uses the 1-channel ViT
    cond_encoder = ConditionalTestInputEncoder(vit_test).to(DEVICE)

    ###############################
    #   Latent Proposal Model     #
    ###############################

    lvitm = LargeVisionTransformerModel(
        embed_dim=vit_pair.c_token.size(-1),
        num_heads=4,
        mlp_dim=256,
        depth=8,
        num_proposals=num_props,
        z_dim=z_dim
    ).to(DEVICE)

    ###############################
    #   Executor                  #
    ###############################

    executor = Executor(
        embed_dim=vit_pair.c_token.size(-1),
        num_heads=4,
        mlp_dim=256,
        depth=4,
        z_dim=z_dim,
        hidden_channels=64,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    return example_encoder, aggregator, cond_encoder, lvitm, executor



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
#   Compute Context C         #
###############################

def compute_context_C(
        example_encoder: ExamplePairEncoder,
        aggregator: ExamplePairAggregator,
        train_inputs: torch.Tensor,        # (K_train,H,W)
        train_outputs: torch.Tensor,       # (K_train,H,W)
        train_input_masks: torch.Tensor,   # (K_train,H,W)
        train_output_masks: torch.Tensor   # (K_train,H,W)
):
    """
    Encodes all training example pairs and aggregates into context C.
    """

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

    h = torch.stack(h_list, dim=1)   # (B,K_train,D)
    pair_mask = None

    C = aggregator(h, mask=pair_mask)  # (B,D)
    return C


###############################
#   Phase 4 PPO Training      #
###############################

def train_phase4_ppo(data_loader: ARCDataModule):
    # Build components
    example_encoder, aggregator, cond_encoder, lvitm, executor = build_generator_components()
    critic = build_critic()

    # Load Phase 3 checkpoints if available
    try:
        gen_state = torch.load("generator_phase3_adv.pt", map_location=DEVICE)
        print("Loaded generator_phase3_adv.pt into generator components (partial load).")
        # You can do partial loads into submodules here if you saved them modularly.
    except FileNotFoundError:
        print("Phase 3 generator checkpoint not found. Using fresh generator components.")

    try:
        critic.load_state_dict(torch.load("critic_phase3_adv.pt", map_location=DEVICE))
        print("Loaded critic_phase3_adv.pt.")
    except FileNotFoundError:
        print("Phase 3 critic checkpoint not found. Using fresh critic.")

    # Controller with critic
    controller = HybridExecuteController(
        lvitm=lvitm,
        executor=executor,
        cond_encoder=cond_encoder,
        critic=critic
    ).to(DEVICE)

    # PPO modules
    z_dim = 64   # must match LViTM z_dim
    actor = PPOActor(z_dim=z_dim, embed_dim=256).to(DEVICE)
    value_fn = PPOValuer(z_dim=z_dim, embed_dim=256).to(DEVICE)
    ppo_refiner = PPORefiner(actor=actor, value_fn=value_fn, lr=1e-4)

    for epoch in range(PPO_EPOCHS):
        print(f"\n=== Phase 4 PPO Epoch {epoch + 1}/{PPO_EPOCHS} ===")

        for batch_idx, batch in enumerate(data_loader):

            ###############################
            #   Move batch to device      #
            ###############################

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

            # Compute context C
            C = compute_context_C(
                example_encoder=example_encoder,
                aggregator=aggregator,
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                train_input_masks=train_input_masks,
                train_output_masks=train_output_masks
            )  # (1,D)

            # Pick first test input as initial grid
            if test_inputs.dim() == 3:
                test_inputs = test_inputs.unsqueeze(0)
                test_input_masks = test_input_masks.unsqueeze(0)
                test_outputs = test_outputs.unsqueeze(0)

            B, K_test, H, W = test_inputs.shape
            init_grid = test_inputs[:, 0].unsqueeze(1).float()       # (B,1,H,W)
            init_mask = test_input_masks[:, 0]                        # (B,H,W)
            target = test_outputs[:, 0]                               # (B,H,W)

            ###############################
            #   PPO Rollout & Update      #
            ###############################

            final_logits, ppo_stats = controller.ppo_rollout_and_update(
                init_grid=init_grid,
                init_mask=init_mask,
                C=C,
                ppo_refiner=ppo_refiner,
                num_steps=PPO_STEPS,
                gamma=PPO_GAMMA
            )

            # Optional: supervised signal on final prediction
            pred_loss = F.cross_entropy(
                final_logits,          # (B,C_out,H,W)
                target.long()          # (B,H,W)
            )

            pred_loss.backward()
            # NOTE: If you want to train executor/LViTM jointly with PPO, you can
            # attach an optimizer here and step it. For now, PPORefiner.update()
            # already steps the actor/value networks.

            print(f"Epoch {epoch+1}, Batch {batch_idx}: "
                  f"PPO loss={ppo_stats['loss']:.4f}, "
                  f"policy={ppo_stats['policy_loss']:.4f}, "
                  f"value={ppo_stats['value_loss']:.4f}, "
                  f"entropy={ppo_stats['entropy']:.4f}")
            
    return actor, value_fn


if __name__ == "__main__":
    train_phase4_ppo()

# src.training.train_orchestrator
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
