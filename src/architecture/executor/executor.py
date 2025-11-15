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