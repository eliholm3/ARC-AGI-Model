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