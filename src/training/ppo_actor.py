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
