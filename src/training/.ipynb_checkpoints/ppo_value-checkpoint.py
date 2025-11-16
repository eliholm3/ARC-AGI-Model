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
