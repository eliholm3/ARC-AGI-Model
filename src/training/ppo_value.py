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
