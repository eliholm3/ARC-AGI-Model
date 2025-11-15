import torch.nn as nn
from FiLM import FiLM

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