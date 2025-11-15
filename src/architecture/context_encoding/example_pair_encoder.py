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