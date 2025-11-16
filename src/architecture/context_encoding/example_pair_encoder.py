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
        
        # Concatenate input-output as channels
        x = torch.cat([I_i, O_i], dim=1)  # (B, 2, H, W)

        # Combine masks: True = valid (NOT padded)
        mask = torch.logical_or(mask_I, mask_O)  # (B, H, W)

        print("\n[ExamplePairEncoder] I_i:", I_i.shape)
        print("[ExamplePairEncoder] O_i:", O_i.shape)
        print("[ExamplePairEncoder] mask_I unique:", mask_I.unique())
        print("[ExamplePairEncoder] mask_O unique:", mask_O.unique())

        print("[ExamplePairEncoder] key_padded_mask shape:", mask.shape)

        # Pass the VALIDITY mask directly to ViT
        # ViT will handle flattening + inversion internally
        h_i = self.vit.forward_grid(x, mask=mask)  # (B, D)

        h_i = self.norm(h_i)


        print("[ExamplePairEncoder] h_i mean/std:", h_i.mean().item(), h_i.std().item())

        return h_i