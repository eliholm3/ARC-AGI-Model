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
