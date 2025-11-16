import torch
import torch.nn as nn


class AdversarialVisionTransformer(nn.Module):

    def __init__(
            self,
            vit_encoder: nn.Module,
            z_dim: int | None = None,   # proposal latent dim
            c_dim: int | None = None,   # context dim
            mlp_dim: int = 256
    ):
        super().__init__()
        self.vit = vit_encoder
        self.z_dim = z_dim
        self.c_dim = c_dim

        # ViT output embedding dimension
        embed_dim = self.vit.c_token.size(-1)   # typically 128

        # === PROJECTIONS ===
        # Project z and C into ViT embedding dimension
        self.z_proj = nn.Linear(z_dim, embed_dim) if z_dim is not None else None
        self.c_proj = nn.Linear(c_dim, embed_dim) if c_dim is not None else None

        # Final concatenated feature dimension
        # h (128) + z_proj (128) + c_proj (128) = 128 * N_parts
        in_dim = embed_dim
        if z_dim is not None:
            in_dim += embed_dim
        if c_dim is not None:
            in_dim += embed_dim

        # === CRITIC HEAD ===
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 1)
        )


    ###########################################################################
    # FORWARD
    ###########################################################################

    def forward(
            self,
            I_in: torch.Tensor,         # (B, C_in, H, W)
            O_pred: torch.Tensor,       # (B, C_out, H, W) or (B, T, C_out, H, W)
            mask_in: torch.Tensor | None = None,
            mask_out: torch.Tensor | None = None,
            z: torch.Tensor | None = None,
            C: torch.Tensor | None = None
    ) -> torch.Tensor:

        B, C_in, H, W = I_in.shape

        #######################################################################
        # MULTI-PROPOSAL BRANCH
        #######################################################################
        if O_pred.dim() == 5:
            B, T, C_out, H, W = O_pred.shape

            # Expand I and flatten
            I_exp = I_in.unsqueeze(1).expand(B, T, C_in, H, W)
            I_flat = I_exp.reshape(B*T, C_in, H, W)

            # Flatten O_pred
            O_flat = O_pred.reshape(B*T, C_out, H, W)

            # Convert logits → single channel (argmax) if needed
            if O_flat.size(1) > 1:
                O_flat = torch.argmax(O_flat, dim=1, keepdim=True).float()

            # Combine masks if present
            if mask_in is not None or mask_out is not None:
                mask_in  = mask_in  if mask_in  is not None else torch.zeros(B, H, W, dtype=torch.bool, device=O_pred.device)
                mask_out = mask_out if mask_out is not None else torch.zeros(B, H, W, dtype=torch.bool, device=O_pred.device)
                mask = torch.logical_or(mask_in, mask_out)
                mask = mask.unsqueeze(1).expand(B, T, H, W).reshape(B*T, H, W)
            else:
                mask = None

            # Concatenate input + predicted output grid
            x = torch.cat([I_flat, O_flat], dim=1)  # (B*T, 2, H, W)

            # Encode each proposal pair with ViT
            h_flat = self.vit.forward_grid(x, mask=mask)  # (B*T, embed_dim)
            h = h_flat.reshape(B, T, -1)                  # (B, T, embed_dim)

            # ===== Assemble features =====
            feats = [h]

            if z is not None and self.z_proj is not None:
                if z.dim() == 2:
                    z = z.unsqueeze(1).expand(B, T, -1)
                z_proj = self.z_proj(z)
                feats.append(z_proj)


            if C is not None and self.c_proj is not None:
                C_exp = C.unsqueeze(1).expand(B, T, -1)
                C_proj = self.c_proj(C_exp)               # (B, T, embed_dim)
                feats.append(C_proj)

            # Concatenate feature parts
            feat = torch.cat(feats, dim=-1)               # (B, T, in_dim)
            # Flatten (B,T,*) → (B*T,*)
            feat = feat.reshape(B*T, -1)

            # Run MLP
            scores = self.mlp(feat).squeeze(-1)   # (B*T,)

            # Reshape back to (B, T)
            scores = scores.view(B, T)

            return scores



        #######################################################################
        # SINGLE-PROPOSAL BRANCH
        #######################################################################
        B, C_out, H, W = O_pred.shape

        # Convert O_pred logits → continuous class channel
        if O_pred.size(1) > 1:
            classes = torch.arange(O_pred.size(1), device=O_pred.device).view(1, -1, 1, 1)
            probs = O_pred.softmax(dim=1)
            O_pred = (probs * classes).sum(dim=1, keepdim=True)

        # Combine masks
        if mask_in is not None or mask_out is not None:
            mask_in  = mask_in  if mask_in  is not None else torch.zeros(B, H, W, dtype=torch.bool, device=O_pred.device)
            mask_out = mask_out if mask_out is not None else torch.zeros(B, H, W, dtype=torch.bool, device=O_pred.device)
            mask = torch.logical_or(mask_in, mask_out)
        else:
            mask = None

        # Grid input to ViT
        x = torch.cat([I_in, O_pred], dim=1)              # (B, 2, H, W)
        h = self.vit.forward_grid(x, mask=mask)           # (B, embed_dim)

        # ===== Assemble features =====
        feats = [h]
        if z is not None and self.z_proj is not None:
            feats.append(self.z_proj(z))
        if C is not None and self.c_proj is not None:
            feats.append(self.c_proj(C))

        feat = torch.cat(feats, dim=-1)                   # (B, in_dim)

        return self.mlp(feat).squeeze(-1)                 # (B)
