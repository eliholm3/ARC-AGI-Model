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
            I_in: torch.Tensor,  # (B, C_in, H, W)
            O_pred: torch.Tensor,  # (B, C_out, H, W) or (B, P, C_out, H, W)
            mask_in: torch.Tensor | None = None,  # (B, H, W)
            mask_out: torch.Tensor | None = None,  # (B, H, W)
            z: torch.Tensor | None = None,  # (B, z_dim) or (B, P, z_dim)
            C: torch.Tensor | None = None  # (B, c_dim) or (B, P, z_dim)
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
        #   Handle Multiple Proposals   #
        ################################# 

        # Check for multiple proposals
        multi = (O_pred.dim() == 5)

        if multi:
            B, T, C_out, H, W = O_pred.shape

            # Expand input across proposals
            I_exp = I_in.unsqueeze(1).expand(B, T, C_in, H, W)  # (B, T, C_in, H, W)
            I_flat = I_exp.reshape(B * T, C_in, H, W)
            O_flat = I_exp.reshape(B * T, C_out, H, W)

            #####################
            #   Combine masks   #
            #####################

            if mask_in is not None or mask_out is not None:

                # Make default mask
                if mask_in is None:
                    mask_in = torch.zeros(B, H, W, dtype=torch.bool, device=O_pred.device)
                if mask_out is None:
                    mask_out = torch.zeros(B, H, W, dtype=torch.bool, device=O_pred.device)
                
                # Combine
                mask = torch.logical_or(mask_in, mask_out)  # (B, H, W)
                mask = mask.unsqueeze(1).expand(B, T, H, W).reshape(B * T, H, W)
            
            else: 
                mask = None

            # Concatentate input & output along channels
            x = torch.cat([I_flat, O_flat], dim=1)  # (B*T, C_in+C_out, H, W)

            ######################
            #   Encode Context   #
            ######################

            # Encode with shared ViT to get pair embedding
            h_flat = self.vit.forward_grid(x, mask=mask)  # (B*T, D)
            h = h_flat.view(B, T, -1)  # (B,T,D)

            features = [h]

            ################################
            #   Add Meta Vectors (z + C)   #
            ################################

            # Add z if provided
            if z is not None:
                # z: (B, z_dim) or (B, T, z_dim)
                if z.dim() == 2:
                    z = z.unsqueeze(1).expand(B, T, -1)
                features.append(z)

            # Add C if provided
            if C is not None:
                # C: (B, c_dim) -> expand across T
                C_exp = C.unsqueeze(1).expand(B, T, -1)
                features.append(C_exp)

            feat = torch.cat(features, dim=-1)  # (B,T,in_dim)

            #############################
            #   Compute Critic Scores   #
            #############################

            scores = self.mlp(feat).squeeze(-1)  # (B,T)
            return scores

        else:

            ##############################
            #   Handle Single Proposal   #
            ##############################
            
            B, C_out, H, W = O_pred.shape

            #####################
            #   Combine masks   #
            #####################

            if mask_in is not None or mask_out is not None:
                if mask_in is None:
                    mask_in = torch.zeros(B, H, W, dtype=torch.bool, device=O_pred.device)
                if mask_out is None:
                    mask_out = torch.zeros(B, H, W, dtype=torch.bool, device=O_pred.device)
                mask = torch.logical_or(mask_in, mask_out)  # (B,H,W)
            else:
                mask = None

            ######################
            #   Encode Context   #
            ######################

            x = torch.cat([I_in, O_pred], dim=1)  # (B, C_in+C_out, H, W)
            h = self.vit.forward_grid(x, mask=mask)  # (B,D)

            features = [h]

            ################################
            #   Add Meta Vectors (z + C)   #
            ################################

            if z is not None:
                # z: (B, z_dim)
                features.append(z)

            if C is not None:
                # C: (B, c_dim)
                features.append(C)

            feat = torch.cat(features, dim=-1)  # (B,in_dim)

            #############################
            #   Compute Critic Scores   #
            #############################

            scores = self.mlp(feat).squeeze(-1)  # (B,)
            return scores