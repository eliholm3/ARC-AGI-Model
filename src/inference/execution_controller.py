import torch
import torch.nn as nn

from ..architecture.LViTM.body import LargeVisionTransformerModel
from ..architecture.executor.executor import Executor
from ..architecture.adViT.critic import AdversarialVisionTransformer
from ..architecture.context_encoding.conditional_encoder import ConditionalTestInputEncoder


class HybridExecuteController(nn.Module):

    def __init__(
            self,
            lvitm: LargeVisionTransformerModel,
            executor: Executor, 
            cond_encoder: ConditionalTestInputEncoder,
            critic: AdversarialVisionTransformer | None = None
    ):
        ###############################
        #   B = batch size            #    
        #   D = token embedding dim   #
        #   P = num proposals         #
        #   H = height                #
        #   W = width                 #
        ###############################

        super().__init__()
        self.lvitm = lvitm,
        self.executor = executor
        self.cond_encoder = cond_encoder,
        self.critic = critic

    ################################
    #   Parallel Mode (Training)   #
    ################################

    def apply_parallel(
        self,
        I_test: torch.Tensor,  # (B, 1, H, W)
        mask_test: torch.Tensor,  # (B, H, W)
        C: torch.Tensor,  # (B, D)
        examples=None
    ):
        B, _, H, W = I_test.shape

        # Encode test input with context
        tokens, key_padding_mask = self.cond_encoder(I_test, mask_test, C)

        #######################
        #   Reasoning Model   #
        #######################

        # Compute proposals
        Z = self.lvitm(C, tokens, key_padding_mask)

        B, P, z_dim = Z.shape

        ################
        #   Executor   #
        ################

        # Flatten input for executor
        grid_expansion = I_test.unsqueeze(1).expand(B, P, 1, H, W).reshape(B*P, 1, H, W)
        z_flat = Z.reshape(B*P, z_dim)

        # Execute proposals
        out_flat = self.executor(grid_expansion, z_flat)
        num_classes = out_flat.size(1)
        outputs = out_flat.view(B, P, num_classes, H, W)

        ##############
        #   Critic   #
        ##############

        scores = None
        best_idx = None
        if self.critic is not None and examples is not None:
            scores = self.critic(
                I_in=I_test,            # (B, 1, H, W)
                O_pred=outputs,         # (B, P, num_classes, H, W)
                mask_in=mask_test,      # (B, H, W)
                mask_out=mask_test,     # assume same valid region as input
                z=Z,                    # (B, P, z_dim)
                C=C                     # (B, D)
            )     
            best_idx = scores.argmax(dim=1)  # (B,)

        return outputs, scores, best_idx
    
    ###################################
    #   Sequential Mode (Inference)   #
    ###################################

    @torch.no_grad()
    def apply_sequential(
        self,
        init_grid: torch.Tensor,  # (B, 1, H, W)
        init_mask: torch.Tensor,  # (B, H, W)
        C,                        # (B, D)
        examples=None,
        num_steps=3
    ):
        grid = init_grid
        mask = init_mask
        history: list[torch.Tensor] = [grid.clone()]

        #########################
        #   Iterate Proposals   #
        #########################

        for _ in range(num_steps):
            B, _, H, W = grid.shape

            # Encode test input with context
            tokens, key_padding_mask = self.cond_encoder(grid, mask, C)

            #######################
            #   Reasoning Model   #
            #######################

            # Compute proposal
            Z = self.lvitm(C, tokens, key_padding_mask)  # (B, P, z_dim)
            B, P, z_dim = Z.shape

            ################
            #   Executor   #
            ################

            # Flatten input for executor
            grid_rep = grid.unsqueeze(1).expand(B, P, 1, H, W).reshape(B * P, 1, H, W)
            z_flat = Z.reshape(B * P, z_dim)

            # Execute proposals
            out_flat = self.executor(grid_rep, z_flat)  # (B*T, num_classes, H, W)
            num_classes = out_flat.size(1)
            outputs = out_flat.view(B, P, num_classes, H, W)

            ##############
            #   Critic   #
            ##############

            # Choose proposal
            if self.critic is not None:
                scores = self.critic(
                    I_in=grid,
                    O_pred=outputs,
                    mask_in=mask,
                    mask_out=mask,
                    z=Z,
                    C=C
                )  # (B, P)
                best_idx = scores.argmax(dim=1)  # (B,)
            else:
                # Or take first proposal
                best_idx = torch.zeros(B, dtype=torch.long, device=grid.device)

            # Gather best output per batch
            idx = best_idx.view(B, 1, 1, 1, 1).expand(B, 1, num_classes, H, W)
            best_out_logits = outputs.gather(dim=1, index=idx).squeeze(1)  # (B, C_out, H, W)

            # Discretize logits
            best_out_grid = best_out_logits.argmax(dim=1, keepdim=True)    # (B, 1, H, W)

            grid = best_out_grid
            # mask stays the same spatially
            history.append(grid.clone())

        final_grid_logits = best_out_logits  
        return final_grid_logits, history