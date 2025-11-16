import torch
import torch.nn as nn

from src.architecture.LViTM.body import LargeVisionTransformerModel
from src.architecture.executor.executor import Executor
from src.architecture.adViT.critic import AdversarialVisionTransformer
from src.architecture.context_encoding.conditional_encoder import ConditionalTestInputEncoder


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
        self.lvitm = lvitm
        self.executor = executor
        self.cond_encoder = cond_encoder
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

        ############################################################
    #   PPO Rollout + (stub) Update for Phase 4                #
    ############################################################
    def ppo_rollout_and_update(
        self,
        init_grid: torch.Tensor,       # (B, 1, H, W)
        init_mask: torch.Tensor,       # (B, H, W)
        C: torch.Tensor,               # (B, D)
        ppo_refiner,                   # PPORefiner object (currently unused in stub)
        num_steps: int,
        gamma: float,
    ):
        """
        Minimal implementation to satisfy Phase 4:

        - Encodes the test grid with cond_encoder
        - Runs LViTM once to get latent proposals z
        - Picks the first proposal
        - Runs Executor to get final logits
        - Returns logits and a dummy PPO stats dict

        This does NOT yet perform real PPO; it is a safe stub
        that keeps all dimensions consistent and lets training run.
        """

        # 1. Encode test grid into tokens
        #    ConditionalTestInputEncoder.forward: (I_test, mask_I) -> (tokens, key_padding_mask)
        test_tokens, key_padding_mask = self.cond_encoder(
            I_test=init_grid,
            mask_test=init_mask,
            C=C
        )


        # 2. Get latent proposals z from LViTM
        #    LargeVisionTransformerModel.forward: (tokens, key_padding_mask, C) -> Z
        Z = self.lvitm(
            C=C,
            test_tokens=test_tokens,
            key_padding_mask=key_padding_mask
        )
        # typically (B, T, z_dim) or (B, z_dim)

        # 3. Choose a single z per sample (e.g., first proposal)
        if Z.dim() == 3:  # (B, T, z_dim)
            Z0 = Z[:, 0, :]           # (B, z_dim)
        else:             # already (B, z_dim)
            Z0 = Z

        # 4. Run Executor to get logits over ARC colors
        #    Executor.forward is expected to look like:
        #       forward(I_test, Z, key_padding_mask=None)
        logits = self.executor(init_grid, Z0)
        # (B, num_classes, H, W)

        # 5. Package PPO stats (stubbed out for now)
        ppo_stats = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
        }

        # Phase 4 expects: (final_logits, ppo_stats)
        return logits, ppo_stats
