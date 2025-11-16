import torch
import torch.nn as nn
from typing import Dict, Any
import os
from dotenv import load_dotenv

class ARCGenerator(nn.Module):
    """
    For each sample:
      - Uses ALL training example pairs (I_i, O_i) to compute a context vector C.
      - Then, for EACH test input in that sample:
          * encodes the test input conditioned on C,
          * uses LViTM to propose latent transformation vectors Z,
          * chooses a single z (here: the first proposal),
          * runs the Executor to produce logits for that test.
    """

    def __init__(
        self,
        example_pair_encoder: nn.Module,
        aggregator: nn.Module,
        cond_encoder: nn.Module,
        lvitm: nn.Module,
        executor: nn.Module,
    ):
        super().__init__()
        self.example_pair_encoder = example_pair_encoder
        self.aggregator = aggregator
        self.cond_encoder = cond_encoder
        self.lvitm = lvitm
        self.executor = executor

    def forward(
        self,
        train_inputs: torch.Tensor,        # (K_train, H, W)   or (B, K_train, H, W)
        train_outputs: torch.Tensor,       # (K_train, H, W)   or (B, K_train, H, W)
        train_input_masks: torch.Tensor,   # (K_train, H, W)   or (B, K_train, H, W)
        train_output_masks: torch.Tensor,  # (K_train, H, W)   or (B, K_train, H, W)
        test_inputs: torch.Tensor,         # (K_test, H, W)    or (B, K_test, H, W)
        test_input_masks: torch.Tensor,    # (K_test, H, W)    or (B, K_test, H, W)
    ) -> Dict[str, Any]:

        ##################################
        #   Normalize Batch Dimensions   #
        ##################################

        load_dotenv()
        DEBUGGING = os.getenv("DEBUGGING")

        # Ensure batch dimension exists
        if train_inputs.dim() == 3:
            train_inputs       = train_inputs.unsqueeze(0)
            train_outputs      = train_outputs.unsqueeze(0)
            train_input_masks  = train_input_masks.unsqueeze(0)
            train_output_masks = train_output_masks.unsqueeze(0)

        if test_inputs.dim() == 3:
            test_inputs       = test_inputs.unsqueeze(0)
            test_input_masks  = test_input_masks.unsqueeze(0)


        B, K_train, H, W = train_inputs.shape
        _, K_test, H_t, W_t = test_inputs.shape
        assert H == H_t and W == W_t, "Train and test grids must share padded size per sample."

        # Ensure masks are boolean
        train_input_masks = train_input_masks.bool()
        train_output_masks = train_output_masks.bool()
        test_input_masks = test_input_masks.bool()

        if DEBUGGING:  # toggle debug
            print("\n[ARCGenerator] train_inputs:", train_inputs.shape)
            print("[ARCGenerator] test_inputs:", test_inputs.shape)
            print("[ARCGenerator] train_input_masks:", train_input_masks.shape)
            print("[ARCGenerator] test_input_masks:", test_input_masks.shape)


        #########################################
        #   Encode ALL training example pairs   #
        #########################################

        # For each training pair (I_i, O_i), we get a context embedding h_i.
        h_list = []

        for k in range(K_train):
            # Shapes: (B, 1, H, W)
            I_k = train_inputs[:, k].unsqueeze(1).float()
            O_k = train_outputs[:, k].unsqueeze(1).float()

            mask_I_k = train_input_masks[:, k]   # (B, H, W)
            mask_O_k = train_output_masks[:, k]  # (B, H, W)

            h_k = self.example_pair_encoder(
                I_i=I_k,
                O_i=O_k,
                mask_I=mask_I_k,
                mask_O=mask_O_k
            )  # (B, D)
            h_list.append(h_k)

        # Stack: (B, K_train, D)
        h = torch.stack(h_list, dim=1)

        # Optional: pair_mask could be used if we ever have invalid pairs.
        pair_mask = None  # (B, K_train) if needed

        ##############################
        #   Aggregate to context C   #
        ##############################

        # Single context vector per sample, shared by all test pairs.
        C = self.aggregator(h, mask=pair_mask)  # (B, D)

        if DEBUGGING:
            print("[ARCGenerator] C mean/std:", C.mean().item(), C.std().item())


        # ----------------------------------
        #   Loop over ALL test inputs
        # ----------------------------------
        all_logits = []
        all_Z = []
        all_z_chosen = []

        for j in range(K_test):
            # Test grid j: (B, 1, H, W)
            I_test_j = test_inputs[:, j].unsqueeze(1).float()
            mask_test_j = test_input_masks[:, j]   # (B, H, W)

            # Encode test input with context C
            test_tokens_j, key_padding_mask_j = self.cond_encoder(
                I_test_j,      # (B,1,H,W)
                mask_test_j,   # (B,H,W)
                C              # (B,D)
            )
            # test_tokens_j: (B, S, D)
            # key_padding_mask_j: (B, S) or None

            # LViTM proposes latent transformation vectors for this test input
            Z_j = self.lvitm(
                C=C,
                test_tokens=test_tokens_j,
                key_padding_mask=key_padding_mask_j
            )  # (B, P, z_dim)

            Bz, P, z_dim = Z_j.shape
            assert Bz == B, "Batch size mismatch between context and proposals."

            # Choose a single proposal z for this test input
            # Minimal baseline: pick first proposal
            z_chosen_j = Z_j[:, 0, :]  # (B, z_dim)

            # Executor predicts the output grid logits
            logits_j = self.executor(
                grid=I_test_j,   # (B,1,H,W)
                z=z_chosen_j     # (B,z_dim)
            )  # (B, num_classes, H, W)

            all_logits.append(logits_j)          # each: (B, C_out, H, W)
            all_Z.append(Z_j)                    # each: (B, P, z_dim)
            all_z_chosen.append(z_chosen_j)      # each: (B, z_dim)

        # -----------------------------------------
        #   Stack results across ALL test inputs
        # -----------------------------------------
        # logits: (B, K_test, num_classes, H, W)
        logits = torch.stack(all_logits, dim=1)

        # Sanitize logits to avoid NaNs/Infs propagating into losses
        logits = torch.nan_to_num(
            logits,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )

        # Z_all: (B, K_test, P, z_dim)
        Z_all = torch.stack(all_Z, dim=1)

        # z_chosen: (B, K_test, z_dim)
        z_chosen = torch.stack(all_z_chosen, dim=1)

        if DEBUGGING:
            print("[ARCGenerator] logits mean/std:", logits.mean().item(), logits.std().item())


        return {
            "logits": logits,         # (B, K_test, C_out, H, W)
            "Z_all": Z_all,           # (B, K_test, P, z_dim)
            "z_chosen": z_chosen,     # (B, K_test, z_dim)
            "C": C                    # (B, D)
        }

"""
Usage:

batch = next(iter(arc_loader))

out = arc_generator(
    train_inputs=batch["train_inputs"],         # (K_train, H, W)
    train_outputs=batch["train_outputs"],       # (K_train, H, W)
    train_input_masks=batch["train_input_masks"],
    train_output_masks=batch["train_output_masks"],
    test_inputs=batch["test_inputs"],           # (K_test, H, W)
    test_input_masks=batch["test_input_masks"],
)

logits = out["logits"]  # (1, K_test, num_classes, H, W)


Loss:

import torch.nn.functional as F

# target: (1, K_test, H, W) → (1 * K_test, H, W)
target = batch["test_outputs"].unsqueeze(0) if batch["test_outputs"].dim() == 3 else batch["test_outputs"]
B, K_test, H, W = target.shape
target_flat = target.view(B * K_test, H, W)

# logits: (1, K_test, C_out, H, W) → (1 * K_test, C_out, H, W)
logits_flat = logits.view(B * K_test, logits.size(2), H, W)

loss = F.cross_entropy(logits_flat, target_flat)
loss.backward()
optimizer.step()

"""