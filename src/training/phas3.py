import math
import torch
import torch.nn.functional as F
from torch import autograd

from arc_generator import ARCGenerator

from available_functions import (
    VisionTransformer,
    ExamplePairEncoder,
    ExamplePairAggregator,
    ConditionalTestInputEncoder,
    LargeVisionTransformerModel,
    Executor,
    AdversarialVisionTransformer,
    arc_loader
)


###############################
#   Hyperparameters           #
###############################

EPOCHS = 5

GEN_LR = 1e-4
CRITIC_LR = 1e-4

LAMBDA_GP = 10.0
LAMBDA_ADV = 0.1

N_CRITIC = 5  # critic steps per generator step

NUM_CLASSES = 10  # ARC colors
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###############################
#   Helper: grad toggle       #
###############################

def set_requires_grad(module, flag: bool):
    """
    Enables or disables gradients for a module.
    """
    for p in module.parameters():
        p.requires_grad = flag


###############################
#   Build Generator           #
###############################

def build_generator():
    """
    Rebuilds the ARCGenerator and submodules.

    NOTE: This should match your Phase 1 configuration.
    """

    # Shared ViT encoder for generator-side modules
    vit_gen = VisionTransformer(
        img_size=30,      # adjust if needed
        patch_size=1,
        embed_dim=128,
        num_heads=4,
        depth=6,
        mlp_dim=256,
        in_channels=2     # ExamplePairEncoder concatenates I and O
    ).to(DEVICE)

    example_encoder = ExamplePairEncoder(vit_gen).to(DEVICE)
    aggregator = ExamplePairAggregator(embed_dim=vit_gen.c_token.size(-1)).to(DEVICE)
    cond_encoder = ConditionalTestInputEncoder(vit_gen).to(DEVICE)

    lvitm = LargeVisionTransformerModel(
        embed_dim=vit_gen.c_token.size(-1),
        num_heads=4,
        mlp_dim=256,
        depth=8,
        num_proposals=4,
        z_dim=64
    ).to(DEVICE)

    executor = Executor(
        embed_dim=vit_gen.c_token.size(-1),
        num_heads=4,
        mlp_dim=256,
        depth=4,
        z_dim=64,
        hidden_channels=64,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    generator = ARCGenerator(
        example_pair_encoder=example_encoder,
        aggregator=aggregator,
        cond_encoder=cond_encoder,
        lvitm=lvitm,
        executor=executor
    ).to(DEVICE)

    return generator


###############################
#   Build Critic              #
###############################

def build_critic():
    """
    Builds the critic with its own ViT encoder.

    IMPORTANT:
    critic ViT in_channels = 1 (input grid) + NUM_CLASSES (output logits)
    """

    vit_critic = VisionTransformer(
        img_size=30,      # adjust if needed
        patch_size=1,
        embed_dim=128,
        num_heads=4,
        depth=6,
        mlp_dim=256,
        in_channels=1 + NUM_CLASSES
    ).to(DEVICE)

    critic = AdversarialVisionTransformer(
        vit_encoder=vit_critic,
        z_dim=None,
        c_dim=None,
        hidden_dim=256
    ).to(DEVICE)

    return critic


###############################
#   One-hot encoder           #
###############################

def one_hot_from_int(
        grid,       # (B, H, W) int
        num_classes # scalar
):
    """
    Converts integer grid to one-hot channels: (B, num_classes, H, W)
    """

    B, H, W = grid.shape
    # (B, H, W, C)
    oh = F.one_hot(grid.long().clamp(min=0, max=num_classes - 1), num_classes=num_classes)
    oh = oh.permute(0, 3, 1, 2).float()  # (B, C, H, W)
    return oh


###############################
#   Gradient Penalty (GP)     #
###############################

def gradient_penalty(
        critic,
        I_real,    # (B, 1, H, W)
        O_real,    # (B, C_out, H, W)
        O_fake,    # (B, C_out, H, W)
        mask_in,   # (B, H, W)
        mask_out   # (B, H, W)
):
    """
    WGAN-GP gradient penalty on interpolated outputs.
    """

    B, C_out, H, W = O_real.shape

    epsilon = torch.rand(B, 1, 1, 1, device=O_real.device)
    O_interp = epsilon * O_real + (1.0 - epsilon) * O_fake
    O_interp.requires_grad_(True)

    I_interp = I_real

    scores = critic(
        I_in=I_interp,
        O_pred=O_interp,
        mask_in=mask_in,
        mask_out=mask_out,
        z=None,
        C=None
    )  # (B,)

    grads = autograd.grad(
        outputs=scores.sum(),
        inputs=O_interp,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # (B, C_out, H, W)

    grads = grads.view(B, -1)
    grad_norm = grads.norm(2, dim=1)  # (B,)

    gp = ((grad_norm - 1.0) ** 2).mean()
    return gp


###############################
#   Phase 3 Training Loop     #
###############################

def train_phase3_adversarial(
        generator,
        critic,
        data_loader
):
    """
    Joint adversarial training of generator + critic.
    Supervised CE + WGAN-GP adversarial loss.
    """

    # Optionally load checkpoints from Phase 1 and Phase 2
    try:
        generator.load_state_dict(torch.load("phase1_generator.pt", map_location=DEVICE))
        print("Loaded Phase 1 generator checkpoint.")
    except FileNotFoundError:
        print("Phase 1 generator checkpoint not found; training generator from scratch.")

    try:
        critic.load_state_dict(torch.load("critic_phase2_warmup.pt", map_location=DEVICE))
        print("Loaded Phase 2 critic checkpoint.")
    except FileNotFoundError:
        print("Phase 2 critic checkpoint not found; training critic from scratch.")

    gen_opt = torch.optim.Adam(generator.parameters(), lr=GEN_LR)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)

    for epoch in range(EPOCHS):
        print(f"\n=== Phase 3 Epoch {epoch + 1}/{EPOCHS} ===")
        total_gen_loss = 0.0
        total_critic_loss = 0.0
        n_gen_steps = 0
        n_critic_steps = 0

        for batch_idx, batch in enumerate(data_loader):

            ###############################
            #   Move batch to device      #
            ###############################

            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(DEVICE)

            train_inputs       = batch["train_inputs"]        # (K_train, H, W)
            train_outputs      = batch["train_outputs"]       # (K_train, H, W)
            train_input_masks  = batch["train_input_masks"]   # (K_train, H, W)
            train_output_masks = batch["train_output_masks"]  # (K_train, H, W)

            test_inputs        = batch["test_inputs"]         # (K_test, H, W)
            test_outputs       = batch["test_outputs"]        # (K_test, H, W)
            test_input_masks   = batch["test_input_masks"]    # (K_test, H, W)
            test_output_masks  = batch["test_output_masks"]   # (K_test, H, W)

            ###############################
            #   Forward: Generator        #
            ###############################

            # ARCGenerator handles B=1 internally
            gen_out = generator(
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                train_input_masks=train_input_masks,
                train_output_masks=train_output_masks,
                test_inputs=test_inputs,
                test_input_masks=test_input_masks,
            )

            logits = gen_out["logits"]  # (1, K_test, C_out, H, W)
            B, K_test, C_out, H, W = logits.shape
            assert B == 1, "Phase 3 currently assumes batch_size=1 per task."

            # -------------------------------
            #   Critic updates (N_CRITIC)    #
            # -------------------------------

            for _ in range(N_CRITIC):
                set_requires_grad(critic, True)
                set_requires_grad(generator, False)

                critic_opt.zero_grad()

                # Use detached logits for critic training
                logits_det = logits.detach().view(K_test, C_out, H, W)  # (B*K_test, C_out,H,W)

                # Real outputs: one-hot
                targets = test_outputs.view(K_test, H, W)               # (K_test,H,W)
                O_real = one_hot_from_int(targets, NUM_CLASSES)        # (K_test, C_out, H, W)

                # Inputs
                I_real = test_inputs.view(K_test, H, W).unsqueeze(1).float()  # (K_test,1,H,W)

                # Masks
                mask_in = test_input_masks.view(K_test, H, W).bool()
                mask_out = test_output_masks.view(K_test, H, W).bool()

                # Critic scores
                score_real = critic(
                    I_in=I_real,
                    O_pred=O_real,
                    mask_in=mask_in,
                    mask_out=mask_out,
                    z=None,
                    C=None
                )  # (K_test,)

                score_fake = critic(
                    I_in=I_real,
                    O_pred=logits_det,
                    mask_in=mask_in,
                    mask_out=mask_out,
                    z=None,
                    C=None
                )  # (K_test,)

                wasserstein = score_fake.mean() - score_real.mean()
                gp = gradient_penalty(
                    critic=critic,
                    I_real=I_real,
                    O_real=O_real,
                    O_fake=logits_det,
                    mask_in=mask_in,
                    mask_out=mask_out
                )

                critic_loss = wasserstein + LAMBDA_GP * gp

                critic_loss.backward()
                critic_opt.step()

                total_critic_loss += critic_loss.item()
                n_critic_steps += 1

            ###############################
            #   Generator update          #
            ###############################

            set_requires_grad(critic, False)
            set_requires_grad(generator, True)

            gen_opt.zero_grad()

            # Supervised CE loss across all test pairs
            logits_flat = logits.view(B * K_test, C_out, H, W)         # (K_test,C,H,W)
            targets_flat = test_outputs.view(B * K_test, H, W).long()  # (K_test,H,W)

            ce_loss = F.cross_entropy(logits_flat, targets_flat)

            # Adversarial loss: - E[critic(I, O_fake)]
            logits_for_adv = logits.view(K_test, C_out, H, W)          # (K_test,C,H,W)

            I_for_adv = test_inputs.view(K_test, H, W).unsqueeze(1).float()
            mask_in_adv = test_input_masks.view(K_test, H, W).bool()
            mask_out_adv = test_output_masks.view(K_test, H, W).bool()

            fake_scores = critic(
                I_in=I_for_adv,
                O_pred=logits_for_adv,
                mask_in=mask_in_adv,
                mask_out=mask_out_adv,
                z=None,
                C=None
            )  # (K_test,)

            gen_adv_loss = -fake_scores.mean()

            gen_loss = ce_loss + LAMBDA_ADV * gen_adv_loss
            gen_loss.backward()
            gen_opt.step()

            total_gen_loss += gen_loss.item()
            n_gen_steps += 1

        avg_gen = total_gen_loss / max(n_gen_steps, 1)
        avg_critic = total_critic_loss / max(n_critic_steps, 1)

        print(f"Epoch {epoch + 1}: Gen Loss = {avg_gen:.4f}, Critic Loss = {avg_critic:.4f}")

    return generator, critic


###############################
#   Main Entrypoint           #
###############################

if __name__ == "__main__":
    generator = build_generator()
    critic = build_critic()

    generator, critic = train_phase3_adversarial(
        generator=generator,
        critic=critic,
        data_loader=arc_loader
    )

    torch.save(generator.state_dict(), "generator_phase3_adv.pt")
    torch.save(critic.state_dict(), "critic_phase3_adv.pt")

    print("\nSaved Phase 3 generator and critic checkpoints.")
