import torch
from torch import autograd
from tqdm import tqdm

from src.training.utils_debug import report_param_stats
from src.training.metrics import ensure_dir, save_loss_plot, append_metrics_csv

from src.architecture.ViT.body import VisionTransformer
from src.architecture.adViT.critic import AdversarialVisionTransformer
from src.data_pipeline.dataloader import ARCDataModule


###############################
#   Hyperparameters           #
###############################

CRITIC_LR = 1e-4
CRITIC_EPOCHS = 50
LAMBDA_GP = 10.0
NUM_CLASSES = 11  # ARC colors
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###############################
#   Build Critic + ViT        #
###############################

def build_critic():
    from src.architecture.ViT.body import VisionTransformer
    from src.architecture.adViT.critic import AdversarialVisionTransformer

    img_size   = 30
    patch_size = 1
    embed_dim  = 256
    num_heads  = 4
    depth_vit  = 6
    mlp_dim    = 256

    # IMPORTANT: ALWAYS 2 CHANNELS
    #   ch1 = I_test  (1 channel)
    #   ch2 = O_real or O_fake (1 channel)
    vit_critic = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth_vit,
        mlp_dim=mlp_dim,
        in_channels=2       
    ).to(DEVICE)

    critic = AdversarialVisionTransformer(
        vit_encoder=vit_critic,
        z_dim=None,
        c_dim=None,
        mlp_dim=256
    ).to(DEVICE)

    return critic



###############################
#   Fake Output Generator     #
###############################

def make_fake_outputs(
        real_outputs,   # (K_test, H, W)
        mask            # (K_test, H, W)
):
    """
    Creates fake outputs by sampling random colors where mask == 1.
    """

    K_test, H, W = real_outputs.shape
    fake = torch.randint(
        low=0,
        high=NUM_CLASSES,
        size=(K_test, H, W),
        device=real_outputs.device
    )

    # Keep padding as 0; only randomize valid cells
    fake = torch.where(mask.bool(), fake, torch.zeros_like(fake))

    return fake


###############################
#   Gradient Penalty (GP)     #
###############################

def gradient_penalty(
        critic,
        I_real,    # (B, 1, H, W)
        O_real,    # (B, 1, H, W)
        O_fake,    # (B, 1, H, W)
        mask_in,   # (B, H, W)
        mask_out   # (B, H, W)
):
    """
    WGAN-GP gradient penalty on interpolated outputs.
    Only interpolates in the output space for simplicity.
    """

    B, _, H, W = O_real.shape

    # Interpolate between real and fake outputs
    epsilon = torch.rand(B, 1, 1, 1, device=O_real.device)
    O_interp = epsilon * O_real + (1.0 - epsilon) * O_fake

    # Enable gradient tracking
    O_interp.requires_grad_(True)

    # Input grid stays fixed (we could also interpolate I, but ARC rules are usually about O)
    I_interp = I_real

    # Critic score on interpolated pair
    scores = critic(
        I_in=I_interp,
        O_pred=O_interp,
        mask_in=mask_in,
        mask_out=mask_out,
        z=None,
        C=None
    )  # (B,)

    # Compute gradients of scores w.r.t O_interp
    grads = autograd.grad(
        outputs=scores.sum(),
        inputs=O_interp,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # (B, 1, H, W)

    grads = grads.view(B, -1)  # (B, H*W)
    grad_norm = grads.norm(2, dim=1)  # (B,)

    gp = ((grad_norm - 1.0) ** 2).mean()
    return gp


###############################
#   Critic Warmup Loop        #
###############################

def train_critic_phase2(critic, data_loader):

    critic.train()
    optimizer = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)
    ensure_dir("checkpoints")
    critic_losses = []

    for epoch in tqdm(range(CRITIC_EPOCHS), "Critic Warmup Epoch:"):
        total_loss = 0.0
        total_batches = 0

        # print(f"\n=== Critic Warmup Epoch {epoch + 1}/{CRITIC_EPOCHS} ===")

        for batch in data_loader:

            ###############################
            #   Move tensors to device    #
            ###############################

            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(DEVICE)

            test_inputs = batch["test_inputs"]          # (K_test, H, W)
            test_outputs = batch["test_outputs"]        # (K_test, H, W)
            test_input_masks = batch["test_input_masks"]    # (K_test, H, W)
            test_output_masks = batch["test_output_masks"]  # (K_test, H, W)

            K_test, H, W = test_inputs.shape
            if K_test == 0:
                continue

            ################################
            #   Prepare real / fake pairs  #
            ################################

            # Real input/output (B = K_test)
            I_real = test_inputs.unsqueeze(1).float()       # (B, 1, H, W)
            O_real = test_outputs.unsqueeze(1).float()      # (B, 1, H, W)

            mask_in = test_input_masks.bool()               # (B, H, W)
            mask_out = test_output_masks.bool()             # (B, H, W)

            # Fake outputs
            O_fake_raw = make_fake_outputs(
                real_outputs=test_outputs,
                mask=test_output_masks
            )  # (B, H, W)
            O_fake = O_fake_raw.unsqueeze(1).float()        # (B, 1, H, W)

            ###############################
            #   Critic Scores             #
            ###############################

            # Real scores
            score_real = critic(
                I_in=I_real,
                O_pred=O_real,
                mask_in=mask_in,
                mask_out=mask_out,
                z=None,
                C=None
            )  # (B,)

            # Fake scores
            score_fake = critic(
                I_in=I_real,
                O_pred=O_fake,
                mask_in=mask_in,
                mask_out=mask_out,
                z=None,
                C=None
            )  # (B,)

            wasserstein = score_fake.mean() - score_real.mean()

            # Gradient penalty
            gp = gradient_penalty(
                critic=critic,
                I_real=I_real,
                O_real=O_real,
                O_fake=O_fake,
                mask_in=mask_in,
                mask_out=mask_out
            )

            loss = wasserstein + LAMBDA_GP * gp

            ###############################
            #   Optimize Critic           #
            ###############################

            optimizer.zero_grad()
            loss.backward()

            ###################################
            #   DEBUG: Check Critic Gradients
            ###################################
            # for name, p in critic.named_parameters():
            #     if p.grad is None:
            #         print(f"[PHASE2 WARNING] No grad for {name}")
            #     else:
            #         if torch.isnan(p.grad).any():
            #             print(f"[PHASE2 ERROR] NaN gradient detected in {name}")
            #         if torch.all(p.grad == 0):
            #             print(f"[PHASE2 WARNING] ZERO gradient in {name}")

            # Optional (expensive): print weight + grad stats
            # report_param_stats(critic, name="Phase2 Critic", max_layers=12)

            optimizer.step()


            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        print(f"Epoch {epoch + 1}: Critic Warmup Loss = {avg_loss:.4f}")
        critic_losses.append(avg_loss)
        try:
            save_loss_plot(critic_losses, None, out_path="checkpoints/phase2_critic_loss.png")
            append_metrics_csv("checkpoints/phase2_metrics.csv", {"epoch": epoch + 1, "critic_loss": avg_loss})
        except Exception as e:
            print(f"[phase2] Warning: failed to save critic metrics: {e}")

    return critic


#######################
#   Main Entrypoint   #
#######################

if __name__ == "__main__":
    critic = build_critic()
    critic = train_critic_phase2(critic, ARCDataModule)
    torch.save(critic.state_dict(), "critic_phase2_warmup.pt")
    print("\nSaved critic after Phase 2 warmup: critic_phase2_warmup.pt")
