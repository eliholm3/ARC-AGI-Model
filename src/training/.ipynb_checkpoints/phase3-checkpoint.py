import torch
import torch.nn.functional as F
from torch import autograd

from src.inference.generator import ARCGenerator

from src.training.utils_debug import report_param_stats

from src.architecture.context_encoding.example_pair_encoder import ExamplePairEncoder
from src.architecture.context_encoding.example_pair_aggregator import ExamplePairAggregator
from src.architecture.context_encoding.conditional_encoder import ConditionalTestInputEncoder
from src.architecture.ViT.body import VisionTransformer
from src.architecture.LViTM.body import LargeVisionTransformerModel
from src.architecture.executor.executor import Executor
from src.architecture.adViT.critic import AdversarialVisionTransformer
from src.data_pipeline.dataloader import ARCDataModule



###############################
#   Hyperparameters           #
###############################

EPOCHS = 5

GEN_LR = 1e-4
CRITIC_LR = 1e-4

LAMBDA_GP = 10.0
LAMBDA_ADV = 0.1

N_CRITIC = 5  # critic steps per generator step

NUM_CLASSES = 11  # ARC colors
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
    from src.architecture.ViT.body import VisionTransformer

    img_size   = 30
    patch_size = 1
    embed_dim  = 128
    num_heads  = 4
    depth_vit  = 6
    mlp_dim    = 256
    z_dim      = 64
    num_props  = 4
    NUM_CLASSES = 11  

    # Two separate ViTs
    vit_pair = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth_vit,
        mlp_dim=mlp_dim,
        in_channels=2          # (I, O)
    ).to(DEVICE)

    vit_test = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth_vit,
        mlp_dim=mlp_dim,
        in_channels=1          # I_test
    ).to(DEVICE)

    example_encoder = ExamplePairEncoder(vit_pair).to(DEVICE)

    aggregator = ExamplePairAggregator(embed_dim=embed_dim).to(DEVICE)

    cond_encoder = ConditionalTestInputEncoder(vit_test).to(DEVICE)

    lvitm = LargeVisionTransformerModel(
        embed_dim=embed_dim,
        num_heads=4,
        mlp_dim=256,
        depth=8,
        num_proposals=num_props,
        z_dim=z_dim
    ).to(DEVICE)

    executor = Executor(
        embed_dim=embed_dim,
        num_heads=4,
        mlp_dim=256,
        depth=4,
        z_dim=z_dim,
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
    from src.architecture.ViT.body import VisionTransformer
    from src.architecture.adViT.critic import AdversarialVisionTransformer

    img_size   = 30
    patch_size = 1
    embed_dim  = 128
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
        I_real,    # (B,1,H,W)
        O_real,    # (B,1,H,W)
        O_fake,    # (B,1,H,W)
        mask_in,   # (B,H,W)
        mask_out   # (B,H,W)
):
    B, _, H, W = O_real.shape

    epsilon = torch.rand(B, 1, 1, 1, device=O_real.device)
    O_interp = epsilon * O_real + (1.0 - epsilon) * O_fake
    O_interp.requires_grad_(True)

    scores = critic(
        I_in=I_real,
        O_pred=O_interp,
        mask_in=mask_in,
        mask_out=mask_out,
        z=None,
        C=None
    )

    grads = autograd.grad(
        outputs=scores.sum(),
        inputs=O_interp,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    grads = grads.view(B, -1)
    grad_norm = grads.norm(2, dim=1)
    return ((grad_norm - 1.0)**2).mean()




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

                # Real outputs: single-channel integer grid
                targets = test_outputs.view(K_test, H, W)               # (K_test, H, W)
                O_real = targets.unsqueeze(1).float()                   # (K_test, 1, H, W)

                # Fake outputs: take argmax over class dimension
                logits_det = logits.detach().view(K_test, C_out, H, W)  # (K_test, C_out, H, W)
                O_fake = logits_det.argmax(dim=1, keepdim=True).float() # (K_test, 1, H, W)


                # Inputs
                I_real = test_inputs.view(K_test, H, W).unsqueeze(1).float()  # (K_test,1,H,W)

                # Masks
                mask_in = test_input_masks.view(K_test, H, W).bool()
                mask_out = test_output_masks.view(K_test, H, W).bool()

                score_real = critic(
                    I_in=I_real,
                    O_pred=O_real,
                    mask_in=mask_in,
                    mask_out=mask_out,
                    z=None,
                    C=None
                )

                score_fake = critic(
                    I_in=I_real,
                    O_pred=O_fake,       # FIXED
                    mask_in=mask_in,
                    mask_out=mask_out,
                    z=None,
                    C=None
                )


                wasserstein = score_fake.mean() - score_real.mean()
                gp = gradient_penalty(
                    critic=critic,
                    I_real=I_real,
                    O_real=O_real,
                    O_fake=O_fake,
                    mask_in=mask_in,
                    mask_out=mask_out
                )


                critic_loss = wasserstein + LAMBDA_GP * gp

                critic_loss.backward()

                for name, p in critic.named_parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any():
                            print(f"[PHASE3 CRITIC NaN GRAD] {name}")
                        if torch.all(p.grad == 0):
                            print(f"[PHASE3 CRITIC ZERO GRAD] {name}")

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
            logits_flat = logits.view(B*K_test, C_out, H, W)
            targets_flat = test_outputs.view(B*K_test, H, W).long()

            PAD_TOKEN = -100
            targets = targets_flat.clone()
            pad_mask = ~test_output_masks.view(B*K_test, H, W).bool()
            targets[pad_mask] = PAD_TOKEN

            per_pixel = F.cross_entropy(
                logits_flat,
                targets,
                ignore_index=PAD_TOKEN,
                reduction="none"
            )

            valid = (targets != PAD_TOKEN).float()
            ce_loss = (per_pixel * valid).sum() / valid.sum()

            I_for_adv = test_inputs.view(K_test, H, W).unsqueeze(1).float()
            mask_in_adv = test_input_masks.view(K_test, H, W).bool()
            mask_out_adv = test_output_masks.view(K_test, H, W).bool()

            logits_for_adv = logits.view(K_test, C_out, H, W)
            O_fake_adv = logits_for_adv.argmax(dim=1, keepdim=True).float()

            fake_scores = critic(
                I_in=I_for_adv,
                O_pred=O_fake_adv,        
                mask_in=mask_in_adv,
                mask_out=mask_out_adv,
                z=None,
                C=None
            )


            gen_adv_loss = -fake_scores.mean()

            gen_loss = ce_loss + LAMBDA_ADV * gen_adv_loss

            gen_loss.backward()

            for name, p in generator.named_parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any():
                        print(f"[PHASE3 GEN NaN GRAD] {name}")
                    if torch.all(p.grad == 0):
                        print(f"[PHASE3 GEN ZERO GRAD] {name}")

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
        data_loader=ARCDataModule
    )

    torch.save(generator.state_dict(), "generator_phase3_adv.pt")
    torch.save(critic.state_dict(), "critic_phase3_adv.pt")

    print("\nSaved Phase 3 generator and critic checkpoints.")
