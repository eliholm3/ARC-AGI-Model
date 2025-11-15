import torch
import torch.nn.functional as F

from checkpoints import load_checkpoint

from arc_generator import ARCGenerator
from ppo_actor import PPOActor
from ppo_value import PPOValuer
from ppo_refiner import PPORefiner

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
from controller.hybrid_execute_controller import HybridExecuteController


###############################
#   Device + Constants        #
###############################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10


###############################
#   Generator Builder         #
###############################

def build_generator():
    vit_gen = VisionTransformer(
        img_size=30,
        patch_size=1,
        embed_dim=128,
        num_heads=4,
        depth=6,
        mlp_dim=256,
        in_channels=2
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

    return generator, example_encoder, aggregator, cond_encoder, lvitm, executor


###############################
#   Critic + PPO Builders     #
###############################

def build_critic():
    vit_critic = VisionTransformer(
        img_size=30,
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


def build_ppo(z_dim=64):
    actor = PPOActor(z_dim=z_dim, embed_dim=256).to(DEVICE)
    valuer = PPOValuer(z_dim=z_dim, embed_dim=256).to(DEVICE)
    refiner = PPORefiner(actor=actor, value_fn=valuer, lr=1e-4)
    return actor, valuer, refiner


###############################
#   Context Computation       #
###############################

def compute_context_C(
        example_encoder,
        aggregator,
        train_inputs,
        train_outputs,
        train_input_masks,
        train_output_masks
):
    if train_inputs.dim() == 3:
        train_inputs = train_inputs.unsqueeze(0)
        train_outputs = train_outputs.unsqueeze(0)
        train_input_masks = train_input_masks.unsqueeze(0)
        train_output_masks = train_output_masks.unsqueeze(0)

    B, K_train, H, W = train_inputs.shape

    h_list = []

    for k in range(K_train):
        I_k = train_inputs[:, k].unsqueeze(1).float()
        O_k = train_outputs[:, k].unsqueeze(1).float()

        mask_I_k = train_input_masks[:, k]
        mask_O_k = train_output_masks[:, k]

        h_k = example_encoder(
            I_i=I_k,
            O_i=O_k,
            mask_I=mask_I_k,
            mask_O=mask_O_k
        )  # (B,D)
        h_list.append(h_k)

    h = torch.stack(h_list, dim=1)
    pair_mask = None

    C = aggregator(h, mask=pair_mask)
    return C


###############################
#   Evaluation Loop           #
###############################

@torch.no_grad()
def evaluate_final():
    """
    Evaluates:
        - baseline generator (direct logits)
        - sequential + PPO refinement (first test input)
    Prints exact match and pixel accuracy for both.
    """

    # Build models
    generator, example_encoder, aggregator, cond_encoder, lvitm, executor = build_generator()
    critic = build_critic()
    actor, valuer, ppo_refiner = build_ppo(z_dim=64)

    # Load checkpoints if they exist
    gen_p3 = load_checkpoint("checkpoints/generator_phase3_adv.pt", map_location=DEVICE)
    if gen_p3 is not None:
        generator.load_state_dict(gen_p3, strict=False)

    crit_p3 = load_checkpoint("checkpoints/critic_phase3_adv.pt", map_location=DEVICE)
    if crit_p3 is not None:
        critic.load_state_dict(crit_p3, strict=False)

    actor_p4 = load_checkpoint("checkpoints/ppo_actor_phase4.pt", map_location=DEVICE)
    if actor_p4 is not None:
        actor.load_state_dict(actor_p4, strict=False)

    val_p4 = load_checkpoint("checkpoints/ppo_valuer_phase4.pt", map_location=DEVICE)
    if val_p4 is not None:
        valuer.load_state_dict(val_p4, strict=False)

    controller = HybridExecuteController(
        lvitm=lvitm,
        executor=executor,
        cond_encoder=cond_encoder,
        critic=critic
    ).to(DEVICE)

    # Metrics
    total_exact_baseline = 0
    total_exact_ppo = 0
    total_tests = 0

    total_pix_baseline = 0
    total_pix_ppo = 0
    total_pixels = 0

    for batch in arc_loader:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(DEVICE)

        train_inputs       = batch["train_inputs"]
        train_outputs      = batch["train_outputs"]
        train_input_masks  = batch["train_input_masks"]
        train_output_masks = batch["train_output_masks"]

        test_inputs        = batch["test_inputs"]
        test_outputs       = batch["test_outputs"]
        test_input_masks   = batch["test_input_masks"]
        test_output_masks  = batch.get("test_output_masks", test_input_masks)

        # Baseline forward (all tests)
        gen_out = generator(
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            train_input_masks=train_input_masks,
            train_output_masks=train_output_masks,
            test_inputs=test_inputs,
            test_input_masks=test_input_masks,
        )

        logits = gen_out["logits"]  # (1,K_test,C_out,H,W)
        preds = logits.argmax(dim=2).squeeze(0)  # (K_test,H,W)
        targets = test_outputs                    # (K_test,H,W)

        K_test, H, W = targets.shape

        for j in range(K_test):
            pred_b = preds[j]
            target_b = targets[j]

            # If original sizes exist, crop; else assume full
            if "test_original_size" in batch:
                orig_h, orig_w = batch["test_original_size"].tolist()
                pred_b = pred_b[:orig_h, :orig_w]
                target_b = target_b[:orig_h, :orig_w]

            exact_b = (pred_b == target_b).all().item()
            total_exact_baseline += exact_b
            total_tests += 1

            total_pix_baseline += (pred_b == target_b).sum().item()
            total_pixels += target_b.numel()

        # PPO sequential evaluation on first test input only
        if test_inputs.dim() == 3:
            test_inputs = test_inputs.unsqueeze(0)
            test_input_masks = test_input_masks.unsqueeze(0)
            test_outputs = test_outputs.unsqueeze(0)

        B, K_test, H, W = test_inputs.shape
        init_grid = test_inputs[:, 0].unsqueeze(1).float()      # (B,1,H,W)
        init_mask = test_input_masks[:, 0]                       # (B,H,W)
        target_ppo = test_outputs[:, 0]                          # (B,H,W)

        C = compute_context_C(
            example_encoder,
            aggregator,
            train_inputs,
            train_outputs,
            train_input_masks,
            train_output_masks
        )

        final_logits, _ = controller.ppo_rollout_and_update(
            init_grid=init_grid,
            init_mask=init_mask,
            C=C,
            ppo_refiner=ppo_refiner,
            num_steps=3,
            gamma=0.99
        )

        pred_ppo = final_logits.argmax(dim=1)  # (B,H,W)

        pred_ppo_b = pred_ppo[0]
        target_ppo_b = target_ppo[0]

        if "test_original_size" in batch:
            orig_h, orig_w = batch["test_original_size"].tolist()
            pred_ppo_b = pred_ppo_b[:orig_h, :orig_w]
            target_ppo_b = target_ppo_b[:orig_h, :orig_w]

        exact_ppo = (pred_ppo_b == target_ppo_b).all().item()
        total_exact_ppo += exact_ppo

        total_pix_ppo += (pred_ppo_b == target_ppo_b).sum().item()

    exact_acc_baseline = total_exact_baseline / total_tests if total_tests > 0 else 0.0
    exact_acc_ppo = total_exact_ppo / total_tests if total_tests > 0 else 0.0

    pix_acc_baseline = total_pix_baseline / total_pixels if total_pixels > 0 else 0.0
    pix_acc_ppo = total_pix_ppo / total_pixels if total_pixels > 0 else 0.0

    print("\n===============================")
    print("        FINAL EVALUATION       ")
    print("===============================")
    print(f"Baseline Exact Match: {exact_acc_baseline * 100:.2f}%")
    print(f"PPO Exact Match:      {exact_acc_ppo * 100:.2f}%")
    print(f"Baseline Pixel Acc:   {pix_acc_baseline * 100:.2f}%")
    print(f"PPO Pixel Acc:        {pix_acc_ppo * 100:.2f}%")


if __name__ == "__main__":
    evaluate_final()
