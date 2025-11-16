import torch
import matplotlib.pyplot as plt
import numpy as np

from src.training.evaluate_final import (
    build_generator,
    build_critic,
    build_ppo,
    compute_context_C
)

from src.inference.execution_controller import HybridExecuteController
from src.data_pipeline.dataloader import ARCDataModule

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================
# Utility: Convert integer grid → RGB for visualization
# ===========================================================
ARC_COLORS = [
    (0, 0, 0),        # 0 black (padding)
    (0, 0, 255),      # 1 blue
    (0, 255, 0),      # 2 green
    (255, 0, 0),      # 3 red
    (255, 255, 0),    # 4 yellow
    (255, 165, 0),    # 5 orange
    (255, 0, 255),    # 6 magenta
    (0, 255, 255),    # 7 cyan
    (128, 0, 128),    # 8 purple
    (165, 42, 42),    # 9 brown
    (255, 255, 255)   # 10 white (your num_classes=11)
]


def grid_to_rgb(grid):
    """
    grid: (H, W) integer tensor
    returns: (H, W, 3) uint8 numpy array
    """
    H, W = grid.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            rgb[i, j] = ARC_COLORS[grid[i, j]]
    return rgb



# ===========================================================
# MAIN INFERENCE FUNCTION
# ===========================================================
def run_inference_on_sample(batch, generator, controller, actor, valuer):
    """
    Runs:
      - baseline generator prediction
      - sequential PPO-enhanced prediction
      - visualization
    """

    # Move to device
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

    # Compute context C from example pairs
    C = compute_context_C(
        example_encoder=generator.example_pair_encoder,
        aggregator=generator.aggregator,
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        train_input_masks=train_input_masks,
        train_output_masks=train_output_masks
    )

    # =======================================================
    # (A) BASELINE GENERATOR PREDICTION (Phase1–3)
    # =======================================================
    gen_out = generator(
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        train_input_masks=train_input_masks,
        train_output_masks=train_output_masks,
        test_inputs=test_inputs,
        test_input_masks=test_input_masks
    )
    baseline_logits = gen_out["logits"]  # (1,K_test,C,H,W)
    baseline_pred = baseline_logits.argmax(dim=2).squeeze(0)  # (K_test,H,W)

    # =======================================================
    # (B) PPO-ENHANCED SEQUENTIAL EXECUTION CONTROLLER
    # =======================================================
    if test_inputs.dim() == 3:
        test_inputs = test_inputs.unsqueeze(0)
        test_input_masks = test_input_masks.unsqueeze(0)
        test_outputs = test_outputs.unsqueeze(0)

    init_grid = test_inputs[:, 0].unsqueeze(1).float()  # (1,1,H,W)
    init_mask = test_input_masks[:, 0]                  # (1,H,W)
    target = test_outputs[:, 0]                         # (1,H,W)

    final_logits, history = controller.apply_sequential(
        init_grid=init_grid,
        init_mask=init_mask,
        C=C,
        examples=None,
        num_steps=3
    )

    ppo_pred = final_logits.argmax(dim=1)[0]  # (H,W)

    # =======================================================
    # Visualization
    # =======================================================
    orig_h, orig_w = batch["test_original_size"].tolist()
    baseline_vis = baseline_pred[0, :orig_h, :orig_w].cpu()
    ppo_vis = ppo_pred[:orig_h, :orig_w].cpu()
    target_vis = target[0, :orig_h, :orig_w].cpu()
    test_in_vis = test_inputs[0, 0, :orig_h, :orig_w].cpu()

    # Plot
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].set_title("Test Input")
    axs[0].imshow(grid_to_rgb(test_in_vis))

    axs[1].set_title("Baseline Prediction")
    axs[1].imshow(grid_to_rgb(baseline_vis))

    axs[2].set_title("PPO Final Prediction")
    axs[2].imshow(grid_to_rgb(ppo_vis))

    axs[3].set_title("Ground Truth Output")
    axs[3].imshow(grid_to_rgb(target_vis))

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    return baseline_vis, ppo_vis, target_vis



# ===========================================================
# ENTRYPOINT
# ===========================================================
if __name__ == "__main__":
    print("Loading models...")

    # Build everything
    generator, example_encoder, aggregator, cond_encoder, lvitm, executor = build_generator()
    critic = build_critic()
    actor, valuer, ppo_refiner = build_ppo(z_dim=64)

    # Load checkpoints
    generator.load_state_dict(torch.load("checkpoints/generator_phase3_adv.pt", map_location=DEVICE), strict=False)
    critic.load_state_dict(torch.load("checkpoints/critic_phase3_adv.pt", map_location=DEVICE), strict=False)
    actor.load_state_dict(torch.load("checkpoints/ppo_actor_phase4.pt", map_location=DEVICE), strict=False)
    valuer.load_state_dict(torch.load("checkpoints/ppo_valuer_phase4.pt", map_location=DEVICE), strict=False)

    # Execution Controller
    controller = HybridExecuteController(
        lvitm=lvitm,
        executor=executor,
        cond_encoder=cond_encoder,
        critic=critic
    ).to(DEVICE)

    # Load dataset (small subset)
    print("Loading dataset...")
    data_module = ARCDataModule(
        dir_path="./src/data_pipeline/ARC_data/data/training",
        batch_size=1,
        shuffle=False,
        pad_value=0
    ).prepare()

    # Limit to 3 samples for testing
    data_module.dataset.data = data_module.dataset.data[:3]
    loader = data_module.get_loader()

    # ===========================================================
    # Run inference on 3 tasks
    # ===========================================================
    for i, batch in enumerate(loader):
        print(f"\n=========== SAMPLE {i} ===========")
        baseline, ppo, target = run_inference_on_sample(
            batch, generator, controller, actor, valuer
        )

    print("\nInference complete.")
