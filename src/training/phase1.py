import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# Import your modules
from ..inference.generator import ARCGenerator

# Example imports (adjust paths to match your project)
from available_functions import (
    ExamplePairEncoder,
    ExamplePairAggregator,
    ConditionalTestInputEncoder,
    LargeVisionTransformerModel,
    Executor,
)

# -------------------------------
# Hyperparameters (adjust later)
# -------------------------------
LR = 1e-4
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Build model components
# -------------------------------
def build_model():
    # Shared Vision Transformer must already exist in your code.
    # For example, assume your ViT is built like this (adapt to your actual file):
    from available_functions import VisionTransformer
    
    vit = VisionTransformer(
        img_size=30,
        patch_size=1,
        embed_dim=128,
        num_heads=4,
        depth=6,
        mlp_dim=256,
        in_channels=2  # because ExamplePairEncoder concatenates I and O
    )

    example_encoder = ExamplePairEncoder(vit).to(DEVICE)
    aggregator = ExamplePairAggregator(embed_dim=vit.c_token.size(-1)).to(DEVICE)

    cond_encoder = ConditionalTestInputEncoder(vit).to(DEVICE)

    lvitm = LargeVisionTransformerModel(
        embed_dim=vit.c_token.size(-1),
        num_heads=4,
        mlp_dim=256,
        depth=8,
        num_proposals=4,
        z_dim=64
    ).to(DEVICE)

    executor = Executor(
        embed_dim=vit.c_token.size(-1),
        num_heads=4,
        mlp_dim=256,
        depth=4,
        z_dim=64,
        hidden_channels=64,
        num_classes=10  # ARC colors {0..9}
    ).to(DEVICE)

    generator = ARCGenerator(
        example_pair_encoder=example_encoder,
        aggregator=aggregator,
        cond_encoder=cond_encoder,
        lvitm=lvitm,
        executor=executor
    ).to(DEVICE)

    return generator


# -------------------------------
# Training Loop (Phase 1)
# -------------------------------
def train_phase1(arc_loader):

    generator = build_model()
    optimizer = Adam(generator.parameters(), lr=LR)

    generator.train()

    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

        total_loss = 0.0
        count = 0

        for batch in arc_loader:
            # -------------------------------
            # Move batch to device
            # -------------------------------
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

            # -------------------------------
            # Forward through ARCGenerator
            # -------------------------------
            out = generator(
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                train_input_masks=train_input_masks,
                train_output_masks=train_output_masks,
                test_inputs=test_inputs,
                test_input_masks=test_input_masks,
            )

            logits = out["logits"]   # (B, K_test, num_classes, H, W)
            B, K_test, C_out, H, W = logits.shape

            # -------------------------------
            # Prepare CE loss inputs
            # -------------------------------
            # logits: (B*K_test, C_out, H, W)
            logits_flat = logits.view(B * K_test, C_out, H, W)

            # targets: (B, K_test, H, W) -> (B*K_test, H, W)
            target_flat = test_outputs.view(B * K_test, H, W)

            # -------------------------------
            # Compute pixelwise classification loss
            # -------------------------------
            loss = F.cross_entropy(logits_flat, target_flat)

            # -------------------------------
            # Backprop
            # -------------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

    # Return trained model
    return generator


# -------------------------------------------------
# Example usage (in your training script)
# -------------------------------------------------
if __name__ == "__main__":
    # arc_loader must already exist from your previous code.
    from available_functions import arc_loader

    model = train_phase1(arc_loader)
    torch.save(model.state_dict(), "phase1_generator.pt")
    print("Saved Phase 1 generator.")
