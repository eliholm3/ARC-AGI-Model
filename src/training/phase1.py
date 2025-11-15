import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from src.training.utils_debug import report_param_stats

# Import your modules
from src.inference.generator import ARCGenerator

# Encoders & components
from src.architecture.context_encoding.example_pair_encoder import ExamplePairEncoder
from src.architecture.context_encoding.example_pair_aggregator import ExamplePairAggregator
from src.architecture.context_encoding.conditional_encoder import ConditionalTestInputEncoder
from src.architecture.LViTM.body import LargeVisionTransformerModel
from src.architecture.executor.executor import Executor
from src.architecture.ViT.body import VisionTransformer

# Training constants
LR = 1e-4
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 11


##############################################################
#   Build MODEL for PHASE 1 using TWO separate ViTs          #
##############################################################
def build_model():

    ##############################################################
    #   1. Vision Transformers                                   #
    #      vit_pair = (I, O) example pairs, 2 channels           #
    #      vit_test = I_test only, 1 channel                     #
    ##############################################################
    img_size   = 30
    patch_size = 1
    embed_dim  = 128
    num_heads  = 4
    depth_vit  = 6
    mlp_dim    = 256
    z_dim      = 64
    num_props  = 4

    # Example pair ViT: (I_i, O_i) → 2 channels
    vit_pair = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth_vit,
        mlp_dim=mlp_dim,
        in_channels=2
    ).to(DEVICE)

    # Test grid ViT: (I_test) → 1 channel
    vit_test = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth_vit,
        mlp_dim=mlp_dim,
        in_channels=1
    ).to(DEVICE)

    ##############################################################
    #   2. Context encoding components                           #
    ##############################################################
    example_encoder = ExamplePairEncoder(vit_pair).to(DEVICE)

    aggregator = ExamplePairAggregator(
        embed_dim=vit_pair.c_token.size(-1)
    ).to(DEVICE)

    cond_encoder = ConditionalTestInputEncoder(vit_test).to(DEVICE)

    ##############################################################
    #   3. LVITM (reasoning / proposal generation)               #
    ##############################################################
    lvitm = LargeVisionTransformerModel(
        embed_dim=vit_pair.c_token.size(-1),
        num_heads=4,
        mlp_dim=256,
        depth=8,
        num_proposals=num_props,
        z_dim=z_dim
    ).to(DEVICE)

    ##############################################################
    #   4. Executor                                              #
    ##############################################################
    executor = Executor(
        embed_dim=vit_pair.c_token.size(-1),
        num_heads=4,
        mlp_dim=256,
        depth=4,
        z_dim=z_dim,
        hidden_channels=64,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    ##############################################################
    #   5. Wrap everything into ARCGenerator                     #
    ##############################################################
    generator = ARCGenerator(
        example_pair_encoder=example_encoder,
        aggregator=aggregator,
        cond_encoder=cond_encoder,
        lvitm=lvitm,
        executor=executor
    ).to(DEVICE)

    return generator


##############################################################
#   Phase 1 Supervised Training                              #
##############################################################
def train_phase1(arc_loader):

    generator = build_model()
    optimizer = Adam(generator.parameters(), lr=LR)

    generator.train()

    for epoch in tqdm(range(EPOCHS), "Epoch:"):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

        total_loss = 0.0
        count = 0

        for batch in tqdm(arc_loader, "Batch:"):
            ############################################################
            #   Move batch to device                                  #
            ############################################################
            for k, v in tqdm(batch.items(), "Sample:"):
                if torch.is_tensor(v):
                    batch[k] = v.to(DEVICE)

            train_inputs       = batch["train_inputs"]
            train_outputs      = batch["train_outputs"]
            train_input_masks  = batch["train_input_masks"]
            train_output_masks = batch["train_output_masks"]

            test_inputs        = batch["test_inputs"]
            test_outputs       = batch["test_outputs"]
            test_input_masks   = batch["test_input_masks"]
            test_output_masks  = batch["test_output_masks"] 

            ############################################################
            #   Forward through ARCGenerator                           #
            ############################################################
            out = generator(
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                train_input_masks=train_input_masks,
                train_output_masks=train_output_masks,
                test_inputs=test_inputs,
                test_input_masks=test_input_masks,
            )

            logits = out["logits"]  # (B, K_test, C_out, H, W)
            B, K_test, C_out, H, W = logits.shape

            ############################################################
            #   Clean CE Loss (NO NaNs)
            ############################################################
            logits_flat = logits.view(B * K_test, C_out, H, W)
            target_flat = test_outputs.view(B * K_test, H, W)

            # Correct padded-area masking
            PAD_TOKEN = -100
            targets = target_flat.clone()
            pad_mask = ~test_output_masks.view(B*K_test, H, W).bool()
            targets[pad_mask] = PAD_TOKEN

            # Compute valid-mask CE loss
            per_pixel = F.cross_entropy(
                logits_flat,
                targets,
                ignore_index=PAD_TOKEN,
                reduction="none"
            )

            valid_mask = (targets != PAD_TOKEN).float()
            loss = (per_pixel * valid_mask).sum() / valid_mask.sum()

            ############################################################
            #   Backprop without spam
            ############################################################
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

    return generator


##############################################################
#   Script Entry Point                                        #
# ##############################################################
# if __name__ == "__main__":
#     from src.data_pipeline.dataloader import ARCDataModule

#     model = train_phase1(ARCDataModule)
#     torch.save(model.state_dict(), "phase1_generator.pt")
#     print("Saved Phase 1 generator.")
