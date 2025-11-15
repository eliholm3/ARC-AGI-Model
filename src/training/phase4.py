import torch
import torch.nn.functional as F

from src.training.ppo_actor import PPOActor
from src.training.ppo_value import PPOValuer
from src.training.ppo_refiner import PPORefiner

from src.architecture.context_encoding.example_pair_encoder import ExamplePairEncoder
from src.architecture.context_encoding.example_pair_aggregator import ExamplePairAggregator
from src.architecture.context_encoding.conditional_encoder import ConditionalTestInputEncoder
from src.architecture.ViT.body import VisionTransformer
from src.architecture.LViTM.body import LargeVisionTransformerModel
from src.architecture.executor.executor import Executor
from src.architecture.adViT.critic import AdversarialVisionTransformer
from src.data_pipeline.dataloader import ARCDataModule

from src.inference.execution_controller import HybridExecuteController


###############################
#   Hyperparameters           #
###############################

PPO_EPOCHS = 3
PPO_STEPS = 3
PPO_GAMMA = 0.99

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10


###############################
#   Build Generator Modules   #
###############################

def build_generator_components():
    """
    Rebuilds generator components individually (not the ARCGenerator wrapper).
    Used to compute context C and supply modules to the controller.
    """

    ###########################
    #   Shared Hyperparams    #
    ###########################

    img_size   = 30
    patch_size = 1
    embed_dim  = 128
    num_heads  = 4
    depth_vit  = 6
    mlp_dim    = 256
    z_dim      = 64
    num_props  = 4

    ###############################
    #   Vision Transformers       #
    ###############################

    # For example pairs (I, O) -> 2 channels
    vit_pair = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth_vit,
        mlp_dim=mlp_dim,
        in_channels=2
    ).to(DEVICE)

    # For test input I_test -> 1 channel
    vit_test = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth_vit,
        mlp_dim=mlp_dim,
        in_channels=1
    ).to(DEVICE)

    ###########################################
    #   Context Encoders (Pair + Aggregator)  #
    ###########################################

    # Example pair encoder uses ViT with 2 input channels
    example_encoder = ExamplePairEncoder(vit_pair).to(DEVICE)

    # Aggregator uses the same embedding dimension as ViTs
    aggregator = ExamplePairAggregator(
        embed_dim=vit_pair.c_token.size(-1)
    ).to(DEVICE)

    ###########################################
    #   Conditional Test Input Encoder        #
    ###########################################

    # Conditional encoder uses the 1-channel ViT
    cond_encoder = ConditionalTestInputEncoder(vit_test).to(DEVICE)

    ###############################
    #   Latent Proposal Model     #
    ###############################

    lvitm = LargeVisionTransformerModel(
        embed_dim=vit_pair.c_token.size(-1),
        num_heads=4,
        mlp_dim=256,
        depth=8,
        num_proposals=num_props,
        z_dim=z_dim
    ).to(DEVICE)

    ###############################
    #   Executor                  #
    ###############################

    executor = Executor(
        embed_dim=vit_pair.c_token.size(-1),
        num_heads=4,
        mlp_dim=256,
        depth=4,
        z_dim=z_dim,
        hidden_channels=64,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    return example_encoder, aggregator, cond_encoder, lvitm, executor



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
#   Compute Context C         #
###############################

def compute_context_C(
        example_encoder: ExamplePairEncoder,
        aggregator: ExamplePairAggregator,
        train_inputs: torch.Tensor,        # (K_train,H,W)
        train_outputs: torch.Tensor,       # (K_train,H,W)
        train_input_masks: torch.Tensor,   # (K_train,H,W)
        train_output_masks: torch.Tensor   # (K_train,H,W)
):
    """
    Encodes all training example pairs and aggregates into context C.
    """

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

    h = torch.stack(h_list, dim=1)   # (B,K_train,D)
    pair_mask = None

    C = aggregator(h, mask=pair_mask)  # (B,D)
    return C


###############################
#   Phase 4 PPO Training      #
###############################

def train_phase4_ppo(data_loader: ARCDataModule):
    # Build components
    example_encoder, aggregator, cond_encoder, lvitm, executor = build_generator_components()
    critic = build_critic()

    # Load Phase 3 checkpoints if available
    try:
        gen_state = torch.load("generator_phase3_adv.pt", map_location=DEVICE)
        print("Loaded generator_phase3_adv.pt into generator components (partial load).")
        # You can do partial loads into submodules here if you saved them modularly.
    except FileNotFoundError:
        print("Phase 3 generator checkpoint not found. Using fresh generator components.")

    try:
        critic.load_state_dict(torch.load("critic_phase3_adv.pt", map_location=DEVICE))
        print("Loaded critic_phase3_adv.pt.")
    except FileNotFoundError:
        print("Phase 3 critic checkpoint not found. Using fresh critic.")

    # Controller with critic
    controller = HybridExecuteController(
        lvitm=lvitm,
        executor=executor,
        cond_encoder=cond_encoder,
        critic=critic
    ).to(DEVICE)

    # PPO modules
    z_dim = 64   # must match LViTM z_dim
    actor = PPOActor(z_dim=z_dim, embed_dim=256).to(DEVICE)
    value_fn = PPOValuer(z_dim=z_dim, embed_dim=256).to(DEVICE)
    ppo_refiner = PPORefiner(actor=actor, value_fn=value_fn, lr=1e-4)

    for epoch in range(PPO_EPOCHS):
        print(f"\n=== Phase 4 PPO Epoch {epoch + 1}/{PPO_EPOCHS} ===")

        for batch_idx, batch in enumerate(data_loader):

            ###############################
            #   Move batch to device      #
            ###############################

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

            # Compute context C
            C = compute_context_C(
                example_encoder=example_encoder,
                aggregator=aggregator,
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                train_input_masks=train_input_masks,
                train_output_masks=train_output_masks
            )  # (1,D)

            # Pick first test input as initial grid
            if test_inputs.dim() == 3:
                test_inputs = test_inputs.unsqueeze(0)
                test_input_masks = test_input_masks.unsqueeze(0)
                test_outputs = test_outputs.unsqueeze(0)

            B, K_test, H, W = test_inputs.shape
            init_grid = test_inputs[:, 0].unsqueeze(1).float()       # (B,1,H,W)
            init_mask = test_input_masks[:, 0]                        # (B,H,W)
            target = test_outputs[:, 0]                               # (B,H,W)

            ###############################
            #   PPO Rollout & Update      #
            ###############################

            final_logits, ppo_stats = controller.ppo_rollout_and_update(
                init_grid=init_grid,
                init_mask=init_mask,
                C=C,
                ppo_refiner=ppo_refiner,
                num_steps=PPO_STEPS,
                gamma=PPO_GAMMA
            )

            # Optional: supervised signal on final prediction
            pred_loss = F.cross_entropy(
                final_logits,          # (B,C_out,H,W)
                target.long()          # (B,H,W)
            )

            pred_loss.backward()
            # NOTE: If you want to train executor/LViTM jointly with PPO, you can
            # attach an optimizer here and step it. For now, PPORefiner.update()
            # already steps the actor/value networks.

            print(f"Epoch {epoch+1}, Batch {batch_idx}: "
                  f"PPO loss={ppo_stats['loss']:.4f}, "
                  f"policy={ppo_stats['policy_loss']:.4f}, "
                  f"value={ppo_stats['value_loss']:.4f}, "
                  f"entropy={ppo_stats['entropy']:.4f}")
            
    return actor, value_fn


if __name__ == "__main__":
    train_phase4_ppo()
