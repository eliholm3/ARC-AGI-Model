import argparse
import torch

from checkpoints import save_checkpoint, load_checkpoint

###############################
#   Device                    #
###############################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###############################
#   Imports for each phase    #
###############################

# Phase 1
from phase1 import train_phase1          # def train_phase1(arc_loader) -> generator

# Phase 2
from train_phase2_critic import train_critic_phase2, build_critic
#   def build_critic() -> critic
#   def train_critic_phase2(critic, data_loader) -> critic

# Phase 3
from train_phase3_adv import train_phase3_adversarial, build_generator, build_critic as build_critic_phase3
#   def build_generator() -> generator
#   def build_critic_phase3() -> critic
#   def train_phase3_adversarial(generator, critic, data_loader) -> (generator, critic)

# Phase 4
from train_phase4_ppo import train_phase4_ppo  # def train_phase4_ppo() -> None

# Data loader
from available_functions import arc_loader


###############################
#   Orchestrator              #
###############################

def run_phase1():
    """
    Phase 1: supervised pretraining of generator.
    """
    print("\n===============================")
    print("   PHASE 1: Supervised Train   ")
    print("===============================")

    generator = train_phase1(arc_loader)

    # Save generator weights
    save_checkpoint(generator.state_dict(), "checkpoints/phase1_generator.pt")


def run_phase2():
    """
    Phase 2: critic warmup with WGAN-GP and random fake outputs.
    """
    print("\n===============================")
    print("   PHASE 2: Critic Warmup      ")
    print("===============================")

    critic = build_critic()

    critic = train_critic_phase2(critic, arc_loader)

    save_checkpoint(critic.state_dict(), "checkpoints/critic_phase2_warmup.pt")


def run_phase3():
    """
    Phase 3: joint adversarial training of generator + critic.
    Uses phase1 + phase2 checkpoints if available.
    """
    print("\n===============================")
    print("   PHASE 3: Adversarial Train  ")
    print("===============================")

    # Build fresh models
    generator = build_generator()
    critic = build_critic_phase3()

    # Optionally load phase 1 generator
    ckpt_gen_p1 = load_checkpoint("checkpoints/phase1_generator.pt", map_location=DEVICE)
    if ckpt_gen_p1 is not None:
        generator.load_state_dict(ckpt_gen_p1, strict=False)

    # Optionally load phase 2 critic
    ckpt_crit_p2 = load_checkpoint("checkpoints/critic_phase2_warmup.pt", map_location=DEVICE)
    if ckpt_crit_p2 is not None:
        critic.load_state_dict(ckpt_crit_p2, strict=False)

    generator, critic = train_phase3_adversarial(
        generator=generator,
        critic=critic,
        data_loader=arc_loader
    )

    save_checkpoint(generator.state_dict(), "checkpoints/generator_phase3_adv.pt")
    save_checkpoint(critic.state_dict(), "checkpoints/critic_phase3_adv.pt")


def run_phase4():
    """
    Phase 4: PPO refinement over latent proposals using HybridExecuteController.
    Uses phase3 checkpoints if available.
    """
    print("\n===============================")
    print("   PHASE 4: PPO Refinement     ")
    print("===============================")

    # train_phase4_ppo internally loads generator/critic if needed
    train_phase4_ppo()


###############################
#   Main Entrypoint           #
###############################

def main():
    parser = argparse.ArgumentParser(description="ARC-AGI-2 Training Orchestrator")

    parser.add_argument(
        "--start_phase",
        type=int,
        default=1,
        help="phase to start from (1-4)"
    )
    parser.add_argument(
        "--end_phase",
        type=int,
        default=4,
        help="phase to end at (1-4)"
    )

    args = parser.parse_args()

    for phase in range(args.start_phase, args.end_phase + 1):
        if phase == 1:
            run_phase1()
        elif phase == 2:
            run_phase2()
        elif phase == 3:
            run_phase3()
        elif phase == 4:
            run_phase4()
        else:
            print(f"[orchestrator] Unknown phase: {phase}")


if __name__ == "__main__":
    main()
