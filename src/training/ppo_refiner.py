import torch
import torch.nn.functional as F

from ppo_actor import PPOActor
from ppo_value import PPOValuer


class PPORefiner:
    """
    Performs PPO updates on latent Z proposals.
    """

    def __init__(
        self,
        actor: PPOActor,
        value_fn: PPOValuer,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        lr=1e-4,
    ):
        ###############################
        #   B = batch size            #    
        #   P = num proposals         #
        ###############################

        self.actor = actor
        self.value_fn = value_fn

        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.opt = torch.optim.Adam(
            list(actor.parameters()) + list(value_fn.parameters()),
            lr=lr
        )

    #######################
    #   Gaussian log-prob #
    #######################
    def _log_prob(self, actions, mu, log_std):
        # std
        std = torch.exp(log_std)  # (B, P, z_dim)

        # Gaussian log prob
        # sum over z_dim (the action dimensions)
        logp = -0.5 * (((actions - mu) / std) ** 2 + 2 * log_std + torch.log(2 * torch.pi))
        logp = logp.sum(dim=-1)  # (B, P)

        return logp

    #############################
    #   Refinement step (TTA)   #
    #############################
    @torch.no_grad()
    def refine(self, Z, steps=1, scale=0.1):
        """
        Test-time heuristic refinement:
        small shifts along actor mean.
        """
        Z_new = Z.clone()

        for _ in range(steps):
            mu, _ = self.actor(Z_new)
            Z_new = Z_new + scale * mu

        return Z_new

    ############################
    #        PPO UPDATE        #
    ############################
    def update(self, Z, actions, old_logp, returns, advantages):
        """
        Z:        (B, P, z_dim)
        actions:  (B, P, z_dim)
        old_logp: (B, P)
        returns:  (B, P)
        advantages: (B, P)
        """

        # New policy
        mu, log_std = self.actor(Z)  # (B, P, z_dim)

        # New log prob
        logp = self._log_prob(actions, mu, log_std)  # (B, P)

        # PPO ratio
        ratio = torch.exp(logp - old_logp)  # (B, P)

        # Clipped surrogate objective
        clipped_ratio = torch.clamp(
            ratio,
            1 - self.clip_ratio,
            1 + self.clip_ratio
        )
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages

        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        values = self.value_fn(Z)  # (B, P)
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus
        entropy = 0.5 * (log_std.exp() * torch.sqrt(torch.tensor(2 * torch.pi * torch.e))).mean()
        entropy_loss = -self.entropy_coef * entropy

        loss = policy_loss + self.value_coef * value_loss + entropy_loss

        # Update step
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }
