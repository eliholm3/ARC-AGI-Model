import torch
from torch import nn, autograd


class ARCWGAN_GP_Trainer:
    """
    Trains generator-critic relation
    """

    def __init__(
        self,
        example_pair_encoder: nn.Module,
        aggregator: nn.Module,
        cond_encoder: nn.Module,
        lvittm: nn.Module,
        executor: nn.Module,
        critic: nn.Module,
        gen_optim: torch.optim.Optimizer,
        crit_optim: torch.optim.Optimizer,
        grad_pen_weight: float = 10.0,
        num_critic_updates: int = 3,
        device: torch.device | None = None,
        sup_loss_weight: float = 0.0,  # set >0 to add supervised loss
    ):
        self.example_pair_encoder = example_pair_encoder
        self.aggregator = aggregator
        self.cond_encoder = cond_encoder
        self.lvittm = lvittm
        self.executor = executor
        self.critic = critic

        self.gen_optim = gen_optim
        self.crit_optim = crit_optim
        self.grad_pen_weight = grad_pen_weight
        self.num_critic_updates = num_critic_updates
        self.sup_loss_weight = sup_loss_weight

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # move modules to device
        for m in [
            self.example_pair_encoder,
            self.aggregator,
            self.cond_encoder,
            self.lvittm,
            self.executor,
            self.critic,
        ]:
            m.to(self.device)

        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")

    #################################
    #   Context (C) from examples   #
    #################################

    def _compute_context(
            self, 
            train_inputs,  # (B, N_train, H, W)
            train_outputs,  # (B, N_train, H, W)
            train_orig_size  # (B, 2)
    ):  # (B, D)
        B, N, H, W = train_inputs.shape

        # Flatten across examples
        I_flat = train_inputs.view(B * N, 1, H, W).to(self.device)
        O_flat = train_outputs.view(B * N, 1, H, W).to(self.device)

        train_in_masks = batch["train_in_masks"].to(self.device)     # (B, N, H, W)
        train_out_masks = batch["train_out_masks"].to(self.device)   # (B, N, H, W)

        # Flatten masks
        mask_I_flat = train_in_masks.view(B * N, H, W)
        mask_O_flat = train_out_masks.view(B * N, H, W)


        # Encode each pair to h_i
        h_flat = self.example_pair_encoder(
            I_flat, O_flat,
            mask_I=masks,
            mask_O=masks
        )  # (B*N, D)

        # Reshape to (B, N, D)
        h = h_flat.view(B, N, -1)

        # Aggregate to context C
        C = self.aggregator(h)  # (B, D)
        return C

    # -------------------------
    # Generator forward: I_test -> O_fake
    # -------------------------
    def _generator_forward(self, I_test, test_orig_size, C):
        """
        I_test:        (B, 1, H, W)
        test_orig_size: (B, 2)
        C:             (B, D)
        returns: O_fake (B, num_classes, H, W), Z (B, T, z_dim)
        """
        B, _, H, W = I_test.shape
        I_test = I_test.to(self.device)

        # Build mask for test input
        orig_sizes = test_orig_size.to("cpu")
        masks = [generate_valid_mask(H, W, orig_sizes[b]) for b in range(B)]
        mask_test = torch.stack(masks, dim=0).to(self.device)  # (B,H,W)

        # Conditional encoding of test input
        tokens, kpm = self.cond_encoder(I_test, mask_test, C)

        # LViTM proposals
        Z = self.lvittm(C, tokens, kpm)  # (B, T, z_dim)

        # For now, just take first proposal z_0 per batch element
        z0 = Z[:, 0, :]  # (B, z_dim)

        # Executor applies z0
        O_fake = self.executor(I_test, z0)  # (B, num_classes, H, W)
        return O_fake, Z, mask_test

    # -------------------------
    # Gradient penalty
    # -------------------------
    def _gradient_penalty(self, I_test, real_out, fake_out, mask_test, C):
        """
        I_test:   (B, 1, H, W)
        real_out: (B, C_out, H, W)
        fake_out: (B, C_out, H, W)
        mask_test: (B, H, W)
        C:       (B, D)
        """
        B = real_out.size(0)
        device = self.device

        alpha = torch.rand(B, 1, 1, 1, device=device)
        interpolated = real_out + alpha * (fake_out - real_out)
        interpolated.requires_grad_(True)

        # Critic score on interpolated outputs
        scores = self.critic(
            I_in=I_test,
            O_pred=interpolated,
            mask_in=mask_test,
            mask_out=mask_test,
            z=None,
            C=C
        )  # (B,)

        # Compute gradient of scores wrt interpolated
        grads = autograd.grad(
            outputs=scores.sum(),
            inputs=interpolated,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]  # (B, C_out, H, W)

        grads = grads.view(B, -1)
        grad_norm = grads.norm(2, dim=1)  # (B,)
        gp = ((grad_norm - 1.0) ** 2).mean()
        return gp

    # -------------------------
    # One training step on a batch
    # -------------------------
    def train_step(self, batch):
        """
        batch: dict from DataLoader, e.g.:
          "train_inputs":        (B, N_train, H, W)
          "train_outputs":       (B, N_train, H, W)
          "test_inputs":         (B, N_test, H, W)
          "test_outputs":        (B, N_test, H, W)
          "train_original_size": (B, 2)
          "test_original_size":  (B, 2)
        """

        # Put batch on device & prepare shapes
        train_inputs = batch["train_inputs"].to(self.device)      # (B,N_t,H,W)
        train_outputs = batch["train_outputs"].to(self.device)
        test_inputs = batch["test_inputs"].to(self.device)        # (B,N_test,H,W)
        test_outputs = batch["test_outputs"].to(self.device)
        train_orig_size = batch["train_original_size"].to(self.device)  # (B,2)
        test_orig_size = batch["test_original_size"].to(self.device)

        B, N_test, H, W = test_inputs.shape
        assert N_test == 1, "Current trainer assumes exactly 1 test pair per task."
        I_test = test_inputs[:, 0].unsqueeze(1)   # (B,1,H,W)
        O_real = test_outputs[:, 0]               # (B,H,W)
        O_real = O_real.unsqueeze(1)              # (B,1,H,W) or adapt to num_classes

        # -----------------------
        # Compute context C once
        # -----------------------
        C = self._compute_context(train_inputs, train_outputs, train_orig_size)  # (B,D)

        # -----------------------
        # Critic updates
        # -----------------------
        self.critic.train()
        self.example_pair_encoder.eval()
        self.aggregator.eval()
        self.cond_encoder.eval()
        self.lvittm.eval()
        self.executor.eval()

        crit_loss_last = crit_wass_last = crit_gp_last = None

        for _ in range(self.num_critic_updates):
            # Fake output (no grad into generator here)
            with torch.no_grad():
                O_fake, Z, mask_test = self._generator_forward(I_test, test_orig_size, C)
            # Critic scores
            real_scores = self.critic(
                I_in=I_test,
                O_pred=O_real,
                mask_in=mask_test,
                mask_out=mask_test,
                z=None,
                C=C
            )  # (B,)
            fake_scores = self.critic(
                I_in=I_test,
                O_pred=O_fake.detach(),
                mask_in=mask_test,
                mask_out=mask_test,
                z=None,
                C=C
            )  # (B,)

            crit_wass = fake_scores.mean() - real_scores.mean()
            crit_gp = self._gradient_penalty(I_test, O_real, O_fake.detach(), mask_test, C)
            crit_loss = crit_wass + self.grad_pen_weight * crit_gp

            self.crit_optim.zero_grad()
            crit_loss.backward()
            self.crit_optim.step()

            crit_loss_last = crit_loss.item()
            crit_wass_last = crit_wass.item()
            crit_gp_last = crit_gp.item()

        # -----------------------
        # Generator update
        # -----------------------
        self.critic.eval()
        self.example_pair_encoder.train()
        self.aggregator.train()
        self.cond_encoder.train()
        self.lvittm.train()
        self.executor.train()

        # Fresh fake output (with grad)
        O_fake, Z, mask_test = self._generator_forward(I_test, test_orig_size, C)

        # Adversarial loss
        fake_scores_for_gen = self.critic(
            I_in=I_test,
            O_pred=O_fake,
            mask_in=mask_test,
            mask_out=mask_test,
            z=None,
            C=C
        )  # (B,)
        gen_adv_loss = -fake_scores_for_gen.mean()

        # Optional supervised loss (e.g., cross-entropy over colors)
        # assumes O_fake is (B, num_classes, H, W) and O_real is color indices (B,H,W)
        gen_sup_loss = torch.tensor(0.0, device=self.device)
        if self.sup_loss_weight > 0.0:
            # Here you might need your true color labels, not binary grids
            # Placeholder: treat O_real as indices already
            target = O_real.squeeze(1).long()  # (B,H,W)
            gen_sup_loss = self.ce_loss(O_fake, target)

        gen_loss = gen_adv_loss + self.sup_loss_weight * gen_sup_loss

        self.gen_optim.zero_grad()
        gen_loss.backward()
        self.gen_optim.step()

        return {
            "crit_loss": crit_loss_last,
            "crit_wass": crit_wass_last,
            "crit_grad_pen": crit_gp_last,
            "gen_loss": gen_loss.item(),
            "gen_adv_loss": gen_adv_loss.item(),
            "gen_sup_loss": gen_sup_loss.item() if self.sup_loss_weight > 0 else 0.0,
        }
