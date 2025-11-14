import torch 
from torch import autograd, nn
from typing import Union, Literal, Dict

class WGAN_GP_Trainer:
    def __init__(
            self,
            generator: nn.Module,
            critic: nn.Module,
            gen_optim: torch.optim.Optimizer,
            crit_optim: torch.optim.Optimizer,
            latent_dim=128,
            grad_pen_weight=10.0,
            num_critic_updates=3,
            device: Union[Literal['cuda', 'cpu'], None] = None
            ) -> None:
        self.generator = generator
        self.critic = critic
        self.gen_optim = gen_optim
        self.crit_optim = crit_optim
        self.latent_dim = latent_dim
        self.grad_pen_weight = grad_pen_weight
        self.num_critic_updates = num_critic_updates
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Move to device
        self.generator.to(self.device)
        self.critic.to(self.device)

    def gradient_penalty(
            self, 
            real_images: torch.Tensor, # (B, C, H, W)
            fake_images: torch.Tensor # (B, C, H, W)
            ) -> torch.Tensor: # Scalar
        
        batch_size = real_images.size(0)

        # Each image int he batch gets a random number between 0 & 1 and stored in vector alpha
        # Uniform(0,1) per WGAN-GP paper
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)

        # Calculate set of interpolated images
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff
        interpolated.requires_grad_(True)

        # Critic scores each interpolated image
        interpolated_pred = self.critic(interpolated) # (B, ) or (B,1)
        # Squeeze if critic returns shape (B,1)
        if interpolated_pred.dim() > 1:
            interpolated_pred = interpolated_pred.view(-1)

        # Gradients of prediction calculated with respect to input image
        grads = autograd.grad(
            outputs=interpolated_pred,
            inputs=interpolated,
            grad_outputs=torch.ones_like(
                interpolated_pred, 
                device=self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0] # (B, C, H, W)

        grads = grads.view(batch_size, -1)
        # Calculate L2 norm of gradient vector
        grad_norm = grads.norm(2, dim=1)
        # Return average squared distance between the L2 nrom and I
        gp = ((grad_norm - 1.0) ** 2).mean()
        return gp
    
    
    def train_step(self, real_images: torch.Tensor) -> Dict[str, torch.Tensor]:

        # Enable training mode
        self.generator.train()
        self.critic.train()

        # Move to device 
        real_images = real_images.to(self.device)
        batch_size = real_images.size(0)

        # tracking variables
        crit_loss_tracker = None
        crit_wass_tracker = None
        crit_grad_pen_tracker = None
        gen_loss_tracker = None

        ##########################################################
        #                     Critic updates                     #
        ##########################################################

        for _ in range(self.num_critic_updates):
            # Sample latent vectors
            z = torch.randn(batch_size, self.latent_dim, device=self.device)

            # Generate fake images
            fake_images = self.generator(z)
            # Ensure same dtype as real images
            fake_images = fake_images.to(self.device)

            # Critic predictions
            real_pred = self.critic(real_images)
            fake_pred = self.critic(fake_images.detach())

            # flatten predictions
            if real_pred.dim() > 1:
                real_pred = real_pred.view(-1)
            if fake_pred.dim() > 1:
                fake_pred = fake_pred.view(-1)
            
            # Wasserstein critic loss
            crit_wass = fake_pred.mean() - real_pred.mean()

            # Gradient penalty
            crit_grad_pen = self.gradient_penalty(real_images, fake_images)

            # Critic loss = wieghted sum of penalties + wass
            crit_loss = crit_wass + self.grad_pen_weight * crit_grad_pen

            # Update critic
            self.crit_optim.zero_grad()
            crit_loss.backward()
            self.crit_optim.step()

            # Keep last values for logging
            crit_loss_tracker = crit_loss.item()
            crit_wass_tracker = crit_wass.item()
            crit_grad_pen_tracker = crit_grad_pen.item()
        
        ##########################################################
        #                   Generator update                     #
        ##########################################################

        # Sample latent space
        z = torch.randn(batch_size, self.latent_dim, device=self.device)

        # Generate fake images
        fake_images = self.generator(z)

        # Critic prediction
        fake_pred_for_gen = self.critic(fake_images)

        # Flatten
        if fake_pred_for_gen.dim() > 1:
            fake_pred_for_gen = fake_pred_for_gen.view(-1)

        # Generator loss = E[critic(fake)]
        gen_loss = -fake_pred_for_gen.mean()

        # Update generator
        self.gen_optim.zero_grad()
        gen_loss.backward()
        self.gen_optim.step()

        # Keep last values for logging
        gen_loss_tracker = gen_loss.item()

        return {
            "crit_loss": crit_loss_tracker,
            "crit_wass": crit_wass_tracker,
            "crit_grad_pen": crit_grad_pen_tracker,
            "gen_loss": gen_loss_tracker
        }