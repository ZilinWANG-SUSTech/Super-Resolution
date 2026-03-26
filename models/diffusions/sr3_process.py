import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class SR3DiffusionProcess(nn.Module):
    """
    Continuous-Time DDPM as proposed in the SR3 paper.
    Uses continuous noise level conditioning (gamma) instead of discrete timesteps.
    """
    def __init__(self, train_steps: int = 1000, sample_steps: int = 50):
        super().__init__()
        self.train_steps = train_steps
        self.sample_steps = sample_steps

        # 1. Register computation variables for training (Linear Schedule)
        betas = torch.linspace(start=1e-4, end=0.005, steps=self.train_steps)
        alphas = 1. - betas
        self.register_buffer("alphas_bar", torch.cumprod(alphas, dim=0))
        
        # 2. Register computation variables for sampling (Algorithm 2 in SR3)
        sample_betas = torch.linspace(start=1e-4, end=0.1, steps=self.sample_steps)
        sample_alphas = 1. - sample_betas
        self.register_buffer("sample_alphas_bar", torch.cumprod(sample_alphas, dim=0))
        self.register_buffer("sample_one_over_sqrt_alphas", torch.sqrt(1. / sample_alphas))
        self.register_buffer("sample_betas_over_sqrt_one_minus_alphas_bar", sample_betas / torch.sqrt(1. - self.sample_alphas_bar))
        self.register_buffer("sample_sigmas", torch.sqrt(sample_betas))

    def _extract(self, values: torch.Tensor, times: torch.Tensor, dimension_num: int):
        """Extracts values from a 1D tensor for a batch of indices and reshapes for broadcasting."""
        B, *_ = times.shape
        selected_values = torch.gather(values, dim=0, index=times)
        return selected_values.reshape((B, *[1 for _ in range(dimension_num-1)]))

    def compute_training_loss(self, net: nn.Module, x_start: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion process matching SR3 Algorithm 1.
        x_start and cond are expected to be in [-1, 1] range.
        """
        B, *_ = x_start.shape
        dimension_num = len(x_start.shape)
        
        # Sample random integer timestep
        time = torch.randint(low=1, high=self.train_steps, size=(B, ), device=x_start.device)
        
        # Calculate continuous gamma (noise level)
        gamma_high = self._extract(self.alphas_bar, time-1, dimension_num)
        gamma_low = self._extract(self.alphas_bar, time, dimension_num)
        
        # Random interpolation for continuous noise injection
        gamma = (gamma_high - gamma_low) * torch.rand_like(gamma_high) + gamma_low

        # Sample pure noise
        epsilon = torch.randn_like(x_start)

        # Add noise to the clean High-Res image
        x_t = torch.sqrt(gamma) * x_start + torch.sqrt(1 - gamma) * epsilon

        # Concatenate noisy image and condition (e.g., Bicubic LR)
        net_input = torch.cat((x_t, cond), dim=1)
        
        # SR3 UNet expects sqrt(gamma) as the time/noise conditioning signal
        noise_level = torch.sqrt(gamma).squeeze() 
        noise_pred = net(net_input, noise_level)
        
        # MSE Loss
        loss = F.mse_loss(noise_pred, epsilon, reduction='mean')
        return loss

    @torch.no_grad()
    def sample(self, net: nn.Module, shape: tuple, device: torch.device, cond: torch.Tensor) -> torch.Tensor:
        """
        Reverse diffusion process matching SR3 Algorithm 2.
        """
        B = shape[0]
        dimension_num = len(shape)

        # Start from pure Gaussian noise
        x_t = torch.randn(shape, device=device)
        
        for step in tqdm(reversed(range(self.sample_steps)), total=self.sample_steps, desc="SR3 Sampling", leave=False):
            
            time = step * x_t.new_ones((B, ), dtype=torch.int64)
            gamma = self._extract(self.sample_alphas_bar, time, dimension_num)
            
            # Predict noise residual
            net_input = torch.cat((x_t, cond), dim=1)
            noise_level = torch.sqrt(gamma).squeeze()
            epsilon = net(net_input, noise_level)

            # Retrieve scheduled constants for current step
            one_over_sqrt_alpha = self._extract(self.sample_one_over_sqrt_alphas, time, dimension_num)
            beta_over_sqrt_one_minus_alpha_bar = self._extract(self.sample_betas_over_sqrt_one_minus_alphas_bar, time, dimension_num)
            sigma = self._extract(self.sample_sigmas, time, dimension_num)
            
            # Add random noise if not the last step (Langevin dynamics)
            z = torch.randn_like(x_t) if step > 0 else 0
            
            # Denoise step
            x_t = one_over_sqrt_alpha * (x_t - beta_over_sqrt_one_minus_alpha_bar * epsilon) + sigma * z

        # The engine will automatically clamp this output and convert it to [0, 1]
        return x_t