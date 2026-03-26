import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseDiffusionProcess(nn.Module, ABC):
    """
    Base class for all diffusion processes (DDPM, DDIM, LDM, etc.).
    This ensures a unified API for the SRDiffusionEngine.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def compute_training_loss(self, net: nn.Module, x_start: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Compute the diffusion loss for training.
        Args:
            net: The denoising network (e.g., UNet).
            x_start: The clean high-resolution image (GT).
        Returns:
            Calculated loss tensor.
        """
        pass

    @abstractmethod
    def sample(self, net: nn.Module, shape: tuple, device: torch.device, *args, **kwargs) -> torch.Tensor:
        """
        Generate samples from pure noise (Inference/Testing).
        Args:
            net: The denoising network.
            shape: Expected output shape (B, C, H, W).
            device: Target device.
        Returns:
            Generated image tensor mapped to [0, 1].
        """
        pass