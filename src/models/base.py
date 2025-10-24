"""Base interface for VQ-VAE models."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict


class BaseVQVAE(nn.Module, ABC):
    """Abstract base class for VQ-VAE models.

    All VQ-VAE implementations should inherit from this class and implement
    the required abstract methods.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full forward pass through VQ-VAE.

        Args:
            x: (B, d) input data

        Returns:
            Dictionary containing at minimum:
                - x_recon: (B, d) reconstructed data
                - z: (B, k) encoder output
                - z_q: (B, k) quantized latent codes
                - indices: (B,) codebook indices

            May also contain additional quantizer-specific information.
        """
        raise NotImplementedError

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation.

        Args:
            x: (B, d) input data

        Returns:
            z: (B, k) latent codes
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent codes to reconstruction.

        Args:
            z: (B, k) latent codes

        Returns:
            x_recon: (B, d) reconstructed data
        """
        raise NotImplementedError

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method for reconstruction.

        Args:
            x: (B, d) input data

        Returns:
            x_recon: (B, d) reconstructed data
        """
        return self.forward(x)['x_recon']
