"""Base interface for decoders."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseDecoder(nn.Module, ABC):
    """Abstract base class for VQ-VAE decoders.

    All decoder implementations should inherit from this class and implement
    the required abstract methods.

    Args:
        latent_dim: Latent dimension (k)
        output_dim: Output dimension (d)
        trainable: Whether decoder parameters should be trainable
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        trainable: bool = True
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self._trainable = trainable

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent codes to reconstruction.

        Args:
            z: (B, latent_dim) latent codes

        Returns:
            x_recon: (B, output_dim) reconstructed data
        """
        raise NotImplementedError

    def freeze(self):
        """Freeze all parameters (stop gradient flow)."""
        for param in self.parameters():
            param.requires_grad = False
        self._trainable = False

    def unfreeze(self):
        """Unfreeze all parameters (enable gradient flow)."""
        for param in self.parameters():
            param.requires_grad = True
        self._trainable = True

    @property
    def trainable(self) -> bool:
        """Return whether decoder is trainable."""
        return self._trainable

    def __repr__(self) -> str:
        """String representation."""
        return (f"{self.__class__.__name__}("
                f"latent_dim={self.latent_dim}, "
                f"output_dim={self.output_dim}, "
                f"trainable={self._trainable})")
