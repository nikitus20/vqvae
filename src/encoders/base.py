"""Base interface for encoders."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for VQ-VAE encoders.

    All encoder implementations should inherit from this class and implement
    the required abstract methods.

    Args:
        input_dim: Input dimension (d)
        latent_dim: Latent dimension (k)
        trainable: Whether encoder parameters should be trainable
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        trainable: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self._trainable = trainable

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation.

        Args:
            x: (B, input_dim) input data

        Returns:
            z: (B, latent_dim) latent codes
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
        """Return whether encoder is trainable."""
        return self._trainable

    def __repr__(self) -> str:
        """String representation."""
        return (f"{self.__class__.__name__}("
                f"input_dim={self.input_dim}, "
                f"latent_dim={self.latent_dim}, "
                f"trainable={self._trainable})")
