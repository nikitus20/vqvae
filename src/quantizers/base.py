"""Base interface for quantizers."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any


class BaseQuantizer(nn.Module, ABC):
    """Abstract base class for VQ-VAE quantizers.

    All quantizer implementations should inherit from this class and implement
    the required abstract methods.

    Args:
        dim: Latent dimension (k)
        codebook_size: Number of codebook entries (n)
        trainable: Whether codebook should be trainable
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        trainable: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self._trainable = trainable

    @abstractmethod
    def forward(
        self,
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Quantize latent codes.

        Args:
            z: (B, dim) encoder outputs

        Returns:
            z_q: (B, dim) quantized vectors
            indices: (B,) codebook indices
            info: Dictionary with additional information (e.g., for STE)
        """
        raise NotImplementedError

    @abstractmethod
    def get_codebook(self) -> torch.Tensor:
        """Get current codebook.

        Returns:
            codebook: (codebook_size, dim) tensor
        """
        raise NotImplementedError

    def freeze(self):
        """Freeze codebook (stop gradient flow)."""
        for param in self.parameters():
            param.requires_grad = False
        self._trainable = False

    def unfreeze(self):
        """Unfreeze codebook (enable gradient flow)."""
        for param in self.parameters():
            param.requires_grad = True
        self._trainable = True

    @property
    def trainable(self) -> bool:
        """Return whether quantizer is trainable."""
        return self._trainable

    def __repr__(self) -> str:
        """String representation."""
        return (f"{self.__class__.__name__}("
                f"dim={self.dim}, "
                f"codebook_size={self.codebook_size}, "
                f"trainable={self._trainable})")
