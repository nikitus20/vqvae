"""Vector Quantizer with Straight-Through Estimator."""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional
from .base import BaseQuantizer


class VectorQuantizer(BaseQuantizer):
    """Vector quantizer with Straight-Through Estimator.

    Performs nearest-neighbor quantization in latent space using a learned
    codebook. Gradients flow through the quantization via the STE trick.

    Args:
        dim: Latent dimension (k)
        codebook_size: Number of codewords (n)
        trainable: Whether codebook should be trainable
        init_codebook: Optional pre-initialized codebook (codebook_size, dim).
                      If None, uses uniform initialization.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        trainable: bool = True,
        init_codebook: Optional[torch.Tensor] = None
    ):
        super().__init__(dim, codebook_size, trainable)

        # Initialize codebook
        if init_codebook is not None:
            # Use provided initialization
            assert init_codebook.shape == (codebook_size, dim), \
                f"init_codebook shape {init_codebook.shape} != expected {(codebook_size, dim)}"
            codebook_init = init_codebook.clone()
        else:
            # Default: uniform initialization
            from ..initialization import uniform_init
            codebook_init = uniform_init(codebook_size, dim)

        # Create trainable codebook parameter
        self.codebook = nn.Parameter(codebook_init, requires_grad=trainable)

    def forward(
        self,
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Vector quantization with Straight-Through Estimator.

        Args:
            z: (B, dim) encoder outputs

        Returns:
            z_q: (B, dim) quantized vectors (with straight-through gradient)
            indices: (B,) nearest codeword indices
            info: Dictionary with 'z_q_hard' for correct loss computation
        """
        # Compute distances: ||z - e_i||² for all codebook entries
        distances = torch.cdist(z, self.codebook)  # (B, codebook_size)

        # Find nearest codeword indices
        indices = distances.argmin(dim=1)  # (B,)

        # Lookup quantized values (hard quantization)
        z_q_hard = self.codebook[indices]  # (B, dim)

        # Straight-Through Estimator: always apply for consistency
        # Forward: use z_q_hard
        # Backward: gradient flows to both z and codebook
        z_q = z + (z_q_hard - z).detach()

        return z_q, indices, {'z_q_hard': z_q_hard}

    def get_codebook(self) -> torch.Tensor:
        """Get current codebook.

        Returns:
            codebook: (codebook_size, dim) tensor
        """
        return self.codebook.data

    def freeze(self):
        """Freeze codebook (stop gradient flow)."""
        self.codebook.requires_grad = False
        self._trainable = False

    def unfreeze(self):
        """Unfreeze codebook (enable gradient flow)."""
        self.codebook.requires_grad = True
        self._trainable = True
