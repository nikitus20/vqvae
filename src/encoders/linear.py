"""Linear encoders for VQ-VAE."""

import torch
import torch.nn as nn
from typing import Optional
from .base import BaseEncoder


class PCAEncoder(BaseEncoder):
    """Fixed PCA encoder: z = U_k^T @ x.

    Uses pre-computed PCA eigenvectors as a fixed (non-trainable) encoder.
    This is optimal for Linear Gaussian models.

    Args:
        U_k: (d, k) PCA eigenvectors (top k)
        trainable: Whether to make encoder trainable (default: False for PCA)
    """

    def __init__(
        self,
        U_k: torch.Tensor,
        trainable: bool = False
    ):
        d, k = U_k.shape
        super().__init__(input_dim=d, latent_dim=k, trainable=trainable)

        # Register encoder matrix as buffer (not trained by default)
        # Store U_k naturally: (d, k) for z = x @ U_k
        self.register_buffer('encoder_weight', U_k)  # (d, k)

        # Set trainability
        if not trainable:
            self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode: z = U_k^T @ x.

        Args:
            x: (B, d) input data

        Returns:
            z: (B, k) latent codes
        """
        # z = x @ U_k, which is equivalent to U_k^T @ x^T then transpose
        z = x @ self.encoder_weight  # (B, d) @ (d, k) -> (B, k)
        return z


class LearnedLinearEncoder(BaseEncoder):
    """Learned linear encoder: z = W @ x + b.

    A simple linear projection that can be learned.

    Args:
        input_dim: Input dimension (d)
        latent_dim: Latent dimension (k)
        trainable: Whether encoder should be trainable (default: True)
        bias: Whether to include bias term
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        trainable: bool = True,
        bias: bool = False
    ):
        super().__init__(input_dim, latent_dim, trainable)

        # Linear layer
        self.linear = nn.Linear(input_dim, latent_dim, bias=bias)

        # Initialize with small random values
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

        # Set trainability
        if not trainable:
            self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode: z = W @ x + b.

        Args:
            x: (B, d) input data

        Returns:
            z: (B, k) latent codes
        """
        return self.linear(x)
