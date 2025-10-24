"""Linear decoders for VQ-VAE."""

import torch
import torch.nn as nn
from typing import Optional
from .base import BaseDecoder


class PCADecoder(BaseDecoder):
    """Fixed PCA decoder: x = U_k @ z.

    Uses pre-computed PCA eigenvectors as a fixed (non-trainable) decoder.
    This is optimal for Linear Gaussian models and is the transpose of PCAEncoder.

    Args:
        U_k: (d, k) PCA eigenvectors (top k)
        trainable: Whether to make decoder trainable (default: False for PCA)
    """

    def __init__(
        self,
        U_k: torch.Tensor,
        trainable: bool = False
    ):
        d, k = U_k.shape
        super().__init__(latent_dim=k, output_dim=d, trainable=trainable)

        # Register decoder matrix as buffer (not trained by default)
        # decoder_weight = U_k for efficient batch multiplication
        self.register_buffer('decoder_weight', U_k)  # (d, k)

        # Set trainability
        if not trainable:
            self.freeze()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode: x = U_k @ z.

        Args:
            z: (B, k) latent codes

        Returns:
            x_recon: (B, d) reconstructed data
        """
        # x = z @ U_k^T, which is equivalent to U_k @ z^T then transpose
        x_recon = z @ self.decoder_weight.T  # (B, k) @ (k, d) -> (B, d)
        return x_recon


class LearnedLinearDecoder(BaseDecoder):
    """Learned linear decoder: x = W @ z + b.

    A simple linear projection that can be learned.

    Args:
        latent_dim: Latent dimension (k)
        output_dim: Output dimension (d)
        trainable: Whether decoder should be trainable (default: True)
        bias: Whether to include bias term
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        trainable: bool = True,
        bias: bool = False
    ):
        super().__init__(latent_dim, output_dim, trainable)

        # Linear layer
        self.linear = nn.Linear(latent_dim, output_dim, bias=bias)

        # Initialize with small random values
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

        # Set trainability
        if not trainable:
            self.freeze()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode: x = W @ z + b.

        Args:
            z: (B, k) latent codes

        Returns:
            x_recon: (B, d) reconstructed data
        """
        return self.linear(z)
