"""Vector Quantizer with multiple initialization methods."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class VectorQuantizer(nn.Module):
    """Vector quantizer with Straight-Through Estimator.

    Supports three initialization methods:
    1. 'uniform': Standard uniform initialization [-1/n, 1/n]
    2. 'kmeans': K-means clustering on initialization data
    3. 'rd_gaussian': R(D)-optimal Gaussian initialization

    Args:
        dim: Latent dimension (k)
        codebook_size: Number of codewords (n)
        init_method: Initialization method ('uniform', 'kmeans', 'rd_gaussian')
        init_data: Optional data for kmeans/rd_gaussian initialization (n_samples, dim)
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        init_method: str = 'uniform',
        init_data: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.init_method = init_method

        # Initialize codebook
        if init_method == 'uniform':
            codebook_init = self._uniform_init()
        elif init_method == 'kmeans':
            if init_data is None:
                raise ValueError("init_data required for kmeans initialization")
            codebook_init = self._kmeans_init(init_data)
        elif init_method == 'rd_gaussian':
            if init_data is None:
                raise ValueError("init_data required for rd_gaussian initialization")
            codebook_init = self._rd_gaussian_init(init_data)
        else:
            raise ValueError(f"Unknown init_method: {init_method}")

        # Create trainable codebook parameter
        self.codebook = nn.Parameter(codebook_init)

    def _uniform_init(self) -> torch.Tensor:
        """Standard uniform initialization: Uniform[-1/n, 1/n] per dimension.

        This is the standard practice but tends to be too small.

        Returns:
            codebook: (codebook_size, dim) tensor
        """
        n = self.codebook_size
        # Uniform in [-1/n, 1/n]
        codebook = torch.rand(n, self.dim) * (2 / n) - (1 / n)
        return codebook

    def _kmeans_init(self, data: torch.Tensor) -> torch.Tensor:
        """K-means clustering initialization.

        Uses sklearn's KMeans for robust clustering.

        Args:
            data: (N, dim) initialization data

        Returns:
            codebook: (codebook_size, dim) tensor with cluster centers
        """
        from sklearn.cluster import KMeans

        # Convert to numpy for sklearn
        data_np = data.cpu().numpy()

        # Run k-means
        kmeans = KMeans(
            n_clusters=self.codebook_size,
            n_init=10,
            max_iter=100,
            random_state=42
        )
        kmeans.fit(data_np)

        # Get cluster centers
        centers = torch.from_numpy(kmeans.cluster_centers_).float()

        # Move to same device as data
        if data.is_cuda:
            centers = centers.cuda()

        return centers

    def _rd_gaussian_init(self, data: torch.Tensor) -> torch.Tensor:
        """R(D)-optimal Gaussian initialization.

        Based on rate-distortion theory for Gaussian sources.
        For codebook size n and dimension k:
            R = log₂(n) / k  (rate in bits/dimension)
            σ_ẑ² = σ_z² × (1 - 2^(-2R))  (optimal reproduction variance)

        Steps:
        1. Estimate σ_z² = mean(Var[z_i]) from data
        2. Compute rate R = log₂(n) / k
        3. Compute target variance σ_ẑ² = σ_z² × (1 - 2^(-2R))
        4. Sample codebook from N(0, σ_ẑ² I_k)

        Args:
            data: (N, dim) initialization data

        Returns:
            codebook: (codebook_size, dim) tensor sampled from optimal distribution
        """
        # Estimate variance of each dimension
        sigma_z_sq = data.var(dim=0).mean().item()

        # Compute rate in bits per dimension
        R = np.log2(self.codebook_size) / self.dim

        # Compute R(D)-optimal reproduction variance
        # For Gaussian: σ_ẑ² = σ_z² × (1 - 2^(-2R))
        compression_factor = 1 - 2 ** (-2 * R)
        sigma_zhat_sq = sigma_z_sq * compression_factor

        # Sample codebook from N(0, σ_ẑ² I)
        codebook = torch.randn(self.codebook_size, self.dim) * np.sqrt(sigma_zhat_sq)

        # Move to same device as data
        if data.is_cuda:
            codebook = codebook.cuda()

        return codebook

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vector quantization with Straight-Through Estimator.

        Args:
            z: (B, dim) encoder outputs

        Returns:
            z_q: (B, dim) quantized vectors (with straight-through gradient)
            indices: (B,) nearest codeword indices
        """
        # Compute distances: ||z - e_i||² for all codebook entries
        # Using: ||z - e||² = ||z||² + ||e||² - 2⟨z, e⟩
        distances = torch.cdist(z, self.codebook)  # (B, codebook_size)

        # Find nearest codeword indices
        indices = distances.argmin(dim=1)  # (B,)

        # Lookup quantized values
        z_q_hard = self.codebook[indices]  # (B, dim)

        # Straight-Through Estimator:
        # Forward: use z_q_hard
        # Backward: copy gradient from z_q to z (if z requires grad)
        # AND preserve gradient to codebook
        if z.requires_grad:
            # Standard STE: copy gradients through
            z_q = z + (z_q_hard - z).detach()
        else:
            # Fixed encoder case: just use z_q_hard to preserve codebook gradients
            z_q = z_q_hard

        return z_q, indices

    def get_codebook(self) -> torch.Tensor:
        """Get current codebook.

        Returns:
            codebook: (codebook_size, dim) tensor
        """
        return self.codebook.data

    def __repr__(self) -> str:
        """String representation."""
        return (f"VectorQuantizer(dim={self.dim}, "
                f"codebook_size={self.codebook_size}, "
                f"init_method='{self.init_method}')")
