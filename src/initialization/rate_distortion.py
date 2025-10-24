"""Rate-distortion optimal codebook initialization methods."""

import torch
import numpy as np
from typing import Optional


def compute_rd_variance(
    data: torch.Tensor,
    rate: float
) -> float:
    """Compute R(D)-optimal reproduction variance for Gaussian source.

    Based on rate-distortion theory for Gaussian sources:
        σ_ẑ² = σ_z² × (1 - 2^(-2R))

    where:
        - σ_z²: Source variance (estimated from data)
        - R: Rate in bits per dimension
        - σ_ẑ²: Optimal reproduction variance

    Args:
        data: (N, dim) data samples to estimate source variance
        rate: Rate in bits per dimension (R = log₂(n) / k)

    Returns:
        sigma_zhat_sq: Optimal reproduction variance σ_ẑ²
    """
    # Estimate variance of each dimension
    sigma_z_sq = data.var(dim=0).mean().item()

    # Compute R(D)-optimal reproduction variance
    # For Gaussian: σ_ẑ² = σ_z² × (1 - 2^(-2R))
    compression_factor = 1 - 2 ** (-2 * rate)
    sigma_zhat_sq = sigma_z_sq * compression_factor

    return sigma_zhat_sq


def rd_gaussian_init(
    codebook_size: int,
    dim: int,
    data: torch.Tensor,
    rate: Optional[float] = None
) -> torch.Tensor:
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
        codebook_size: Number of codewords (n)
        dim: Latent dimension (k)
        data: (N, dim) initialization data
        rate: Optional rate override (bits/dim). If None, computed as log₂(n)/k

    Returns:
        codebook: (codebook_size, dim) tensor sampled from optimal distribution
    """
    # Validate input
    assert data.shape[1] == dim, \
        f"Data dimension {data.shape[1]} doesn't match expected dim {dim}"

    device = data.device

    # Compute rate in bits per dimension
    if rate is None:
        rate = np.log2(codebook_size) / dim

    # Compute R(D)-optimal variance
    sigma_zhat_sq = compute_rd_variance(data, rate)

    # Sample codebook from N(0, σ_ẑ² I)
    codebook = torch.randn(codebook_size, dim, device=device) * np.sqrt(sigma_zhat_sq)

    return codebook
