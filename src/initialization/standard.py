"""Standard codebook initialization methods."""

import torch
import numpy as np
from typing import Optional


def uniform_init(
    codebook_size: int,
    dim: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Standard uniform initialization: Uniform[-1/n, 1/n] per dimension.

    This is the standard practice but tends to produce small initial values.

    Args:
        codebook_size: Number of codewords (n)
        dim: Latent dimension (k)
        device: Device to create tensor on

    Returns:
        codebook: (codebook_size, dim) tensor
    """
    n = codebook_size
    # Uniform in [-1/n, 1/n]
    codebook = torch.rand(n, dim, device=device) * (2 / n) - (1 / n)
    return codebook


def random_normal_init(
    codebook_size: int,
    dim: int,
    std: float = 1.0,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Random normal initialization: N(0, std²I).

    Args:
        codebook_size: Number of codewords (n)
        dim: Latent dimension (k)
        std: Standard deviation
        device: Device to create tensor on

    Returns:
        codebook: (codebook_size, dim) tensor
    """
    codebook = torch.randn(codebook_size, dim, device=device) * std
    return codebook


def kmeans_init(
    codebook_size: int,
    dim: int,
    data: torch.Tensor,
    n_init: int = 10,
    max_iter: int = 100,
    random_state: int = 42
) -> torch.Tensor:
    """K-means clustering initialization.

    Uses sklearn's KMeans for robust clustering on provided data.

    Args:
        codebook_size: Number of codewords (n)
        dim: Latent dimension (k)
        data: (N, dim) initialization data
        n_init: Number of k-means runs with different initializations
        max_iter: Maximum iterations per k-means run
        random_state: Random seed for reproducibility

    Returns:
        codebook: (codebook_size, dim) tensor with cluster centers
    """
    from sklearn.cluster import KMeans

    # Validate input
    assert data.shape[1] == dim, \
        f"Data dimension {data.shape[1]} doesn't match expected dim {dim}"
    assert len(data) >= codebook_size, \
        f"Need at least {codebook_size} samples for k-means, got {len(data)}"

    # Convert to numpy for sklearn
    device = data.device
    data_np = data.cpu().numpy()

    # Run k-means
    kmeans = KMeans(
        n_clusters=codebook_size,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state
    )
    kmeans.fit(data_np)

    # Get cluster centers
    centers = torch.from_numpy(kmeans.cluster_centers_).float()

    # Move to same device as data
    if device.type == 'cuda':
        centers = centers.cuda()

    return centers
