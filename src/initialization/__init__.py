"""Codebook initialization methods for VQ-VAE quantizers."""

from .standard import uniform_init, kmeans_init, random_normal_init
from .rate_distortion import rd_gaussian_init, compute_rd_variance

__all__ = [
    'uniform_init',
    'kmeans_init',
    'random_normal_init',
    'rd_gaussian_init',
    'compute_rd_variance',
]
