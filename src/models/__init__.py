"""VQ-VAE model components."""

from .base import BaseVQVAE
from .vqvae import VQVAE

# Legacy imports for backward compatibility
from .legacy import LinearGaussianVQVAE

__all__ = ['BaseVQVAE', 'VQVAE', 'LinearGaussianVQVAE']
