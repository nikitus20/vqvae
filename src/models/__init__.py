"""VQ-VAE model components."""

from .quantizer import VectorQuantizer
from .vqvae import LinearGaussianVQVAE

__all__ = ['VectorQuantizer', 'LinearGaussianVQVAE']
