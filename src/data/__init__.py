"""Data generation modules for VQ-VAE experiments."""

from .base import BaseDataset
from .gaussian import GaussianDataset

# For backward compatibility
from .gaussian import GaussianDataset as LinearGaussianDataset

__all__ = ['BaseDataset', 'GaussianDataset', 'LinearGaussianDataset']
