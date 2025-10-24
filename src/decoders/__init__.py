"""Decoder modules for VQ-VAE."""

from .base import BaseDecoder
from .linear import PCADecoder, LearnedLinearDecoder

__all__ = ['BaseDecoder', 'PCADecoder', 'LearnedLinearDecoder']
