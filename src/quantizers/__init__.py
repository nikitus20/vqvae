"""Quantizer modules for VQ-VAE."""

from .base import BaseQuantizer
from .vq import VectorQuantizer
from .rotation_vq import RotationVectorQuantizer

__all__ = ['BaseQuantizer', 'VectorQuantizer', 'RotationVectorQuantizer']
