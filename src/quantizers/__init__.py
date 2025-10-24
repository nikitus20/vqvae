"""Quantizer modules for VQ-VAE."""

from .base import BaseQuantizer
from .vq import VectorQuantizer
from .projection_vq import ProjectionVectorQuantizer
from .rotation_vq import RotationVectorQuantizer

__all__ = [
    'BaseQuantizer',
    'VectorQuantizer',
    'ProjectionVectorQuantizer',
    'RotationVectorQuantizer'
]
