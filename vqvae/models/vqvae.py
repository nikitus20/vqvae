"""Complete VQ-VAE model combining encoder, codebook, and decoder."""

import numpy as np
from typing import Dict, Tuple
from .encoder import LinearEncoder
from .codebook import Codebook


class VQVAE:
    """
    VQ-VAE model: X → Encoder → Z → Quantize → Q → Decoder → X_hat

    The encoder and decoder share weights (decoder = encoder^T).

    Args:
        D: Ambient dimension
        m: Latent dimension
        K: Number of codebook entries
        orthonormal_encoder: If True, encoder has orthonormal rows
    """

    def __init__(self, D: int, m: int, K: int, orthonormal_encoder: bool = True):
        self.encoder = LinearEncoder(D, m, orthonormal=orthonormal_encoder)
        self.codebook = Codebook(K, m)
        self.D = D
        self.m = m
        self.K = K

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode: Z = X @ W^T"""
        return self.encoder.encode(X)

    def quantize(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Quantize: find nearest codes"""
        return self.codebook.nearest_code(Z)

    def decode(self, Q: np.ndarray) -> np.ndarray:
        """Decode: X_hat = Q @ W"""
        return self.encoder.decode(Q)

    def forward(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Full forward pass.

        Args:
            X: (N, D) input data

        Returns:
            Dictionary containing:
                - Z: (N, m) latent codes
                - k: (N,) quantization indices
                - Q: (N, m) quantized codes
                - X_hat: (N, D) reconstructions
        """
        Z = self.encode(X)
        k, Q = self.quantize(Z)
        X_hat = self.decode(Q)

        return {
            'Z': Z,
            'k': k,
            'Q': Q,
            'X_hat': X_hat
        }

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Full reconstruction pipeline."""
        return self.forward(X)['X_hat']
