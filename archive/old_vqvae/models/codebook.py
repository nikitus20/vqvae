"""Codebook for vector quantization."""

import numpy as np
from typing import Tuple


class Codebook:
    """
    Codebook E ∈ R^{K×m} for vector quantization.

    Args:
        K: Number of codes
        m: Code dimension
    """

    def __init__(self, K: int, m: int):
        self.K = K
        self.m = m
        self.E = np.random.randn(K, m) * 0.1

    def nearest_code(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find nearest code for each latent vector.

        Args:
            Z: (N, m) latent vectors

        Returns:
            k: (N,) indices of nearest codes
            Q: (N, m) nearest code vectors (Q = E[k])
        """
        # Compute squared distances: ||z_i - e_j||^2 = ||z_i||^2 - 2⟨z_i, e_j⟩ + ||e_j||^2
        Z_sq = np.sum(Z**2, axis=1, keepdims=True)  # (N, 1)
        E_sq = np.sum(self.E**2, axis=1)  # (K,)
        D2 = Z_sq - 2 * Z @ self.E.T + E_sq  # (N, K)

        k = np.argmin(D2, axis=1)  # (N,)
        Q = self.E[k]  # (N, m)
        return k, Q

    def update(self, grad_E: np.ndarray, lr: float):
        """
        Update codebook parameters.

        Args:
            grad_E: (K, m) gradient
            lr: learning rate
        """
        self.E -= lr * grad_E

    def init_from_kmeans(self, Z: np.ndarray, iters: int = 25, seed: int = 0):
        """
        Initialize codebook using k-means++ and Lloyd iterations.

        Args:
            Z: (N, m) latent vectors
            iters: Number of Lloyd iterations
            seed: Random seed
        """
        rng = np.random.default_rng(seed)
        N, m = Z.shape

        # k-means++ initialization
        centers = np.empty((self.K, m), dtype=Z.dtype)
        i0 = rng.integers(0, N)
        centers[0] = Z[i0]
        d2 = np.sum((Z - centers[0])**2, axis=1)

        for k in range(1, self.K):
            probs = d2 / np.maximum(d2.sum(), 1e-12)
            idx = rng.choice(N, p=probs)
            centers[k] = Z[idx]
            d2 = np.minimum(d2, np.sum((Z - centers[k])**2, axis=1))

        # Lloyd iterations
        self.E = centers.copy()
        for _ in range(iters):
            k, _ = self.nearest_code(Z)
            for j in range(self.K):
                idxj = np.where(k == j)[0]
                if idxj.size > 0:
                    self.E[j] = Z[idxj].mean(axis=0)
