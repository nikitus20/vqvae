"""Linear encoder with optional orthonormal constraint."""

import numpy as np


class LinearEncoder:
    """
    Linear encoder W: R^D → R^m
    Maps data from ambient dimension D to latent dimension m.

    Args:
        D: Ambient dimension
        m: Latent dimension
        orthonormal: If True, rows of W form orthonormal basis (Stiefel manifold)
    """

    def __init__(self, D: int, m: int, orthonormal: bool = True):
        self.D = D
        self.m = m
        self.orthonormal = orthonormal
        # W has shape (m, D) so each row is a basis vector
        self.W = np.random.randn(m, D) / np.sqrt(D)

        if orthonormal:
            self.project()

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode data: Z = X @ W^T

        Args:
            X: (N, D) data matrix

        Returns:
            Z: (N, m) encoded data
        """
        return X @ self.W.T

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """
        Decode latents: X_hat = Z @ W

        Args:
            Z: (N, m) latent codes

        Returns:
            X_hat: (N, D) reconstructed data
        """
        return Z @ self.W

    def project(self):
        """Project W onto Stiefel manifold (orthonormal rows)."""
        if self.orthonormal:
            U, _, Vt = np.linalg.svd(self.W, full_matrices=False)
            self.W = U @ Vt

    def update(self, grad_W: np.ndarray, lr: float):
        """
        Update encoder parameters.

        Args:
            grad_W: (m, D) gradient
            lr: learning rate
        """
        if self.orthonormal:
            # Riemannian gradient on Stiefel manifold
            sym = grad_W.T @ self.W + self.W.T @ grad_W
            grad_W_riem = grad_W - 0.5 * sym.T @ self.W
            self.W -= lr * grad_W_riem
            self.project()
        else:
            self.W -= lr * grad_W

    def init_from_pca(self, Sigma: np.ndarray):
        """
        Initialize encoder from top-m eigenvectors of covariance.

        Args:
            Sigma: (D, D) covariance matrix
        """
        evals, evecs = np.linalg.eigh(Sigma)
        idx = np.argsort(evals)[::-1]
        self.W = evecs[:, idx[:self.m]].T.copy()
