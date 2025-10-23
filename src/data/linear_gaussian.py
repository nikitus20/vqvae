"""Linear Gaussian data generation for VQ-VAE experiments.

This module implements the Linear Gaussian model:
    X = A @ Y + W
where:
    - Y ~ N(0, Σ) is latent variable (k-dimensional)
    - A ∈ R^(d×k) is signal subspace (d > k)
    - W ~ N(0, σ²I_d) is noise
    - X ∈ R^d is observed data

The optimal encoder/decoder is given by PCA: Φ = Ψ = U_k (top k eigenvectors).
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Tuple, Optional


class LinearGaussianDataset:
    """Generate and manage Linear Gaussian data for VQ-VAE experiments.

    Args:
        d: Ambient dimension
        k: Latent dimension (rank of signal)
        sigma_noise: Noise standard deviation
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        d: int = 64,
        k: int = 8,
        sigma_noise: float = 0.1,
        n_samples: int = 10000,
        seed: int = 42
    ):
        self.d = d
        self.k = k
        self.sigma_noise = sigma_noise
        self.n_samples = n_samples
        self.seed = seed

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Generate ground truth parameters
        self.A, self.Sigma_y = self._generate_ground_truth()

        # Generate data
        self.X, self.Y = self._generate_data()

        # Compute theoretical quantities (PCA solution)
        self._compute_theory()

    def _generate_ground_truth(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate signal subspace A and latent covariance Σ.

        Returns:
            A: (d, k) orthonormal matrix (signal subspace)
            Sigma_y: (k, k) positive definite covariance matrix
        """
        # Generate orthonormal A via QR decomposition
        A_random = torch.randn(self.d, self.k)
        A, _ = torch.linalg.qr(A_random)  # (d, k) orthonormal

        # Generate positive definite Σ_y = B @ B^T + εI
        B = torch.randn(self.k, self.k)
        Sigma_y = B @ B.T + 0.1 * torch.eye(self.k)  # Add small regularization

        return A, Sigma_y

    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate data: X = A @ Y + W.

        Returns:
            X: (n_samples, d) observed data
            Y: (n_samples, k) latent variables
        """
        # Sample latent Y ~ N(0, Σ_y)
        mean_y = torch.zeros(self.k)
        dist_y = torch.distributions.MultivariateNormal(mean_y, self.Sigma_y)
        Y = dist_y.sample((self.n_samples,))  # (n_samples, k)

        # Compute signal: A @ Y^T
        signal = Y @ self.A.T  # (n_samples, d)

        # Add noise: W ~ N(0, σ²I)
        noise = torch.randn(self.n_samples, self.d) * self.sigma_noise

        # Observed data
        X = signal + noise

        return X, Y

    def _compute_theory(self):
        """Compute theoretical quantities for analysis.

        Computes:
            - Σ_X: Data covariance
            - U_k: Top k eigenvectors (PCA solution)
            - z_true: Optimal latent representation
            - Σ_z: Latent covariance
            - σ_z²: Mean latent variance (for R(D) initialization)
        """
        # Empirical data covariance
        X_centered = self.X - self.X.mean(dim=0, keepdim=True)
        self.Sigma_X = (X_centered.T @ X_centered) / (self.n_samples - 1)

        # PCA: Get top k eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(self.Sigma_X)
        # Sort in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Top k eigenvectors (PCA solution)
        self.U_k = eigenvectors[:, :self.k]  # (d, k)
        self.eigenvalues_k = eigenvalues[:self.k]  # (k,)

        # Optimal latent representation: z = U_k^T @ X
        self.z_true = self.X @ self.U_k  # (n_samples, k)

        # Latent covariance: Σ_z = U_k^T @ Σ_X @ U_k
        self.Sigma_z = self.U_k.T @ self.Sigma_X @ self.U_k  # (k, k)

        # Mean latent variance (for R(D) initialization)
        self.sigma_z_squared = torch.diag(self.Sigma_z).mean().item()

        # Store useful quantities
        self.signal_variance = torch.diag(self.A @ self.Sigma_y @ self.A.T).mean().item()
        self.noise_variance = self.sigma_noise ** 2
        self.snr = self.signal_variance / (self.noise_variance + 1e-10)  # Avoid division by zero

    def get_dataloader(self, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
        """Create PyTorch DataLoader for training.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data

        Returns:
            DataLoader yielding (X, Y) batches
        """
        dataset = TensorDataset(self.X, self.Y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_initialization_batch(self, n_init: int = 1000) -> torch.Tensor:
        """Get batch for codebook initialization.

        Returns z = U_k^T @ X for first n_init samples.
        This is used to estimate σ_z² for R(D) initialization.

        Args:
            n_init: Number of samples for initialization

        Returns:
            z: (n_init, k) latent codes
        """
        n_init = min(n_init, self.n_samples)
        X_init = self.X[:n_init]
        z_init = X_init @ self.U_k  # Project to latent space
        return z_init

    def __len__(self) -> int:
        """Number of samples in dataset."""
        return self.n_samples

    def __repr__(self) -> str:
        """String representation."""
        return (f"LinearGaussianDataset(d={self.d}, k={self.k}, "
                f"σ_noise={self.sigma_noise:.3f}, n={self.n_samples}, "
                f"σ_z²={self.sigma_z_squared:.3f}, SNR={self.snr:.2f})")
