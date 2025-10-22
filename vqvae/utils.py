"""Utility functions for data generation, metrics, and visualization."""

import numpy as np
from typing import Dict, Tuple


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def geomspace_diag(lmax: float, lmin: float, r: int) -> np.ndarray:
    """Generate geometrically spaced eigenvalues."""
    return np.geomspace(lmax, lmin, r)


def make_lowrank_cov(D: int, r: int, lmax: float = 4.0, lmin: float = 0.25,
                     rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate low-rank covariance matrix.

    Args:
        D: Ambient dimension
        r: Rank
        lmax: Maximum eigenvalue
        lmin: Minimum eigenvalue
        rng: Random number generator

    Returns:
        Sigma: (D, D) covariance matrix with rank r
    """
    rng = rng or np.random.default_rng(0)
    U, _ = np.linalg.qr(rng.normal(size=(D, r)))
    lam = geomspace_diag(lmax, lmin, r)
    Sigma = (U * lam) @ U.T
    return Sigma


def sample_gaussian(N: int, mean: np.ndarray, cov: np.ndarray,
                   rng: np.random.Generator = None) -> np.ndarray:
    """
    Sample from multivariate Gaussian.

    Args:
        N: Number of samples
        mean: (D,) mean vector
        cov: (D, D) covariance matrix
        rng: Random number generator

    Returns:
        X: (N, D) samples
    """
    rng = rng or np.random.default_rng(0)
    return rng.multivariate_normal(mean=mean, cov=cov, size=N)


def make_lowrank_data(D: int, r: int, N: int, lmax: float = 4.0, lmin: float = 0.25,
                     seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data from low-rank Gaussian.

    Args:
        D: Ambient dimension
        r: Rank
        N: Number of samples
        lmax: Maximum eigenvalue
        lmin: Minimum eigenvalue
        seed: Random seed

    Returns:
        X: (N, D) data matrix
        Sigma: (D, D) true covariance
    """
    rng = np.random.default_rng(seed)
    Sigma = make_lowrank_cov(D, r, lmax, lmin, rng)
    X = sample_gaussian(N, np.zeros(D), Sigma, rng)
    return X, Sigma


def metrics(W: np.ndarray, E: np.ndarray, X: np.ndarray) -> Dict[str, float]:
    """
    Compute VQ-VAE metrics.

    Args:
        W: (m, D) encoder matrix
        E: (K, m) codebook
        X: (N, D) data

    Returns:
        Dictionary with metrics:
            - recon_mse: Reconstruction MSE
            - in_sub_dist: In-subspace distortion
            - usage_entropy: Code usage entropy (nats)
            - dead_frac: Fraction of dead codes
    """
    from .models.vqvae import VQVAE

    # Use VQVAE interface
    vqvae = VQVAE(W.shape[1], W.shape[0], E.shape[0])
    vqvae.encoder.W = W
    vqvae.codebook.E = E

    result = vqvae.forward(X)
    Z = result['Z']
    Q = result['Q']
    X_hat = result['X_hat']
    k = result['k']

    # Reconstruction MSE
    recon_mse = np.mean(np.sum((X - X_hat)**2, axis=1))

    # In-subspace distortion
    in_sub_dist = np.mean(np.sum((Z - Q)**2, axis=1))

    # Code usage statistics
    p = np.bincount(k, minlength=E.shape[0]) / len(k)
    usage_entropy = -np.sum(p[p > 0] * np.log(p[p > 0]))
    dead_frac = float(np.mean(p == 0.0))

    return {
        "recon_mse": recon_mse,
        "in_sub_dist": in_sub_dist,
        "usage_entropy": usage_entropy,
        "dead_frac": dead_frac
    }
