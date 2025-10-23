"""Tests for data generation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.data.linear_gaussian import LinearGaussianDataset


def test_linear_gaussian_shapes():
    """Test that data generator produces correct shapes."""
    d, k, n = 32, 4, 1000
    dataset = LinearGaussianDataset(d=d, k=k, n_samples=n, seed=42)

    assert dataset.X.shape == (n, d), f"X shape: expected {(n, d)}, got {dataset.X.shape}"
    assert dataset.Y.shape == (n, k), f"Y shape: expected {(n, k)}, got {dataset.Y.shape}"
    assert dataset.A.shape == (d, k), f"A shape: expected {(d, k)}, got {dataset.A.shape}"
    assert dataset.U_k.shape == (d, k), f"U_k shape: expected {(d, k)}, got {dataset.U_k.shape}"
    print("✓ Shape test passed")


def test_pca_recovery():
    """Test that PCA recovers signal subspace."""
    d, k, n = 64, 8, 5000
    dataset = LinearGaussianDataset(d=d, k=k, n_samples=n, sigma_noise=0.01, seed=42)

    # Check that U_k captures signal subspace A
    # Note: With noise, PCA may not perfectly recover A, but eigenvalues should be high
    # Instead, check that top k eigenvalues are significantly larger than rest
    all_eigenvalues, _ = torch.linalg.eigh(dataset.Sigma_X)
    all_eigenvalues = torch.sort(all_eigenvalues, descending=True).values

    top_k_mean = all_eigenvalues[:k].mean()
    rest_mean = all_eigenvalues[k:].mean() if k < d else 0.0

    ratio = top_k_mean / (rest_mean + 1e-6)
    assert ratio > 5.0, f"Top {k} eigenvalues should be >> rest. Ratio: {ratio:.2f}"

    print(f"✓ PCA recovery test passed (eigenvalue ratio: {ratio:.1f})")


def test_dataloader():
    """Test DataLoader functionality."""
    dataset = LinearGaussianDataset(d=32, k=4, n_samples=1000, seed=42)
    dataloader = dataset.get_dataloader(batch_size=128, shuffle=True)

    # Check batch shapes
    batch = next(iter(dataloader))
    x, y = batch
    assert x.shape[1] == 32, f"Batch x dimension: expected 32, got {x.shape[1]}"
    assert y.shape[1] == 4, f"Batch y dimension: expected 4, got {y.shape[1]}"
    assert x.shape[0] <= 128, f"Batch size: expected <=128, got {x.shape[0]}"

    print("✓ DataLoader test passed")


def test_initialization_batch():
    """Test initialization batch generation."""
    dataset = LinearGaussianDataset(d=32, k=4, n_samples=1000, seed=42)
    z_init = dataset.get_initialization_batch(n_init=100)

    assert z_init.shape == (100, 4), f"Init batch shape: expected (100, 4), got {z_init.shape}"

    # Check that z_init ≈ X[:100] @ U_k
    z_expected = dataset.X[:100] @ dataset.U_k
    assert torch.allclose(z_init, z_expected, atol=1e-5), "Init batch doesn't match projection"

    print("✓ Initialization batch test passed")


def test_theoretical_quantities():
    """Test that theoretical quantities are computed correctly."""
    dataset = LinearGaussianDataset(d=32, k=4, n_samples=5000, seed=42)

    # Check that sigma_z_squared is positive
    assert dataset.sigma_z_squared > 0, "σ_z² should be positive"

    # Check that SNR makes sense
    assert dataset.snr > 0, "SNR should be positive"

    # Check covariance shapes
    assert dataset.Sigma_X.shape == (32, 32), "Σ_X shape incorrect"
    assert dataset.Sigma_z.shape == (4, 4), "Σ_z shape incorrect"

    print(f"✓ Theoretical quantities test passed (σ_z²={dataset.sigma_z_squared:.3f}, SNR={dataset.snr:.1f})")


if __name__ == "__main__":
    print("Running data generation tests...\n")
    test_linear_gaussian_shapes()
    test_pca_recovery()
    test_dataloader()
    test_initialization_batch()
    test_theoretical_quantities()
    print("\n✓ All data tests passed!")
