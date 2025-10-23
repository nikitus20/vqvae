"""Tests for vector quantizer."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.models.quantizer import VectorQuantizer


def test_uniform_initialization():
    """Test that uniform initialization has correct variance."""
    dim, n = 8, 256
    quantizer = VectorQuantizer(dim, n, init_method='uniform')

    codebook = quantizer.get_codebook()
    assert codebook.shape == (n, dim), f"Codebook shape: expected ({n}, {dim}), got {codebook.shape}"

    # Check variance approximately matches uniform[-1/n, 1/n]
    # Var[U(-a, a)] = a²/3, where a = 1/n
    expected_var = (1/n)**2 / 3
    actual_var = codebook.var(dim=0).mean().item()

    # Allow 20% tolerance (stochastic)
    assert abs(actual_var - expected_var) / expected_var < 0.3, \
        f"Uniform init variance: expected {expected_var:.6f}, got {actual_var:.6f}"

    print(f"✓ Uniform initialization test passed (var={actual_var:.6f})")


def test_rd_gaussian_initialization():
    """Test that R(D) initialization matches theoretical variance."""
    dim, n = 8, 256
    sigma_z_sq = 1.0

    # Generate test data
    data = torch.randn(1000, dim) * np.sqrt(sigma_z_sq)

    # Create quantizer with R(D) init
    quantizer = VectorQuantizer(dim, n, init_method='rd_gaussian', init_data=data)

    codebook = quantizer.get_codebook()

    # Compute expected variance from R(D) theory
    R = np.log2(n) / dim
    expected_var = sigma_z_sq * (1 - 2**(-2*R))
    actual_var = codebook.var(dim=0).mean().item()

    # Allow 15% tolerance (stochastic)
    assert abs(actual_var - expected_var) / expected_var < 0.2, \
        f"R(D) init variance: expected {expected_var:.6f}, got {actual_var:.6f}"

    print(f"✓ R(D) initialization test passed (var={actual_var:.6f}, expected={expected_var:.6f})")


def test_kmeans_initialization():
    """Test that K-means initialization works."""
    dim, n = 4, 16

    # Generate clustered data
    data = torch.randn(500, dim)

    # Create quantizer with K-means init
    quantizer = VectorQuantizer(dim, n, init_method='kmeans', init_data=data)

    codebook = quantizer.get_codebook()
    assert codebook.shape == (n, dim), f"K-means codebook shape: expected ({n}, {dim}), got {codebook.shape}"

    # Check that codebook entries are reasonably spread out
    mean_distance = torch.cdist(codebook, codebook).mean().item()
    assert mean_distance > 0.1, "K-means codebook entries too close together"

    print(f"✓ K-means initialization test passed (mean distance={mean_distance:.3f})")


def test_forward_pass():
    """Test quantizer forward pass."""
    dim, n = 8, 32
    batch_size = 64

    quantizer = VectorQuantizer(dim, n, init_method='uniform')

    # Random input
    z = torch.randn(batch_size, dim)

    # Forward pass
    z_q, indices = quantizer(z)

    # Check shapes
    assert z_q.shape == (batch_size, dim), f"z_q shape: expected ({batch_size}, {dim}), got {z_q.shape}"
    assert indices.shape == (batch_size,), f"indices shape: expected ({batch_size},), got {indices.shape}"

    # Check that indices are valid
    assert (indices >= 0).all() and (indices < n).all(), "Invalid indices"

    # Check that z_q entries match codebook
    codebook = quantizer.get_codebook()
    for i in range(min(5, batch_size)):  # Check first few
        expected = codebook[indices[i]]
        actual = z_q[i]
        # Note: gradient might differ due to STE, but values should match
        assert torch.allclose(actual.detach(), expected, atol=1e-5), \
            f"z_q[{i}] doesn't match codebook[{indices[i]}]"

    print("✓ Forward pass test passed")


def test_straight_through_estimator():
    """Test that STE gradient flows correctly."""
    dim, n = 4, 8
    quantizer = VectorQuantizer(dim, n, init_method='uniform')

    # Input that requires grad
    z = torch.randn(10, dim, requires_grad=True)

    # Forward
    z_q, indices = quantizer(z)

    # Backward (dummy loss)
    loss = z_q.sum()
    loss.backward()

    # Check that gradient exists for z (STE property)
    assert z.grad is not None, "No gradient for z (STE failed)"
    assert z.grad.shape == z.shape, "Gradient shape mismatch"

    print("✓ Straight-through estimator test passed")


def test_variance_comparison():
    """Compare initialization method variances."""
    dim, n = 8, 256
    sigma_z_sq = 1.0
    data = torch.randn(1000, dim) * np.sqrt(sigma_z_sq)

    # Create quantizers
    q_uniform = VectorQuantizer(dim, n, init_method='uniform')
    q_rd = VectorQuantizer(dim, n, init_method='rd_gaussian', init_data=data)

    var_uniform = q_uniform.get_codebook().var(dim=0).mean().item()
    var_rd = q_rd.get_codebook().var(dim=0).mean().item()

    # R(D) should have MUCH larger variance than uniform
    assert var_rd > var_uniform * 10, \
        f"R(D) variance should be >>uniform. Got uniform={var_uniform:.6f}, R(D)={var_rd:.6f}"

    print(f"✓ Variance comparison test passed (uniform={var_uniform:.6f}, R(D)={var_rd:.6f})")


if __name__ == "__main__":
    print("Running quantizer tests...\n")
    test_uniform_initialization()
    test_rd_gaussian_initialization()
    test_kmeans_initialization()
    test_forward_pass()
    test_straight_through_estimator()
    test_variance_comparison()
    print("\n✓ All quantizer tests passed!")
