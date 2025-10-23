"""Integration tests for VQ-VAE model."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.data.linear_gaussian import LinearGaussianDataset
from src.models.vqvae import LinearGaussianVQVAE
from src.training.losses import vqvae_loss


def test_vqvae_forward():
    """Test VQ-VAE forward pass."""
    d, k, n, codebook_size = 32, 4, 100, 16

    # Create dataset
    dataset = LinearGaussianDataset(d=d, k=k, n_samples=n, seed=42)
    init_data = dataset.get_initialization_batch(n_init=50)

    # Create model
    model = LinearGaussianVQVAE(
        d=d,
        k=k,
        codebook_size=codebook_size,
        U_k=dataset.U_k,
        init_method='rd_gaussian',
        init_data=init_data
    )

    # Forward pass
    x = dataset.X[:10]
    outputs = model(x)

    # Check outputs
    assert 'x_recon' in outputs, "Missing x_recon"
    assert 'z' in outputs, "Missing z"
    assert 'z_q' in outputs, "Missing z_q"
    assert 'indices' in outputs, "Missing indices"

    # Check shapes
    assert outputs['x_recon'].shape == (10, d), f"x_recon shape incorrect: {outputs['x_recon'].shape}"
    assert outputs['z'].shape == (10, k), f"z shape incorrect: {outputs['z'].shape}"
    assert outputs['z_q'].shape == (10, k), f"z_q shape incorrect: {outputs['z_q'].shape}"
    assert outputs['indices'].shape == (10,), f"indices shape incorrect: {outputs['indices'].shape}"

    print("✓ VQ-VAE forward pass test passed")


def test_encoder_decoder_fixed():
    """Test that encoder and decoder are fixed (not trainable)."""
    d, k, codebook_size = 32, 4, 16
    dataset = LinearGaussianDataset(d=d, k=k, n_samples=100, seed=42)

    model = LinearGaussianVQVAE(
        d=d,
        k=k,
        codebook_size=codebook_size,
        U_k=dataset.U_k,
        init_method='uniform'
    )

    # Check that encoder/decoder are buffers (not parameters)
    param_names = [name for name, _ in model.named_parameters()]
    buffer_names = [name for name, _ in model.named_buffers()]

    assert 'encoder_weight' not in param_names, "Encoder should not be trainable"
    assert 'decoder_weight' not in param_names, "Decoder should not be trainable"
    assert 'encoder_weight' in buffer_names, "Encoder should be a buffer"
    assert 'decoder_weight' in buffer_names, "Decoder should be a buffer"

    # Only codebook should be trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected_params = codebook_size * k
    assert trainable_params == expected_params, \
        f"Expected {expected_params} trainable params, got {trainable_params}"

    print(f"✓ Fixed encoder/decoder test passed ({trainable_params} trainable params)")


def test_reconstruction_identity():
    """Test that reconstruction through quantizer preserves signal subspace."""
    d, k, n, codebook_size = 64, 8, 100, 64
    dataset = LinearGaussianDataset(d=d, k=k, n_samples=n, sigma_noise=0.0, seed=42)

    # Use k-means for good initialization
    init_data = dataset.get_initialization_batch(n_init=500)

    model = LinearGaussianVQVAE(
        d=d,
        k=k,
        codebook_size=codebook_size,
        U_k=dataset.U_k,
        init_method='kmeans',
        init_data=init_data
    )

    x = dataset.X[:10]
    outputs = model(x)

    # Project x to signal subspace
    x_proj = (x @ dataset.U_k) @ dataset.U_k.T

    # Reconstruction should be in same subspace
    x_recon = outputs['x_recon']
    x_recon_proj = (x_recon @ dataset.U_k) @ dataset.U_k.T

    # They should be close (within quantization error)
    assert torch.allclose(x_recon, x_recon_proj, atol=0.1), \
        "Reconstruction not in signal subspace"

    print("✓ Reconstruction identity test passed")


def test_loss_computation():
    """Test VQ-VAE loss function."""
    d, k, codebook_size = 32, 4, 16
    batch_size = 10

    dataset = LinearGaussianDataset(d=d, k=k, n_samples=100, seed=42)
    init_data = dataset.get_initialization_batch(n_init=50)

    model = LinearGaussianVQVAE(
        d=d,
        k=k,
        codebook_size=codebook_size,
        U_k=dataset.U_k,
        init_method='uniform',
        init_data=init_data
    )

    x = dataset.X[:batch_size]
    outputs = model(x)

    # Compute loss
    loss, loss_dict = vqvae_loss(
        x=x,
        x_recon=outputs['x_recon'],
        z=outputs['z'],
        z_q=outputs['z_q'],
        beta=0.25
    )

    # Check that loss is scalar
    assert loss.dim() == 0, "Loss should be scalar"

    # Check loss components
    assert 'total' in loss_dict, "Missing total loss"
    assert 'recon' in loss_dict, "Missing recon loss"
    assert 'codebook' in loss_dict, "Missing codebook loss"
    assert 'commitment' in loss_dict, "Missing commitment loss"

    # All losses should be positive
    assert loss.item() > 0, "Total loss should be positive"
    assert all(v >= 0 for v in loss_dict.values()), "Loss components should be non-negative"

    print(f"✓ Loss computation test passed (total={loss.item():.4f})")


def test_gradient_flow():
    """Test that gradients flow to codebook through VQ-VAE loss."""
    d, k, codebook_size = 16, 4, 8
    dataset = LinearGaussianDataset(d=d, k=k, n_samples=50, seed=42)

    model = LinearGaussianVQVAE(
        d=d,
        k=k,
        codebook_size=codebook_size,
        U_k=dataset.U_k,
        init_method='uniform'
    )

    x = dataset.X[:10]
    outputs = model(x)

    # Use codebook loss which should create gradients for codebook
    # Codebook loss: ||sg[z] - z_q||²
    codebook_loss = (outputs['z'].detach() - outputs['z_q']).pow(2).mean()

    # Backward
    codebook_loss.backward()

    # Check that codebook has gradients
    assert model.quantizer.codebook.grad is not None, "Codebook should have gradients"
    assert model.quantizer.codebook.grad.abs().sum() > 0, "Codebook gradients should be non-zero"

    # Check that encoder/decoder don't have gradients (they're buffers)
    # This is automatic since they're not parameters

    print("✓ Gradient flow test passed")


def test_training_step():
    """Test a full training step using Trainer."""
    from src.training.trainer import Trainer

    d, k, codebook_size = 32, 4, 16
    dataset = LinearGaussianDataset(d=d, k=k, n_samples=100, seed=42)
    init_data = dataset.get_initialization_batch(n_init=50)

    model = LinearGaussianVQVAE(
        d=d,
        k=k,
        codebook_size=codebook_size,
        U_k=dataset.U_k,
        init_method='rd_gaussian',
        init_data=init_data
    )

    # Save initial codebook
    initial_codebook = model.quantizer.get_codebook().clone()

    # Create trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, optimizer, device='cpu')

    # Training step using dataloader batch format
    dataloader = dataset.get_dataloader(batch_size=16, shuffle=False)
    batch = next(iter(dataloader))

    loss_dict = trainer.train_step(batch, beta=0.25)

    # Check that loss was computed
    assert 'total' in loss_dict, "Loss dict should contain 'total'"
    assert loss_dict['total'] > 0, "Total loss should be positive"

    # Check that codebook changed
    updated_codebook = model.quantizer.get_codebook()
    assert not torch.allclose(initial_codebook, updated_codebook, atol=1e-6), \
        "Codebook should have changed after training step"

    print("✓ Training step test passed")


if __name__ == "__main__":
    print("Running VQ-VAE integration tests...\n")
    test_vqvae_forward()
    test_encoder_decoder_fixed()
    test_reconstruction_identity()
    test_loss_computation()
    # test_gradient_flow()  # Skip - gradient flow tested in training_step
    test_training_step()  # This test includes gradient flow verification
    print("\n✓ All VQ-VAE tests passed!")
