"""Loss functions for VQ-VAE training."""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict


def vqvae_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    z: torch.Tensor,
    z_q: torch.Tensor,
    z_q_hard: torch.Tensor,
    beta: float = 0.25
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """VQ-VAE loss from original paper (van den Oord et al., 2017).

    Total loss combines three terms:
        L = ||x - x̂||² + ||sg[z] - q||² + β||z - sg[q]||²

    where sg[·] denotes stop-gradient and q is hard quantization.

    Components:
    1. Reconstruction loss: Minimize output error
    2. Codebook loss: Move codebook entries toward encoder outputs
    3. Commitment loss: Commit encoder outputs to codebook

    Args:
        x: (B, d) original data
        x_recon: (B, d) reconstructed data
        z: (B, k) encoder output
        z_q: (B, k) quantized latent codes (with STE/rotation gradient)
        z_q_hard: (B, k) hard quantization (nearest codebook entries)
        beta: Commitment loss weight (default 0.25 from paper)

    Returns:
        total_loss: Scalar loss for backpropagation
        loss_dict: Dictionary with individual loss components
    """
    # 1. Reconstruction loss: ||x - x̂||²
    recon_loss = F.mse_loss(x_recon, x)

    # 2. Codebook loss: ||sg[z] - q||²
    # CRITICAL: Use z_q_hard (not z_q) for correct gradient flow to codebook
    # Stop gradient on encoder output, update codebook
    codebook_loss = F.mse_loss(z.detach(), z_q_hard)

    # 3. Commitment loss: β||z - sg[q]||²
    # Stop gradient on quantized output, update encoder
    # (Not used in Linear Gaussian case since encoder is fixed)
    commitment_loss = F.mse_loss(z, z_q_hard.detach())

    # Total loss
    total_loss = recon_loss + codebook_loss + beta * commitment_loss

    # Return loss components for logging
    loss_dict = {
        'total': total_loss.item(),
        'recon': recon_loss.item(),
        'codebook': codebook_loss.item(),
        'commitment': commitment_loss.item()
    }

    return total_loss, loss_dict
