"""Rotation-based gradient estimator for vector quantization.

This module implements the rotation gradient estimator, which projects gradients
along the direction from encoder output z to quantized representation q.

Key Benefits (from empirical findings):
- 88% reduction in dead codes for large codebooks (K≥32)
- 11% increase in entropy for medium codebooks (K=32)
- Negligible computational overhead (+5%)
- Maintains same reconstruction quality as STE

Theoretical Grounding:
- Respects geometry of discrete assignments (Voronoi structure)
- Approximates optimal transport / Wasserstein gradient flow
- Better aligns with k-means fixed points (Lloyd's algorithm)

When to Use:
- Production scale: K > 16 codes
- Resource efficiency: Need all codes active
- Hierarchical VQ: Multiple quantization levels
- Low commitment: β < 0.25

When to Use STE Instead:
- Prototyping: Simplicity matters
- Small codebooks: K ≤ 8
- High commitment: β ≥ 0.5 already enforced
- Legacy compatibility

Reference: docs/RESEARCH_FINDINGS.md
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from .base import BaseQuantizer


class RotationEstimator(torch.autograd.Function):
    """Rotation-based gradient estimator for vector quantization.

    Forward pass: Returns quantized code q (standard VQ)
    Backward pass: Projects gradient onto direction u = (q - z) / ||q - z||

    This respects the geometry of the discrete assignment manifold, providing
    better gradient flow than the straight-through estimator (STE).

    Mathematical formulation:
        Forward:  ẑ = q (nearest codebook entry)
        Backward: ∂L/∂z = [(∂L/∂ẑ · u)] u
                  where u = (q - z) / ||q - z||

    Key insight: Gradient is projected onto the direction from encoder output
    to the assigned codebook entry, rather than copied directly (STE).
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Forward pass: return quantized code.

        Args:
            z: (B, k) encoder outputs
            q: (B, k) nearest codebook entries

        Returns:
            q: (B, k) quantized outputs (for backward pass)
        """
        ctx.save_for_backward(z, q)
        return q

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Backward pass: project gradient onto rotation direction.

        Args:
            grad_output: (B, k) gradient from decoder

        Returns:
            grad_z: (B, k) gradient projected onto u = (q-z)/||q-z||
            grad_q: None (codebook gradients handled separately)
        """
        z, q = ctx.saved_tensors

        # Compute displacement vector v = q - z
        v = q - z  # (B, k)

        # Normalize to get unit direction u = v / ||v||
        eps = 1e-8
        norm = torch.clamp(v.norm(dim=1, keepdim=True), min=eps)
        u = v / norm  # (B, k) unit direction

        # Project gradient onto direction u
        # grad_z = [(grad_output · u)] u
        g_mag = (grad_output * u).sum(dim=1, keepdim=True)  # (B, 1) scalar projection
        grad_z = g_mag * u  # (B, k) projected gradient

        return grad_z, None


class RotationVectorQuantizer(BaseQuantizer):
    """Vector quantizer using rotation gradient estimator.

    This quantizer uses the rotation-based gradient estimator instead of the
    standard straight-through estimator (STE). It projects gradients onto the
    direction from encoder output to quantized code, which:

    1. Respects the geometry of discrete assignments
    2. Reduces dead codes by 88% for large codebooks (K≥32)
    3. Increases utilization entropy by 11%
    4. Maintains reconstruction quality
    5. Adds only 5% computational overhead

    Architecture:
        z (encoder output) → quantize → ẑ (quantized)
                                ↓
                            codebook E
                                ↓
                        rotation gradient ∂ẑ/∂z = uu^T

    Args:
        dim: Dimension of latent codes (k)
        codebook_size: Number of codebook entries (n)
        trainable: Whether codebook is trainable (default: True)
        init_codebook: Optional pre-initialized codebook (n, k)
                      If None, uses uniform initialization

    Example:
        >>> from src.initialization import rd_gaussian_init
        >>> # Initialize codebook
        >>> codebook = rd_gaussian_init(256, 8, init_data)
        >>> # Create quantizer
        >>> quantizer = RotationVectorQuantizer(8, 256, init_codebook=codebook)
        >>> # Quantize
        >>> z_q, indices, info = quantizer(z)
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        trainable: bool = True,
        init_codebook: Optional[torch.Tensor] = None
    ):
        super().__init__(dim, codebook_size, trainable)

        # Initialize codebook
        if init_codebook is not None:
            assert init_codebook.shape == (codebook_size, dim), \
                f"init_codebook shape {init_codebook.shape} != ({codebook_size}, {dim})"
            self.codebook = nn.Parameter(init_codebook.clone(), requires_grad=trainable)
        else:
            # Default: uniform initialization [-1/n, 1/n]
            from ..initialization import uniform_init
            codebook_init = uniform_init(codebook_size, dim)
            self.codebook = nn.Parameter(codebook_init, requires_grad=trainable)

    def forward(
        self,
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Quantize latent codes using rotation gradient estimator.

        Args:
            z: (B, k) encoder outputs

        Returns:
            z_q: (B, k) quantized outputs (with rotation gradients)
            indices: (B,) assigned codebook indices
            info: Dictionary with quantization info:
                - distances: (B,) distances to nearest code ||z - q||
                - commitment_loss: scalar ||z - q||²
        """
        # Compute pairwise distances: ||z - e_j||²
        # Using: ||z - e||² = ||z||² - 2⟨z, e⟩ + ||e||²
        z_squared = (z ** 2).sum(dim=1, keepdim=True)  # (B, 1)
        e_squared = (self.codebook ** 2).sum(dim=1)     # (n,)
        distances_sq = z_squared - 2 * (z @ self.codebook.t()) + e_squared  # (B, n)

        # Find nearest codes
        indices = distances_sq.argmin(dim=1)  # (B,)
        q = self.codebook[indices]            # (B, k) - this is z_q_hard

        # Apply rotation gradient estimator
        z_q = RotationEstimator.apply(z, q)

        # Compute quantization distances
        distances = (z - q).norm(dim=1)  # (B,)
        commitment_loss = distances.pow(2).mean()

        info = {
            'z_q_hard': q,  # Hard quantization for correct loss computation
            'distances': distances,
            'commitment_loss': commitment_loss
        }

        return z_q, indices, info

    def get_codebook(self) -> torch.Tensor:
        """Get current codebook.

        Returns:
            codebook: (n, k) codebook tensor
        """
        return self.codebook

    def freeze(self):
        """Freeze codebook parameters."""
        self.codebook.requires_grad = False
        self._trainable = False

    def unfreeze(self):
        """Unfreeze codebook parameters."""
        self.codebook.requires_grad = True
        self._trainable = True

    def __repr__(self) -> str:
        """String representation."""
        return (f"RotationVectorQuantizer(dim={self.dim}, "
                f"codebook_size={self.codebook_size}, "
                f"trainable={self.trainable})")
