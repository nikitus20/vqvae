"""Projection-based gradient estimator for vector quantization."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from .base import BaseQuantizer


class ProjectionEstimator(torch.autograd.Function):
    """Projection-based gradient estimator for VQ.

    Projects gradient onto the direction from encoder output z to quantized code q.

    Math:
        Forward:  z_q = q
        Backward: ∂L/∂z = (u^T g) u  where u = (q - z)/||q - z||, g = ∂L/∂z_q

    This is a rank-1 projection that enforces gradient flow only along the
    quantization direction, potentially helping encoder outputs move toward codebook.
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(z, q)
        return q

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        z, q = ctx.saved_tensors

        # Direction from z to q
        v = q - z
        norm = v.norm(dim=1, keepdim=True).clamp(min=1e-8)
        u = v / norm  # (B, k) unit direction

        # Project gradient onto u: grad_z = (u^T g) u
        proj = (grad_output * u).sum(dim=1, keepdim=True)  # (B, 1)
        grad_z = proj * u  # (B, k)

        return grad_z, None


class ProjectionVectorQuantizer(BaseQuantizer):
    """Vector quantizer using projection gradient estimator.

    Projects gradients onto the direction from encoder output to quantized code.

    Args:
        dim: Latent dimension (k)
        codebook_size: Number of codewords (n)
        trainable: Whether codebook is trainable
        init_codebook: Optional initialization (n, k)
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        trainable: bool = True,
        init_codebook: Optional[torch.Tensor] = None
    ):
        super().__init__(dim, codebook_size, trainable)

        if init_codebook is not None:
            assert init_codebook.shape == (codebook_size, dim)
            self.codebook = nn.Parameter(init_codebook.clone(), requires_grad=trainable)
        else:
            from ..initialization import uniform_init
            self.codebook = nn.Parameter(
                uniform_init(codebook_size, dim), requires_grad=trainable
            )

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Quantize with projection gradient estimator.

        Args:
            z: (B, k) encoder outputs

        Returns:
            z_q: (B, k) quantized outputs
            indices: (B,) codebook indices
            info: {'z_q_hard', 'distances', 'commitment_loss'}
        """
        # Find nearest codes
        z_sq = (z ** 2).sum(dim=1, keepdim=True)  # (B, 1)
        e_sq = (self.codebook ** 2).sum(dim=1)     # (n,)
        dist_sq = z_sq - 2 * (z @ self.codebook.t()) + e_sq  # (B, n)

        indices = dist_sq.argmin(dim=1)  # (B,)
        q = self.codebook[indices]       # (B, k)

        # Apply projection estimator
        z_q = ProjectionEstimator.apply(z, q)

        # Metrics
        distances = (z - q).norm(dim=1)
        commitment_loss = distances.pow(2).mean()

        return z_q, indices, {
            'z_q_hard': q,
            'distances': distances,
            'commitment_loss': commitment_loss
        }

    def get_codebook(self) -> torch.Tensor:
        return self.codebook

    def freeze(self):
        self.codebook.requires_grad = False
        self._trainable = False

    def unfreeze(self):
        self.codebook.requires_grad = True
        self._trainable = True
