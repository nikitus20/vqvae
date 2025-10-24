"""Legacy VQ-VAE models for backward compatibility.

⚠️  DEPRECATED: This module is maintained for backward compatibility only.
    New code should use the composable VQVAE with separate encoder/decoder components:

    from src.encoders import PCAEncoder
    from src.decoders import PCADecoder
    from src.quantizers import VectorQuantizer
    from src.models import VQVAE

    encoder = PCAEncoder(U_k, trainable=False)
    decoder = PCADecoder(U_k, trainable=False)
    quantizer = VectorQuantizer(k, n, init_codebook=codebook)
    model = VQVAE(encoder, quantizer, decoder)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from ..quantizers import VectorQuantizer


class LinearGaussianVQVAE(nn.Module):
    """VQ-VAE with fixed PCA encoder/decoder for Linear Gaussian experiments.

    ⚠️  DEPRECATED: Use composable VQVAE with PCAEncoder, VectorQuantizer, and PCADecoder instead.

    This class is maintained only for backward compatibility with old code.
    It will be removed in a future version.

    Architecture:
    1. Encoder: z = U_k^T @ x  [FIXED - optimal PCA solution]
    2. Quantizer: ẑ = Q(z)     [TRAINABLE - learned codebook]
    3. Decoder: x̂ = U_k @ ẑ    [FIXED - transpose of encoder]

    For the Linear Gaussian model X = AY + W, the optimal encoder/decoder
    is given by PCA: U_k = top k eigenvectors of data covariance.

    Only the codebook is trained; encoder and decoder are fixed.

    Args:
        d: Ambient dimension
        k: Latent dimension
        codebook_size: Number of codewords
        U_k: (d, k) PCA eigenvectors (fixed encoder/decoder)
        init_method: Codebook initialization ('uniform', 'kmeans', 'rd_gaussian')
        init_data: Optional initialization data (for kmeans/rd_gaussian)
    """

    def __init__(
        self,
        d: int,
        k: int,
        codebook_size: int,
        U_k: torch.Tensor,
        init_method: str = 'uniform',
        init_data: Optional[torch.Tensor] = None
    ):
        super().__init__()

        self.d = d
        self.k = k
        self.codebook_size = codebook_size

        # Fixed encoder/decoder (PCA solution)
        # Register as buffers (not trained, but part of state_dict)
        # Note: Store in natural orientation for matrix multiplication
        self.register_buffer('encoder_weight', U_k)    # (d, k) for z = x @ U_k
        self.register_buffer('decoder_weight', U_k.T)  # (k, d) for x̂ = z @ U_k^T

        # Create codebook using initialization method
        if init_method == 'uniform':
            from ..initialization import uniform_init
            init_codebook = uniform_init(codebook_size, k)
        elif init_method == 'kmeans':
            if init_data is None:
                raise ValueError("init_data required for kmeans initialization")
            from ..initialization import kmeans_init
            init_codebook = kmeans_init(codebook_size, k, init_data)
        elif init_method == 'rd_gaussian':
            if init_data is None:
                raise ValueError("init_data required for rd_gaussian initialization")
            from ..initialization import rd_gaussian_init
            init_codebook = rd_gaussian_init(codebook_size, k, init_data)
        else:
            raise ValueError(f"Unknown init_method: {init_method}")

        # Trainable quantizer
        self.quantizer = VectorQuantizer(
            dim=k,
            codebook_size=codebook_size,
            init_codebook=init_codebook
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode: z = U_k^T @ x.

        Args:
            x: (B, d) input data

        Returns:
            z: (B, k) latent codes
        """
        # z = x @ U_k, which is equivalent to U_k^T @ x^T then transpose
        z = x @ self.encoder_weight  # (B, d) @ (d, k) -> (B, k)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode: x̂ = U_k @ z.

        Args:
            z: (B, k) latent codes

        Returns:
            x_recon: (B, d) reconstructed data
        """
        # x̂ = z @ U_k^T, which is equivalent to U_k @ z^T then transpose
        x_recon = z @ self.decoder_weight  # (B, k) @ (k, d) -> (B, d)
        return x_recon

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full forward pass through VQ-VAE.

        Args:
            x: (B, d) input data

        Returns:
            Dictionary containing:
                - x_recon: (B, d) reconstructed data
                - z: (B, k) encoder output
                - z_q: (B, k) quantized latent codes
                - indices: (B,) codebook indices
        """
        # Encode
        z = self.encode(x)

        # Quantize
        z_q, indices, _ = self.quantizer(z)

        # Decode
        x_recon = self.decode(z_q)

        return {
            'x_recon': x_recon,
            'z': z,
            'z_q': z_q,
            'indices': indices
        }

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method for reconstruction.

        Args:
            x: (B, d) input data

        Returns:
            x_recon: (B, d) reconstructed data
        """
        return self.forward(x)['x_recon']

    def __repr__(self) -> str:
        """String representation."""
        return (f"LinearGaussianVQVAE(d={self.d}, k={self.k}, "
                f"codebook_size={self.codebook_size})")
