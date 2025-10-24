"""Composable VQ-VAE model."""

import torch
import torch.nn as nn
from typing import Dict
from .base import BaseVQVAE
from ..encoders.base import BaseEncoder
from ..quantizers.base import BaseQuantizer
from ..decoders.base import BaseDecoder


class VQVAE(BaseVQVAE):
    """Composable VQ-VAE model.

    Combines encoder, quantizer, and decoder into a full VQ-VAE architecture.
    Each component can be any implementation of the respective base class.

    Architecture:
        x → encoder → z → quantizer → z_q → decoder → x_recon

    Args:
        encoder: Encoder module (BaseEncoder)
        quantizer: Quantizer module (BaseQuantizer)
        decoder: Decoder module (BaseDecoder)

    Example:
        >>> # Create components
        >>> encoder = PCAEncoder(U_k, trainable=False)
        >>> quantizer = VectorQuantizer(k, n, init_codebook=codebook)
        >>> decoder = PCADecoder(U_k, trainable=False)
        >>> # Compose model
        >>> model = VQVAE(encoder, quantizer, decoder)
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        quantizer: BaseQuantizer,
        decoder: BaseDecoder
    ):
        super().__init__()

        # Validate dimensions match
        assert encoder.latent_dim == quantizer.dim, \
            f"Encoder latent_dim {encoder.latent_dim} != quantizer dim {quantizer.dim}"
        assert quantizer.dim == decoder.latent_dim, \
            f"Quantizer dim {quantizer.dim} != decoder latent_dim {decoder.latent_dim}"

        # Store components
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder

        # Store dimensions
        self.input_dim = encoder.input_dim
        self.latent_dim = encoder.latent_dim
        self.output_dim = decoder.output_dim
        self.codebook_size = quantizer.codebook_size

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation.

        Args:
            x: (B, d) input data

        Returns:
            z: (B, k) latent codes
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent codes to reconstruction.

        Args:
            z: (B, k) latent codes

        Returns:
            x_recon: (B, d) reconstructed data
        """
        return self.decoder(z)

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
                - (optional) additional quantizer-specific info
        """
        # Encode
        z = self.encoder(x)

        # Quantize
        z_q, indices, quantizer_info = self.quantizer(z)

        # Decode
        x_recon = self.decoder(z_q)

        # Combine outputs
        outputs = {
            'x_recon': x_recon,
            'z': z,
            'z_q': z_q,
            'indices': indices,
        }

        # Add any additional quantizer info
        outputs.update(quantizer_info)

        return outputs

    def __repr__(self) -> str:
        """String representation."""
        encoder_name = self.encoder.__class__.__name__
        quantizer_name = self.quantizer.__class__.__name__
        decoder_name = self.decoder.__class__.__name__

        return (f"VQVAE(\n"
                f"  encoder={encoder_name}(d={self.input_dim}, k={self.latent_dim}),\n"
                f"  quantizer={quantizer_name}(k={self.latent_dim}, n={self.codebook_size}),\n"
                f"  decoder={decoder_name}(k={self.latent_dim}, d={self.output_dim})\n"
                f")")
