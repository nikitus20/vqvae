"""Metrics tracking for VQ-VAE experiments."""

import torch
import pandas as pd
from collections import defaultdict
from typing import Dict
from pathlib import Path


class MetricsTracker:
    """Track and compute experimental metrics for VQ-VAE.

    Args:
        codebook_size: Number of codebook entries
    """

    def __init__(self, codebook_size: int):
        self.codebook_size = codebook_size
        self.metrics = defaultdict(list)

    def update(self, step: int, **kwargs):
        """Add metrics for a given step.

        Args:
            step: Training step number
            **kwargs: Metric name-value pairs
        """
        self.metrics['step'].append(step)
        for key, value in kwargs.items():
            self.metrics[key].append(value)

    def compute_utilization_metrics(self, indices: torch.Tensor) -> Dict[str, float]:
        """Compute codebook utilization metrics.

        Args:
            indices: (N,) tensor of assigned codeword indices

        Returns:
            Dictionary containing:
                - perplexity: exp(entropy) of usage distribution
                - dead_codes: number of unused codes
                - min_usage: minimum usage probability
                - max_usage: maximum usage probability
                - usage_std: std of usage distribution
                - usage_entropy: entropy in nats
        """
        # Count usage of each code
        counts = torch.bincount(
            indices.flatten(),
            minlength=self.codebook_size
        ).float()
        usage = counts / counts.sum()

        # Perplexity: exp(H[usage])
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        entropy = -(usage * torch.log(usage + epsilon)).sum()
        perplexity = torch.exp(entropy)

        # Dead codes (usage below threshold)
        dead_threshold = 1e-4
        dead_codes = (usage < dead_threshold).sum()

        return {
            'perplexity': perplexity.item(),
            'dead_codes': dead_codes.item(),
            'min_usage': usage[usage > 0].min().item() if (usage > 0).any() else 0.0,
            'max_usage': usage.max().item(),
            'usage_std': usage.std().item(),
            'usage_entropy': entropy.item()
        }

    def compute_distortion_metrics(
        self,
        z: torch.Tensor,
        z_q: torch.Tensor
    ) -> Dict[str, float]:
        """Compute quantization distortion.

        Args:
            z: (N, k) encoder outputs
            z_q: (N, k) quantized outputs

        Returns:
            Dictionary containing:
                - quantization_error: E[||z - ẑ||²]
                - quant_error_per_dim: mean squared error per dimension
        """
        squared_error = (z - z_q).pow(2)

        return {
            'quantization_error': squared_error.mean().item(),
            'quant_error_per_dim': squared_error.mean(dim=0).mean().item()
        }

    def compute_codebook_stats(self, codebook: torch.Tensor) -> Dict[str, float]:
        """Compute statistics of codebook.

        Args:
            codebook: (n, k) codebook tensor

        Returns:
            Dictionary containing:
                - codebook_mean_norm: average ||c_j||
                - codebook_std_norm: std of ||c_j||
                - codebook_mean_distance: average pairwise distance
        """
        # Norms of codebook entries
        norms = codebook.norm(dim=1)

        # Pairwise distances (subsample if codebook is large)
        if len(codebook) > 100:
            idx = torch.randperm(len(codebook))[:100]
            sub_codebook = codebook[idx]
        else:
            sub_codebook = codebook

        # Compute pairwise distances
        pdist = torch.cdist(sub_codebook, sub_codebook)
        # Exclude diagonal (distance to self = 0)
        mean_distance = pdist[pdist > 0].mean()

        return {
            'codebook_mean_norm': norms.mean().item(),
            'codebook_std_norm': norms.std().item(),
            'codebook_mean_distance': mean_distance.item()
        }

    def save(self, path: str):
        """Save metrics to CSV file.

        Args:
            path: Path to save CSV
        """
        df = pd.DataFrame(self.metrics)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def load(self, path: str):
        """Load metrics from CSV file.

        Args:
            path: Path to CSV file
        """
        df = pd.read_csv(path)
        self.metrics = df.to_dict('list')

    def get_final_metrics(self) -> Dict[str, float]:
        """Get metrics from final step.

        Returns:
            Dictionary with final metric values
        """
        if len(self.metrics['step']) == 0:
            return {}

        final_metrics = {}
        for key, values in self.metrics.items():
            if key != 'step' and len(values) > 0:
                final_metrics[key] = values[-1]

        return final_metrics

    def __repr__(self) -> str:
        """String representation."""
        n_steps = len(self.metrics.get('step', []))
        n_metrics = len(self.metrics) - 1  # Exclude 'step'
        return f"MetricsTracker(steps={n_steps}, metrics={n_metrics})"
