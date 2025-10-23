"""Trainer class for VQ-VAE experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict
from .losses import vqvae_loss


class Trainer:
    """Training loop for VQ-VAE experiments.

    Args:
        model: VQ-VAE model
        optimizer: PyTorch optimizer
        device: Device for training ('cpu' or 'cuda')
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

    def train_step(
        self,
        batch: torch.Tensor,
        beta: float = 0.25
    ) -> Dict[str, float]:
        """Execute single training step.

        Args:
            batch: Batch from DataLoader (tuple of (x, y))
            beta: Commitment loss weight

        Returns:
            Dictionary with loss components
        """
        self.model.train()

        # Extract x from batch (batch is (x, y) tuple)
        if isinstance(batch, (list, tuple)):
            x = batch[0].to(self.device)
        else:
            x = batch.to(self.device)

        # Forward pass
        outputs = self.model(x)

        # Compute loss
        loss, loss_dict = vqvae_loss(
            x=x,
            x_recon=outputs['x_recon'],
            z=outputs['z'],
            z_q=outputs['z_q'],
            beta=beta
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_dict

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        metrics_tracker: 'MetricsTracker'
    ) -> Dict[str, float]:
        """Evaluate model on full dataset.

        Args:
            dataloader: DataLoader for evaluation
            metrics_tracker: MetricsTracker instance for computing metrics

        Returns:
            Dictionary with all metrics
        """
        self.model.eval()

        all_indices = []
        all_z = []
        all_z_q = []
        total_recon_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            # Extract x from batch
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(self.device)
            else:
                x = batch.to(self.device)

            # Forward pass
            outputs = self.model(x)

            # Collect for metrics
            all_indices.append(outputs['indices'].cpu())
            all_z.append(outputs['z'].cpu())
            all_z_q.append(outputs['z_q'].cpu())

            # Compute reconstruction loss
            recon_loss = F.mse_loss(outputs['x_recon'], x)
            total_recon_loss += recon_loss.item()
            n_batches += 1

        # Concatenate all batches
        indices = torch.cat(all_indices, dim=0)
        z = torch.cat(all_z, dim=0)
        z_q = torch.cat(all_z_q, dim=0)

        # Compute comprehensive metrics
        utilization_metrics = metrics_tracker.compute_utilization_metrics(indices)
        distortion_metrics = metrics_tracker.compute_distortion_metrics(z, z_q)
        codebook_stats = metrics_tracker.compute_codebook_stats(
            self.model.quantizer.get_codebook().cpu()
        )

        # Combine all metrics
        metrics = {
            'recon_loss': total_recon_loss / n_batches,
            **utilization_metrics,
            **distortion_metrics,
            **codebook_stats
        }

        return metrics

    def save_checkpoint(self, path: str):
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
