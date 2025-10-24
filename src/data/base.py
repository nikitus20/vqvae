"""Base interface for datasets."""

import torch
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class BaseDataset(ABC):
    """Abstract base class for VQ-VAE datasets.

    All dataset implementations should inherit from this class and implement
    the required abstract methods.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        raise NotImplementedError

    @abstractmethod
    def get_dataloader(
        self,
        batch_size: int = 64,
        shuffle: bool = True
    ) -> DataLoader:
        """Create PyTorch DataLoader for training.

        Args:
            batch_size: Batch size for data loading
            shuffle: Whether to shuffle data

        Returns:
            DataLoader instance
        """
        raise NotImplementedError

    @abstractmethod
    def get_initialization_batch(
        self,
        n_init: int = 1000
    ) -> torch.Tensor:
        """Get batch for codebook initialization.

        Args:
            n_init: Number of samples for initialization

        Returns:
            Tensor of shape (n_init, latent_dim) for codebook initialization
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def ambient_dim(self) -> int:
        """Return ambient dimension (d) of data."""
        raise NotImplementedError

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Return latent dimension (k) for encoding."""
        raise NotImplementedError
