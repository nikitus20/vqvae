"""Utility functions for VQ-VAE research framework."""

import torch
import numpy as np
import random


def set_seed(seed: int):
    """Set all random seeds for reproducibility.

    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)

    Args:
        seed: Random seed value

    Example:
        >>> from src.utils import set_seed
        >>> set_seed(42)
        >>> # Now all random operations are reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters

    Example:
        >>> from src.utils import count_parameters
        >>> from src.models import VQVAE
        >>> num_params = count_parameters(model)
        >>> print(f"Model has {num_params:,} trainable parameters")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the appropriate device for computation.

    Args:
        prefer_cuda: Whether to prefer CUDA if available

    Returns:
        torch.device: Either 'cuda' or 'cpu'

    Example:
        >>> from src.utils import get_device
        >>> device = get_device()
        >>> model = model.to(device)
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
