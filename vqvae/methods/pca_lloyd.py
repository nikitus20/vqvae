"""Method 1: PCA + Lloyd Max (k-means) - Baseline."""

import numpy as np
from typing import Dict, Any
from ..models.vqvae import VQVAE
from ..utils import metrics


def train_pca_lloyd(
    X_train: np.ndarray,
    X_eval: np.ndarray,
    m: int,
    K: int,
    lloyd_iters: int = 30,
    seed: int = 0
) -> Dict[str, Any]:
    """
    Baseline method: PCA for encoder + Lloyd Max for codebook.

    1. Compute covariance Sigma from training data
    2. Set W = top-m eigenvectors of Sigma
    3. Compute Z = X_train @ W^T
    4. Run k-means (Lloyd) on Z to get codebook E

    Args:
        X_train: (N, D) training data
        X_eval: (N_eval, D) evaluation data
        m: Latent dimension
        K: Number of codebook entries
        lloyd_iters: Number of Lloyd iterations for k-means
        seed: Random seed

    Returns:
        Dictionary with:
            - model: Trained VQVAE model
            - metrics: Evaluation metrics
            - logs: Training logs (empty for this method)
    """
    D = X_train.shape[1]

    # Create model
    model = VQVAE(D, m, K, orthonormal_encoder=True)

    # 1. PCA for encoder
    Sigma = np.cov(X_train, rowvar=False, bias=True)
    model.encoder.init_from_pca(Sigma)

    # 2. Lloyd Max for codebook
    Z_train = model.encode(X_train)
    model.codebook.init_from_kmeans(Z_train, iters=lloyd_iters, seed=seed)

    # 3. Evaluate
    mets = metrics(model.encoder.W, model.codebook.E, X_eval)

    return {
        "model": model,
        "metrics": mets,
        "logs": []
    }
