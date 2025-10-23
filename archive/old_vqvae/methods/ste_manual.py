"""Method 3: SGD with STE using manually computed exact gradients."""

import numpy as np
from typing import Dict, Any
from dataclasses import dataclass

from ..models.vqvae import VQVAE
from ..utils import metrics


@dataclass
class STEConfig:
    """Configuration for manual STE training."""
    steps: int = 200
    batch: int = 1024
    alpha: float = 1.0
    beta: float = 0.25
    lrW: float = 0.08
    lrE: float = 0.2
    seed: int = 0
    log_every: int = 20


def train_ste_manual(
    X_train: np.ndarray,
    X_eval: np.ndarray,
    m: int,
    K: int,
    cfg: STEConfig = None
) -> Dict[str, Any]:
    """
    Train VQ-VAE using manually computed STE gradients.

    Gradients derived analytically:
    - Codebook: ∇E_j = (2α/|B_j|) Σ_{i∈B_j} (E_j - Z_i)
    - Encoder: ∇W = -2(Q^T R + (W R^T)^T X + β(Z-Q)^T X)/B + Riemannian projection

    Args:
        X_train: (N, D) training data
        X_eval: (N_eval, D) evaluation data
        m: Latent dimension
        K: Number of codebook entries
        cfg: Configuration object

    Returns:
        Dictionary with:
            - model: Trained VQVAE model
            - metrics: Evaluation metrics
            - logs: Training logs
    """
    if cfg is None:
        cfg = STEConfig()

    rng = np.random.default_rng(cfg.seed)
    D = X_train.shape[1]

    # Initialize model with PCA + k-means
    model = VQVAE(D, m, K, orthonormal_encoder=True)
    Sigma = np.cov(X_train, rowvar=False, bias=True)
    model.encoder.init_from_pca(Sigma)
    Z0 = model.encode(X_train[:8000])
    model.codebook.init_from_kmeans(Z0, iters=20, seed=cfg.seed)

    # Aliases
    W = model.encoder.W
    E = model.codebook.E

    logs = []
    N = X_train.shape[0]

    for t in range(cfg.steps):
        # Sample batch
        idx = rng.integers(0, N, size=cfg.batch)
        xb = X_train[idx]

        # Forward pass
        Z = xb @ W.T  # (B, m)
        k_idx, Q = model.codebook.nearest_code(Z)  # (B, m)
        Xhat = Q @ W  # (B, D)

        # Residual
        R = xb - Xhat  # (B, D)

        # Compute losses
        rec = np.mean(np.sum(R**2, axis=1))
        dist = np.sum((Z - Q)**2, axis=1)
        codebook_loss = cfg.alpha * np.mean(dist)
        commit_loss = cfg.beta * np.mean(dist)
        total_loss = rec + codebook_loss + commit_loss

        # ============ Codebook gradient ============
        # ∇E_j = (2α/|B_j|) Σ_{i∈B_j} (E_j - Z_i)
        grad_E = np.zeros_like(E)
        for j in range(K):
            idxj = np.where(k_idx == j)[0]
            if idxj.size > 0:
                grad_E[j] = 2 * cfg.alpha * (E[j] - Z[idxj].mean(axis=0))

        # ============ Encoder gradient ============
        # Three terms from chain rule:
        # 1. Reconstruction: -2 Q^T R / B
        # 2. Reconstruction through W: -2 (W R^T)^T X / B
        # 3. Commitment: 2β (Z - Q)^T X / B

        WR = R @ W.T  # (B, m)

        term1 = -2 * (Q.T @ R) / cfg.batch  # (m, D)
        term2 = -2 * (WR.T @ xb) / cfg.batch  # (m, D)
        term3 = 2 * cfg.beta * ((Z - Q).T @ xb) / cfg.batch  # (m, D)

        G = term1 + term2 + term3  # (m, D)

        # Riemannian projection to Stiefel tangent space
        # grad_riem = G - (1/2)(G^T W + W^T G)^T W
        # Note: sym = (G^T @ W + W^T @ G) is (D, D), so we need (W @ sym)
        sym = G.T @ W + W.T @ G  # (D, D)
        grad_W_riem = G - 0.5 * (W @ sym)  # (m, D)

        # ============ SGD update ============
        E -= cfg.lrE * grad_E
        W -= cfg.lrW * grad_W_riem

        # Project W back to Stiefel
        model.encoder.W = W
        model.encoder.project()
        W = model.encoder.W

        # ============ Logging ============
        if (t % cfg.log_every == 0) or (t == cfg.steps - 1):
            mets = metrics(W, E, X_eval)
            log_entry = {
                "step": t,
                "loss": float(total_loss),
                "rec": float(rec),
                "codebook": float(codebook_loss),
                "commit": float(commit_loss),
                **mets
            }
            logs.append(log_entry)
            print(f"[Manual] step {t:4d} | loss {total_loss:.4f} | "
                  f"recon {rec:.4f} | mse {mets['recon_mse']:.4f}")

    # Final metrics
    mets_final = metrics(W, E, X_eval)

    return {
        "model": model,
        "metrics": mets_final,
        "logs": logs
    }
