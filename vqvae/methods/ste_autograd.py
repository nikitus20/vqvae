"""Method 2: SGD with STE using PyTorch autograd."""

import numpy as np
from typing import Dict, Any, List

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes for when torch is not available
    class nn:
        class Module:
            pass
        class Parameter:
            pass

from ..models.vqvae import VQVAE
from ..utils import metrics


if TORCH_AVAILABLE:
    class VQSTEModelTorch(nn.Module):
        """PyTorch VQ-VAE model with Straight-Through Estimator."""

        def __init__(self, D: int, m: int, K: int, W_init: np.ndarray,
                     E_init: np.ndarray, alpha: float = 1.0, beta: float = 0.25):
            super().__init__()
            self.D, self.m, self.K = D, m, K
            self.W = nn.Parameter(torch.tensor(W_init, dtype=torch.float32))
            self.E = nn.Parameter(torch.tensor(E_init, dtype=torch.float32))
            self.alpha = alpha
            self.beta = beta

        def encode(self, x):
            """Z = X @ W^T"""
            return x @ self.W.t()

        def quantize(self, z):
            """Find nearest codes with STE."""
            z2 = (z**2).sum(dim=1, keepdim=True)
            e2 = (self.E**2).sum(dim=1)
            D2 = z2 - 2*z @ self.E.t() + e2
            k = torch.argmin(D2, dim=1)
            q = self.E[k]
            # Straight-through estimator
            z_q = z + (q - z).detach()
            return z_q, q, k

        def decode(self, z_q):
            """X_hat = Z_q @ W"""
            return z_q @ self.W

        def forward(self, x):
            z = self.encode(x)
            z_q, q, k = self.quantize(z)
            x_hat = self.decode(z_q)
            return z, q, z_q, x_hat, k

        def loss(self, x):
            z, q, z_q, x_hat, k = self.forward(x)
            rec = ((x - x_hat)**2).sum(dim=1).mean()
            codebook = self.alpha * ((z.detach() - q)**2).sum(dim=1).mean()
            commit = self.beta * ((z - q.detach())**2).sum(dim=1).mean()
            total = rec + codebook + commit
            return total, {"rec": rec.item(), "codebook": codebook.item(), "commit": commit.item()}


    def project_stiefel_torch(W: torch.Tensor) -> torch.Tensor:
        """Project W onto Stiefel manifold (orthonormal rows)."""
        with torch.no_grad():
            U, _, Vt = torch.linalg.svd(W, full_matrices=False)
            return (U @ Vt).detach()
else:
    VQSTEModelTorch = None
    def project_stiefel_torch(W):
        return None


def train_ste_autograd(
    X_train: np.ndarray,
    X_eval: np.ndarray,
    m: int,
    K: int,
    steps: int = 200,
    batch: int = 1024,
    alpha: float = 1.0,
    beta: float = 0.25,
    lrW: float = 0.05,
    lrE: float = 0.2,
    seed: int = 0,
    log_every: int = 20
) -> Dict[str, Any]:
    """
    Train VQ-VAE using PyTorch autograd with Straight-Through Estimator.

    Args:
        X_train: (N, D) training data
        X_eval: (N_eval, D) evaluation data
        m: Latent dimension
        K: Number of codebook entries
        steps: Number of SGD steps
        batch: Batch size
        alpha: Codebook loss weight
        beta: Commitment loss weight
        lrW: Learning rate for encoder
        lrE: Learning rate for codebook
        seed: Random seed
        log_every: Log metrics every N steps

    Returns:
        Dictionary with:
            - model: Trained VQVAE model (NumPy)
            - metrics: Evaluation metrics
            - logs: Training logs
    """
    if not TORCH_AVAILABLE:
        print("PyTorch not available; skipping autograd variant.")
        return {"model": None, "metrics": None, "logs": []}

    device = torch.device("cpu")
    D = X_train.shape[1]

    # Initialize with PCA + k-means
    from ..models.vqvae import VQVAE as VQVAENumpy
    init_model = VQVAENumpy(D, m, K, orthonormal_encoder=True)
    Sigma = np.cov(X_train, rowvar=False, bias=True)
    init_model.encoder.init_from_pca(Sigma)
    Z0 = init_model.encode(X_train[:8000])
    init_model.codebook.init_from_kmeans(Z0, iters=20, seed=seed)

    W0 = init_model.encoder.W
    E0 = init_model.codebook.E

    # Create PyTorch model
    model_torch = VQSTEModelTorch(D, m, K, W0, E0, alpha=alpha, beta=beta).to(device)
    optW = torch.optim.SGD([model_torch.W], lr=lrW)
    optE = torch.optim.SGD([model_torch.E], lr=lrE)

    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)

    logs = []
    N = X_train.shape[0]

    for t in range(steps):
        # Sample batch
        idx = np.random.randint(0, N, size=batch)
        xb = Xtr[idx]

        # Forward and backward
        loss, parts = model_torch.loss(xb)
        optW.zero_grad()
        optE.zero_grad()
        loss.backward()
        optW.step()
        optE.step()

        # Project encoder to Stiefel
        model_torch.W.data[:] = project_stiefel_torch(model_torch.W.data)

        # Log metrics
        if (t % log_every == 0) or (t == steps - 1):
            with torch.no_grad():
                W_np = model_torch.W.detach().cpu().numpy()
                E_np = model_torch.E.detach().cpu().numpy()
                mets = metrics(W_np, E_np, X_eval)
                log_entry = {
                    "step": t,
                    "loss": float(loss.item()),
                    **parts,
                    **mets
                }
                logs.append(log_entry)
                print(f"[Autograd] step {t:4d} | loss {loss.item():.4f} | "
                      f"recon {parts['rec']:.4f} | mse {mets['recon_mse']:.4f}")

    # Convert back to NumPy model
    final_model = VQVAENumpy(D, m, K, orthonormal_encoder=True)
    final_model.encoder.W = model_torch.W.detach().cpu().numpy()
    final_model.codebook.E = model_torch.E.detach().cpu().numpy()

    mets_final = metrics(final_model.encoder.W, final_model.codebook.E, X_eval)

    return {
        "model": final_model,
        "metrics": mets_final,
        "logs": logs
    }
