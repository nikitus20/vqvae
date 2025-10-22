
import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np

# Optional torch (autograd variant)
try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

import matplotlib.pyplot as plt

# =========================
# Utilities
# =========================

def set_seed(seed: int):
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def geomspace_diag(lmax: float, lmin: float, r: int) -> np.ndarray:
    return np.geomspace(lmax, lmin, r)

def make_lowrank_cov(D: int, r: int, lmax: float=4.0, lmin: float=0.25, rng: np.random.Generator=None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    U, _ = np.linalg.qr(rng.normal(size=(D, r)))
    lam = geomspace_diag(lmax, lmin, r)
    Sigma = (U * lam) @ U.T
    return Sigma

def sample_gaussian(N: int, mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator=None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    return rng.multivariate_normal(mean=mean, cov=cov, size=N)

def stiefel_project_rows(W: np.ndarray) -> np.ndarray:
    # Rows orthonormal: W W^T = I_m
    U, _, Vt = np.linalg.svd(W, full_matrices=False)
    return U @ Vt

def nearest_code(Z: np.ndarray, E: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Z: (N, m), E: (K, m)
    # returns indices k (N,), and Q = E[k]
    D2 = np.sum(Z**2, axis=1, keepdims=True) - 2*Z@E.T + np.sum(E**2, axis=1)
    k = np.argmin(D2, axis=1)
    Q = E[k]
    return k, Q

def metrics(W: np.ndarray, E: np.ndarray, X: np.ndarray) -> Dict[str, float]:
    Z = X @ W.T
    k, Q = nearest_code(Z, E)
    Xhat = Q @ W  # == (W^T Q)^T
    mse = np.mean(np.sum((X - Xhat)**2, axis=1))
    in_sub = np.mean(np.sum((Z - Q)**2, axis=1))
    p = np.bincount(k, minlength=E.shape[0]) / len(k)
    usage_entropy = -np.sum(p[p>0]*np.log(p[p>0]))
    dead = float(np.mean(p==0.0))
    return {"Recon MSE": mse, "In-sub distortion": in_sub, "Usage entropy (nats)": usage_entropy, "Dead-code frac": dead}

def kmeans_lloyd(Z: np.ndarray, K: int, iters: int=25, seed: int=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    N, m = Z.shape
    # k-means++ init
    centers = np.empty((K, m), dtype=Z.dtype)
    i0 = rng.integers(0, N)
    centers[0] = Z[i0]
    d2 = np.sum((Z - centers[0])**2, axis=1)
    for k in range(1, K):
        probs = d2 / np.maximum(d2.sum(), 1e-12)
        idx = rng.choice(N, p=probs)
        centers[k] = Z[idx]
        d2 = np.minimum(d2, np.sum((Z - centers[k])**2, axis=1))
    E = centers.copy()
    for _ in range(iters):
        D2 = np.sum(Z**2, axis=1, keepdims=True) - 2*Z@E.T + np.sum(E**2, axis=1)
        k = np.argmin(D2, axis=1)
        for j in range(K):
            idxj = np.where(k==j)[0]
            if idxj.size > 0:
                E[j] = Z[idxj].mean(axis=0)
    return E

# =========================
# Variant 1: PCA + Lloyd
# =========================

def pca_top_m(Sigma: np.ndarray, m: int) -> np.ndarray:
    evals, evecs = np.linalg.eigh(Sigma)
    idx = np.argsort(evals)[::-1]
    W = evecs[:, idx[:m]].T  # rows are top-m eigenvectors
    return W

def run_baseline_pca_lloyd(X_train: np.ndarray, X_eval: np.ndarray, m: int, K: int, lloyd_iters: int=30, seed: int=0) -> Dict[str, Any]:
    Sigma = np.cov(X_train, rowvar=False, bias=True)
    W_pca = pca_top_m(Sigma, m)
    Z = X_train @ W_pca.T
    E = kmeans_lloyd(Z, K, iters=lloyd_iters, seed=seed)
    mets = metrics(W_pca, E, X_eval)
    return {"W": W_pca, "E": E, "metrics": mets}

# =========================
# Variant 2: VQ-VAE + STE (Autograd, PyTorch)
# =========================

class VQSTEModelTorch(nn.Module):
    def __init__(self, D: int, m: int, K: int, W_init: np.ndarray, E_init: np.ndarray, alpha=1.0, beta=0.25):
        super().__init__()
        self.D, self.m, self.K = D, m, K
        W0 = torch.tensor(W_init, dtype=torch.float32)
        E0 = torch.tensor(E_init, dtype=torch.float32)
        self.W = nn.Parameter(W0)
        self.E = nn.Parameter(E0)
        self.alpha = alpha
        self.beta = beta

    def st_forward(self, x):
        z = x @ self.W.t()
        z2 = (z**2).sum(dim=1, keepdim=True)
        e2 = (self.E**2).sum(dim=1)
        D2 = z2 - 2*z @ self.E.t() + e2
        k = torch.argmin(D2, dim=1)
        q = self.E[k]
        z_q = z + (q - z).detach()
        xhat = z_q @ self.W
        return z, q, z_q, xhat, k

    def loss(self, x):
        z, q, z_q, xhat, k = self.st_forward(x)
        rec = ((x - xhat)**2).sum(dim=1).mean()
        codebook = self.alpha * ((z.detach() - q)**2).sum(dim=1).mean()
        commit = self.beta * ((z - q.detach())**2).sum(dim=1).mean()
        return rec + codebook + commit, {"rec": rec.item(), "codebook": codebook.item(), "commit": commit.item()}

def project_stiefel_torch(W: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        U, _, Vt = torch.linalg.svd(W, full_matrices=False)
        return (U @ Vt).detach()

def run_vqvae_ste_autograd(X_train: np.ndarray, X_eval: np.ndarray, m: int, K: int, steps=200, batch=1024, alpha=1.0, beta=0.25, lrW=0.05, lrE=0.2, seed: int=0):
    if not TORCH_AVAILABLE:
        print("Torch not available; skipping autograd variant.")
        return {"W": None, "E": None, "metrics": None, "logs": []}

    device = torch.device("cpu")
    D = X_train.shape[1]
    Sigma = np.cov(X_train, rowvar=False, bias=True)
    W0 = pca_top_m(Sigma, m)
    Z0 = X_train[:8000] @ W0.T
    E0 = kmeans_lloyd(Z0, K, iters=20, seed=seed)

    model = VQSTEModelTorch(D, m, K, W0, E0, alpha=alpha, beta=beta).to(device)
    optW = torch.optim.SGD([model.W], lr=lrW)
    optE = torch.optim.SGD([model.E], lr=lrE)

    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    Xev = torch.tensor(X_eval, dtype=torch.float32, device=device)

    logs = []
    B = batch
    N = X_train.shape[0]
    for t in range(steps):
        idx = np.random.randint(0, N, size=B)
        xb = Xtr[idx]
        loss, parts = model.loss(xb)
        optW.zero_grad(); optE.zero_grad()
        loss.backward()
        optW.step(); optE.step()
        model.W.data[:] = project_stiefel_torch(model.W.data)
        if (t % 20 == 0) or (t == steps-1):
            with torch.no_grad():
                W_np = model.W.detach().cpu().numpy()
                E_np = model.E.detach().cpu().numpy()
                mets = metrics(W_np, E_np, X_eval)
                logs.append({"step": t, "loss": float(loss.item()), **parts, **mets})
                print(f"[Auto] step {t:4d} | loss {loss.item():.4f} | recon {parts['rec']:.4f} | mse {mets['Recon MSE']:.4f}")

    W_final = model.W.detach().cpu().numpy()
    E_final = model.E.detach().cpu().numpy()
    mets_final = metrics(W_final, E_final, X_eval)
    return {"W": W_final, "E": E_final, "metrics": mets_final, "logs": logs}

# =========================
# Variant 3: VQ-VAE + STE (Manual grads, NumPy)
# =========================

from dataclasses import dataclass

@dataclass
class STEConfig:
    steps: int = 200
    batch: int = 1024
    alpha: float = 1.0
    beta: float = 0.25
    lrW: float = 0.08
    lrE: float = 0.2
    seed: int = 0

def run_vqvae_ste_manual(X_train: np.ndarray, X_eval: np.ndarray, m: int, K: int, cfg: STEConfig) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    D = X_train.shape[1]
    Sigma = np.cov(X_train, rowvar=False, bias=True)
    W = pca_top_m(Sigma, m)
    Z0 = X_train[:8000] @ W.T
    E = kmeans_lloyd(Z0, K, iters=20, seed=cfg.seed)
    logs = []

    N = X_train.shape[0]
    for t in range(cfg.steps):
        idx = rng.integers(0, N, size=cfg.batch)
        xb = X_train[idx]
        Z = xb @ W.T
        k_idx, Q = nearest_code(Z, E)
        Xhat = Q @ W
        r = xb - Xhat

        rec = np.mean(np.sum(r**2, axis=1))
        dist = np.sum((Z - Q)**2, axis=1)
        codebook = cfg.alpha * np.mean(dist)
        commit = cfg.beta * np.mean(dist)
        total = rec + codebook + commit

        # Codebook gradient
        grad_E = np.zeros_like(E)
        for j in range(K):
            idxj = np.where(k_idx == j)[0]
            if idxj.size > 0:
                grad_E[j] = 2*cfg.alpha*(E[j] - Z[idxj].mean(axis=0))

        # Encoder gradient
        Wr = r @ W.T
        term1 = -2 * (Q.T @ r) / cfg.batch
        term2 = -2 * (Wr.T @ xb) / cfg.batch
        term3 = 2*cfg.beta * ((Z - Q).T @ xb) / cfg.batch
        G = term1 + term2 + term3

        # Riemannian projection to Stiefel
        sym = (G.T @ W) + (W.T @ G)
        grad_W_riem = G - 0.5 * (sym.T @ W)

        # SGD
        E -= cfg.lrE * grad_E
        W -= cfg.lrW * grad_W_riem
        W = stiefel_project_rows(W)

        if (t % 20 == 0) or (t == cfg.steps-1):
            mets = metrics(W, E, X_eval)
            logs.append({"step": t, "loss": float(total), "rec": float(rec), "codebook": float(codebook), "commit": float(commit), **mets})
            print(f"[Manual] step {t:4d} | loss {total:.4f} | recon {rec:.4f} | mse {mets['Recon MSE']:.4f}")

    mets_final = metrics(W, E, X_eval)
    return {"W": W, "E": E, "metrics": mets_final, "logs": logs}

# =========================
# Plotting helpers
# =========================

def plot_training(logs, title_prefix: str, outdir: str):
    if not logs:
        return
    steps = [d["step"] for d in logs]

    if "loss" in logs[0]:
        plt.figure()
        plt.plot(steps, [d.get("loss", np.nan) for d in logs], label="Total")
        if "rec" in logs[0]:
            plt.plot(steps, [d.get("rec", np.nan) for d in logs], label="Recon")
        if "codebook" in logs[0]:
            plt.plot(steps, [d.get("codebook", np.nan) for d in logs], label="Codebook")
        if "commit" in logs[0]:
            plt.plot(steps, [d.get("commit", np.nan) for d in logs], label="Commit")
        plt.xlabel("Step"); plt.ylabel("Loss"); plt.title(f"{title_prefix}: losses"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{title_prefix}_losses.png"), dpi=140); plt.close()

    if "Recon MSE" in logs[0]:
        plt.figure()
        plt.plot(steps, [d["Recon MSE"] for d in logs], label="Recon MSE")
        plt.xlabel("Step"); plt.ylabel("Recon MSE"); plt.title(f"{title_prefix}: reconstruction MSE"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{title_prefix}_mse.png"), dpi=140); plt.close()

    if "In-sub distortion" in logs[0]:
        plt.figure()
        plt.plot(steps, [d["In-sub distortion"] for d in logs], label="In-subspace distortion")
        plt.xlabel("Step"); plt.ylabel("In-subspace distortion"); plt.title(f"{title_prefix}: latent distortion"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{title_prefix}_insub.png"), dpi=140); plt.close()

    if "Usage entropy (nats)" in logs[0]:
        plt.figure()
        plt.plot(steps, [d["Usage entropy (nats)"] for d in logs], label="Usage entropy")
        plt.xlabel("Step"); plt.ylabel("Entropy (nats)"); plt.title(f"{title_prefix}: code usage entropy"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{title_prefix}_entropy.png"), dpi=140); plt.close()

    if "Dead-code frac" in logs[0]:
        plt.figure()
        plt.plot(steps, [d["Dead-code frac"] for d in logs], label="Dead-code fraction")
        plt.xlabel("Step"); plt.ylabel("Dead-code frac"); plt.title(f"{title_prefix}: dead-code fraction"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{title_prefix}_dead.png"), dpi=140); plt.close()

# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--rank", type=int, default=6)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--n_train", type=int, default=20000)
    parser.add_argument("--n_eval", type=int, default=4000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--lrW", type=float, default=0.08)
    parser.add_argument("--lrE", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="results_out")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    Sigma = make_lowrank_cov(args.D, args.rank, rng=rng)
    X_train = sample_gaussian(args.n_train, np.zeros(args.D), Sigma, rng=rng)
    X_eval  = sample_gaussian(args.n_eval,  np.zeros(args.D), Sigma, rng=rng)

    # Baseline
    print("== Baseline: PCA + Lloyd ==")
    base = run_baseline_pca_lloyd(X_train, X_eval, args.m, args.K, seed=args.seed)
    print("Baseline metrics:", base["metrics"])

    # Autograd
    print("== VQ-VAE + STE (Autograd) ==")
    auto = run_vqvae_ste_autograd(
        X_train, X_eval, args.m, args.K,
        steps=args.steps, batch=args.batch, alpha=args.alpha, beta=args.beta,
        lrW=args.lrW, lrE=args.lrE, seed=args.seed
    )
    if auto["logs"]:
        import csv
        with open(os.path.join(args.outdir, "autograd_logs.csv"), "w", newline="") as f:
            w = None
            for row in auto["logs"]:
                if w is None:
                    w = csv.DictWriter(f, fieldnames=list(row.keys()))
                    w.writeheader()
                w.writerow(row)
        plot_training(auto["logs"], "autograd", args.outdir)
        print("Autograd final metrics:", auto["metrics"])

    # Manual
    print("== VQ-VAE + STE (Manual grads) ==")
    cfg = STEConfig(steps=args.steps, batch=args.batch, alpha=args.alpha, beta=args.beta, lrW=args.lrW, lrE=args.lrE, seed=args.seed)
    manual = run_vqvae_ste_manual(X_train, X_eval, args.m, args.K, cfg)
    if manual["logs"]:
        import csv
        with open(os.path.join(args.outdir, "manual_logs.csv"), "w", newline=True) as f:
            w = None
            for row in manual["logs"]:
                if w is None:
                    w = csv.DictWriter(f, fieldnames=list(row.keys()))
                    w.writeheader()
                w.writerow(row)
        plot_training(manual["logs"], "manual", args.outdir)
        print("Manual final metrics:", manual["metrics"])

    # Summary
    import json
    summary = {
        "baseline": base["metrics"],
        "autograd": auto["metrics"] if auto["metrics"] is not None else None,
        "manual": manual["metrics"]
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved results to", args.outdir)

if __name__ == "__main__":
    main()
