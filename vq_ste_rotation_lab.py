import argparse, math, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import json, os, yaml
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- utils ----------
def set_seed(s=42):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)

def make_subspace_data(d=256, r=4, N=20000, sigma=0.1):
    A = np.random.randn(d, r)
    U, _ = np.linalg.qr(A)                  # true subspace
    s = np.random.randn(N, r)
    x = s @ U.T + sigma * np.random.randn(N, d)
    return x.astype(np.float32), U

class Codebook(nn.Module):
    def __init__(self, K, dim, scale=0.1):
        super().__init__()
        self.code = nn.Parameter(torch.randn(K, dim) * scale)
    def forward(self): return self.code

def kmeans_pp_init(z, K):
    N, dim = z.shape
    centers = np.empty((K, dim), dtype=z.dtype)
    idx = np.random.randint(N); centers[0] = z[idx]
    d2 = np.full(N, np.inf)
    for k in range(1, K):
        d2 = np.minimum(d2, np.sum((z - centers[k-1])**2, axis=1))
        probs = d2 / d2.sum()
        idx = np.random.choice(N, p=probs); centers[k] = z[idx]
    return centers

# ---------- rotation surrogate ----------
class RotationEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, q):
        ctx.save_for_backward(z, q); return q
    @staticmethod
    def backward(ctx, grad_output):
        z, q = ctx.saved_tensors
        v = q - z; eps = 1e-8
        norm = torch.clamp(v.norm(dim=1, keepdim=True), min=eps)
        u = v / norm
        g_mag = (grad_output * u).sum(dim=1, keepdim=True)
        grad_z = g_mag * u
        grad_q = torch.zeros_like(q)
        return grad_z, grad_q

# ---------- model ----------
class LinearVQ(nn.Module):
    def __init__(self, d, r, K, beta=0.25, estimator="ste", orthonormal_W=False):
        super().__init__()
        self.W = nn.Parameter(torch.randn(r, d) / math.sqrt(d))
        self.codebook = Codebook(K, r)
        self.beta = beta; self.estimator = estimator; self.orthonormal_W = orthonormal_W
    def encode(self, x): return F.linear(x, self.W)
    def decode(self, z):   return F.linear(z, self.W.t())
    def assign(self, z):
        code = self.codebook()
        d2 = (z**2).sum(1,keepdim=True) + (code**2).sum(1).unsqueeze(0) - 2*z@code.t()
        idx = torch.argmin(d2, dim=1); q = code[idx]; return q, idx
    def forward(self, x):
        z = self.encode(x); q, idx = self.assign(z)
        if self.estimator == "ste":        zhat = z + (q - z).detach()
        elif self.estimator == "rotation": zhat = RotationEstimator.apply(z, q)
        else: raise ValueError("estimator")
        x_hat = self.decode(zhat)
        rec = F.mse_loss(x_hat, x)
        codebook = F.mse_loss(q, z.detach())
        commit   = F.mse_loss(z, q.detach())
        loss = rec + codebook + self.beta*commit
        if self.orthonormal_W and self.training:
            with torch.no_grad():
                W_t = self.W.data.t(); Q,_ = torch.linalg.qr(W_t, mode='reduced'); self.W.data.copy_(Q.t())
        return loss, rec, z, q, idx

# ---------- metrics ----------
def principal_angles(U, W_rows):
    Vhat = W_rows.t().cpu().numpy()
    Qhat, _ = np.linalg.qr(Vhat)
    M = U.T @ Qhat
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return float(np.mean(np.arccos(s)))

def entropy_dead(idxs, K):
    counts = torch.bincount(idxs, minlength=K).float()
    p = counts / counts.sum(); eps=1e-12
    ent = float(-(p*(p+eps).log()).sum()); dead = int((counts==0).sum())
    return ent, dead

# ---------- logging ----------
def setup_experiment_dir(base_dir="runs", exp_name=None):
    date_str = datetime.now().strftime("%Y%m%d")
    exp_id = datetime.now().strftime("%H%M%S")
    if exp_name:
        exp_id = f"{exp_name}_{exp_id}"
    exp_dir = Path(base_dir) / date_str / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir

def save_config(exp_dir, config):
    with open(exp_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

def save_logs(exp_dir, logs):
    with open(exp_dir / "logs.json", 'w') as f:
        json.dump(logs, f, indent=2)

def save_final_metrics(exp_dir, logs):
    final_metrics = {k: v[-1] for k, v in logs.items()}
    with open(exp_dir / "metrics_final.json", 'w') as f:
        json.dump(final_metrics, f, indent=2)

def save_seed(exp_dir, seed):
    with open(exp_dir / "seed.txt", 'w') as f:
        f.write(str(seed))

def plot_metrics(exp_dir, logs):
    epochs = list(range(1, len(logs["val_rec"]) + 1))

    # Reconstruction MSE
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, logs["val_rec"], 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Reconstruction MSE', fontsize=12)
    plt.title('Reconstruction MSE over Epochs', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(exp_dir / "rec_mse.png", dpi=100)
    plt.close()

    # Mean Principal Angle
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, logs["angle"], 'r-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Mean Principal Angle (radians)', fontsize=12)
    plt.title('Mean Principal Angle over Epochs', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(exp_dir / "mean_principal_angle.png", dpi=100)
    plt.close()

    # Code Usage Entropy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, logs["entropy"], 'g-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Code Usage Entropy', fontsize=12)
    plt.title('Code Usage Entropy over Epochs', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(exp_dir / "code_usage_entropy.png", dpi=100)
    plt.close()

    # Dead Codes
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, logs["dead"], 'm-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Number of Dead Codes', fontsize=12)
    plt.title('Dead Codes over Epochs', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(exp_dir / "dead_codes.png", dpi=100)
    plt.close()

# ---------- train ----------
def run(estimator="ste", d=256, r=4, K=16, sigma=0.1, Ntr=20000, Nval=4000,
        epochs=40, bs=256, lr=2e-3, beta=0.25, seed=42, orth=False, verbose=True,
        exp_dir=None):
    set_seed(seed)

    # Setup experiment directory and logging
    if exp_dir is None:
        exp_name = f"{estimator}_K{K}_sigma{sigma}_{'orth' if orth else 'noorth'}"
        exp_dir = setup_experiment_dir(exp_name=exp_name)

    # Save configuration
    config = {
        "estimator": estimator,
        "d": d, "r": r, "K": K,
        "sigma": sigma,
        "Ntr": Ntr, "Nval": Nval,
        "epochs": epochs,
        "batch_size": bs,
        "learning_rate": lr,
        "beta": beta,
        "seed": seed,
        "orthonormal_W": orth
    }
    save_config(exp_dir, config)
    save_seed(exp_dir, seed)

    # Redirect stdout to file
    stdout_file = open(exp_dir / "stdout.txt", 'w')

    Xtr, U = make_subspace_data(d, r, Ntr, sigma)
    Xva, _ = make_subspace_data(d, r, Nval, sigma)
    tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr)), batch_size=bs, shuffle=True, drop_last=True)
    va_loader = DataLoader(TensorDataset(torch.from_numpy(Xva)), batch_size=bs, shuffle=False)

    model = LinearVQ(d,r,K,beta,estimator,orthonormal_W=orth)
    with torch.no_grad():
        z0 = model.encode(torch.from_numpy(Xtr[:4000])).cpu().numpy()
        model.codebook.code.data.copy_(torch.from_numpy(kmeans_pp_init(z0, K)))

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    logs = {"val_rec":[], "angle":[], "entropy":[], "dead":[]}

    for ep in range(epochs):
        model.train()
        for (xb,) in tr_loader:
            loss, rec, z, q, idx = model(xb)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            vr=0.0; all_idx=[]
            for (xb,) in va_loader:
                loss, rec, z, q, idx = model(xb)
                vr += rec.item()*xb.size(0); all_idx.append(idx)
            vr /= len(va_loader.dataset)
            all_idx = torch.cat(all_idx,dim=0)
            ent, dead = entropy_dead(all_idx, K)
            ang = principal_angles(U, model.W.data)

        logs["val_rec"].append(vr); logs["angle"].append(ang)
        logs["entropy"].append(ent); logs["dead"].append(dead)

        if verbose and (ep+1)%10==0:
            msg = f"[{ep+1:03d}] rec={vr:.5f} angle={ang:.4f} ent={ent:.3f} dead={dead}"
            print(msg)
            stdout_file.write(msg + "\n")
            stdout_file.flush()

    # Save logs and plots
    save_logs(exp_dir, logs)
    save_final_metrics(exp_dir, logs)
    plot_metrics(exp_dir, logs)

    stdout_file.close()

    if verbose:
        print(f"\nExperiment saved to: {exp_dir}")
        print("Final metrics:", {k: v[-1] for k,v in logs.items()})

    return model, logs, exp_dir

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--estimator", choices=["ste","rotation"], default="ste")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--orth", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--bs", type=int, default=256)
    args = p.parse_args()

    _, logs, exp_dir = run(
        estimator=args.estimator,
        epochs=args.epochs,
        K=args.K,
        sigma=args.sigma,
        orth=args.orth,
        seed=args.seed,
        beta=args.beta,
        lr=args.lr,
        bs=args.bs
    )