# VQ-VAE STE vs Rotation Lab Guide

## 0. TL;DR Checklist

- [ ] Create/activate Python env with PyTorch + NumPy + Matplotlib
- [ ] Put `vq_ste_rotation_lab.py` at repo root (content provided below)
- [ ] Run baseline experiments:
  ```bash
  python vq_ste_rotation_lab.py --estimator ste --epochs 60 --K 16 --sigma 0.1 --orth
  python vq_ste_rotation_lab.py --estimator rotation --epochs 60 --K 16 --sigma 0.1 --orth
  ```
- [ ] Save logs as JSON and plots (four metrics over epochs)
- [ ] Fill in the Results Template (Section 9) and send back

## 1. Project Structure

```
.
├─ vq_ste_rotation_lab.py         # training script (STE or rotation)
├─ configs/
│  ├─ default.yaml                # base hyperparams
│  └─ sweeps.yaml                 # curated experiment grid
├─ runs/
│  └─ {date}/{exp_id}/
│      ├─ config.json
│      ├─ logs.json               # per-epoch metrics
│      ├─ metrics_final.json      # last-epoch summary
│      ├─ seed.txt
│      ├─ stdout.txt
│      ├─ rec_mse.png             # plots
│      ├─ mean_principal_angle.png
│      ├─ code_usage_entropy.png
│      └─ dead_codes.png
└─ analysis/
   └─ summarize.ipynb             # optional plotting/aggregation notebook
```

> **Note:** If any item is missing, create it. Use the filenames verbatim.

## 2. Environment Setup

### Requirements
- Python ≥ 3.9
- Packages: `torch` (CPU ok), `numpy`, `matplotlib`, `pyyaml` (for configs), `tqdm` (optional)

### Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch numpy matplotlib pyyaml tqdm
```

## 3. Data & Model Specification

### Data Generation
- **Ambient dimension:** d = 256
- **Intrinsic dimension:** r = 4
- **Truth model:** x = U⋆s + ε
  - U⋆: orthonormal (QR of Gaussian)
  - s ~ N(0, Ir)
  - ε ~ N(0, σ²Id)

### Student Model
- **Encoder:** W ∈ ℝʳˣᵈ (linear)
- **Decoder:** D = W⊤ (tied)
- **VQ layer:** codebook {qₖ}ᵏ₌₁ᴷ ⊂ ℝʳ, nearest-neighbor hard assignment ẑ = q_{k*}(z)

### Loss Function
```
‖x - W⊤ẑ‖² + ‖sg(z) - ẑ‖² + β‖z - sg(ẑ)‖²
```
where z = Wx

### Gradient Estimators
- **STE:** backward uses ∂ẑ/∂z ≈ I
- **Rotation surrogate:** backward transports only the component of the decoder gradient along the vector z → ẑ (constant-Jacobian per sample)
- **Subspace regularity (optional):** `--orth` projects rows of W to the Stiefel manifold each step (QR)

## 4. Training Script

Place at repo root as `vq_ste_rotation_lab.py`:

```python
import argparse, math, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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

# ---------- train ----------
def run(estimator="ste", d=256, r=4, K=16, sigma=0.1, Ntr=20000, Nval=4000,
        epochs=40, bs=256, lr=2e-3, beta=0.25, seed=42, orth=False, verbose=True):
    set_seed(seed)
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
            print(f"[{ep+1:03d}] rec={vr:.5f} angle={ang:.4f} ent={ent:.3f} dead={dead}")

    return model, logs

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--estimator", choices=["ste","rotation"], default="ste")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--orth", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    _, logs = run(estimator=args.estimator, epochs=args.epochs, K=args.K,
                  sigma=args.sigma, orth=args.orth, seed=args.seed)
    print("Final:", {k: v[-1] for k,v in logs.items()})
```

## 5. Core Metrics (Must Log)

1. **Reconstruction (val):** E[‖x - W⊤ẑ‖²]
2. **Mean principal angle:** Between row(W) and span(U⋆⊤) - lower is better
3. **Code usage entropy:** Higher ≈ more uniform usage
4. **Dead codes:** Count of unused centroids - lower is better

## 6. How to Run

### STE Baseline
```bash
python vq_ste_rotation_lab.py --estimator ste --epochs 60 --K 16 --sigma 0.1 --orth --seed 1
```

### Rotation Surrogate
```bash
python vq_ste_rotation_lab.py --estimator rotation --epochs 60 --K 16 --sigma 0.1 --orth --seed 1
```

**Important:** Repeat with seeds 1, 2, 3, 4, 5. Save per-seed logs and a seed-averaged summary.

## 7. Curated Experiment Suite

### A. Subspace Recovery vs Code Usage (Primary)
- **Config:** d=256, r=4, K=16, sigma=0.1, epochs=60, orth=True
- **Runs:** STE vs Rotation, seeds {1..5}
- **Expectation:**
  - Both recover subspace well (principal angle → small)
  - Rotation yields higher code entropy and fewer dead codes at comparable recon error

### B. Codebook Size Sensitivity
- Keep A but vary K ∈ {8, 16, 32}
- **Expectation:** As K grows, STE more prone to dead codes; Rotation more stable

### C. Noise Robustness
- Keep A but vary sigma ∈ {0.0, 0.1, 0.3}
- **Expectation:** Higher noise degrades subspace angle similarly; rotation should retain better utilization

### D. Commitment Strength
- Keep A but vary beta ∈ {0.0, 0.25, 0.5}
- **Expectation:** Larger β → STE can over-shrink z→q without boundary awareness → more dead codes; rotation mitigates

### Optional "Purist" Runs
Turn `--orth` off to see the effect of unconstrained W. Keep other settings fixed.

### Config File (`configs/sweeps.yaml`)
```yaml
# configs/sweeps.yaml
common:
  d: 256
  r: 4
  epochs: 60
  orth: true
  seed_list: [1,2,3,4,5]

grids:
  - name: A_subspace_usage
    params:
      estimator: [ste, rotation]
      K: [16]
      sigma: [0.1]

  - name: B_codebook_size
    params:
      estimator: [ste, rotation]
      K: [8,16,32]
      sigma: [0.1]

  - name: C_noise
    params:
      estimator: [ste, rotation]
      K: [16]
      sigma: [0.0,0.1,0.3]
```

## 8. Saving Artifacts & Plotting

### For Each Run
Save the following:
- `config.json` - experiment configuration
- `logs.json` - epoch arrays for all four metrics
- `metrics_final.json` - last epoch metrics
- `seed.txt` - random seed used
- `stdout.txt` - console output

Generate 4 plots (epoch vs metric):
- `rec_mse.png`
- `mean_principal_angle.png`
- `code_usage_entropy.png`
- `dead_codes.png`

### Aggregation Across Seeds
- Compute mean ± std at the final epoch for each metric
- For each grid, produce a small CSV:

```csv
exp,estimator,K,sigma,beta,mean_rec,std_rec,mean_angle,std_angle,mean_entropy,std_entropy,mean_dead,std_dead
A,ste,16,0.1,0.25,...,...,...
A,rotation,16,0.1,0.25,...,...,...
...
```

## 9. Results Template

```markdown
# SUMMARY (first pass)

Config: d=256, r=4, epochs=60, orth=True
Seeds: 1..5

A) Subspace & Usage (K=16, sigma=0.1)
- STE:
  - final rec MSE:    mean=____ std=____
  - mean princ angle: mean=____ rad std=____
  - code entropy:     mean=____ std=____
  - dead codes:       mean=____ std=____
- Rotation:
  - final rec MSE:    mean=____ std=____
  - mean princ angle: mean=____ rad std=____
  - code entropy:     mean=____ std=____
  - dead codes:       mean=____ std=____
Comment: (2–3 lines: did rotation reduce dead codes / raise entropy without hurting angle/MSE?)

B) Codebook Size
- K=8:    STE (...) vs Rotation (...)
- K=16:   STE (...) vs Rotation (...)
- K=32:   STE (...) vs Rotation (...)
Comment: (1–2 lines on scaling and stability)

C) Noise (K=16)
- sigma=0.0: STE (...) vs Rotation (...)
- sigma=0.1: STE (...) vs Rotation (...)
- sigma=0.3: STE (...) vs Rotation (...)
Comment: (1–2 lines on robustness)

Plots: (attach the four PNGs per run or a seed-mean plot)
```

## 10. Interpretation Guide

### Key Questions

**Does STE recover the correct subspace?**
- Yes on this data (acts like tied linear AE/PCA in encoder path)
- Mean principal angle should go near zero for both methods

**Does Rotation improve code utilization?**
- Expect higher entropy and fewer dead codes for rotation at similar recon error and principal angle
- This indicates that transporting boundary geometry back to W helps stabilize assignments

**Trade-offs:**
- If rotation slightly increases MSE early but achieves better utilization and similar final MSE/angle, we prefer it for VQ stability

## 11. Sanity Checks

1. Set `sigma=0.0, K=8, epochs=20`: both methods should rapidly lower angle and MSE; rotation shouldn't break anything
2. Increase K to 32 with `beta=0.5`: if STE shows many dead codes while rotation keeps using more codes, the hypothesis is supported

## 12. Next Steps

After first batch of experiments:
1. Add finite-difference gradient alignment on a tiny config (K=2, r=2) to visualize STE bias near Voronoi edges
2. Add PCA initialization for W vs random init; compare convergence speed
3. Try untied decoder; log if conclusions persist
4. Swap Gaussian latent for a 4-mixture to see boundary complexity effects

## 13. Reproducibility Guidelines

- Always set and log `--seed`
- For faster runs: reduce Ntr/Nval in the script or reduce epochs
- Keep `--orth` on for the main tables (clean theory)
- Run a small ablation with it off

## 14. Deliverables

Please provide:
1. The filled Results Template (Section 9)
2. `runs/{date}/.../metrics_final.json` for the compared rows
3. The four plots for the A) experiment (K=16, sigma=0.1), averaged over seeds if possible
