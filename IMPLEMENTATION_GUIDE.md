# VQ-VAE Research: Implementation Guide for Claude Code

## Project Overview

This is a research project investigating **Vector Quantized Variational Autoencoders (VQ-VAE)** with focus on:
1. **Idea 7**: R(D)-optimal codebook initialization
2. **Idea 1**: Population dynamics and dead codewords

The goal is to build clean, modular experimental code to test theoretical predictions.

---

## Theoretical Background

### Linear Gaussian Model

**Data Generation**:
```
X = A @ Y + W
```
where:
- `Y ~ N(0, Σ)` is latent variable (k-dimensional)
- `A ∈ R^(d×k)` is signal subspace (d > k)
- `W ~ N(0, σ²I_d)` is noise
- `X ∈ R^d` is observed data

**Optimal Encoder/Decoder**:
- PCA solution: `Φ = Ψ = U_k` (top k eigenvectors of data covariance)
- Latent representation: `z = U_k^T @ X ∈ R^k`
- This is EXACT for linear Gaussian case

**VQ-VAE Components**:
1. Encoder: `z = Φ^T @ X` (linear projection)
2. Quantizer: `ẑ = argmin_j ||z - c_j||²` (nearest neighbor in codebook)
3. Decoder: `X̂ = Ψ @ ẑ` (linear reconstruction)

### Rate-Distortion Theory for Initialization

For Gaussian `z ~ N(0, σ_z² I_k)` and codebook size `n`:

**Rate**: `R = log₂(n) / k` bits per dimension

**R(D)-Optimal Reproduction Variance**:
```python
σ_ẑ² = σ_z² × (1 - 2^(-2R))
```

**Key Theoretical Predictions**:
1. **Standard uniform init** `[-1/n, 1/n]` is TOO SMALL (variance ~ 1/n²)
2. **R(D) init** `N(0, σ_ẑ² I_k)` matches optimal scale
3. **Expected initial distortion**:
   - Standard: `D₀ ≈ k σ_z²` (no compression!)
   - R(D): `D₀ ≈ k σ_z² × 2^(-2R)` (near-optimal!)
4. **Dead codes**: Standard init causes many dead codes due to tiny variance

---

## Directory Structure

```
vqvae-research/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── linear_gaussian.py       # Linear Gaussian data generator
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vqvae.py                 # Main VQ-VAE model
│   │   ├── encoder_decoder.py       # Linear encoder/decoder for Gaussian
│   │   └── quantizer.py             # Vector quantizer with different inits
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py               # Training loop
│   │   └── losses.py                # VQ-VAE loss functions
│   ├── initialization/
│   │   ├── __init__.py
│   │   ├── standard.py              # Uniform and k-means init
│   │   └── rate_distortion.py       # R(D)-based initialization
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py               # Compute all metrics
│       └── visualization.py         # Plotting utilities
├── experiments/
│   └── idea_7_initialization/
│       ├── __init__.py
│       ├── config.yaml              # Hyperparameters
│       ├── run_linear_gaussian.py   # Main experiment script
│       └── analysis.py              # Result analysis
├── results/
│   └── idea_7_linear_gaussian/      # Experiment outputs (created at runtime)
│       ├── metrics.csv
│       ├── plots/
│       └── checkpoints/
├── notebooks/
│   └── exploratory_analysis.ipynb   # Interactive analysis
├── requirements.txt
├── README.md
└── IMPLEMENTATION_GUIDE.md          # This file
```

---

## Implementation Specifications

### 1. Data Generator (`src/data/linear_gaussian.py`)

**Class**: `LinearGaussianDataset`

**Constructor Parameters**:
```python
def __init__(
    self,
    d: int = 64,           # Ambient dimension
    k: int = 8,            # Latent dimension
    sigma_noise: float = 0.1,  # Noise std
    n_samples: int = 10000,    # Number of samples
    seed: int = 42         # Random seed
)
```

**Methods**:
```python
def _generate_ground_truth() -> Tuple[Tensor, Tensor]:
    """
    Generate A (d×k orthonormal) and Σ (k×k positive definite).
    
    A: Use QR decomposition of random matrix
    Σ: Generate as B @ B^T + εI for stability
    """

def _generate_data() -> Tuple[Tensor, Tensor]:
    """
    Generate X = AY + W.
    Returns: (X, Y) both of shape (n_samples, d) and (n_samples, k)
    """

def _compute_theory():
    """
    Compute theoretical quantities:
    - Σ_X = A @ Σ @ A^T + σ²I
    - U_k = top k eigenvectors of Σ_X (PCA solution)
    - z_true = X @ U_k (optimal latent representation)
    - Σ_z = U_k^T @ Σ_X @ U_k
    - σ_z² = mean(diag(Σ_z))
    
    Store all as attributes for analysis.
    """

def get_dataloader(batch_size: int = 64) -> DataLoader:
    """Return PyTorch DataLoader."""

def get_initialization_batch(n_init: int = 1000) -> Tensor:
    """
    Return z = X @ U_k for first n_init samples.
    Used to estimate σ_z² for R(D) initialization.
    """
```

**Important**: Store theoretical quantities (`U_k`, `σ_z²`, etc.) as attributes for validation.

---

### 2. Vector Quantizer (`src/models/quantizer.py`)

**Class**: `VectorQuantizer(nn.Module)`

**Constructor**:
```python
def __init__(
    self,
    dim: int,              # Latent dimension k
    codebook_size: int,    # Number of codewords n
    init_method: str = 'uniform',  # 'uniform', 'kmeans', 'rd_gaussian'
    init_data: Optional[Tensor] = None  # For kmeans/R(D)
)
```

**Initialization Methods**:

```python
def _uniform_init() -> Tensor:
    """
    Standard practice: Uniform[-1/n, 1/n] per dimension.
    Return: (codebook_size, dim) tensor
    """
    n = self.codebook_size
    return torch.rand(n, self.dim) * (2/n) - (1/n)

def _kmeans_init(data: Tensor) -> Tensor:
    """
    K-means clustering on data.
    data: (N, dim) tensor
    Return: (codebook_size, dim) tensor
    
    Use sklearn.cluster.KMeans for simplicity.
    """
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=self.codebook_size, random_state=42)
    kmeans.fit(data.cpu().numpy())
    return torch.from_numpy(kmeans.cluster_centers_).float()

def _rd_gaussian_init(data: Tensor) -> Tensor:
    """
    R(D)-optimal initialization.
    
    Steps:
    1. Estimate σ_z² = mean(data.var(dim=0))
    2. Compute rate: R = log2(codebook_size) / dim
    3. Compute target variance: σ_ẑ² = σ_z² × (1 - 2^(-2R))
    4. Sample: N(0, σ_ẑ² I_k)
    
    Return: (codebook_size, dim) tensor
    """
    # Estimate variance
    sigma_z_sq = data.var(dim=0).mean().item()
    
    # Compute R(D) target variance
    R = np.log2(self.codebook_size) / self.dim
    sigma_zhat_sq = sigma_z_sq * (1 - 2**(-2*R))
    
    # Sample codebook
    return torch.randn(self.codebook_size, self.dim) * np.sqrt(sigma_zhat_sq)
```

**Forward Method**:
```python
def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Vector quantization with straight-through estimator.
    
    Args:
        z: (B, dim) encoder outputs
    
    Returns:
        z_q: (B, dim) quantized vectors (with ST gradient)
        indices: (B,) nearest codeword indices
    """
    # Compute distances: (B, codebook_size)
    distances = torch.cdist(z, self.codebook)
    
    # Get nearest codeword indices
    indices = distances.argmin(dim=1)
    
    # Lookup quantized values
    z_q = self.codebook[indices]
    
    # Straight-through estimator: 
    # Forward: use z_q
    # Backward: copy gradient from z_q to z
    z_q = z + (z_q - z).detach()
    
    return z_q, indices
```

**Gradient Updates**:
The codebook is an `nn.Parameter`, so it gets updated via:
```python
# Codebook loss (move codewords toward assigned encoder outputs)
codebook_loss = F.mse_loss(z.detach(), z_q)

# Or equivalently in VQ-VAE paper:
codebook_loss = (z.detach() - z_q).pow(2).mean()
```

---

### 3. VQ-VAE Model (`src/models/vqvae.py`)

**Class**: `LinearGaussianVQVAE(nn.Module)`

```python
class LinearGaussianVQVAE(nn.Module):
    def __init__(
        self,
        d: int,                    # Ambient dimension
        k: int,                    # Latent dimension
        codebook_size: int,        # Number of codewords
        U_k: Tensor,               # PCA eigenvectors (d, k) - FIXED
        init_method: str = 'uniform',
        init_data: Optional[Tensor] = None
    ):
        """
        Linear Gaussian VQ-VAE with PCA encoder/decoder.
        
        Architecture:
        1. Encoder: z = U_k^T @ x  [FIXED, not trained]
        2. Quantizer: ẑ = Q(z)     [trained]
        3. Decoder: x̂ = U_k @ ẑ    [FIXED, not trained]
        """
        super().__init__()
        
        # Fixed encoder/decoder (PCA solution)
        self.register_buffer('encoder', U_k.T)  # (k, d)
        self.register_buffer('decoder', U_k)    # (d, k)
        
        # Trainable quantizer
        self.quantizer = VectorQuantizer(
            dim=k,
            codebook_size=codebook_size,
            init_method=init_method,
            init_data=init_data
        )
    
    def encode(self, x: Tensor) -> Tensor:
        """z = U_k^T @ x"""
        return x @ self.decoder  # (B, d) @ (d, k) -> (B, k)
    
    def decode(self, z: Tensor) -> Tensor:
        """x̂ = U_k @ z"""
        return z @ self.encoder.T  # (B, k) @ (k, d) -> (B, d)
    
    def forward(self, x: Tensor) -> dict:
        """
        Full forward pass.
        
        Returns dict with:
            - x_recon: reconstructed x
            - z: encoder output
            - z_q: quantized z
            - indices: codebook indices
        """
        z = self.encode(x)
        z_q, indices = self.quantizer(z)
        x_recon = self.decode(z_q)
        
        return {
            'x_recon': x_recon,
            'z': z,
            'z_q': z_q,
            'indices': indices
        }
```

**Important**: Encoder and decoder are FIXED (not trained) in linear Gaussian experiments. Only quantizer is trained.

---

### 4. Loss Function (`src/training/losses.py`)

```python
def vqvae_loss(
    x: Tensor,
    x_recon: Tensor,
    z: Tensor,
    z_q: Tensor,
    beta: float = 0.25
) -> Tuple[Tensor, dict]:
    """
    VQ-VAE loss from original paper.
    
    L = ||x - x̂||² + ||sg[z] - z_q||² + β||z - sg[z_q]||²
    
    where sg[·] is stop-gradient.
    
    Components:
    1. Reconstruction loss: ||x - x̂||²
    2. Codebook loss: ||z_e - z_q||² (move codebook toward encoder)
    3. Commitment loss: β||z_e - z_q||² (move encoder toward codebook)
    
    Args:
        x: original data (B, d)
        x_recon: reconstructed data (B, d)
        z: encoder output (B, k)
        z_q: quantized z (B, k)
        beta: commitment loss weight
    
    Returns:
        total_loss: scalar
        loss_dict: dict with individual loss components
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(x_recon, x)
    
    # Codebook loss (sg on encoder output)
    codebook_loss = F.mse_loss(z.detach(), z_q)
    
    # Commitment loss (sg on quantized output)
    commitment_loss = F.mse_loss(z, z_q.detach())
    
    # Total loss
    total_loss = recon_loss + codebook_loss + beta * commitment_loss
    
    loss_dict = {
        'total': total_loss.item(),
        'recon': recon_loss.item(),
        'codebook': codebook_loss.item(),
        'commitment': commitment_loss.item()
    }
    
    return total_loss, loss_dict
```

---

### 5. Metrics Tracker (`src/evaluation/metrics.py`)

```python
class MetricsTracker:
    """Track and compute all experimental metrics."""
    
    def __init__(self, codebook_size: int):
        self.codebook_size = codebook_size
        self.metrics = defaultdict(list)
    
    def update(self, step: int, **kwargs):
        """Add metrics for a step."""
        self.metrics['step'].append(step)
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def compute_utilization_metrics(
        self, 
        indices: Tensor
    ) -> dict:
        """
        Compute codebook utilization metrics.
        
        Args:
            indices: (B,) tensor of assigned codeword indices
        
        Returns dict with:
            - perplexity: exp(entropy) of usage distribution
            - dead_codes: number of unused codes
            - min_usage: minimum usage probability
            - max_usage: maximum usage probability
            - usage_std: std of usage distribution
        """
        # Count usage
        counts = torch.bincount(
            indices.flatten(), 
            minlength=self.codebook_size
        ).float()
        usage = counts / counts.sum()
        
        # Perplexity: exp(H[usage])
        entropy = -(usage * torch.log(usage + 1e-10)).sum()
        perplexity = torch.exp(entropy)
        
        # Dead codes (usage < threshold)
        dead_threshold = 1e-4
        dead_codes = (usage < dead_threshold).sum()
        
        return {
            'perplexity': perplexity.item(),
            'dead_codes': dead_codes.item(),
            'min_usage': usage.min().item(),
            'max_usage': usage.max().item(),
            'usage_std': usage.std().item(),
            'usage_entropy': entropy.item()
        }
    
    def compute_distortion_metrics(
        self,
        z: Tensor,
        z_q: Tensor
    ) -> dict:
        """
        Compute quantization distortion.
        
        Args:
            z: encoder outputs (B, k)
            z_q: quantized outputs (B, k)
        
        Returns:
            - quantization_error: E[||z - ẑ||²]
            - per_dimension: mean squared error per dimension
        """
        se = (z - z_q).pow(2)
        
        return {
            'quantization_error': se.mean().item(),
            'quant_error_per_dim': se.mean(dim=0).mean().item()
        }
    
    def compute_codebook_stats(
        self,
        codebook: Tensor
    ) -> dict:
        """
        Statistics of codebook itself.
        
        Args:
            codebook: (n, k) tensor
        
        Returns:
            - mean_norm: average ||c_j||
            - std_norm: std of ||c_j||
            - mean_distance: average pairwise distance
        """
        norms = codebook.norm(dim=1)
        
        # Pairwise distances (expensive, subsample if needed)
        if len(codebook) > 100:
            idx = torch.randperm(len(codebook))[:100]
            sub_codebook = codebook[idx]
        else:
            sub_codebook = codebook
        
        pdist = torch.cdist(sub_codebook, sub_codebook)
        mean_distance = pdist[pdist > 0].mean()
        
        return {
            'codebook_mean_norm': norms.mean().item(),
            'codebook_std_norm': norms.std().item(),
            'codebook_mean_distance': mean_distance.item()
        }
    
    def save(self, path: str):
        """Save metrics to CSV."""
        import pandas as pd
        df = pd.DataFrame(self.metrics)
        df.to_csv(path, index=False)
    
    def load(self, path: str):
        """Load metrics from CSV."""
        import pandas as pd
        df = pd.read_csv(path)
        self.metrics = df.to_dict('list')
```

---

### 6. Training Loop (`src/training/trainer.py`)

```python
class Trainer:
    """Training loop for VQ-VAE experiments."""
    
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
        batch: Tensor,
        beta: float = 0.25
    ) -> dict:
        """Single training step."""
        self.model.train()
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
        metrics_tracker: MetricsTracker
    ) -> dict:
        """Evaluate on full dataset."""
        self.model.eval()
        
        all_indices = []
        all_z = []
        all_z_q = []
        total_recon_loss = 0.0
        
        for batch in dataloader:
            x = batch[0].to(self.device)  # batch is (x, y) tuple
            outputs = self.model(x)
            
            all_indices.append(outputs['indices'])
            all_z.append(outputs['z'])
            all_z_q.append(outputs['z_q'])
            total_recon_loss += F.mse_loss(outputs['x_recon'], x).item()
        
        # Concatenate
        indices = torch.cat(all_indices, dim=0)
        z = torch.cat(all_z, dim=0)
        z_q = torch.cat(all_z_q, dim=0)
        
        # Compute metrics
        utilization = metrics_tracker.compute_utilization_metrics(indices)
        distortion = metrics_tracker.compute_distortion_metrics(z, z_q)
        codebook_stats = metrics_tracker.compute_codebook_stats(
            self.model.quantizer.codebook
        )
        
        metrics = {
            'recon_loss': total_recon_loss / len(dataloader),
            **utilization,
            **distortion,
            **codebook_stats
        }
        
        return metrics
```

---

### 7. Experiment Script (`experiments/idea_7_initialization/run_linear_gaussian.py`)

```python
"""
Experiment: Test R(D) initialization on Linear Gaussian model.

Compare three initialization methods:
1. Uniform: standard [-1/n, 1/n]
2. K-means: cluster first batch
3. R(D) Gaussian: theory-based initialization
"""

import torch
import numpy as np
from pathlib import Path
import yaml
from typing import Dict

from src.data.linear_gaussian import LinearGaussianDataset
from src.models.vqvae import LinearGaussianVQVAE
from src.training.trainer import Trainer
from src.evaluation.metrics import MetricsTracker
from src.evaluation.visualization import plot_training_curves, plot_utilization


def run_experiment(config: Dict, init_method: str) -> MetricsTracker:
    """
    Run single experiment with specified initialization.
    
    Args:
        config: experiment configuration
        init_method: 'uniform', 'kmeans', or 'rd_gaussian'
    
    Returns:
        MetricsTracker with all recorded metrics
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {init_method}")
    print(f"{'='*60}\n")
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Generate data
    dataset = LinearGaussianDataset(
        d=config['d'],
        k=config['k'],
        sigma_noise=config['sigma_noise'],
        n_samples=config['n_samples'],
        seed=config['seed']
    )
    
    dataloader = dataset.get_dataloader(batch_size=config['batch_size'])
    
    # Get initialization data
    init_data = dataset.get_initialization_batch(n_init=1000)
    
    # Create model
    model = LinearGaussianVQVAE(
        d=config['d'],
        k=config['k'],
        codebook_size=config['codebook_size'],
        U_k=dataset.U_k,
        init_method=init_method,
        init_data=init_data
    )
    
    # Optimizer (only codebook is trainable)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    # Trainer and metrics
    trainer = Trainer(model, optimizer, device='cpu')
    metrics_tracker = MetricsTracker(codebook_size=config['codebook_size'])
    
    # IMPORTANT: Evaluate BEFORE training (test initialization quality)
    print("Evaluating initialization...")
    init_metrics = trainer.evaluate(dataloader, metrics_tracker)
    metrics_tracker.update(step=0, **init_metrics)
    
    # Print theoretical prediction
    R = np.log2(config['codebook_size']) / config['k']
    sigma_z_sq = dataset.sigma_z_squared
    theoretical_distortion = config['k'] * sigma_z_sq * (2 ** (-2 * R))
    
    print(f"\nInitialization Results:")
    print(f"  Quantization Error: {init_metrics['quantization_error']:.6f}")
    print(f"  Theoretical Minimum: {theoretical_distortion:.6f}")
    print(f"  Ratio: {init_metrics['quantization_error'] / theoretical_distortion:.2f}x")
    print(f"  Dead Codes: {init_metrics['dead_codes']} / {config['codebook_size']}")
    print(f"  Perplexity: {init_metrics['perplexity']:.2f} (max={config['codebook_size']})")
    
    # Training loop
    print(f"\nTraining for {config['num_steps']} steps...")
    for step in range(1, config['num_steps'] + 1):
        # Sample batch
        batch = next(iter(dataloader))
        
        # Train step
        loss_dict = trainer.train_step(batch, beta=config['beta'])
        
        # Evaluate periodically
        if step % config['eval_every'] == 0:
            metrics = trainer.evaluate(dataloader, metrics_tracker)
            metrics_tracker.update(step=step, **metrics, **loss_dict)
            
            if step % (config['eval_every'] * 10) == 0:
                print(f"Step {step}/{config['num_steps']}: "
                      f"Quant Error = {metrics['quantization_error']:.6f}, "
                      f"Dead Codes = {metrics['dead_codes']}, "
                      f"Perplexity = {metrics['perplexity']:.2f}")
    
    # Final evaluation
    final_metrics = trainer.evaluate(dataloader, metrics_tracker)
    print(f"\nFinal Results:")
    print(f"  Quantization Error: {final_metrics['quantization_error']:.6f}")
    print(f"  Dead Codes: {final_metrics['dead_codes']}")
    print(f"  Perplexity: {final_metrics['perplexity']:.2f}")
    
    return metrics_tracker, dataset


def main():
    """Run all experiments and compare results."""
    
    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create results directory
    results_dir = Path('results/idea_7_linear_gaussian')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiments for each initialization method
    methods = ['uniform', 'kmeans', 'rd_gaussian']
    all_results = {}
    dataset = None
    
    for method in methods:
        tracker, dataset = run_experiment(config, method)
        all_results[method] = tracker
        
        # Save individual results
        method_dir = results_dir / method
        method_dir.mkdir(exist_ok=True)
        tracker.save(method_dir / 'metrics.csv')
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plot_dir = results_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)
    
    plot_training_curves(all_results, dataset, save_dir=plot_dir)
    plot_utilization(all_results, save_dir=plot_dir)
    
    print(f"\nResults saved to: {results_dir}")
    print("Experiment complete!")


if __name__ == "__main__":
    main()
```

---

### 8. Configuration (`experiments/idea_7_initialization/config.yaml`)

```yaml
# Linear Gaussian Experiment Configuration

# Data generation
d: 64                    # Ambient dimension
k: 8                     # Latent dimension
sigma_noise: 0.1         # Noise standard deviation
n_samples: 10000         # Number of training samples
seed: 42                 # Random seed

# Model
codebook_size: 256       # Number of codewords (n = 2^8)
                         # Rate R = log2(256)/8 = 1 bit/dim

# Training
batch_size: 128
lr: 0.001                # Learning rate
beta: 0.25               # Commitment loss weight
num_steps: 5000          # Training steps
eval_every: 50           # Evaluation frequency

# Expected theoretical values (for reference)
# R = 1 bit/dim
# For σ_z² ≈ 1:
#   - R(D) bound: k × σ_z² × 2^(-2R) ≈ 8 × 1 × 0.25 = 2.0
#   - Standard init: k × σ_z² ≈ 8 (no compression)
```

---

### 9. Visualization (`src/evaluation/visualization.py`)

```python
"""Plotting utilities for experiment results."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict
import pandas as pd


def plot_training_curves(
    all_results: Dict[str, 'MetricsTracker'],
    dataset: 'LinearGaussianDataset',
    save_dir: Path
):
    """
    Plot training curves comparing initialization methods.
    
    Creates figure with 4 subplots:
    1. Quantization error over time
    2. Perplexity over time
    3. Dead codes over time
    4. Reconstruction loss over time
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Curves: Initialization Comparison', fontsize=16)
    
    # Compute theoretical bound
    k = dataset.k
    sigma_z_sq = dataset.sigma_z_squared
    codebook_size = list(all_results.values())[0].codebook_size
    R = np.log2(codebook_size) / k
    theoretical_bound = k * sigma_z_sq * (2 ** (-2 * R))
    
    methods = list(all_results.keys())
    colors = {'uniform': 'red', 'kmeans': 'blue', 'rd_gaussian': 'green'}
    labels = {'uniform': 'Uniform', 'kmeans': 'K-Means', 'rd_gaussian': 'R(D) Gaussian'}
    
    for method in methods:
        metrics = all_results[method].metrics
        df = pd.DataFrame(metrics)
        
        # Plot 1: Quantization error
        axes[0, 0].plot(
            df['step'], 
            df['quantization_error'],
            color=colors[method],
            label=labels[method],
            linewidth=2
        )
        
        # Plot 2: Perplexity
        axes[0, 1].plot(
            df['step'],
            df['perplexity'],
            color=colors[method],
            label=labels[method],
            linewidth=2
        )
        
        # Plot 3: Dead codes
        axes[1, 0].plot(
            df['step'],
            df['dead_codes'],
            color=colors[method],
            label=labels[method],
            linewidth=2
        )
        
        # Plot 4: Reconstruction loss
        axes[1, 1].plot(
            df['step'],
            df['recon_loss'],
            color=colors[method],
            label=labels[method],
            linewidth=2
        )
    
    # Formatting
    axes[0, 0].axhline(
        theoretical_bound, 
        color='black', 
        linestyle='--', 
        label=f'R(D) Bound = {theoretical_bound:.3f}'
    )
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Quantization Error')
    axes[0, 0].set_title('Quantization Error (MSE)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    axes[0, 1].axhline(
        codebook_size, 
        color='black', 
        linestyle='--', 
        label=f'Max = {codebook_size}'
    )
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Perplexity')
    axes[0, 1].set_title('Codebook Utilization (Perplexity)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Number of Dead Codes')
    axes[1, 0].set_title('Dead Codes (usage < 1e-4)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Reconstruction Loss')
    axes[1, 1].set_title('Reconstruction Loss (MSE)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_dir / 'training_curves.png'}")


def plot_utilization(
    all_results: Dict[str, 'MetricsTracker'],
    save_dir: Path
):
    """
    Plot codebook utilization comparison at initialization and final.
    
    Bar plots showing:
    - Initial perplexity
    - Final perplexity
    - Initial dead codes
    - Final dead codes
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Codebook Utilization: Initialization vs. Final', fontsize=14)
    
    methods = list(all_results.keys())
    labels = {'uniform': 'Uniform', 'kmeans': 'K-Means', 'rd_gaussian': 'R(D) Gaussian'}
    
    # Extract initial and final metrics
    init_perplexity = []
    final_perplexity = []
    init_dead = []
    final_dead = []
    
    for method in methods:
        metrics = all_results[method].metrics
        init_perplexity.append(metrics['perplexity'][0])
        final_perplexity.append(metrics['perplexity'][-1])
        init_dead.append(metrics['dead_codes'][0])
        final_dead.append(metrics['dead_codes'][-1])
    
    x = np.arange(len(methods))
    width = 0.35
    
    # Perplexity
    axes[0].bar(
        x - width/2, 
        init_perplexity, 
        width, 
        label='Initial',
        alpha=0.8
    )
    axes[0].bar(
        x + width/2, 
        final_perplexity, 
        width, 
        label='Final',
        alpha=0.8
    )
    axes[0].set_ylabel('Perplexity')
    axes[0].set_title('Codebook Perplexity')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([labels[m] for m in methods])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Dead codes
    axes[1].bar(
        x - width/2,
        init_dead,
        width,
        label='Initial',
        alpha=0.8
    )
    axes[1].bar(
        x + width/2,
        final_dead,
        width,
        label='Final',
        alpha=0.8
    )
    axes[1].set_ylabel('Number of Dead Codes')
    axes[1].set_title('Dead Codes (usage < 1e-4)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([labels[m] for m in methods])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'utilization_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_dir / 'utilization_comparison.png'}")
```

---

## Testing & Validation

### Unit Tests

Create `tests/` directory with:

```python
# tests/test_data.py
def test_linear_gaussian_generation():
    """Test data generator produces correct shapes and statistics."""
    dataset = LinearGaussianDataset(d=32, k=4, n_samples=1000)
    assert dataset.X.shape == (1000, 32)
    assert dataset.Y.shape == (1000, 4)
    assert dataset.U_k.shape == (32, 4)
    
    # Check PCA recovers signal subspace
    overlap = (dataset.U_k.T @ dataset.A).abs()
    # Should be close to identity (up to sign/rotation)
    assert (overlap.max(dim=0).values > 0.9).all()


# tests/test_quantizer.py
def test_initialization_variances():
    """Test that initialization methods produce correct variances."""
    dim, n = 8, 256
    
    # Generate test data
    sigma_z_sq = 1.0
    data = torch.randn(1000, dim) * np.sqrt(sigma_z_sq)
    
    # Uniform init
    quantizer_unif = VectorQuantizer(dim, n, 'uniform')
    var_unif = quantizer_unif.codebook.var(dim=0).mean()
    expected_unif = (1/3) * (2/n)**2
    assert torch.isclose(var_unif, torch.tensor(expected_unif), rtol=0.1)
    
    # R(D) init
    quantizer_rd = VectorQuantizer(dim, n, 'rd_gaussian', init_data=data)
    var_rd = quantizer_rd.codebook.var(dim=0).mean()
    R = np.log2(n) / dim
    expected_rd = sigma_z_sq * (1 - 2**(-2*R))
    assert torch.isclose(var_rd, torch.tensor(expected_rd), rtol=0.1)
```

Run with: `pytest tests/`

---

## Expected Results

Based on theory, we expect:

### Initialization Quality (Step 0)

| Metric | Uniform | K-Means | R(D) Gaussian |
|--------|---------|---------|---------------|
| Quantization Error | ~8.0 | ~2.5 | **~2.0** |
| Dead Codes | 100-150 | 10-30 | **0-5** |
| Perplexity | 50-100 | 150-200 | **220-256** |

### Final Performance (After Training)

| Metric | Uniform | K-Means | R(D) Gaussian |
|--------|---------|---------|---------------|
| Quantization Error | ~2.0 | ~2.0 | **~2.0** |
| Dead Codes | 20-50 | 5-15 | **0-5** |
| Perplexity | 150-200 | 200-240 | **240-256** |
| Steps to Converge | 3000+ | 1500 | **500-1000** |

**Key Findings**:
1. R(D) init starts much closer to optimal
2. R(D) init avoids dead codes from the beginning
3. R(D) init converges 2-3× faster
4. All methods converge to similar final performance (but R(D) is more reliable)

---

## Dependencies

**requirements.txt**:
```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
pandas>=2.0.0
pyyaml>=6.0
jupyter>=1.0.0
pytest>=7.4.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Running the Experiments

1. **Setup**:
```bash
cd vqvae-research
pip install -r requirements.txt
```

2. **Run linear Gaussian experiment**:
```bash
python experiments/idea_7_initialization/run_linear_gaussian.py
```

3. **View results**:
```bash
ls results/idea_7_linear_gaussian/plots/
# Should see: training_curves.png, utilization_comparison.png
```

4. **Interactive analysis**:
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

---

## Success Criteria

The implementation is correct if:

1. **Theory matches practice**: 
   - R(D) init distortion matches prediction: `D₀ ≈ k σ_z² × 2^(-2R)`
   - Standard init distortion is ~4× worse

2. **Dead codes**:
   - R(D) init: 0-5 dead codes initially
   - Uniform init: 100+ dead codes initially

3. **Convergence**:
   - All methods reach similar final performance
   - R(D) init converges 2-3× faster

4. **Reproducibility**:
   - Running twice with same seed produces identical results
   - Metrics saved to CSV for later analysis

---

## Next Steps After Implementation

Once this is working:

1. **Extend to MNIST**: Test on real image data
2. **Idea 1**: Add population dynamics analysis
3. **Theoretical extensions**: 
   - Analyze training dynamics with ODE
   - Prove convergence rates
4. **Paper**: Write up results with theory + experiments

---

## Contact & Questions

If anything is unclear:
1. Check `VQVAE_theory.md` for theoretical background
2. Check `Ideas.md` for research motivation
3. Review this guide's examples

Key design principles:
- **Modularity**: Each component is independent and testable
- **Theory-driven**: Every metric connects to theoretical prediction
- **Reproducibility**: Fixed seeds, logged hyperparameters
- **Clarity**: Code should be readable and well-documented

Good luck! This should provide a solid foundation for the research.
