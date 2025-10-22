# VQ-VAE: Unified Vector Quantization Framework

A clean, unified framework for comparing different Vector Quantization methods in VQ-VAE. This project implements three approaches for training VQ-VAE models and compares their performance on synthetic low-rank data.

## Architecture

The framework follows a modular VQVAE design:

```
X → Encoder (W) → Z → Quantize (E) → Q → Decoder (W^T) → X_hat
```

- **Encoder**: Linear map `W: R^D → R^m` with optional orthonormal constraint (Stiefel manifold)
- **Codebook**: `E ∈ R^{K×m}` containing K code vectors
- **Decoder**: Transpose of encoder `W^T: R^m → R^D`

## Three Methods

### 1. PCA + Lloyd Max (Baseline)
Classic approach:
- Initialize encoder with top-m PCA eigenvectors
- Run k-means (Lloyd algorithm) on latent codes
- No gradient-based training

### 2. Autograd STE
Gradient descent using PyTorch's automatic differentiation:
- Straight-Through Estimator (STE) for non-differentiable quantization
- Autograd computes all gradients
- Stiefel projection after each update

### 3. Manual STE
Gradient descent with analytically derived gradients:
- Hand-computed STE gradients for encoder and codebook
- Riemannian gradient projection for Stiefel manifold
- Exact mathematical updates

## Project Structure

```
vqvae/
├── models/
│   ├── encoder.py       # Linear encoder with orthonormal constraint
│   ├── codebook.py      # Codebook with k-means initialization
│   └── vqvae.py         # Complete VQ-VAE model
├── methods/
│   ├── pca_lloyd.py     # Method 1: PCA + Lloyd
│   ├── ste_autograd.py  # Method 2: Autograd STE
│   └── ste_manual.py    # Method 3: Manual STE
├── utils.py             # Data generation, metrics, utilities
└── experiment.py        # Main experiment runner

run_experiment.py        # Entry point
```

## Installation

```bash
# Core dependencies
pip install numpy matplotlib

# Optional: For autograd method
pip install torch
```

## Usage

### Quick Start

Run comparison of all three methods:

```bash
python run_experiment.py
```

### Custom Configuration

```bash
python run_experiment.py \
  --D 64 \              # Ambient dimension
  --rank 6 \            # True data rank
  --m 4 \               # Latent dimension
  --K 32 \              # Codebook size
  --n_train 20000 \     # Training samples
  --n_eval 4000 \       # Evaluation samples
  --steps 200 \         # SGD steps
  --batch 1024 \        # Batch size
  --alpha 1.0 \         # Codebook loss weight
  --beta 0.25 \         # Commitment loss weight
  --lrW 0.08 \          # Encoder learning rate
  --lrE 0.2 \           # Codebook learning rate
  --seed 0 \            # Random seed
  --outdir results_out  # Output directory
```

### Output

The experiment generates:
- `summary.json` - Final metrics for all methods
- `comparison.png` - Bar chart comparing methods
- `manual_ste.png` - Training curves for Manual STE
- `autograd_ste.png` - Training curves for Autograd STE (if PyTorch available)
- `*.json` - Detailed training logs

## Metrics

The framework tracks:
- **Reconstruction MSE**: `||X - X_hat||^2`
- **In-Subspace Distortion**: `||Z - Q||^2`
- **Usage Entropy**: Code utilization (higher = better)
- **Dead Code Fraction**: Unused codebook entries

## Example Output

```
======================================================================
SUMMARY
======================================================================

Final Metrics:
Method               Recon MSE       In-Sub Dist     Usage Ent       Dead Frac
--------------------------------------------------------------------------------
PCA + Lloyd          2.983021        2.331891        2.7321          0.0000
Autograd STE         2.891234        2.245678        2.7456          0.0000
Manual STE           2.887543        2.241289        2.7489          0.0000
```

## Key Features

✓ **Unified Interface**: All methods use same VQVAE architecture
✓ **Clean Separation**: Models, methods, and experiments are modular
✓ **Easy Extension**: Add new methods by implementing training function
✓ **Comprehensive Metrics**: Track reconstruction, distortion, and code usage
✓ **Reproducible**: Seeded random generation for consistent results

## Adding New Methods

To add a new training method:

1. Create `vqvae/methods/your_method.py`
2. Implement `train_your_method(X_train, X_eval, m, K, ...)`
3. Return dict with `{"model": vqvae, "metrics": dict, "logs": list}`
4. Import in `vqvae/methods/__init__.py`
5. Add to `experiment.py`

## Mathematical Details

### Straight-Through Estimator (STE)

The quantization operation is non-differentiable:
```
Q = E[argmin_k ||Z - E_k||^2]
```

STE approximates gradients by copying gradients through quantization:
```
∂L/∂Z ≈ ∂L/∂Q  (straight-through)
```

### Loss Function

Total loss combines three terms:
```
L = ||X - X_hat||^2 + α||Z - Q||^2 + β||Z - Q||^2
    \_____________/   \___________/   \___________/
    reconstruction    codebook loss   commitment loss
```

- **Reconstruction**: Minimize output error
- **Codebook**: Pull codes toward encoder outputs
- **Commitment**: Pull encoder outputs toward codes

## Experimental Results

We've conducted comprehensive experiments comparing all four methods. Key findings:

### Performance Summary (D=64, m=4, K=32, 500 steps)

| Method | Recon MSE | Improvement | Code Utilization |
|--------|-----------|-------------|------------------|
| **Autograd STE** | **2.4475** | **-1.68%** | 99.1% |
| **Rotation** | **2.4479** | **-1.66%** | 99.1% |
| **Manual STE** | 2.4577 | -1.27% | 99.0% |
| **PCA + Lloyd** | 2.4893 | baseline | 99.0% |

### Key Insights

1. ✅ **Manual STE matches Autograd**: Verified within 1.4% error across all metrics
2. 🔄 **STE ≈ Rotation**: No practical difference for linear encoders (0.02%)
3. 📊 **Excellent scalability**: Works well from K=8 to K=64
4. 🎯 **Commitment loss flexible**: β ∈ [0, 1] gives only 0.2% variation
5. 💯 **Perfect code utilization**: >98% of theoretical maximum

### Detailed Analysis

See [ANALYSIS.md](ANALYSIS.md) for comprehensive experimental results including:
- STE implementation verification
- Convergence analysis (500 steps)
- Codebook size scaling (K=8,16,32,64)
- Commitment loss ablation (β=0.0-1.0)

See [SUMMARY.md](SUMMARY.md) for a quick overview.

### Run Experiments

```bash
# Quick comparison of all methods
python run_experiment.py

# Comprehensive experiments (convergence, scaling, ablation)
python run_comprehensive_experiments.py

# Results saved to comprehensive_results/ with plots
```

## References

This implementation is based on:
- van den Oord et al., "Neural Discrete Representation Learning" (VQ-VAE, 2017)
- Bengio et al., "Estimating or Propagating Gradients Through Stochastic Neurons" (STE, 2013)

## License

MIT
