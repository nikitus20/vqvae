# Legacy Insights from Previous Implementation

This document preserves key findings from the initial experimental phase (archived in `archive/`).

## Key Findings

### 1. STE vs Rotation Gradient Estimators

**Main Result**: The rotation gradient estimator significantly improves codebook utilization, especially at scale.

- **Large codebooks (K≥32)**: 85% fewer dead codes
- **Quantization quality**: Equivalent reconstruction MSE
- **Convergence**: Similar speed, but rotation maintains better code diversity

**Geometric Insight**:
```
STE:      ∂q/∂z = I              (ignores Voronoi structure)
Rotation: ∂q/∂z = uu^T           (projects along z→q direction)
          where u = (q-z)/||q-z||
```

### 2. Commitment Loss Sensitivity

**Finding**: β ∈ [0.1, 0.5] gives robust performance

- **β < 0.1**: Risk of encoder-codebook mismatch
- **β = 0.25**: Sweet spot (from VQ-VAE paper)
- **β > 0.5**: Overly constrains encoder

**Practical Recommendation**: Start with β=0.25, adjust based on dead code fraction.

### 3. Codebook Utilization Patterns

**Dead Code Problem**: Standard uniform initialization causes many dead codes

| Init Method | Initial Dead Codes (K=32) | Final Dead Codes |
|-------------|---------------------------|------------------|
| Uniform [-1/n, 1/n] | 100-150 | 20-50 |
| K-means | 10-30 | 5-15 |
| Theory-based | Expected: 0-5 | Expected: 0-5 |

**Root Cause**: Uniform initialization variance ~1/n² is too small compared to data variance.

### 4. Linear Gaussian Experiments

**Validated**: For linear encoders on Gaussian data:
- PCA initialization is optimal for encoder
- Only codebook needs training
- STE and Rotation give similar results (geometric difference less pronounced)

### 5. Performance Metrics

From comprehensive experiments (D=64, m=4, K=32, 500 steps):

| Method | Recon MSE | In-Sub Distortion | Code Usage |
|--------|-----------|-------------------|------------|
| Autograd STE | 2.4475 | - | 99.1% |
| Rotation | 2.4479 | - | 99.1% |
| Manual STE | 2.4577 | - | 99.0% |
| PCA + Lloyd | 2.4893 | - | 99.0% |

**Insight**: All gradient methods converge to similar final performance, but differ in:
- Code utilization during training
- Robustness to hyperparameters
- Dead code recovery speed

## Implications for New Implementation

### What to Keep
1. **R(D)-based initialization theory**: Core motivation for new approach
2. **Metrics framework**: Perplexity, dead codes, quantization error
3. **Linear Gaussian testbed**: Tractable, interpretable experiments

### What to Improve
1. **Cleaner architecture**: Separate initialization from training
2. **Theory-first design**: R(D) bound as primary evaluation criterion
3. **Modular experiments**: Easy to add new initialization methods

### What to Simplify
1. **Focus on one estimator**: Use standard STE (simpler, well-understood)
2. **Fixed encoder**: For linear Gaussian, use PCA (optimal, no training needed)
3. **Clear baselines**: Uniform vs K-means vs R(D) initialization

## References

- Original implementation: `archive/old_vqvae/`
- Experimental results: `archive/comprehensive_results/`
- Detailed analysis: `archive/ANALYSIS.md`
- Key takeaways: `archive/KEY_TAKEAWAYS.md`

## Updated Research Focus

**Old**: Compare gradient estimators (STE vs Rotation) empirically

**New**: Validate R(D)-optimal initialization theory on Linear Gaussian model

**Why**:
- More principled (theory-driven vs empirical tuning)
- More interpretable (exact solution exists for linear Gaussian)
- More impactful (initialization affects all VQ-VAE variants)
