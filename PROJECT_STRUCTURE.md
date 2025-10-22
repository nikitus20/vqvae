# VQ-VAE Project Structure

Complete guide to all files in this repository.

## Core Framework (`vqvae/`)

### Models (`vqvae/models/`)
- **`encoder.py`**: Linear encoder W: R^D → R^m with Stiefel manifold constraint
  - `LinearEncoder` class with encode/decode/project methods
  - Supports both orthonormal and unconstrained variants
  - PCA initialization for principled starting point

- **`codebook.py`**: Vector quantization codebook E ∈ R^{K×m}
  - `Codebook` class with nearest-neighbor assignment
  - K-means++ initialization for stable convergence
  - Lloyd iteration support for baseline method

- **`vqvae.py`**: Complete VQ-VAE architecture
  - `VQVAE` class combining encoder + codebook + decoder
  - Clean forward pass interface
  - Unified model interface for all methods

### Training Methods (`vqvae/methods/`)
- **`pca_lloyd.py`**: Method 1 - PCA + Lloyd Max baseline
  - Single-pass optimization (no gradient descent)
  - Closed-form PCA solution
  - K-means for codebook initialization

- **`ste_autograd.py`**: Method 2 - Autograd STE
  - PyTorch implementation with automatic differentiation
  - Straight-Through Estimator gradient approximation
  - Full autograd validation of manual gradients

- **`ste_manual.py`**: Method 3 - Manual STE
  - Analytically derived gradients
  - Pure NumPy implementation (no PyTorch)
  - Riemannian gradient for Stiefel manifold

- **`rotation_autograd.py`**: Method 4 - Rotation Estimator
  - Geometric gradient projection approach
  - Projects gradients onto displacement direction
  - PyTorch custom autograd function

### Utilities (`vqvae/`)
- **`utils.py`**: Data generation and metrics
  - Low-rank Gaussian data generation
  - Comprehensive metrics (MSE, distortion, entropy, dead codes)
  - Seeding and reproducibility utilities

- **`experiment.py`**: Basic experiment runner
  - Single experiment comparing all methods
  - Plotting and visualization
  - JSON results export

- **`__init__.py`**: Package exports

## Experiment Scripts

### Entry Points
- **`run_experiment.py`**: Quick comparison runner
  - Runs all 4 methods on standard settings
  - Generates comparison plots
  - ~30 seconds runtime

- **`run_comprehensive_experiments.py`**: Full experimental suite
  - 4 experiments: verification, convergence, scaling, ablation
  - 13,700+ training steps total
  - ~5 minutes runtime
  - Systematic hyperparameter sweeps

### Visualization
- **`create_summary_figure.py`**: Generate comprehensive summary plot
  - 9-panel figure with all key results
  - High-resolution export (200 DPI)
  - Publication-ready formatting

## Documentation

### Main Docs
- **`README.md`**: Project overview and quick start
  - Architecture description
  - Usage instructions
  - Installation guide
  - Quick results summary

- **`ANALYSIS.md`**: Comprehensive experimental analysis (~5000 words)
  - Detailed results for all 4 experiments
  - Statistical comparisons
  - Theoretical insights
  - Surprising findings
  - Practical recommendations

- **`SUMMARY.md`**: Quick reference guide
  - TL;DR of all experiments
  - Key numbers and tables
  - When to use each method
  - Best hyperparameters

- **`PROJECT_STRUCTURE.md`**: This file
  - Complete file listing
  - Purpose of each component
  - Quick navigation guide

## Old Experiments (Archive)

These files were used during development and are now superseded by the unified framework:

- **`vq_experiments_three_variants.py`**: Original 3-method comparison
  - Contained PCA+Lloyd, Autograd STE, Manual STE
  - Now integrated into `vqvae/methods/`

- **`vq_ste_rotation_lab.py`**: STE vs Rotation experiments
  - Early rotation estimator implementation
  - Now improved in `vqvae/methods/rotation_autograd.py`

- **`vq_comprehensive_experiments.py`**: Earlier comprehensive tests
  - Scaling and ablation studies
  - Now superseded by `run_comprehensive_experiments.py`

- **`vq_2d_visualization.py`**: 2D toy visualizations
  - For debugging and intuition building

- **`vq_multimodal_experiment.py`**: Multi-modal data tests
  - Not included in final framework

**Note**: These files can be safely deleted; all functionality is in the new unified framework.

## Results and Outputs

### Generated Directories
- **`results_out/`**: Output from `run_experiment.py`
  - Quick comparison results
  - Training logs (JSON)
  - Individual method plots

- **`comprehensive_results/YYYYMMDD_HHMMSS/`**: Output from comprehensive experiments
  - Timestamped experiment directory
  - All 4 experiment results
  - High-quality plots
  - Detailed JSON metrics

### Key Output Files
```
comprehensive_results/YYYYMMDD_HHMMSS/
├── ste_verification.json          # Autograd vs Manual comparison
├── ste_verification.png            # Verification plots
├── convergence_results.json        # 500-step convergence data
├── convergence_comparison.png      # Training curves
├── codebook_size_results.json      # K=8,16,32,64 scaling
├── codebook_size_scaling.png       # Scaling plots
├── commitment_results.json         # β ablation data
├── commitment_ablation.png         # β sensitivity plots
└── all_experiments.json            # Master results file

comprehensive_results/
└── EXPERIMENTAL_SUMMARY.png        # 9-panel summary figure
```

## File Size Reference

| Category | Files | Total Lines | Est. Size |
|----------|-------|-------------|-----------|
| Core Framework | 9 | ~1200 | 40 KB |
| Experiments | 3 | ~800 | 25 KB |
| Documentation | 4 | ~2000 | 80 KB |
| Old Code | 5 | ~2500 | 90 KB |
| **Total** | **21** | **~6500** | **~235 KB** |

## Quick Navigation

### I want to...

**...understand the architecture**
→ Read `README.md` then `vqvae/models/vqvae.py`

**...see the results**
→ Look at `comprehensive_results/EXPERIMENTAL_SUMMARY.png` or read `SUMMARY.md`

**...run experiments**
→ Execute `python run_experiment.py` or `python run_comprehensive_experiments.py`

**...understand the theory**
→ Read `ANALYSIS.md` section on gradient estimators

**...implement my own method**
→ Copy `vqvae/methods/ste_manual.py` as template

**...add a new experiment**
→ Edit `run_comprehensive_experiments.py` and add new function

**...debug something**
→ Check `vqvae/utils.py` for metrics, add logging to method files

## Code Statistics

### Lines of Code (excluding comments/blanks)
- **Core models**: ~200 lines
- **Training methods**: ~600 lines
- **Experiments**: ~500 lines
- **Utilities**: ~150 lines

### Complexity
- **McCabe complexity**: Low (<10 per function)
- **Dependencies**: NumPy (required), PyTorch (optional), Matplotlib (visualization)
- **Test coverage**: Validated through comprehensive experiments

## Version History

### v1.0 (2025-09-30)
- ✅ Unified framework with 4 methods
- ✅ Comprehensive experiments (4 studies)
- ✅ Full documentation (3 docs)
- ✅ Validation: Manual STE matches Autograd within 1.4%

### Future (v2.0)
- [ ] Nonlinear encoders (MLP/CNN)
- [ ] Real image data (MNIST, CIFAR)
- [ ] Larger scale (D=512, K=512)
- [ ] Additional estimators (Gumbel-Softmax)

## Citation

If you use this code in your research:

```bibtex
@misc{vqvae2025,
  title={VQ-VAE: Unified Framework for Vector Quantization Methods},
  author={[Your Name]},
  year={2025},
  howpublished={\url{https://github.com/[your-repo]}}
}
```

## Maintenance

### Key Files to Update
- Modify methods: Edit `vqvae/methods/*.py`
- Add experiments: Edit `run_comprehensive_experiments.py`
- Update docs: Edit `README.md`, `ANALYSIS.md`, `SUMMARY.md`

### Testing
```bash
# Quick smoke test
python run_experiment.py --steps 10 --n_train 1000

# Full validation
python run_comprehensive_experiments.py --seed 42
```

## Contact

For questions, issues, or contributions:
- Open an issue on GitHub
- Check documentation first (README.md, ANALYSIS.md)
- Cite relevant experiments from ANALYSIS.md

---

**Last Updated**: 2025-09-30
**Framework Version**: 1.0
**Total Experiments**: 13,700+ steps across 4 studies
