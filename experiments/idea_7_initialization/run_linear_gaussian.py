"""
Experiment: R(D)-Optimal Initialization for Linear Gaussian VQ-VAE

This script compares three codebook initialization methods:
1. Uniform: Standard [-1/n, 1/n] initialization
2. K-means: Clustering on initialization batch
3. R(D) Gaussian: Theory-based optimal initialization

Key Questions:
- Does R(D) init start closer to optimal distortion?
- Does R(D) init avoid dead codes?
- How much faster does R(D) init converge?
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import yaml
from typing import Dict

from src.data.linear_gaussian import LinearGaussianDataset
from src.models.vqvae import LinearGaussianVQVAE
from src.training.trainer import Trainer
from src.evaluation.metrics import MetricsTracker
from src.evaluation.visualization import plot_training_curves, plot_utilization


def run_experiment(config: Dict, init_method: str, dataset: LinearGaussianDataset) -> MetricsTracker:
    """Run single experiment with specified initialization method.

    Args:
        config: Experiment configuration dictionary
        init_method: Initialization method ('uniform', 'kmeans', 'rd_gaussian')
        dataset: LinearGaussianDataset instance

    Returns:
        MetricsTracker with all recorded metrics
    """
    print(f"\n{'='*70}")
    print(f"  Running Experiment: {init_method.upper()}")
    print(f"{'='*70}\n")

    # Set seed for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Get data
    dataloader = dataset.get_dataloader(batch_size=config['batch_size'])
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

    print(f"Model: {model}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Optimizer (only codebook is trainable)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # Trainer and metrics
    trainer = Trainer(model, optimizer, device='cpu')
    metrics_tracker = MetricsTracker(codebook_size=config['codebook_size'])

    # ========================================================================
    # IMPORTANT: Evaluate BEFORE training to test initialization quality
    # ========================================================================
    print("\nEvaluating initialization quality...")
    init_metrics = trainer.evaluate(dataloader, metrics_tracker)
    metrics_tracker.update(step=0, **init_metrics)

    # Compute theoretical prediction
    R = np.log2(config['codebook_size']) / config['k']
    sigma_z_sq = dataset.sigma_z_squared
    theoretical_distortion = config['k'] * sigma_z_sq * (2 ** (-2 * R))

    print(f"\nInitialization Results:")
    print(f"  Quantization Error:     {init_metrics['quantization_error']:.6f}")
    print(f"  Theoretical Minimum:    {theoretical_distortion:.6f}")
    print(f"  Ratio (actual/theory):  {init_metrics['quantization_error'] / theoretical_distortion:.2f}x")
    print(f"  Dead Codes:             {init_metrics['dead_codes']:.0f} / {config['codebook_size']}")
    print(f"  Perplexity:             {init_metrics['perplexity']:.2f} (max={config['codebook_size']})")

    # ========================================================================
    # Training Loop
    # ========================================================================
    print(f"\nTraining for {config['num_steps']} steps...")
    print(f"{'-'*70}")

    for step in range(1, config['num_steps'] + 1):
        # Sample batch
        batch = next(iter(dataloader))

        # Train step
        loss_dict = trainer.train_step(batch, beta=config['beta'])

        # Evaluate periodically
        if step % config['eval_every'] == 0:
            metrics = trainer.evaluate(dataloader, metrics_tracker)
            metrics_tracker.update(step=step, **metrics, **loss_dict)

            # Print progress
            if step % (config['eval_every'] * 10) == 0:
                print(f"Step {step:4d}/{config['num_steps']:4d} | "
                      f"Quant Error: {metrics['quantization_error']:.6f} | "
                      f"Dead Codes: {metrics['dead_codes']:3.0f} | "
                      f"Perplexity: {metrics['perplexity']:6.2f}")

    # ========================================================================
    # Final Evaluation
    # ========================================================================
    final_metrics = trainer.evaluate(dataloader, metrics_tracker)
    print(f"\n{'-'*70}")
    print(f"Final Results:")
    print(f"  Quantization Error:     {final_metrics['quantization_error']:.6f}")
    print(f"  Theoretical Minimum:    {theoretical_distortion:.6f}")
    print(f"  Gap:                    {(final_metrics['quantization_error'] - theoretical_distortion):.6f}")
    print(f"  Dead Codes:             {final_metrics['dead_codes']:.0f} / {config['codebook_size']}")
    print(f"  Perplexity:             {final_metrics['perplexity']:.2f}")
    print(f"  Usage Entropy:          {final_metrics['usage_entropy']:.3f} nats")

    return metrics_tracker


def main():
    """Run all experiments and generate comparison plots."""

    print("\n" + "="*70)
    print("  VQ-VAE INITIALIZATION EXPERIMENT (Linear Gaussian)")
    print("  Idea 7: R(D)-Optimal Codebook Initialization")
    print("="*70)

    # Load configuration
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"\nConfiguration:")
    print(f"  Data:      d={config['d']}, k={config['k']}, σ={config['sigma_noise']}, n={config['n_samples']}")
    print(f"  Model:     codebook_size={config['codebook_size']}")
    print(f"  Training:  steps={config['num_steps']}, lr={config['lr']}, β={config['beta']}")

    # Create results directory
    results_dir = Path('results/idea_7_linear_gaussian')
    results_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Generate Data Once (shared across all experiments)
    # ========================================================================
    print(f"\nGenerating Linear Gaussian data...")
    dataset = LinearGaussianDataset(
        d=config['d'],
        k=config['k'],
        sigma_noise=config['sigma_noise'],
        n_samples=config['n_samples'],
        seed=config['seed']
    )
    print(f"  {dataset}")
    print(f"  Signal-to-Noise Ratio:  {dataset.snr:.2f}")

    # ========================================================================
    # Run Experiments for Each Initialization Method
    # ========================================================================
    methods = ['uniform', 'kmeans', 'rd_gaussian']
    all_results = {}

    for method in methods:
        tracker = run_experiment(config, method, dataset)
        all_results[method] = tracker

        # Save individual results
        method_dir = results_dir / method
        method_dir.mkdir(exist_ok=True)
        tracker.save(method_dir / 'metrics.csv')
        print(f"\n  Saved metrics to: {method_dir / 'metrics.csv'}")

    # ========================================================================
    # Generate Comparison Plots
    # ========================================================================
    print(f"\n{'='*70}")
    print("  Generating comparison plots...")
    print(f"{'='*70}\n")

    plot_dir = results_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)

    plot_training_curves(all_results, dataset, save_dir=plot_dir)
    plot_utilization(all_results, save_dir=plot_dir)

    # ========================================================================
    # Summary Table
    # ========================================================================
    print(f"\n{'='*70}")
    print("  SUMMARY COMPARISON")
    print(f"{'='*70}\n")

    print(f"{'Method':<15} {'Init Q.Err':<12} {'Final Q.Err':<12} {'Init Dead':<10} {'Final Dead':<10} {'Final Perp':<12}")
    print(f"{'-'*70}")

    for method in methods:
        metrics = all_results[method].metrics
        init_qe = metrics['quantization_error'][0]
        final_qe = metrics['quantization_error'][-1]
        init_dead = metrics['dead_codes'][0]
        final_dead = metrics['dead_codes'][-1]
        final_perp = metrics['perplexity'][-1]

        method_name = {'uniform': 'Uniform', 'kmeans': 'K-Means', 'rd_gaussian': 'R(D) Gaussian'}[method]
        print(f"{method_name:<15} {init_qe:<12.6f} {final_qe:<12.6f} {init_dead:<10.0f} {final_dead:<10.0f} {final_perp:<12.2f}")

    print(f"\nResults saved to: {results_dir}")
    print(f"Plots saved to:   {plot_dir}")
    print(f"\n{'='*70}")
    print("  Experiment Complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
