"""Plotting utilities for VQ-VAE experiment results."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .metrics import MetricsTracker
    from ..data.linear_gaussian import LinearGaussianDataset


def plot_training_curves(
    all_results: Dict[str, 'MetricsTracker'],
    dataset: 'LinearGaussianDataset',
    save_dir: Path
):
    """Plot training curves comparing initialization methods.

    Creates 2x2 figure with:
    1. Quantization error over time (with R(D) bound)
    2. Perplexity over time
    3. Dead codes over time
    4. Reconstruction loss over time

    Args:
        all_results: Dict mapping method name to MetricsTracker
        dataset: LinearGaussianDataset with theoretical quantities
        save_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Curves: Initialization Comparison', fontsize=16, y=0.995)

    # Compute theoretical R(D) bound
    k = dataset.k
    sigma_z_sq = dataset.sigma_z_squared
    codebook_size = list(all_results.values())[0].codebook_size
    R = np.log2(codebook_size) / k
    theoretical_bound = k * sigma_z_sq * (2 ** (-2 * R))

    # Colors and labels for methods
    method_styles = {
        'uniform': {'color': '#d62728', 'label': 'Uniform', 'linewidth': 2},
        'kmeans': {'color': '#1f77b4', 'label': 'K-Means', 'linewidth': 2},
        'rd_gaussian': {'color': '#2ca02c', 'label': 'R(D) Gaussian', 'linewidth': 2.5}
    }

    # Plot each method
    for method, tracker in all_results.items():
        df = pd.DataFrame(tracker.metrics)
        style = method_styles.get(method, {'color': 'gray', 'label': method, 'linewidth': 2})

        # Plot 1: Quantization error
        if 'quantization_error' in df.columns:
            axes[0, 0].plot(df['step'], df['quantization_error'], **style)

        # Plot 2: Perplexity
        if 'perplexity' in df.columns:
            axes[0, 1].plot(df['step'], df['perplexity'], **style)

        # Plot 3: Dead codes
        if 'dead_codes' in df.columns:
            axes[1, 0].plot(df['step'], df['dead_codes'], **style)

        # Plot 4: Reconstruction loss
        if 'recon_loss' in df.columns:
            axes[1, 1].plot(df['step'], df['recon_loss'], **style)

    # Format Plot 1: Quantization Error
    axes[0, 0].axhline(
        theoretical_bound,
        color='black',
        linestyle='--',
        linewidth=1.5,
        label=f'R(D) Bound = {theoretical_bound:.3f}'
    )
    axes[0, 0].set_xlabel('Training Step', fontsize=11)
    axes[0, 0].set_ylabel('Quantization Error (MSE)', fontsize=11)
    axes[0, 0].set_title('Quantization Error: $\mathbb{E}[||z - \hat{z}||^2]$', fontsize=12)
    axes[0, 0].legend(fontsize=10, loc='best')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')

    # Format Plot 2: Perplexity
    axes[0, 1].axhline(
        codebook_size,
        color='black',
        linestyle='--',
        linewidth=1.5,
        label=f'Max = {codebook_size}'
    )
    axes[0, 1].set_xlabel('Training Step', fontsize=11)
    axes[0, 1].set_ylabel('Perplexity', fontsize=11)
    axes[0, 1].set_title('Codebook Utilization (Perplexity)', fontsize=12)
    axes[0, 1].legend(fontsize=10, loc='best')
    axes[0, 1].grid(True, alpha=0.3)

    # Format Plot 3: Dead Codes
    axes[1, 0].set_xlabel('Training Step', fontsize=11)
    axes[1, 0].set_ylabel('Number of Dead Codes', fontsize=11)
    axes[1, 0].set_title('Dead Codes (usage < $10^{-4}$)', fontsize=12)
    axes[1, 0].legend(fontsize=10, loc='best')
    axes[1, 0].grid(True, alpha=0.3)

    # Format Plot 4: Reconstruction Loss
    axes[1, 1].set_xlabel('Training Step', fontsize=11)
    axes[1, 1].set_ylabel('Reconstruction Loss (MSE)', fontsize=11)
    axes[1, 1].set_title('Reconstruction Loss: $||x - \hat{x}||^2$', fontsize=12)
    axes[1, 1].legend(fontsize=10, loc='best')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    save_path = save_dir / 'training_curves.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")


def plot_utilization(
    all_results: Dict[str, 'MetricsTracker'],
    save_dir: Path
):
    """Plot codebook utilization comparison (initial vs final).

    Creates bar plots showing:
    - Initial and final perplexity
    - Initial and final dead codes

    Args:
        all_results: Dict mapping method name to MetricsTracker
        save_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Codebook Utilization: Initialization vs. Final', fontsize=14)

    methods = list(all_results.keys())
    method_labels = {
        'uniform': 'Uniform',
        'kmeans': 'K-Means',
        'rd_gaussian': 'R(D) Gaussian'
    }

    # Extract initial and final metrics
    init_perplexity = []
    final_perplexity = []
    init_dead = []
    final_dead = []

    for method in methods:
        metrics = all_results[method].metrics
        init_perplexity.append(metrics['perplexity'][0] if metrics['perplexity'] else 0)
        final_perplexity.append(metrics['perplexity'][-1] if metrics['perplexity'] else 0)
        init_dead.append(metrics['dead_codes'][0] if metrics['dead_codes'] else 0)
        final_dead.append(metrics['dead_codes'][-1] if metrics['dead_codes'] else 0)

    x = np.arange(len(methods))
    width = 0.35

    # Plot 1: Perplexity
    bars1 = axes[0].bar(
        x - width/2,
        init_perplexity,
        width,
        label='Initial',
        alpha=0.8,
        color='#ff7f0e'
    )
    bars2 = axes[0].bar(
        x + width/2,
        final_perplexity,
        width,
        label='Final',
        alpha=0.8,
        color='#1f77b4'
    )
    axes[0].set_ylabel('Perplexity', fontsize=11)
    axes[0].set_title('Codebook Perplexity', fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([method_labels.get(m, m) for m in methods])
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0].text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.1f}',
                ha='center',
                va='bottom',
                fontsize=9
            )

    # Plot 2: Dead Codes
    bars3 = axes[1].bar(
        x - width/2,
        init_dead,
        width,
        label='Initial',
        alpha=0.8,
        color='#ff7f0e'
    )
    bars4 = axes[1].bar(
        x + width/2,
        final_dead,
        width,
        label='Final',
        alpha=0.8,
        color='#1f77b4'
    )
    axes[1].set_ylabel('Number of Dead Codes', fontsize=11)
    axes[1].set_title('Dead Codes (usage < $10^{-4}$)', fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([method_labels.get(m, m) for m in methods])
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            axes[1].text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=9
            )

    plt.tight_layout()
    save_path = save_dir / 'utilization_comparison.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")
