#!/usr/bin/env python3
"""
Comprehensive VQ-VAE experiments comparing all methods.

This script runs systematic comparisons of:
1. PCA + Lloyd Max (baseline)
2. Autograd STE
3. Manual STE
4. Rotation Estimator

Experiments include:
- Convergence comparison (longer training)
- Codebook size variation
- Learning rate sensitivity
- Commitment loss ablation
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from vqvae.utils import set_seed, make_lowrank_cov, sample_gaussian, metrics
from vqvae.methods import (train_pca_lloyd, train_ste_autograd, train_ste_manual,
                            train_rotation_autograd, STEConfig)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def run_convergence_experiment(outdir: Path, seed: int = 42):
    """Compare convergence of all gradient-based methods with extended training."""
    print("\n" + "="*80)
    print("EXPERIMENT 1: Convergence Comparison (500 steps)")
    print("="*80)

    set_seed(seed)

    # Generate data
    D, rank, m, K = 64, 6, 4, 32
    n_train, n_eval = 20000, 4000

    rng = np.random.default_rng(seed)
    Sigma = make_lowrank_cov(D, rank, lmax=4.0, lmin=0.25, rng=rng)
    X_train = sample_gaussian(n_train, np.zeros(D), Sigma, rng=rng)
    X_eval = sample_gaussian(n_eval, np.zeros(D), Sigma, rng=rng)

    results = {}

    # Baseline
    print("\n[1/4] PCA + Lloyd...")
    baseline = train_pca_lloyd(X_train, X_eval, m, K, seed=seed)
    results['baseline'] = baseline
    print(f"Baseline: recon_mse={baseline['metrics']['recon_mse']:.4f}, "
          f"entropy={baseline['metrics']['usage_entropy']:.3f}")

    # Autograd STE
    if TORCH_AVAILABLE:
        print("\n[2/4] Autograd STE (500 steps)...")
        autograd = train_ste_autograd(
            X_train, X_eval, m, K,
            steps=500, batch=1024, alpha=1.0, beta=0.25,
            lrW=0.05, lrE=0.2, seed=seed, log_every=25
        )
        results['autograd_ste'] = autograd
        if autograd['metrics']:
            print(f"Autograd STE: recon_mse={autograd['metrics']['recon_mse']:.4f}, "
                  f"entropy={autograd['metrics']['usage_entropy']:.3f}")

    # Manual STE
    print("\n[3/4] Manual STE (500 steps)...")
    cfg = STEConfig(steps=500, batch=1024, alpha=1.0, beta=0.25,
                    lrW=0.08, lrE=0.2, seed=seed, log_every=25)
    manual = train_ste_manual(X_train, X_eval, m, K, cfg)
    results['manual_ste'] = manual
    print(f"Manual STE: recon_mse={manual['metrics']['recon_mse']:.4f}, "
          f"entropy={manual['metrics']['usage_entropy']:.3f}")

    # Rotation
    if TORCH_AVAILABLE:
        print("\n[4/4] Rotation Estimator (500 steps)...")
        rotation = train_rotation_autograd(
            X_train, X_eval, m, K,
            steps=500, batch=1024, alpha=1.0, beta=0.25,
            lrW=0.05, lrE=0.2, seed=seed, log_every=25
        )
        results['rotation'] = rotation
        if rotation['metrics']:
            print(f"Rotation: recon_mse={rotation['metrics']['recon_mse']:.4f}, "
                  f"entropy={rotation['metrics']['usage_entropy']:.3f}")

    # Plot convergence curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Convergence Comparison (500 steps)', fontsize=14)

    methods = []
    if TORCH_AVAILABLE and 'autograd_ste' in results and results['autograd_ste']['logs']:
        methods.append(('autograd_ste', 'Autograd STE', 'blue'))
    if 'manual_ste' in results and results['manual_ste']['logs']:
        methods.append(('manual_ste', 'Manual STE', 'green'))
    if TORCH_AVAILABLE and 'rotation' in results and results['rotation']['logs']:
        methods.append(('rotation', 'Rotation', 'red'))

    # Plot reconstruction MSE
    ax = axes[0, 0]
    for method_key, label, color in methods:
        logs = results[method_key]['logs']
        steps = [e['step'] for e in logs]
        mse = [e['recon_mse'] for e in logs]
        ax.plot(steps, mse, label=label, color=color, linewidth=2)
    if 'baseline' in results:
        ax.axhline(results['baseline']['metrics']['recon_mse'],
                   linestyle='--', color='black', label='PCA+Lloyd', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('Reconstruction Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot in-subspace distortion
    ax = axes[0, 1]
    for method_key, label, color in methods:
        logs = results[method_key]['logs']
        steps = [e['step'] for e in logs]
        dist = [e['in_sub_dist'] for e in logs]
        ax.plot(steps, dist, label=label, color=color, linewidth=2)
    if 'baseline' in results:
        ax.axhline(results['baseline']['metrics']['in_sub_dist'],
                   linestyle='--', color='black', label='PCA+Lloyd', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('In-Subspace Distortion')
    ax.set_title('Latent Distortion')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot usage entropy
    ax = axes[1, 0]
    for method_key, label, color in methods:
        logs = results[method_key]['logs']
        steps = [e['step'] for e in logs]
        ent = [e['usage_entropy'] for e in logs]
        ax.plot(steps, ent, label=label, color=color, linewidth=2)
    if 'baseline' in results:
        ax.axhline(results['baseline']['metrics']['usage_entropy'],
                   linestyle='--', color='black', label='PCA+Lloyd', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Usage Entropy (nats)')
    ax.set_title('Codebook Utilization')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot dead code fraction
    ax = axes[1, 1]
    for method_key, label, color in methods:
        logs = results[method_key]['logs']
        steps = [e['step'] for e in logs]
        dead = [e['dead_frac'] for e in logs]
        ax.plot(steps, dead, label=label, color=color, linewidth=2)
    if 'baseline' in results:
        ax.axhline(results['baseline']['metrics']['dead_frac'],
                   linestyle='--', color='black', label='PCA+Lloyd', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Dead Code Fraction')
    ax.set_title('Dead Codes')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outdir / 'convergence_comparison.png', dpi=150)
    plt.close()

    # Save results
    with open(outdir / 'convergence_results.json', 'w') as f:
        summary = {k: v['metrics'] for k, v in results.items() if v['metrics']}
        json.dump(summary, f, indent=2, default=float)

    return results


def run_codebook_size_experiment(outdir: Path, seed: int = 42):
    """Test different codebook sizes K."""
    print("\n" + "="*80)
    print("EXPERIMENT 2: Codebook Size Variation (K = 8, 16, 32, 64)")
    print("="*80)

    K_values = [8, 16, 32, 64]
    D, rank, m = 64, 6, 4
    n_train, n_eval = 20000, 4000

    results = {k: {} for k in K_values}

    for K in K_values:
        print(f"\n--- K = {K} ---")
        set_seed(seed)

        rng = np.random.default_rng(seed)
        Sigma = make_lowrank_cov(D, rank, lmax=4.0, lmin=0.25, rng=rng)
        X_train = sample_gaussian(n_train, np.zeros(D), Sigma, rng=rng)
        X_eval = sample_gaussian(n_eval, np.zeros(D), Sigma, rng=rng)

        # Baseline
        baseline = train_pca_lloyd(X_train, X_eval, m, K, seed=seed)
        results[K]['baseline'] = baseline['metrics']

        # Manual STE
        cfg = STEConfig(steps=300, batch=1024, alpha=1.0, beta=0.25,
                        lrW=0.08, lrE=0.2, seed=seed, log_every=50)
        manual = train_ste_manual(X_train, X_eval, m, K, cfg)
        results[K]['manual_ste'] = manual['metrics']

        # Rotation
        if TORCH_AVAILABLE:
            rotation = train_rotation_autograd(
                X_train, X_eval, m, K,
                steps=300, batch=1024, alpha=1.0, beta=0.25,
                lrW=0.05, lrE=0.2, seed=seed, log_every=50
            )
            if rotation['metrics']:
                results[K]['rotation'] = rotation['metrics']

        print(f"K={K}: Baseline MSE={results[K]['baseline']['recon_mse']:.4f}, "
              f"Manual MSE={results[K]['manual_ste']['recon_mse']:.4f}")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Codebook Size Scaling', fontsize=14)

    K_vals = list(K_values)

    # Reconstruction MSE
    ax = axes[0, 0]
    baseline_mse = [results[k]['baseline']['recon_mse'] for k in K_vals]
    manual_mse = [results[k]['manual_ste']['recon_mse'] for k in K_vals]
    ax.plot(K_vals, baseline_mse, 'o-', label='PCA+Lloyd', linewidth=2, markersize=8)
    ax.plot(K_vals, manual_mse, 's-', label='Manual STE', linewidth=2, markersize=8)
    if TORCH_AVAILABLE and 'rotation' in results[K_vals[0]]:
        rotation_mse = [results[k]['rotation']['recon_mse'] for k in K_vals]
        ax.plot(K_vals, rotation_mse, '^-', label='Rotation', linewidth=2, markersize=8)
    ax.set_xlabel('Codebook Size (K)')
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('Reconstruction Error vs K')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    # Usage entropy
    ax = axes[0, 1]
    baseline_ent = [results[k]['baseline']['usage_entropy'] for k in K_vals]
    manual_ent = [results[k]['manual_ste']['usage_entropy'] for k in K_vals]
    ax.plot(K_vals, baseline_ent, 'o-', label='PCA+Lloyd', linewidth=2, markersize=8)
    ax.plot(K_vals, manual_ent, 's-', label='Manual STE', linewidth=2, markersize=8)
    if TORCH_AVAILABLE and 'rotation' in results[K_vals[0]]:
        rotation_ent = [results[k]['rotation']['usage_entropy'] for k in K_vals]
        ax.plot(K_vals, rotation_ent, '^-', label='Rotation', linewidth=2, markersize=8)
    # Theoretical max entropy
    max_ent = [np.log(k) for k in K_vals]
    ax.plot(K_vals, max_ent, '--', color='gray', label='Max (log K)', linewidth=2)
    ax.set_xlabel('Codebook Size (K)')
    ax.set_ylabel('Usage Entropy (nats)')
    ax.set_title('Code Utilization vs K')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    # Dead codes
    ax = axes[1, 0]
    baseline_dead = [results[k]['baseline']['dead_frac'] for k in K_vals]
    manual_dead = [results[k]['manual_ste']['dead_frac'] for k in K_vals]
    ax.plot(K_vals, baseline_dead, 'o-', label='PCA+Lloyd', linewidth=2, markersize=8)
    ax.plot(K_vals, manual_dead, 's-', label='Manual STE', linewidth=2, markersize=8)
    if TORCH_AVAILABLE and 'rotation' in results[K_vals[0]]:
        rotation_dead = [results[k]['rotation']['dead_frac'] for k in K_vals]
        ax.plot(K_vals, rotation_dead, '^-', label='Rotation', linewidth=2, markersize=8)
    ax.set_xlabel('Codebook Size (K)')
    ax.set_ylabel('Dead Code Fraction')
    ax.set_title('Dead Codes vs K')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    # In-subspace distortion
    ax = axes[1, 1]
    baseline_dist = [results[k]['baseline']['in_sub_dist'] for k in K_vals]
    manual_dist = [results[k]['manual_ste']['in_sub_dist'] for k in K_vals]
    ax.plot(K_vals, baseline_dist, 'o-', label='PCA+Lloyd', linewidth=2, markersize=8)
    ax.plot(K_vals, manual_dist, 's-', label='Manual STE', linewidth=2, markersize=8)
    if TORCH_AVAILABLE and 'rotation' in results[K_vals[0]]:
        rotation_dist = [results[k]['rotation']['in_sub_dist'] for k in K_vals]
        ax.plot(K_vals, rotation_dist, '^-', label='Rotation', linewidth=2, markersize=8)
    ax.set_xlabel('Codebook Size (K)')
    ax.set_ylabel('In-Subspace Distortion')
    ax.set_title('Latent Distortion vs K')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    plt.tight_layout()
    plt.savefig(outdir / 'codebook_size_scaling.png', dpi=150)
    plt.close()

    # Save results
    with open(outdir / 'codebook_size_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)

    return results


def run_commitment_ablation(outdir: Path, seed: int = 42):
    """Test different commitment loss weights β."""
    print("\n" + "="*80)
    print("EXPERIMENT 3: Commitment Loss Ablation (β = 0.0, 0.1, 0.25, 0.5, 1.0)")
    print("="*80)

    beta_values = [0.0, 0.1, 0.25, 0.5, 1.0]
    D, rank, m, K = 64, 6, 4, 32
    n_train, n_eval = 20000, 4000

    results = {b: {} for b in beta_values}

    for beta in beta_values:
        print(f"\n--- β = {beta} ---")
        set_seed(seed)

        rng = np.random.default_rng(seed)
        Sigma = make_lowrank_cov(D, rank, lmax=4.0, lmin=0.25, rng=rng)
        X_train = sample_gaussian(n_train, np.zeros(D), Sigma, rng=rng)
        X_eval = sample_gaussian(n_eval, np.zeros(D), Sigma, rng=rng)

        # Manual STE
        cfg = STEConfig(steps=300, batch=1024, alpha=1.0, beta=beta,
                        lrW=0.08, lrE=0.2, seed=seed, log_every=50)
        manual = train_ste_manual(X_train, X_eval, m, K, cfg)
        results[beta]['manual_ste'] = manual['metrics']

        # Rotation
        if TORCH_AVAILABLE:
            rotation = train_rotation_autograd(
                X_train, X_eval, m, K,
                steps=300, batch=1024, alpha=1.0, beta=beta,
                lrW=0.05, lrE=0.2, seed=seed, log_every=50
            )
            if rotation['metrics']:
                results[beta]['rotation'] = rotation['metrics']

        print(f"β={beta}: Manual MSE={results[beta]['manual_ste']['recon_mse']:.4f}, "
              f"entropy={results[beta]['manual_ste']['usage_entropy']:.3f}")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Commitment Loss Ablation', fontsize=14)

    beta_vals = list(beta_values)

    # Reconstruction MSE
    ax = axes[0, 0]
    manual_mse = [results[b]['manual_ste']['recon_mse'] for b in beta_vals]
    ax.plot(beta_vals, manual_mse, 's-', label='Manual STE', linewidth=2, markersize=8)
    if TORCH_AVAILABLE and 'rotation' in results[beta_vals[0]]:
        rotation_mse = [results[b]['rotation']['recon_mse'] for b in beta_vals]
        ax.plot(beta_vals, rotation_mse, '^-', label='Rotation', linewidth=2, markersize=8)
    ax.set_xlabel('Commitment Weight (β)')
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('Reconstruction Error vs β')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Usage entropy
    ax = axes[0, 1]
    manual_ent = [results[b]['manual_ste']['usage_entropy'] for b in beta_vals]
    ax.plot(beta_vals, manual_ent, 's-', label='Manual STE', linewidth=2, markersize=8)
    if TORCH_AVAILABLE and 'rotation' in results[beta_vals[0]]:
        rotation_ent = [results[b]['rotation']['usage_entropy'] for b in beta_vals]
        ax.plot(beta_vals, rotation_ent, '^-', label='Rotation', linewidth=2, markersize=8)
    ax.set_xlabel('Commitment Weight (β)')
    ax.set_ylabel('Usage Entropy (nats)')
    ax.set_title('Code Utilization vs β')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Dead codes
    ax = axes[1, 0]
    manual_dead = [results[b]['manual_ste']['dead_frac'] for b in beta_vals]
    ax.plot(beta_vals, manual_dead, 's-', label='Manual STE', linewidth=2, markersize=8)
    if TORCH_AVAILABLE and 'rotation' in results[beta_vals[0]]:
        rotation_dead = [results[b]['rotation']['dead_frac'] for b in beta_vals]
        ax.plot(beta_vals, rotation_dead, '^-', label='Rotation', linewidth=2, markersize=8)
    ax.set_xlabel('Commitment Weight (β)')
    ax.set_ylabel('Dead Code Fraction')
    ax.set_title('Dead Codes vs β')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # In-subspace distortion
    ax = axes[1, 1]
    manual_dist = [results[b]['manual_ste']['in_sub_dist'] for b in beta_vals]
    ax.plot(beta_vals, manual_dist, 's-', label='Manual STE', linewidth=2, markersize=8)
    if TORCH_AVAILABLE and 'rotation' in results[beta_vals[0]]:
        rotation_dist = [results[b]['rotation']['in_sub_dist'] for b in beta_vals]
        ax.plot(beta_vals, rotation_dist, '^-', label='Rotation', linewidth=2, markersize=8)
    ax.set_xlabel('Commitment Weight (β)')
    ax.set_ylabel('In-Subspace Distortion')
    ax.set_title('Latent Distortion vs β')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outdir / 'commitment_ablation.png', dpi=150)
    plt.close()

    # Save results
    with open(outdir / 'commitment_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)

    return results


def verify_ste_implementations(outdir: Path, seed: int = 42):
    """Verify that autograd and manual STE give identical results."""
    if not TORCH_AVAILABLE:
        print("\nPyTorch not available - skipping STE verification")
        return None

    print("\n" + "="*80)
    print("VERIFICATION: Autograd vs Manual STE (same seed, same hyperparams)")
    print("="*80)

    set_seed(seed)

    D, rank, m, K = 64, 6, 4, 16
    n_train, n_eval = 10000, 2000

    rng = np.random.default_rng(seed)
    Sigma = make_lowrank_cov(D, rank, lmax=4.0, lmin=0.25, rng=rng)
    X_train = sample_gaussian(n_train, np.zeros(D), Sigma, rng=rng)
    X_eval = sample_gaussian(n_eval, np.zeros(D), Sigma, rng=rng)

    # Same hyperparameters
    steps, batch = 200, 512
    alpha, beta = 1.0, 0.25
    lrW, lrE = 0.08, 0.2

    print("\n[1/2] Autograd STE...")
    autograd = train_ste_autograd(
        X_train, X_eval, m, K,
        steps=steps, batch=batch, alpha=alpha, beta=beta,
        lrW=lrW, lrE=lrE, seed=seed, log_every=20
    )

    print("\n[2/2] Manual STE...")
    cfg = STEConfig(steps=steps, batch=batch, alpha=alpha, beta=beta,
                    lrW=lrW, lrE=lrE, seed=seed, log_every=20)
    manual = train_ste_manual(X_train, X_eval, m, K, cfg)

    # Compare final metrics
    print("\n" + "-"*80)
    print("COMPARISON:")
    print("-"*80)
    print(f"{'Metric':<25} {'Autograd STE':<20} {'Manual STE':<20} {'Diff':<15}")
    print("-"*80)

    for key in ['recon_mse', 'in_sub_dist', 'usage_entropy', 'dead_frac']:
        auto_val = autograd['metrics'][key]
        manual_val = manual['metrics'][key]
        diff = abs(auto_val - manual_val)
        print(f"{key:<25} {auto_val:<20.6f} {manual_val:<20.6f} {diff:<15.6f}")

    # Plot convergence comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('STE Implementation Verification', fontsize=14)

    auto_logs = autograd['logs']
    manual_logs = manual['logs']

    auto_steps = [e['step'] for e in auto_logs]
    manual_steps = [e['step'] for e in manual_logs]

    # Reconstruction MSE
    ax = axes[0, 0]
    ax.plot(auto_steps, [e['recon_mse'] for e in auto_logs],
            'o-', label='Autograd', markersize=4, linewidth=2)
    ax.plot(manual_steps, [e['recon_mse'] for e in manual_logs],
            's-', label='Manual', markersize=4, linewidth=2, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('Reconstruction Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss
    ax = axes[0, 1]
    ax.plot(auto_steps, [e['loss'] for e in auto_logs],
            'o-', label='Autograd', markersize=4, linewidth=2)
    ax.plot(manual_steps, [e['loss'] for e in manual_logs],
            's-', label='Manual', markersize=4, linewidth=2, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Usage entropy
    ax = axes[1, 0]
    ax.plot(auto_steps, [e['usage_entropy'] for e in auto_logs],
            'o-', label='Autograd', markersize=4, linewidth=2)
    ax.plot(manual_steps, [e['usage_entropy'] for e in manual_logs],
            's-', label='Manual', markersize=4, linewidth=2, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Usage Entropy (nats)')
    ax.set_title('Code Utilization')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # In-subspace distortion
    ax = axes[1, 1]
    ax.plot(auto_steps, [e['in_sub_dist'] for e in auto_logs],
            'o-', label='Autograd', markersize=4, linewidth=2)
    ax.plot(manual_steps, [e['in_sub_dist'] for e in manual_logs],
            's-', label='Manual', markersize=4, linewidth=2, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('In-Subspace Distortion')
    ax.set_title('Latent Distortion')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outdir / 'ste_verification.png', dpi=150)
    plt.close()

    # Save comparison
    comparison = {
        'autograd': autograd['metrics'],
        'manual': manual['metrics'],
        'differences': {k: abs(autograd['metrics'][k] - manual['metrics'][k])
                       for k in autograd['metrics'].keys()}
    }
    with open(outdir / 'ste_verification.json', 'w') as f:
        json.dump(comparison, f, indent=2, default=float)

    return comparison


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive VQ-VAE experiments')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--outdir', type=str, default='comprehensive_results',
                       help='Output directory')
    parser.add_argument('--experiments', nargs='+',
                       choices=['convergence', 'codebook', 'commitment', 'verification', 'all'],
                       default=['all'], help='Which experiments to run')
    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / timestamp
    outdir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("COMPREHENSIVE VQ-VAE EXPERIMENTS")
    print("="*80)
    print(f"Output directory: {outdir}")
    print(f"Random seed: {args.seed}")
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    print("="*80)

    experiments_to_run = args.experiments
    if 'all' in experiments_to_run:
        experiments_to_run = ['verification', 'convergence', 'codebook', 'commitment']

    all_results = {}

    if 'verification' in experiments_to_run:
        all_results['verification'] = verify_ste_implementations(outdir, args.seed)

    if 'convergence' in experiments_to_run:
        all_results['convergence'] = run_convergence_experiment(outdir, args.seed)

    if 'codebook' in experiments_to_run:
        all_results['codebook'] = run_codebook_size_experiment(outdir, args.seed)

    if 'commitment' in experiments_to_run:
        all_results['commitment'] = run_commitment_ablation(outdir, args.seed)

    # Save master summary
    with open(outdir / 'all_experiments.json', 'w') as f:
        json.dump({'metadata': {'seed': args.seed, 'timestamp': timestamp},
                   'results': all_results}, f, indent=2, default=float)

    print("\n" + "="*80)
    print(f"✓ All experiments completed! Results saved to: {outdir}")
    print("="*80)


if __name__ == "__main__":
    main()
