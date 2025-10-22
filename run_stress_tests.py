#!/usr/bin/env python3
"""
Stress-test experiments to find where VQ methods diverge.

This script explores extreme hyperparameters and challenging settings to identify
when different gradient estimators (STE vs Rotation) and implementations
(Autograd vs Manual) start producing different results.
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


def run_learning_rate_sweep(outdir: Path, seed: int = 42):
    """Test extreme learning rates to find instability."""
    print("\n" + "="*80)
    print("STRESS TEST 1: Learning Rate Sweep")
    print("="*80)

    # Standard settings
    D, rank, m, K = 64, 6, 4, 32
    n_train, n_eval = 20000, 4000
    steps, batch = 200, 1024

    # Extreme learning rates
    lr_configs = [
        # (lrW, lrE, description)
        (0.001, 0.01, "very_low"),
        (0.01, 0.05, "low"),
        (0.05, 0.2, "standard"),
        (0.2, 0.5, "high"),
        (0.5, 1.0, "very_high"),
        (1.0, 2.0, "extreme"),
        (0.001, 1.0, "mismatched_low_W"),
        (0.5, 0.05, "mismatched_high_W"),
    ]

    results = {}

    for lrW, lrE, desc in lr_configs:
        print(f"\n--- Learning Rate: {desc} (lrW={lrW}, lrE={lrE}) ---")
        set_seed(seed)

        rng = np.random.default_rng(seed)
        Sigma = make_lowrank_cov(D, rank, lmax=4.0, lmin=0.25, rng=rng)
        X_train = sample_gaussian(n_train, np.zeros(D), Sigma, rng=rng)
        X_eval = sample_gaussian(n_eval, np.zeros(D), Sigma, rng=rng)

        config_results = {}

        # Manual STE
        print(f"  [1/3] Manual STE...")
        cfg = STEConfig(steps=steps, batch=batch, alpha=1.0, beta=0.25,
                        lrW=lrW, lrE=lrE, seed=seed, log_every=50)
        try:
            manual = train_ste_manual(X_train, X_eval, m, K, cfg)
            config_results['manual_ste'] = {
                'metrics': manual['metrics'],
                'converged': not np.isnan(manual['metrics']['recon_mse']),
                'final_mse': manual['metrics']['recon_mse'],
                'logs': manual['logs']
            }
            print(f"     MSE={manual['metrics']['recon_mse']:.4f}")
        except Exception as e:
            config_results['manual_ste'] = {'error': str(e), 'converged': False}
            print(f"     ERROR: {e}")

        # Autograd STE
        if TORCH_AVAILABLE:
            print(f"  [2/3] Autograd STE...")
            try:
                autograd = train_ste_autograd(
                    X_train, X_eval, m, K,
                    steps=steps, batch=batch, alpha=1.0, beta=0.25,
                    lrW=lrW, lrE=lrE, seed=seed, log_every=50
                )
                if autograd['metrics']:
                    config_results['autograd_ste'] = {
                        'metrics': autograd['metrics'],
                        'converged': not np.isnan(autograd['metrics']['recon_mse']),
                        'final_mse': autograd['metrics']['recon_mse'],
                        'logs': autograd['logs']
                    }
                    print(f"     MSE={autograd['metrics']['recon_mse']:.4f}")
            except Exception as e:
                config_results['autograd_ste'] = {'error': str(e), 'converged': False}
                print(f"     ERROR: {e}")

        # Rotation
        if TORCH_AVAILABLE:
            print(f"  [3/3] Rotation...")
            try:
                rotation = train_rotation_autograd(
                    X_train, X_eval, m, K,
                    steps=steps, batch=batch, alpha=1.0, beta=0.25,
                    lrW=lrW, lrE=lrE, seed=seed, log_every=50
                )
                if rotation['metrics']:
                    config_results['rotation'] = {
                        'metrics': rotation['metrics'],
                        'converged': not np.isnan(rotation['metrics']['recon_mse']),
                        'final_mse': rotation['metrics']['recon_mse'],
                        'logs': rotation['logs']
                    }
                    print(f"     MSE={rotation['metrics']['recon_mse']:.4f}")
            except Exception as e:
                config_results['rotation'] = {'error': str(e), 'converged': False}
                print(f"     ERROR: {e}")

        results[desc] = config_results

    # Analyze divergence
    print("\n" + "-"*80)
    print("DIVERGENCE ANALYSIS:")
    print("-"*80)

    for desc, res in results.items():
        manual_converged = res.get('manual_ste', {}).get('converged', False)
        autograd_converged = res.get('autograd_ste', {}).get('converged', False)
        rotation_converged = res.get('rotation', {}).get('converged', False)

        print(f"\n{desc}:")
        print(f"  Manual STE:   {'✓' if manual_converged else '✗'}")
        print(f"  Autograd STE: {'✓' if autograd_converged else '✗'}")
        print(f"  Rotation:     {'✓' if rotation_converged else '✗'}")

        if manual_converged and autograd_converged:
            diff = abs(res['manual_ste']['final_mse'] - res['autograd_ste']['final_mse'])
            rel_diff = diff / res['manual_ste']['final_mse'] * 100
            print(f"  Manual vs Autograd: {rel_diff:.2f}% difference")

        if autograd_converged and rotation_converged:
            diff = abs(res['autograd_ste']['final_mse'] - res['rotation']['final_mse'])
            rel_diff = diff / res['autograd_ste']['final_mse'] * 100
            print(f"  STE vs Rotation: {rel_diff:.2f}% difference")

    # Save results
    with open(outdir / 'learning_rate_sweep.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)

    # Plot convergence for different LRs
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Learning Rate Sweep: Training Dynamics', fontsize=16)

    for idx, (lrW, lrE, desc) in enumerate(lr_configs):
        if idx >= 8:
            break
        ax = axes[idx // 4, idx % 4]

        res = results[desc]

        if 'manual_ste' in res and 'logs' in res['manual_ste']:
            logs = res['manual_ste']['logs']
            steps_list = [e['step'] for e in logs]
            mse = [e['recon_mse'] for e in logs]
            ax.plot(steps_list, mse, label='Manual STE', linewidth=2, color='green')

        if 'autograd_ste' in res and 'logs' in res['autograd_ste']:
            logs = res['autograd_ste']['logs']
            steps_list = [e['step'] for e in logs]
            mse = [e['recon_mse'] for e in logs]
            ax.plot(steps_list, mse, label='Autograd STE', linewidth=2, color='blue', alpha=0.7)

        if 'rotation' in res and 'logs' in res['rotation']:
            logs = res['rotation']['logs']
            steps_list = [e['step'] for e in logs]
            mse = [e['recon_mse'] for e in logs]
            ax.plot(steps_list, mse, label='Rotation', linewidth=2, color='red', alpha=0.7)

        ax.set_title(f'{desc}\nlrW={lrW}, lrE={lrE}')
        ax.set_xlabel('Step')
        ax.set_ylabel('Recon MSE')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outdir / 'learning_rate_sweep.png', dpi=150)
    plt.close()

    return results


def run_batch_size_experiments(outdir: Path, seed: int = 42):
    """Test small batch sizes (noisy gradients) vs large."""
    print("\n" + "="*80)
    print("STRESS TEST 2: Batch Size Variation (Gradient Noise)")
    print("="*80)

    D, rank, m, K = 64, 6, 4, 32
    n_train, n_eval = 20000, 4000
    steps = 300

    batch_sizes = [32, 64, 128, 256, 512, 1024, 2048]
    results = {}

    for batch in batch_sizes:
        print(f"\n--- Batch Size: {batch} ---")
        set_seed(seed)

        rng = np.random.default_rng(seed)
        Sigma = make_lowrank_cov(D, rank, lmax=4.0, lmin=0.25, rng=rng)
        X_train = sample_gaussian(n_train, np.zeros(D), Sigma, rng=rng)
        X_eval = sample_gaussian(n_eval, np.zeros(D), Sigma, rng=rng)

        batch_results = {}

        # Manual STE
        cfg = STEConfig(steps=steps, batch=batch, alpha=1.0, beta=0.25,
                        lrW=0.08, lrE=0.2, seed=seed, log_every=50)
        manual = train_ste_manual(X_train, X_eval, m, K, cfg)
        batch_results['manual_ste'] = {
            'final_mse': manual['metrics']['recon_mse'],
            'logs': manual['logs']
        }
        print(f"  Manual STE: {manual['metrics']['recon_mse']:.4f}")

        # Autograd STE
        if TORCH_AVAILABLE:
            autograd = train_ste_autograd(
                X_train, X_eval, m, K,
                steps=steps, batch=batch, alpha=1.0, beta=0.25,
                lrW=0.08, lrE=0.2, seed=seed, log_every=50
            )
            if autograd['metrics']:
                batch_results['autograd_ste'] = {
                    'final_mse': autograd['metrics']['recon_mse'],
                    'logs': autograd['logs']
                }
                print(f"  Autograd STE: {autograd['metrics']['recon_mse']:.4f}")

        # Rotation
        if TORCH_AVAILABLE:
            rotation = train_rotation_autograd(
                X_train, X_eval, m, K,
                steps=steps, batch=batch, alpha=1.0, beta=0.25,
                lrW=0.08, lrE=0.2, seed=seed, log_every=50
            )
            if rotation['metrics']:
                batch_results['rotation'] = {
                    'final_mse': rotation['metrics']['recon_mse'],
                    'logs': rotation['logs']
                }
                print(f"  Rotation: {rotation['metrics']['recon_mse']:.4f}")

        results[batch] = batch_results

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Batch Size Sensitivity', fontsize=14)

    # Final MSE vs batch size
    ax = axes[0]
    manual_mse = [results[b]['manual_ste']['final_mse'] for b in batch_sizes]
    ax.plot(batch_sizes, manual_mse, 'o-', label='Manual STE', linewidth=2, markersize=8)

    if TORCH_AVAILABLE:
        autograd_mse = [results[b]['autograd_ste']['final_mse'] for b in batch_sizes]
        rotation_mse = [results[b]['rotation']['final_mse'] for b in batch_sizes]
        ax.plot(batch_sizes, autograd_mse, 's-', label='Autograd STE', linewidth=2, markersize=8)
        ax.plot(batch_sizes, rotation_mse, '^-', label='Rotation', linewidth=2, markersize=8)

    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Final Reconstruction MSE')
    ax.set_title('Performance vs Batch Size')
    ax.set_xscale('log', base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Variance across training
    ax = axes[1]
    for batch in [32, 128, 512, 2048]:
        logs = results[batch]['manual_ste']['logs']
        steps_list = [e['step'] for e in logs]
        mse = [e['recon_mse'] for e in logs]
        ax.plot(steps_list, mse, label=f'Batch={batch}', linewidth=2)

    ax.set_xlabel('Step')
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('Training Dynamics (Manual STE)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outdir / 'batch_size_variation.png', dpi=150)
    plt.close()

    with open(outdir / 'batch_size_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)

    return results


def run_difficult_data_experiments(outdir: Path, seed: int = 42):
    """Test on challenging data: high noise, high dimension, rank mismatch."""
    print("\n" + "="*80)
    print("STRESS TEST 3: Difficult Data Scenarios")
    print("="*80)

    n_train, n_eval = 20000, 4000
    steps, batch = 300, 1024

    scenarios = [
        # (D, rank, m, K, sigma, description)
        (64, 6, 4, 32, 0.0, "no_noise"),
        (64, 6, 4, 32, 0.5, "high_noise"),
        (64, 6, 4, 32, 1.0, "very_high_noise"),
        (128, 6, 4, 32, 0.1, "high_dimension"),
        (64, 10, 4, 32, 0.1, "rank_mismatch_low_m"),
        (64, 2, 4, 32, 0.1, "rank_mismatch_high_m"),
        (64, 6, 4, 8, 0.1, "small_codebook"),
    ]

    results = {}

    for D, rank, m, K, sigma, desc in scenarios:
        print(f"\n--- Scenario: {desc} (D={D}, r={rank}, m={m}, K={K}, σ={sigma}) ---")
        set_seed(seed)

        rng = np.random.default_rng(seed)

        # Add isotropic noise to make data harder
        Sigma = make_lowrank_cov(D, rank, lmax=4.0, lmin=0.25, rng=rng)
        if sigma > 0:
            Sigma += sigma * np.eye(D)

        X_train = sample_gaussian(n_train, np.zeros(D), Sigma, rng=rng)
        X_eval = sample_gaussian(n_eval, np.zeros(D), Sigma, rng=rng)

        scenario_results = {}

        # Manual STE
        cfg = STEConfig(steps=steps, batch=batch, alpha=1.0, beta=0.25,
                        lrW=0.08, lrE=0.2, seed=seed, log_every=50)
        manual = train_ste_manual(X_train, X_eval, m, K, cfg)
        scenario_results['manual_ste'] = manual['metrics']
        print(f"  Manual STE: MSE={manual['metrics']['recon_mse']:.4f}, "
              f"entropy={manual['metrics']['usage_entropy']:.3f}")

        # Autograd STE
        if TORCH_AVAILABLE:
            autograd = train_ste_autograd(
                X_train, X_eval, m, K,
                steps=steps, batch=batch, alpha=1.0, beta=0.25,
                lrW=0.08, lrE=0.2, seed=seed, log_every=50
            )
            if autograd['metrics']:
                scenario_results['autograd_ste'] = autograd['metrics']
                print(f"  Autograd STE: MSE={autograd['metrics']['recon_mse']:.4f}, "
                      f"entropy={autograd['metrics']['usage_entropy']:.3f}")

        # Rotation
        if TORCH_AVAILABLE:
            rotation = train_rotation_autograd(
                X_train, X_eval, m, K,
                steps=steps, batch=batch, alpha=1.0, beta=0.25,
                lrW=0.08, lrE=0.2, seed=seed, log_every=50
            )
            if rotation['metrics']:
                scenario_results['rotation'] = rotation['metrics']
                print(f"  Rotation: MSE={rotation['metrics']['recon_mse']:.4f}, "
                      f"entropy={rotation['metrics']['usage_entropy']:.3f}")

        results[desc] = scenario_results

    # Analyze divergence
    print("\n" + "-"*80)
    print("METHOD DIVERGENCE IN DIFFICULT SCENARIOS:")
    print("-"*80)

    for desc, res in results.items():
        if 'manual_ste' in res and 'autograd_ste' in res:
            diff = abs(res['manual_ste']['recon_mse'] - res['autograd_ste']['recon_mse'])
            rel_diff = diff / res['manual_ste']['recon_mse'] * 100
            print(f"\n{desc}:")
            print(f"  Manual vs Autograd: {rel_diff:.2f}% MSE difference")

        if 'autograd_ste' in res and 'rotation' in res:
            diff = abs(res['autograd_ste']['recon_mse'] - res['rotation']['recon_mse'])
            rel_diff = diff / res['autograd_ste']['recon_mse'] * 100
            print(f"  STE vs Rotation: {rel_diff:.2f}% MSE difference")

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance on Difficult Data', fontsize=14)

    scenario_names = [s[-1] for s in scenarios]

    # MSE
    ax = axes[0, 0]
    manual_mse = [results[name]['manual_ste']['recon_mse'] for name in scenario_names]
    ax.bar(range(len(scenario_names)), manual_mse, alpha=0.8, label='Manual STE')
    if TORCH_AVAILABLE:
        autograd_mse = [results[name]['autograd_ste']['recon_mse'] for name in scenario_names]
        rotation_mse = [results[name]['rotation']['recon_mse'] for name in scenario_names]
        ax.bar(range(len(scenario_names)), autograd_mse, alpha=0.6, label='Autograd STE')
        ax.bar(range(len(scenario_names)), rotation_mse, alpha=0.4, label='Rotation')
    ax.set_xticks(range(len(scenario_names)))
    ax.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('Final MSE')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Entropy
    ax = axes[0, 1]
    manual_ent = [results[name]['manual_ste']['usage_entropy'] for name in scenario_names]
    ax.bar(range(len(scenario_names)), manual_ent, alpha=0.8, label='Manual STE')
    if TORCH_AVAILABLE:
        autograd_ent = [results[name]['autograd_ste']['usage_entropy'] for name in scenario_names]
        rotation_ent = [results[name]['rotation']['usage_entropy'] for name in scenario_names]
        ax.bar(range(len(scenario_names)), autograd_ent, alpha=0.6, label='Autograd STE')
        ax.bar(range(len(scenario_names)), rotation_ent, alpha=0.4, label='Rotation')
    ax.set_xticks(range(len(scenario_names)))
    ax.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax.set_ylabel('Usage Entropy')
    ax.set_title('Code Utilization')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Dead codes
    ax = axes[1, 0]
    manual_dead = [results[name]['manual_ste']['dead_frac'] for name in scenario_names]
    ax.bar(range(len(scenario_names)), manual_dead, alpha=0.8, label='Manual STE')
    if TORCH_AVAILABLE:
        autograd_dead = [results[name]['autograd_ste']['dead_frac'] for name in scenario_names]
        rotation_dead = [results[name]['rotation']['dead_frac'] for name in scenario_names]
        ax.bar(range(len(scenario_names)), autograd_dead, alpha=0.6, label='Autograd STE')
        ax.bar(range(len(scenario_names)), rotation_dead, alpha=0.4, label='Rotation')
    ax.set_xticks(range(len(scenario_names)))
    ax.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax.set_ylabel('Dead Code Fraction')
    ax.set_title('Dead Codes')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Method divergence
    ax = axes[1, 1]
    if TORCH_AVAILABLE:
        manual_auto_diff = [abs(results[name]['manual_ste']['recon_mse'] -
                               results[name]['autograd_ste']['recon_mse']) /
                           results[name]['manual_ste']['recon_mse'] * 100
                           for name in scenario_names]
        ste_rot_diff = [abs(results[name]['autograd_ste']['recon_mse'] -
                           results[name]['rotation']['recon_mse']) /
                       results[name]['autograd_ste']['recon_mse'] * 100
                       for name in scenario_names]

        x = range(len(scenario_names))
        width = 0.4
        ax.bar([i - width/2 for i in x], manual_auto_diff, width,
               label='Manual vs Autograd', alpha=0.8)
        ax.bar([i + width/2 for i in x], ste_rot_diff, width,
               label='STE vs Rotation', alpha=0.8)
        ax.set_xticks(range(len(scenario_names)))
        ax.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax.set_ylabel('Relative Difference (%)')
        ax.set_title('Method Divergence')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(outdir / 'difficult_data_scenarios.png', dpi=150)
    plt.close()

    with open(outdir / 'difficult_data_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)

    return results


def main():
    parser = argparse.ArgumentParser(description='Stress test VQ methods to find divergence')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--outdir', type=str, default='stress_test_results',
                       help='Output directory')
    parser.add_argument('--tests', nargs='+',
                       choices=['learning_rate', 'batch_size', 'difficult_data', 'all'],
                       default=['all'], help='Which stress tests to run')
    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / timestamp
    outdir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("VQ-VAE STRESS TESTS: Finding Where Methods Diverge")
    print("="*80)
    print(f"Output directory: {outdir}")
    print(f"Random seed: {args.seed}")
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    print("="*80)

    tests_to_run = args.tests
    if 'all' in tests_to_run:
        tests_to_run = ['learning_rate', 'batch_size', 'difficult_data']

    all_results = {}

    if 'learning_rate' in tests_to_run:
        all_results['learning_rate'] = run_learning_rate_sweep(outdir, args.seed)

    if 'batch_size' in tests_to_run:
        all_results['batch_size'] = run_batch_size_experiments(outdir, args.seed)

    if 'difficult_data' in tests_to_run:
        all_results['difficult_data'] = run_difficult_data_experiments(outdir, args.seed)

    # Save master summary
    with open(outdir / 'stress_test_summary.json', 'w') as f:
        json.dump({
            'metadata': {'seed': args.seed, 'timestamp': timestamp},
            'results': all_results
        }, f, indent=2, default=float)

    print("\n" + "="*80)
    print(f"✓ All stress tests completed! Results saved to: {outdir}")
    print("="*80)


if __name__ == "__main__":
    main()
