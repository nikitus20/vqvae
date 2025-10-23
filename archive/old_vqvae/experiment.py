"""Unified experiment runner comparing all three VQ-VAE methods."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from .utils import set_seed, make_lowrank_data, metrics
from .methods import (train_pca_lloyd, train_ste_autograd, train_ste_manual,
                      train_rotation_autograd, STEConfig)


def plot_training_logs(logs: Dict[str, Any], title: str, outdir: Path):
    """Plot training curves from logs."""
    if not logs:
        return

    steps = [entry["step"] for entry in logs]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(title, fontsize=14)

    # Loss
    ax = axes[0, 0]
    if "loss" in logs[0]:
        ax.plot(steps, [entry["loss"] for entry in logs], 'b-', linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Total Loss")
        ax.set_title("Total Loss")
        ax.grid(True, alpha=0.3)

    # Reconstruction loss
    ax = axes[0, 1]
    if "rec" in logs[0]:
        ax.plot(steps, [entry["rec"] for entry in logs], 'g-', linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Reconstruction Loss")
        ax.set_title("Reconstruction Loss")
        ax.grid(True, alpha=0.3)

    # Reconstruction MSE
    ax = axes[0, 2]
    if "recon_mse" in logs[0]:
        ax.plot(steps, [entry["recon_mse"] for entry in logs], 'r-', linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Reconstruction MSE")
        ax.set_title("Reconstruction MSE")
        ax.grid(True, alpha=0.3)

    # In-subspace distortion
    ax = axes[1, 0]
    if "in_sub_dist" in logs[0]:
        ax.plot(steps, [entry["in_sub_dist"] for entry in logs], 'orange', linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("In-Subspace Distortion")
        ax.set_title("In-Subspace Distortion")
        ax.grid(True, alpha=0.3)

    # Usage entropy
    ax = axes[1, 1]
    if "usage_entropy" in logs[0]:
        ax.plot(steps, [entry["usage_entropy"] for entry in logs], 'purple', linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Usage Entropy (nats)")
        ax.set_title("Code Usage Entropy")
        ax.grid(True, alpha=0.3)

    # Dead codes fraction
    ax = axes[1, 2]
    if "dead_frac" in logs[0]:
        ax.plot(steps, [entry["dead_frac"] for entry in logs], 'brown', linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Dead Code Fraction")
        ax.set_title("Dead Code Fraction")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outdir / f"{title.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()


def plot_comparison(results: Dict[str, Dict], outdir: Path):
    """Create comparison plots across all three methods."""
    methods = ["baseline", "autograd", "manual"]
    method_labels = ["PCA + Lloyd", "Autograd STE", "Manual STE"]

    # Extract final metrics
    metrics_list = []
    for method in methods:
        if method in results and results[method]["metrics"] is not None:
            metrics_list.append(results[method]["metrics"])
        else:
            metrics_list.append(None)

    if all(m is None for m in metrics_list):
        return

    # Plot bar chart comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Method Comparison", fontsize=14)

    metric_keys = ["recon_mse", "in_sub_dist", "usage_entropy", "dead_frac"]
    metric_titles = ["Reconstruction MSE", "In-Subspace Distortion",
                     "Usage Entropy (nats)", "Dead Code Fraction"]

    for idx, (key, title) in enumerate(zip(metric_keys, metric_titles)):
        ax = axes[idx // 2, idx % 2]
        values = []
        labels = []

        for method, label, mets in zip(methods, method_labels, metrics_list):
            if mets is not None:
                values.append(mets[key])
                labels.append(label)

        if values:
            bars = ax.bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(values)])
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(True, axis='y', alpha=0.3)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(outdir / "comparison.png", dpi=150)
    plt.close()


def run_experiment(args):
    """Run complete experiment comparing all three methods."""
    set_seed(args.seed)

    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("VQ-VAE Method Comparison Experiment")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  D={args.D}, rank={args.rank}, m={args.m}, K={args.K}")
    print(f"  n_train={args.n_train}, n_eval={args.n_eval}")
    print(f"  seed={args.seed}")
    print("=" * 70)

    # Generate data
    print("\n[1/4] Generating data...")
    rng = np.random.default_rng(args.seed)
    from .utils import make_lowrank_cov, sample_gaussian
    Sigma = make_lowrank_cov(args.D, args.rank, lmax=4.0, lmin=0.25, rng=rng)
    X_train = sample_gaussian(args.n_train, np.zeros(args.D), Sigma, rng=rng)
    X_eval = sample_gaussian(args.n_eval, np.zeros(args.D), Sigma, rng=rng)

    results = {}

    # ========== Method 1: PCA + Lloyd ==========
    print("\n[2/4] Running Method 1: PCA + Lloyd Max...")
    print("-" * 70)
    baseline_result = train_pca_lloyd(
        X_train, X_eval, args.m, args.K,
        lloyd_iters=30, seed=args.seed
    )
    results["baseline"] = baseline_result
    print("Baseline metrics:", baseline_result["metrics"])

    # ========== Method 2: Autograd STE ==========
    print("\n[3/4] Running Method 2: Autograd STE...")
    print("-" * 70)
    autograd_result = train_ste_autograd(
        X_train, X_eval, args.m, args.K,
        steps=args.steps, batch=args.batch,
        alpha=args.alpha, beta=args.beta,
        lrW=args.lrW, lrE=args.lrE,
        seed=args.seed, log_every=20
    )
    results["autograd"] = autograd_result

    if autograd_result["logs"]:
        # Save logs
        with open(outdir / "autograd_logs.json", "w") as f:
            json.dump(autograd_result["logs"], f, indent=2)
        # Plot
        plot_training_logs(autograd_result["logs"], "Autograd STE", outdir)
        print("Autograd final metrics:", autograd_result["metrics"])

    # ========== Method 3: Manual STE ==========
    print("\n[4/4] Running Method 3: Manual STE...")
    print("-" * 70)
    cfg = STEConfig(
        steps=args.steps, batch=args.batch,
        alpha=args.alpha, beta=args.beta,
        lrW=args.lrW, lrE=args.lrE,
        seed=args.seed, log_every=20
    )
    manual_result = train_ste_manual(X_train, X_eval, args.m, args.K, cfg)
    results["manual"] = manual_result

    if manual_result["logs"]:
        # Save logs
        with open(outdir / "manual_logs.json", "w") as f:
            json.dump(manual_result["logs"], f, indent=2)
        # Plot
        plot_training_logs(manual_result["logs"], "Manual STE", outdir)
        print("Manual final metrics:", manual_result["metrics"])

    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary = {
        "config": vars(args),
        "baseline": baseline_result["metrics"],
        "autograd": autograd_result["metrics"] if autograd_result["metrics"] else None,
        "manual": manual_result["metrics"]
    }

    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print("\nFinal Metrics:")
    print(f"{'Method':<20} {'Recon MSE':<15} {'In-Sub Dist':<15} {'Usage Ent':<15} {'Dead Frac':<15}")
    print("-" * 80)

    for method_key, method_name in [("baseline", "PCA + Lloyd"),
                                     ("autograd", "Autograd STE"),
                                     ("manual", "Manual STE")]:
        mets = results[method_key]["metrics"]
        if mets:
            print(f"{method_name:<20} {mets['recon_mse']:<15.6f} {mets['in_sub_dist']:<15.6f} "
                  f"{mets['usage_entropy']:<15.4f} {mets['dead_frac']:<15.4f}")

    # Create comparison plots
    plot_comparison(results, outdir)

    print(f"\n✓ Results saved to: {outdir}")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare VQ-VAE training methods: PCA+Lloyd, Autograd STE, Manual STE"
    )

    # Data parameters
    parser.add_argument("--D", type=int, default=64, help="Ambient dimension")
    parser.add_argument("--rank", type=int, default=6, help="True data rank")
    parser.add_argument("--m", type=int, default=4, help="Latent dimension")
    parser.add_argument("--K", type=int, default=32, help="Number of codebook entries")
    parser.add_argument("--n_train", type=int, default=20000, help="Training samples")
    parser.add_argument("--n_eval", type=int, default=4000, help="Evaluation samples")

    # Training parameters
    parser.add_argument("--steps", type=int, default=200, help="Number of SGD steps")
    parser.add_argument("--batch", type=int, default=1024, help="Batch size")
    parser.add_argument("--alpha", type=float, default=1.0, help="Codebook loss weight")
    parser.add_argument("--beta", type=float, default=0.25, help="Commitment loss weight")
    parser.add_argument("--lrW", type=float, default=0.08, help="Learning rate for encoder")
    parser.add_argument("--lrE", type=float, default=0.2, help="Learning rate for codebook")

    # Misc
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--outdir", type=str, default="results_out", help="Output directory")

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
