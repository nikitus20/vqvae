"""
Comprehensive VQ-VAE experiments comparing STE vs Rotation gradient estimators
with k-means baseline and theoretical analysis.
"""

import argparse
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Import our main VQ model
from vq_ste_rotation_lab import (
    LinearVQ, make_subspace_data, principal_angles,
    entropy_dead, set_seed, RotationEstimator
)

# ---------- Enhanced Metrics ----------
def compute_gradient_alignment(model, data_loader, U_true, num_samples=100):
    """Compute gradient alignment near Voronoi boundaries"""
    model.eval()
    alignments = []

    for i, (x,) in enumerate(data_loader):
        if i * x.size(0) >= num_samples:
            break

        z = model.encode(x)
        z.requires_grad_(True)

        # Get assignment
        q, idx = model.assign(z)

        # Apply estimator
        if model.estimator == "ste":
            zhat = z + (q - z).detach()
        elif model.estimator == "rotation":
            zhat = RotationEstimator.apply(z, q)
        else:
            zhat = q.detach()

        # Compute loss and gradient
        x_hat = model.decode(zhat)
        loss = F.mse_loss(x_hat, x)
        loss.backward()

        if z.grad is not None:
            # Get distances to nearest codes
            with torch.no_grad():
                code = model.codebook()
                dists = torch.cdist(z, code, p=2)
                sorted_dists, _ = torch.sort(dists, dim=1)

                # Find points near boundaries (similar distance to top 2 codes)
                boundary_mask = (sorted_dists[:, 1] - sorted_dists[:, 0]) < 0.1

                if boundary_mask.any():
                    # Compute finite difference gradient
                    eps = 1e-4
                    z_plus = z[boundary_mask].detach() + eps
                    z_minus = z[boundary_mask].detach() - eps

                    # This is simplified - in practice would compute full finite diff
                    # For now, just measure gradient magnitude consistency
                    grad_norm = z.grad[boundary_mask].norm(dim=1)
                    alignments.extend(grad_norm.cpu().numpy().tolist())

    return np.array(alignments) if alignments else np.array([0.0])

def track_codebook_evolution(model, data_loader, epochs=50, lr=0.002):
    """Track how codebooks evolve during training"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    codebook_history = []
    loss_history = []
    usage_history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        all_idx = []

        for (x,) in data_loader:
            loss, rec, z, q, idx = model(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            all_idx.extend(idx.cpu().numpy())

        # Track metrics
        codebook_history.append(model.codebook.code.data.clone().cpu().numpy())
        loss_history.append(epoch_loss / len(data_loader))

        # Track usage
        K = model.codebook.code.size(0)
        counts = np.bincount(all_idx, minlength=K)
        usage_history.append(counts)

    return {
        'codebooks': codebook_history,
        'losses': loss_history,
        'usage': usage_history
    }

def run_kmeans_baseline(data, K=16):
    """Run k-means as baseline"""
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)

    # Compute metrics
    counts = np.bincount(labels, minlength=K)
    p = counts / counts.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    dead = np.sum(counts == 0)
    silhouette = silhouette_score(data, labels) if K > 1 else 0

    return {
        'centers': kmeans.cluster_centers_,
        'labels': labels,
        'entropy': entropy,
        'dead_codes': dead,
        'silhouette': silhouette,
        'inertia': kmeans.inertia_
    }

def compare_with_lloyds(model, data_tensor, K=16):
    """Compare VQ solution with Lloyd's algorithm (k-means)"""
    with torch.no_grad():
        z = model.encode(data_tensor).cpu().numpy()

    # Run k-means
    kmeans_result = run_kmeans_baseline(z, K)

    # Get VQ codebook
    vq_codes = model.codebook.code.data.cpu().numpy()

    # Compute distance between codebooks (Hungarian matching would be better)
    # For now, just compare distributions
    vq_dists = []
    km_dists = []

    for i in range(len(z)):
        vq_dist = np.min(np.linalg.norm(z[i] - vq_codes, axis=1))
        km_dist = np.min(np.linalg.norm(z[i] - kmeans_result['centers'], axis=1))
        vq_dists.append(vq_dist)
        km_dists.append(km_dist)

    return {
        'vq_mean_dist': np.mean(vq_dists),
        'km_mean_dist': np.mean(km_dists),
        'km_entropy': kmeans_result['entropy'],
        'km_dead': kmeans_result['dead_codes'],
        'km_silhouette': kmeans_result['silhouette']
    }

# ---------- Experiment Runners ----------
def run_scaling_experiment(d=256, r=4, sigma=0.1, seeds=[1, 2, 3]):
    """Test scaling behavior with different K"""
    K_values = [4, 8, 16, 32, 64]
    results = {'ste': {}, 'rotation': {}}

    for estimator in ['ste', 'rotation']:
        for K in K_values:
            print(f"\nRunning {estimator} with K={K}")
            K_results = []

            for seed in seeds:
                set_seed(seed)
                Xtr, U = make_subspace_data(d, r, 20000, sigma)
                Xva, _ = make_subspace_data(d, r, 4000, sigma)

                model = LinearVQ(d, r, K, beta=0.25, estimator=estimator, orthonormal_W=True)

                # Initialize codebook
                z0 = model.encode(torch.from_numpy(Xtr[:4000])).detach().cpu().numpy()
                kmeans = KMeans(n_clusters=K, random_state=seed, n_init=1)
                kmeans.fit(z0)
                model.codebook.code.data = torch.from_numpy(kmeans.cluster_centers_).float()

                # Train
                optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
                tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr)),
                                        batch_size=256, shuffle=True)
                va_loader = DataLoader(TensorDataset(torch.from_numpy(Xva)),
                                        batch_size=256, shuffle=False)

                for epoch in range(40):
                    model.train()
                    for (xb,) in tr_loader:
                        loss, _, _, _, _ = model(xb)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Evaluate
                model.eval()
                with torch.no_grad():
                    val_rec = 0
                    all_idx = []
                    for (xb,) in va_loader:
                        _, rec, _, _, idx = model(xb)
                        val_rec += rec.item() * xb.size(0)
                        all_idx.append(idx)

                    val_rec /= len(va_loader.dataset)
                    all_idx = torch.cat(all_idx)
                    ent, dead = entropy_dead(all_idx, K)
                    angle = principal_angles(U, model.W.data)

                K_results.append({
                    'rec': val_rec,
                    'entropy': ent,
                    'dead': dead,
                    'angle': angle
                })

            # Average over seeds
            avg_results = {
                'rec': np.mean([r['rec'] for r in K_results]),
                'rec_std': np.std([r['rec'] for r in K_results]),
                'entropy': np.mean([r['entropy'] for r in K_results]),
                'entropy_std': np.std([r['entropy'] for r in K_results]),
                'dead': np.mean([r['dead'] for r in K_results]),
                'dead_std': np.std([r['dead'] for r in K_results]),
                'angle': np.mean([r['angle'] for r in K_results]),
                'angle_std': np.std([r['angle'] for r in K_results])
            }

            results[estimator][K] = avg_results
            print(f"{estimator} K={K}: rec={avg_results['rec']:.4f}±{avg_results['rec_std']:.4f}, "
                  f"ent={avg_results['entropy']:.3f}±{avg_results['entropy_std']:.3f}, "
                  f"dead={avg_results['dead']:.1f}±{avg_results['dead_std']:.1f}")

    return results

def run_noise_robustness_experiment(d=256, r=4, K=16, seeds=[1, 2, 3]):
    """Test robustness to different noise levels"""
    sigma_values = [0.0, 0.05, 0.1, 0.2, 0.3]
    results = {'ste': {}, 'rotation': {}}

    for estimator in ['ste', 'rotation']:
        for sigma in sigma_values:
            print(f"\nRunning {estimator} with sigma={sigma}")
            sigma_results = []

            for seed in seeds:
                set_seed(seed)
                Xtr, U = make_subspace_data(d, r, 20000, sigma)
                Xva, _ = make_subspace_data(d, r, 4000, sigma)

                model = LinearVQ(d, r, K, beta=0.25, estimator=estimator, orthonormal_W=True)

                # Train (simplified)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
                tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr)),
                                        batch_size=256, shuffle=True)
                va_loader = DataLoader(TensorDataset(torch.from_numpy(Xva)),
                                        batch_size=256, shuffle=False)

                for epoch in range(40):
                    model.train()
                    for (xb,) in tr_loader:
                        loss, _, _, _, _ = model(xb)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Evaluate
                model.eval()
                with torch.no_grad():
                    val_rec = 0
                    all_idx = []
                    for (xb,) in va_loader:
                        _, rec, _, _, idx = model(xb)
                        val_rec += rec.item() * xb.size(0)
                        all_idx.append(idx)

                    val_rec /= len(va_loader.dataset)
                    all_idx = torch.cat(all_idx)
                    ent, dead = entropy_dead(all_idx, K)
                    angle = principal_angles(U, model.W.data)

                sigma_results.append({
                    'rec': val_rec,
                    'entropy': ent,
                    'dead': dead,
                    'angle': angle
                })

            # Average over seeds
            avg_results = {
                'rec': np.mean([r['rec'] for r in sigma_results]),
                'rec_std': np.std([r['rec'] for r in sigma_results]),
                'entropy': np.mean([r['entropy'] for r in sigma_results]),
                'entropy_std': np.std([r['entropy'] for r in sigma_results]),
                'dead': np.mean([r['dead'] for r in sigma_results]),
                'dead_std': np.std([r['dead'] for r in sigma_results]),
                'angle': np.mean([r['angle'] for r in sigma_results]),
                'angle_std': np.std([r['angle'] for r in sigma_results])
            }

            results[estimator][sigma] = avg_results
            print(f"{estimator} sigma={sigma}: rec={avg_results['rec']:.4f}±{avg_results['rec_std']:.4f}, "
                  f"ent={avg_results['entropy']:.3f}±{avg_results['entropy_std']:.3f}, "
                  f"dead={avg_results['dead']:.1f}±{avg_results['dead_std']:.1f}")

    return results

def run_commitment_ablation(d=256, r=4, K=16, sigma=0.1, seeds=[1, 2, 3]):
    """Test different commitment loss weights"""
    beta_values = [0.0, 0.1, 0.25, 0.5, 1.0]
    results = {'ste': {}, 'rotation': {}}

    for estimator in ['ste', 'rotation']:
        for beta in beta_values:
            print(f"\nRunning {estimator} with beta={beta}")
            beta_results = []

            for seed in seeds:
                set_seed(seed)
                Xtr, U = make_subspace_data(d, r, 20000, sigma)
                Xva, _ = make_subspace_data(d, r, 4000, sigma)

                model = LinearVQ(d, r, K, beta=beta, estimator=estimator, orthonormal_W=True)

                # Train
                optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
                tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr)),
                                        batch_size=256, shuffle=True)
                va_loader = DataLoader(TensorDataset(torch.from_numpy(Xva)),
                                        batch_size=256, shuffle=False)

                for epoch in range(40):
                    model.train()
                    for (xb,) in tr_loader:
                        loss, _, _, _, _ = model(xb)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Evaluate
                model.eval()
                with torch.no_grad():
                    val_rec = 0
                    all_idx = []
                    for (xb,) in va_loader:
                        _, rec, _, _, idx = model(xb)
                        val_rec += rec.item() * xb.size(0)
                        all_idx.append(idx)

                    val_rec /= len(va_loader.dataset)
                    all_idx = torch.cat(all_idx)
                    ent, dead = entropy_dead(all_idx, K)
                    angle = principal_angles(U, model.W.data)

                # K-means comparison
                km_comparison = compare_with_lloyds(model, torch.from_numpy(Xva), K)

                beta_results.append({
                    'rec': val_rec,
                    'entropy': ent,
                    'dead': dead,
                    'angle': angle,
                    'km_entropy': km_comparison['km_entropy'],
                    'km_dead': km_comparison['km_dead']
                })

            # Average over seeds
            avg_results = {
                'rec': np.mean([r['rec'] for r in beta_results]),
                'rec_std': np.std([r['rec'] for r in beta_results]),
                'entropy': np.mean([r['entropy'] for r in beta_results]),
                'entropy_std': np.std([r['entropy'] for r in beta_results]),
                'dead': np.mean([r['dead'] for r in beta_results]),
                'dead_std': np.std([r['dead'] for r in beta_results]),
                'angle': np.mean([r['angle'] for r in beta_results]),
                'angle_std': np.std([r['angle'] for r in beta_results]),
                'km_entropy': np.mean([r['km_entropy'] for r in beta_results]),
                'km_dead': np.mean([r['km_dead'] for r in beta_results])
            }

            results[estimator][beta] = avg_results
            print(f"{estimator} beta={beta}: rec={avg_results['rec']:.4f}±{avg_results['rec_std']:.4f}, "
                  f"ent={avg_results['entropy']:.3f}±{avg_results['entropy_std']:.3f}, "
                  f"dead={avg_results['dead']:.1f}±{avg_results['dead_std']:.1f}, "
                  f"km_ent={avg_results['km_entropy']:.3f}")

    return results

def plot_experiment_results(results, experiment_name, x_param, x_label):
    """Create comprehensive plots for experiment results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract data
    ste_results = results['ste']
    rot_results = results['rotation']
    x_values = list(ste_results.keys())

    # Plot 1: Reconstruction MSE
    ax = axes[0, 0]
    ste_rec = [ste_results[x]['rec'] for x in x_values]
    ste_rec_std = [ste_results[x]['rec_std'] for x in x_values]
    rot_rec = [rot_results[x]['rec'] for x in x_values]
    rot_rec_std = [rot_results[x]['rec_std'] for x in x_values]

    ax.errorbar(x_values, ste_rec, yerr=ste_rec_std, label='STE', marker='o', linewidth=2)
    ax.errorbar(x_values, rot_rec, yerr=rot_rec_std, label='Rotation', marker='s', linewidth=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('Reconstruction Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Entropy
    ax = axes[0, 1]
    ste_ent = [ste_results[x]['entropy'] for x in x_values]
    ste_ent_std = [ste_results[x]['entropy_std'] for x in x_values]
    rot_ent = [rot_results[x]['entropy'] for x in x_values]
    rot_ent_std = [rot_results[x]['entropy_std'] for x in x_values]

    ax.errorbar(x_values, ste_ent, yerr=ste_ent_std, label='STE', marker='o', linewidth=2)
    ax.errorbar(x_values, rot_ent, yerr=rot_ent_std, label='Rotation', marker='s', linewidth=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Code Usage Entropy')
    ax.set_title('Codebook Utilization')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Dead Codes
    ax = axes[1, 0]
    ste_dead = [ste_results[x]['dead'] for x in x_values]
    ste_dead_std = [ste_results[x]['dead_std'] for x in x_values]
    rot_dead = [rot_results[x]['dead'] for x in x_values]
    rot_dead_std = [rot_results[x]['dead_std'] for x in x_values]

    ax.errorbar(x_values, ste_dead, yerr=ste_dead_std, label='STE', marker='o', linewidth=2)
    ax.errorbar(x_values, rot_dead, yerr=rot_dead_std, label='Rotation', marker='s', linewidth=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Number of Dead Codes')
    ax.set_title('Dead Codes')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Principal Angle
    ax = axes[1, 1]
    ste_angle = [ste_results[x]['angle'] for x in x_values]
    ste_angle_std = [ste_results[x]['angle_std'] for x in x_values]
    rot_angle = [rot_results[x]['angle'] for x in x_values]
    rot_angle_std = [rot_results[x]['angle_std'] for x in x_values]

    ax.errorbar(x_values, ste_angle, yerr=ste_angle_std, label='STE', marker='o', linewidth=2)
    ax.errorbar(x_values, rot_angle, yerr=rot_angle_std, label='Rotation', marker='s', linewidth=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Mean Principal Angle (rad)')
    ax.set_title('Subspace Recovery')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'{experiment_name} - STE vs Rotation Comparison', fontsize=14)
    plt.tight_layout()

    return fig

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive VQ experiments')
    parser.add_argument('--experiment', choices=['scaling', 'noise', 'commitment', 'all'],
                        default='all', help='Which experiment to run')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3],
                        help='Random seeds for averaging')
    args = parser.parse_args()

    # Create experiment directory
    exp_dir = Path("runs") / datetime.now().strftime("%Y%m%d") / "comprehensive_experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Run experiments
    if args.experiment in ['scaling', 'all']:
        print("\n" + "="*60)
        print("Running Scaling Experiment (K variation)")
        print("="*60)
        scaling_results = run_scaling_experiment(seeds=args.seeds)
        results['scaling'] = scaling_results

        # Plot and save
        fig = plot_experiment_results(scaling_results, 'Scaling Experiment',
                                       'K', 'Codebook Size (K)')
        fig.savefig(exp_dir / 'scaling_results.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Save JSON
        with open(exp_dir / 'scaling_results.json', 'w') as f:
            json.dump(scaling_results, f, indent=2, default=float)

    if args.experiment in ['noise', 'all']:
        print("\n" + "="*60)
        print("Running Noise Robustness Experiment")
        print("="*60)
        noise_results = run_noise_robustness_experiment(seeds=args.seeds)
        results['noise'] = noise_results

        # Plot and save
        fig = plot_experiment_results(noise_results, 'Noise Robustness',
                                       'sigma', 'Noise Level (σ)')
        fig.savefig(exp_dir / 'noise_results.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Save JSON
        with open(exp_dir / 'noise_results.json', 'w') as f:
            json.dump(noise_results, f, indent=2, default=float)

    if args.experiment in ['commitment', 'all']:
        print("\n" + "="*60)
        print("Running Commitment Loss Ablation")
        print("="*60)
        commitment_results = run_commitment_ablation(seeds=args.seeds)
        results['commitment'] = commitment_results

        # Plot and save
        fig = plot_experiment_results(commitment_results, 'Commitment Loss Ablation',
                                       'beta', 'Commitment Weight (β)')
        fig.savefig(exp_dir / 'commitment_results.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Save JSON
        with open(exp_dir / 'commitment_results.json', 'w') as f:
            json.dump(commitment_results, f, indent=2, default=float)

    # Save all results
    with open(exp_dir / 'all_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nAll results saved to: {exp_dir}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF KEY FINDINGS")
    print("="*60)

    if 'scaling' in results:
        print("\n1. Scaling (K variation):")
        for est in ['ste', 'rotation']:
            K32_ent = results['scaling'][est][32]['entropy']
            K32_dead = results['scaling'][est][32]['dead']
            print(f"   {est.upper()}: K=32 → entropy={K32_ent:.3f}, dead={K32_dead:.1f}")

    if 'noise' in results:
        print("\n2. Noise Robustness:")
        for est in ['ste', 'rotation']:
            s03_ent = results['noise'][est][0.3]['entropy']
            s03_rec = results['noise'][est][0.3]['rec']
            print(f"   {est.upper()}: σ=0.3 → entropy={s03_ent:.3f}, rec={s03_rec:.4f}")

    if 'commitment' in results:
        print("\n3. Commitment Loss (β):")
        for est in ['ste', 'rotation']:
            b0_ent = results['commitment'][est][0.0]['entropy']
            b1_ent = results['commitment'][est][1.0]['entropy']
            print(f"   {est.upper()}: β=0 → ent={b0_ent:.3f}, β=1 → ent={b1_ent:.3f}")

if __name__ == "__main__":
    main()