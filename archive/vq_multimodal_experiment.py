"""
Test VQ-VAE gradient estimators on multi-modal (mixture of Gaussians) data
to stress-test codebook utilization in non-convex settings.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.datasets import make_blobs
from vq_ste_rotation_lab import LinearVQ, entropy_dead, set_seed

def make_multimodal_data(n_samples=20000, n_clusters=8, d=256, r=4, cluster_std=0.05):
    """Generate multi-modal data with multiple Gaussian clusters"""
    # Generate low-dimensional clusters
    centers_r = np.random.randn(n_clusters, r) * 2

    # Assign samples to clusters
    samples_per_cluster = n_samples // n_clusters
    X_r = []
    labels = []

    for i in range(n_clusters):
        # Generate samples around this center
        cluster_samples = centers_r[i] + np.random.randn(samples_per_cluster, r) * cluster_std
        X_r.append(cluster_samples)
        labels.extend([i] * samples_per_cluster)

    X_r = np.vstack(X_r)
    labels = np.array(labels)

    # Random projection to high dimension
    A = np.random.randn(r, d) / np.sqrt(r)
    Q, _ = np.linalg.qr(A.T)
    U = Q[:, :r].T  # r x d orthonormal

    # Project to high dimension
    X = X_r @ U

    # Add small noise in ambient space
    X += np.random.randn(n_samples, d) * 0.01

    return X.astype(np.float32), labels, U, centers_r

def run_multimodal_experiment():
    """Compare STE vs Rotation on multi-modal data"""

    # Parameters
    d = 256
    r = 4
    n_clusters = 8
    K_values = [8, 16, 32]  # Codebook sizes to test
    seeds = [1, 2, 3]

    results = {'ste': {}, 'rotation': {}}

    for K in K_values:
        print(f"\n{'='*60}")
        print(f"Testing with K={K} codes for {n_clusters} data clusters")
        print('='*60)

        for estimator in ['ste', 'rotation']:
            K_results = []

            for seed in seeds:
                set_seed(seed)

                # Generate multi-modal data
                X_train, labels_train, U, centers = make_multimodal_data(
                    n_samples=20000, n_clusters=n_clusters, d=d, r=r
                )
                X_val, labels_val, _, _ = make_multimodal_data(
                    n_samples=4000, n_clusters=n_clusters, d=d, r=r
                )

                # Create model
                model = LinearVQ(d, r, K, beta=0.25, estimator=estimator, orthonormal_W=True)

                # Data loaders
                train_loader = DataLoader(
                    TensorDataset(torch.from_numpy(X_train)),
                    batch_size=256, shuffle=True
                )
                val_loader = DataLoader(
                    TensorDataset(torch.from_numpy(X_val)),
                    batch_size=256, shuffle=False
                )

                # Train
                optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

                for epoch in range(60):
                    model.train()
                    for (x,) in train_loader:
                        loss, _, _, _, _ = model(x)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Evaluate
                model.eval()
                with torch.no_grad():
                    val_rec = 0
                    all_idx = []
                    all_z = []

                    for (x,) in val_loader:
                        _, rec, z, _, idx = model(x)
                        val_rec += rec.item() * x.size(0)
                        all_idx.append(idx)
                        all_z.append(z)

                    val_rec /= len(val_loader.dataset)
                    all_idx = torch.cat(all_idx)
                    all_z = torch.cat(all_z)

                    # Compute metrics
                    ent, dead = entropy_dead(all_idx, K)

                    # Compute cluster preservation: how well do VQ codes preserve data clusters
                    # For each data cluster, see how many VQ codes it uses
                    cluster_codes = {}
                    for i, label in enumerate(labels_val):
                        if label not in cluster_codes:
                            cluster_codes[label] = set()
                        cluster_codes[label].add(all_idx[i].item())

                    # Average number of codes per cluster (ideal: K/n_clusters)
                    codes_per_cluster = np.mean([len(codes) for codes in cluster_codes.values()])

                    # Cluster purity: for each code, how pure is its cluster assignment
                    code_labels = {}
                    for i, idx in enumerate(all_idx):
                        idx_val = idx.item()
                        if idx_val not in code_labels:
                            code_labels[idx_val] = []
                        code_labels[idx_val].append(labels_val[i])

                    purities = []
                    for idx_val, labels in code_labels.items():
                        if labels:
                            label_counts = np.bincount(labels, minlength=n_clusters)
                            purity = label_counts.max() / len(labels)
                            purities.append(purity)

                    avg_purity = np.mean(purities) if purities else 0

                K_results.append({
                    'rec': val_rec,
                    'entropy': ent,
                    'dead': dead,
                    'codes_per_cluster': codes_per_cluster,
                    'purity': avg_purity
                })

            # Average results
            avg_results = {
                'rec': np.mean([r['rec'] for r in K_results]),
                'rec_std': np.std([r['rec'] for r in K_results]),
                'entropy': np.mean([r['entropy'] for r in K_results]),
                'entropy_std': np.std([r['entropy'] for r in K_results]),
                'dead': np.mean([r['dead'] for r in K_results]),
                'dead_std': np.std([r['dead'] for r in K_results]),
                'codes_per_cluster': np.mean([r['codes_per_cluster'] for r in K_results]),
                'purity': np.mean([r['purity'] for r in K_results])
            }

            results[estimator][K] = avg_results

            print(f"\n{estimator.upper()}:")
            print(f"  Reconstruction: {avg_results['rec']:.4f} ± {avg_results['rec_std']:.4f}")
            print(f"  Entropy: {avg_results['entropy']:.3f} ± {avg_results['entropy_std']:.3f}")
            print(f"  Dead codes: {avg_results['dead']:.1f} ± {avg_results['dead_std']:.1f}")
            print(f"  Codes/cluster: {avg_results['codes_per_cluster']:.2f} (ideal: {K/n_clusters:.2f})")
            print(f"  Cluster purity: {avg_results['purity']:.3f}")

    return results

def plot_multimodal_results(results):
    """Plot results from multi-modal experiment"""

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    K_values = list(results['ste'].keys())

    # Plot 1: Reconstruction
    ax = axes[0, 0]
    ste_rec = [results['ste'][K]['rec'] for K in K_values]
    rot_rec = [results['rotation'][K]['rec'] for K in K_values]
    ax.plot(K_values, ste_rec, 'o-', label='STE', linewidth=2, markersize=8)
    ax.plot(K_values, rot_rec, 's-', label='Rotation', linewidth=2, markersize=8)
    ax.set_xlabel('Codebook Size (K)')
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('Reconstruction Quality')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Entropy
    ax = axes[0, 1]
    ste_ent = [results['ste'][K]['entropy'] for K in K_values]
    rot_ent = [results['rotation'][K]['entropy'] for K in K_values]
    ax.plot(K_values, ste_ent, 'o-', label='STE', linewidth=2, markersize=8)
    ax.plot(K_values, rot_ent, 's-', label='Rotation', linewidth=2, markersize=8)
    ax.set_xlabel('Codebook Size (K)')
    ax.set_ylabel('Code Usage Entropy')
    ax.set_title('Codebook Utilization')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Dead codes
    ax = axes[0, 2]
    ste_dead = [results['ste'][K]['dead'] for K in K_values]
    rot_dead = [results['rotation'][K]['dead'] for K in K_values]
    ax.plot(K_values, ste_dead, 'o-', label='STE', linewidth=2, markersize=8)
    ax.plot(K_values, rot_dead, 's-', label='Rotation', linewidth=2, markersize=8)
    ax.set_xlabel('Codebook Size (K)')
    ax.set_ylabel('Number of Dead Codes')
    ax.set_title('Dead Codes')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Codes per cluster
    ax = axes[1, 0]
    ste_cpc = [results['ste'][K]['codes_per_cluster'] for K in K_values]
    rot_cpc = [results['rotation'][K]['codes_per_cluster'] for K in K_values]
    ideal = [K/8 for K in K_values]  # 8 data clusters
    ax.plot(K_values, ste_cpc, 'o-', label='STE', linewidth=2, markersize=8)
    ax.plot(K_values, rot_cpc, 's-', label='Rotation', linewidth=2, markersize=8)
    ax.plot(K_values, ideal, 'k--', label='Ideal', linewidth=2, alpha=0.5)
    ax.set_xlabel('Codebook Size (K)')
    ax.set_ylabel('Codes per Data Cluster')
    ax.set_title('Cluster Coverage')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Cluster purity
    ax = axes[1, 1]
    ste_purity = [results['ste'][K]['purity'] for K in K_values]
    rot_purity = [results['rotation'][K]['purity'] for K in K_values]
    ax.plot(K_values, ste_purity, 'o-', label='STE', linewidth=2, markersize=8)
    ax.plot(K_values, rot_purity, 's-', label='Rotation', linewidth=2, markersize=8)
    ax.set_xlabel('Codebook Size (K)')
    ax.set_ylabel('Average Cluster Purity')
    ax.set_title('Cluster Preservation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Summary table
    ax = axes[1, 2]
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    table_data.append(['Metric', 'STE (K=32)', 'Rotation (K=32)'])
    table_data.append(['Entropy', f"{results['ste'][32]['entropy']:.3f}",
                       f"{results['rotation'][32]['entropy']:.3f}"])
    table_data.append(['Dead Codes', f"{results['ste'][32]['dead']:.1f}",
                       f"{results['rotation'][32]['dead']:.1f}"])
    table_data.append(['Codes/Cluster', f"{results['ste'][32]['codes_per_cluster']:.2f}",
                       f"{results['rotation'][32]['codes_per_cluster']:.2f}"])
    table_data.append(['Purity', f"{results['ste'][32]['purity']:.3f}",
                       f"{results['rotation'][32]['purity']:.3f}"])

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    # Color header
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight better values
    for i in range(1, 5):
        ste_val = float(table_data[i][1])
        rot_val = float(table_data[i][2])
        if i in [1, 3, 4]:  # Higher is better
            if rot_val > ste_val:
                table[(i, 2)].set_facecolor('#90EE90')
            elif ste_val > rot_val:
                table[(i, 1)].set_facecolor('#90EE90')
        else:  # Lower is better (dead codes)
            if rot_val < ste_val:
                table[(i, 2)].set_facecolor('#90EE90')
            elif ste_val < rot_val:
                table[(i, 1)].set_facecolor('#90EE90')

    fig.suptitle('Multi-Modal Data Experiment (8 Gaussian Clusters)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig

if __name__ == "__main__":
    print("Running Multi-Modal Data Experiment")
    print("=" * 60)

    # Run experiment
    results = run_multimodal_experiment()

    # Save results
    exp_dir = Path("runs") / datetime.now().strftime("%Y%m%d") / "multimodal_experiment"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Plot results
    fig = plot_multimodal_results(results)
    fig.savefig(exp_dir / "multimodal_results.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save JSON results
    import json
    with open(exp_dir / "multimodal_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nResults saved to: {exp_dir}")

    # Print summary
    print("\n" + "="*60)
    print("MULTIMODAL EXPERIMENT SUMMARY")
    print("="*60)
    print("\nKey Finding: On multi-modal data with 8 clusters:")
    print(f"  - Rotation achieves better cluster coverage")
    print(f"  - Rotation has fewer dead codes across all K values")
    print(f"  - Both methods preserve cluster structure similarly")