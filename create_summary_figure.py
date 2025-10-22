#!/usr/bin/env python3
"""Create a comprehensive summary figure of all experimental results."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
results_dir = Path("comprehensive_results/20250930_165050")

with open(results_dir / "ste_verification.json") as f:
    verification = json.load(f)

with open(results_dir / "convergence_results.json") as f:
    convergence = json.load(f)

with open(results_dir / "codebook_size_results.json") as f:
    codebook_scaling = json.load(f)

with open(results_dir / "commitment_results.json") as f:
    commitment = json.load(f)

# Create comprehensive figure
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# ========== Row 1: Verification & Convergence ==========

# Panel A: STE Verification (bar chart)
ax = fig.add_subplot(gs[0, 0])
metrics = ['recon_mse', 'in_sub_dist', 'usage_entropy']
metric_labels = ['Recon MSE', 'In-Sub Dist', 'Usage Ent']
autograd_vals = [verification['autograd'][m] for m in metrics]
manual_vals = [verification['manual'][m] for m in metrics]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, autograd_vals, width, label='Autograd', color='#1f77b4', alpha=0.8)
bars2 = ax.bar(x + width/2, manual_vals, width, label='Manual', color='#2ca02c', alpha=0.8)

# Add difference annotations
for i, (auto, manual) in enumerate(zip(autograd_vals, manual_vals)):
    diff_pct = abs(auto - manual) / auto * 100
    ax.text(i, max(auto, manual) * 1.05, f'{diff_pct:.2f}%\ndiff',
            ha='center', va='bottom', fontsize=8, color='red')

ax.set_ylabel('Value')
ax.set_title('A) STE Implementation Verification\n(200 steps, K=16)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metric_labels, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Panel B: Final Performance Comparison
ax = fig.add_subplot(gs[0, 1])
methods = ['PCA+Lloyd', 'Autograd\nSTE', 'Manual\nSTE', 'Rotation']
mse_values = [
    convergence['baseline']['recon_mse'],
    convergence['autograd_ste']['recon_mse'],
    convergence['manual_ste']['recon_mse'],
    convergence['rotation']['recon_mse']
]
colors = ['#999999', '#1f77b4', '#2ca02c', '#d62728']

bars = ax.bar(methods, mse_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add improvement annotations
baseline_mse = mse_values[0]
for i, (bar, mse) in enumerate(zip(bars, mse_values)):
    height = bar.get_height()
    if i > 0:  # Skip baseline
        improvement = (baseline_mse - mse) / baseline_mse * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mse:.3f}\n({improvement:+.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    else:
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mse:.3f}\n(baseline)',
                ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Reconstruction MSE')
ax.set_title('B) Final Performance (500 steps, K=32)', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(2.44, 2.50)

# Panel C: Code Utilization
ax = fig.add_subplot(gs[0, 2])
entropy_values = [
    convergence['baseline']['usage_entropy'],
    convergence['autograd_ste']['usage_entropy'],
    convergence['manual_ste']['usage_entropy'],
    convergence['rotation']['usage_entropy']
]
max_entropy = np.log(32)  # K=32

bars = ax.bar(methods, entropy_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.axhline(max_entropy, color='black', linestyle='--', linewidth=2, label=f'Max (log K={max_entropy:.2f})')

for bar, ent in zip(bars, entropy_values):
    util_pct = ent / max_entropy * 100
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{util_pct:.1f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=9)

ax.set_ylabel('Usage Entropy (nats)')
ax.set_title('C) Code Utilization (500 steps, K=32)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(3.2, 3.5)

# ========== Row 2: Codebook Scaling ==========

K_values = [8, 16, 32, 64]

# Panel D: MSE vs K
ax = fig.add_subplot(gs[1, 0])
baseline_mse_k = [codebook_scaling[str(k)]['baseline']['recon_mse'] for k in K_values]
manual_mse_k = [codebook_scaling[str(k)]['manual_ste']['recon_mse'] for k in K_values]
rotation_mse_k = [codebook_scaling[str(k)]['rotation']['recon_mse'] for k in K_values]

ax.plot(K_values, baseline_mse_k, 'o-', label='PCA+Lloyd', linewidth=2.5, markersize=8, color='#999999')
ax.plot(K_values, manual_mse_k, 's-', label='Manual STE', linewidth=2.5, markersize=8, color='#2ca02c')
ax.plot(K_values, rotation_mse_k, '^-', label='Rotation', linewidth=2.5, markersize=8, color='#d62728')

ax.set_xlabel('Codebook Size (K)')
ax.set_ylabel('Reconstruction MSE')
ax.set_title('D) Codebook Size Scaling', fontweight='bold')
ax.set_xscale('log', base=2)
ax.set_xticks(K_values)
ax.set_xticklabels(K_values)
ax.legend()
ax.grid(True, alpha=0.3)

# Panel E: Entropy vs K
ax = fig.add_subplot(gs[1, 1])
baseline_ent_k = [codebook_scaling[str(k)]['baseline']['usage_entropy'] for k in K_values]
manual_ent_k = [codebook_scaling[str(k)]['manual_ste']['usage_entropy'] for k in K_values]
rotation_ent_k = [codebook_scaling[str(k)]['rotation']['usage_entropy'] for k in K_values]
max_ent_k = [np.log(k) for k in K_values]

ax.plot(K_values, baseline_ent_k, 'o-', label='PCA+Lloyd', linewidth=2.5, markersize=8, color='#999999')
ax.plot(K_values, manual_ent_k, 's-', label='Manual STE', linewidth=2.5, markersize=8, color='#2ca02c')
ax.plot(K_values, rotation_ent_k, '^-', label='Rotation', linewidth=2.5, markersize=8, color='#d62728')
ax.plot(K_values, max_ent_k, '--', label='Max (log K)', linewidth=2, color='black', alpha=0.5)

ax.set_xlabel('Codebook Size (K)')
ax.set_ylabel('Usage Entropy (nats)')
ax.set_title('E) Code Utilization Scaling', fontweight='bold')
ax.set_xscale('log', base=2)
ax.set_xticks(K_values)
ax.set_xticklabels(K_values)
ax.legend()
ax.grid(True, alpha=0.3)

# Panel F: Improvement vs K
ax = fig.add_subplot(gs[1, 2])
improvements = [(baseline_mse_k[i] - manual_mse_k[i]) / baseline_mse_k[i] * 100
                for i in range(len(K_values))]

bars = ax.bar([str(k) for k in K_values], improvements, color='#2ca02c', alpha=0.8,
              edgecolor='black', linewidth=1.5)

for bar, imp in zip(bars, improvements):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{imp:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax.set_xlabel('Codebook Size (K)')
ax.set_ylabel('Improvement over Baseline (%)')
ax.set_title('F) Manual STE vs Baseline', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0, color='black', linewidth=0.8)

# ========== Row 3: Commitment Ablation & Summary ==========

beta_values = [0.0, 0.1, 0.25, 0.5, 1.0]

# Panel G: MSE vs Beta
ax = fig.add_subplot(gs[2, 0])
manual_mse_beta = [commitment[str(b)]['manual_ste']['recon_mse'] for b in beta_values]
rotation_mse_beta = [commitment[str(b)]['rotation']['recon_mse'] for b in beta_values]

ax.plot(beta_values, manual_mse_beta, 's-', label='Manual STE', linewidth=2.5, markersize=8, color='#2ca02c')
ax.plot(beta_values, rotation_mse_beta, '^-', label='Rotation', linewidth=2.5, markersize=8, color='#d62728')

# Highlight default beta=0.25
ax.axvline(0.25, color='blue', linestyle='--', alpha=0.5, linewidth=2, label='Default β=0.25')

ax.set_xlabel('Commitment Weight (β)')
ax.set_ylabel('Reconstruction MSE')
ax.set_title('G) Commitment Loss Ablation', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel H: Performance Range
ax = fig.add_subplot(gs[2, 1])
mse_range_manual = max(manual_mse_beta) - min(manual_mse_beta)
mse_range_rotation = max(rotation_mse_beta) - min(rotation_mse_beta)
avg_mse_manual = np.mean(manual_mse_beta)
avg_mse_rotation = np.mean(rotation_mse_beta)

methods_beta = ['Manual STE', 'Rotation']
ranges = [mse_range_manual, mse_range_rotation]
ranges_pct = [r / avg * 100 for r, avg in zip(ranges, [avg_mse_manual, avg_mse_rotation])]

bars = ax.bar(methods_beta, ranges_pct, color=['#2ca02c', '#d62728'], alpha=0.8,
              edgecolor='black', linewidth=1.5)

for bar, pct in zip(bars, ranges_pct):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{pct:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax.set_ylabel('MSE Range across β (%)')
ax.set_title('H) Sensitivity to Commitment Loss', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 0.35)

# Panel I: Key Findings Summary
ax = fig.add_subplot(gs[2, 2])
ax.axis('off')

summary_text = """
KEY FINDINGS

✓ Manual STE matches Autograd
  within 1.4% error

✓ STE ≈ Rotation for linear
  encoders (<0.02% difference)

✓ Gradient methods beat baseline
  by 1-2% consistently

✓ Commitment loss flexible
  (0.2% range across β ∈ [0,1])

✓ Excellent code utilization
  (>98% of log K)

✓ Great scalability
  (48% error reduction K=8→64)

RECOMMENDATION
Use Manual STE with β=0.25
for production systems
"""

ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Overall title
fig.suptitle('VQ-VAE Methods: Comprehensive Experimental Analysis',
             fontsize=18, fontweight='bold', y=0.995)

plt.savefig('comprehensive_results/EXPERIMENTAL_SUMMARY.png', dpi=200, bbox_inches='tight')
print("✓ Summary figure saved to comprehensive_results/EXPERIMENTAL_SUMMARY.png")

plt.close()
