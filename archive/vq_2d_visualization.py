import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import Voronoi, voronoi_plot_2d
import math
from pathlib import Path
from datetime import datetime

# ---------- Gradient Estimators ----------
class STEEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, q):
        ctx.save_for_backward(z, q)
        return q

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradient straight through
        return grad_output, None

class RotationEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, q):
        ctx.save_for_backward(z, q)
        return q

    @staticmethod
    def backward(ctx, grad_output):
        z, q = ctx.saved_tensors
        v = q - z
        eps = 1e-8
        norm = torch.clamp(v.norm(dim=1, keepdim=True), min=eps)
        u = v / norm
        # Project gradient along z->q direction
        g_mag = (grad_output * u).sum(dim=1, keepdim=True)
        grad_z = g_mag * u
        return grad_z, None

# ---------- 2D VQ Model ----------
class VQ2D(nn.Module):
    def __init__(self, K=4, estimator='ste'):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(K, 2) * 0.5)
        self.estimator = estimator

    def forward(self, z):
        # Find nearest codebook entry
        dists = torch.cdist(z, self.codebook, p=2)
        idx = torch.argmin(dists, dim=1)
        q = self.codebook[idx]

        # Apply gradient estimator
        if self.estimator == 'ste':
            q_grad = STEEstimator.apply(z, q)
        elif self.estimator == 'rotation':
            q_grad = RotationEstimator.apply(z, q)
        else:
            q_grad = q.detach()  # No gradient

        return q_grad, idx, q.detach()

# ---------- Data Generation ----------
def make_2d_mixture(N=1000, K=4, sigma=0.1):
    """Generate 2D Gaussian mixture data"""
    # Fixed centers for reproducibility
    centers = np.array([
        [-1, -1], [1, -1], [1, 1], [-1, 1]
    ]) * 0.7

    data = []
    labels = []
    for i in range(N):
        k = i % K
        x = centers[k] + np.random.randn(2) * sigma
        data.append(x)
        labels.append(k)

    return np.array(data, dtype=np.float32), np.array(labels)

# ---------- Gradient Analysis ----------
def compute_gradient_field(model, grid_points, target_loss='reconstruction'):
    """Compute gradient field at grid points"""
    model.eval()
    gradients = []

    for point in grid_points:
        z = torch.tensor(point, dtype=torch.float32, requires_grad=True).unsqueeze(0)
        q_grad, idx, q = model(z)

        if target_loss == 'reconstruction':
            # Reconstruction loss gradient
            loss = torch.sum((q_grad - z) ** 2)
        elif target_loss == 'commitment':
            # Commitment loss gradient
            loss = torch.sum((z - q) ** 2)
        else:
            # Combined loss
            loss = torch.sum((q_grad - z) ** 2) + 0.25 * torch.sum((z - q) ** 2)

        loss.backward()
        if z.grad is not None:
            grad = z.grad.squeeze().detach().numpy()
        else:
            grad = np.zeros(2)  # No gradient
        gradients.append(grad)

        # Clear gradients for next iteration
        if z.grad is not None:
            z.grad.zero_()

    return np.array(gradients)

def compute_finite_diff_gradient(model, point, eps=1e-4):
    """Compute true gradient via finite differences"""
    grad = np.zeros(2)
    z_base = torch.tensor(point, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        q_base, _, _ = model(z_base)
        loss_base = torch.sum((q_base - z_base) ** 2).item()

        for i in range(2):
            z_plus = z_base.clone()
            z_plus[0, i] += eps
            q_plus, _, _ = model(z_plus)
            loss_plus = torch.sum((q_plus - z_plus) ** 2).item()

            grad[i] = (loss_plus - loss_base) / eps

    return grad

# ---------- Visualization ----------
def plot_voronoi_and_gradients(model, data, title="VQ Gradient Field"):
    """Create comprehensive visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Get codebook locations
    codes = model.codebook.detach().numpy()

    # Create grid for gradient field
    x_range = np.linspace(-2, 2, 20)
    y_range = np.linspace(-2, 2, 20)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Compute gradients
    grad_recon = compute_gradient_field(model, grid_points, 'reconstruction')
    grad_commit = compute_gradient_field(model, grid_points, 'commitment')
    grad_combined = compute_gradient_field(model, grid_points, 'combined')

    # Reshape for plotting
    grad_recon = grad_recon.reshape(20, 20, 2)
    grad_commit = grad_commit.reshape(20, 20, 2)
    grad_combined = grad_combined.reshape(20, 20, 2)

    # Plot 1: Data and Voronoi cells
    ax = axes[0, 0]
    ax.scatter(data[:, 0], data[:, 1], alpha=0.3, s=10)
    ax.scatter(codes[:, 0], codes[:, 1], c='red', s=200, marker='*',
               edgecolors='black', linewidths=2, label='Codebook')

    # Add Voronoi boundaries
    if len(codes) > 2:
        vor = Voronoi(codes)
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue',
                        line_width=2, line_alpha=0.6, point_size=0)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_title('Data & Voronoi Cells')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Reconstruction gradient field
    ax = axes[0, 1]
    ax.quiver(xx, yy, grad_recon[:, :, 0], grad_recon[:, :, 1],
              alpha=0.6, scale=50)
    ax.scatter(codes[:, 0], codes[:, 1], c='red', s=200, marker='*',
               edgecolors='black', linewidths=2)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_title('Reconstruction Loss Gradient')
    ax.grid(True, alpha=0.3)

    # Plot 3: Commitment gradient field
    ax = axes[0, 2]
    ax.quiver(xx, yy, grad_commit[:, :, 0], grad_commit[:, :, 1],
              alpha=0.6, scale=50, color='green')
    ax.scatter(codes[:, 0], codes[:, 1], c='red', s=200, marker='*',
               edgecolors='black', linewidths=2)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_title('Commitment Loss Gradient')
    ax.grid(True, alpha=0.3)

    # Plot 4: Combined gradient field
    ax = axes[1, 0]
    ax.quiver(xx, yy, grad_combined[:, :, 0], grad_combined[:, :, 1],
              alpha=0.6, scale=50, color='purple')
    ax.scatter(codes[:, 0], codes[:, 1], c='red', s=200, marker='*',
               edgecolors='black', linewidths=2)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_title('Combined Loss Gradient')
    ax.grid(True, alpha=0.3)

    # Plot 5: Gradient magnitude heatmap
    ax = axes[1, 1]
    grad_mag = np.sqrt(grad_combined[:, :, 0]**2 + grad_combined[:, :, 1]**2)
    im = ax.contourf(xx, yy, grad_mag, levels=20, cmap='viridis')
    ax.scatter(codes[:, 0], codes[:, 1], c='red', s=200, marker='*',
               edgecolors='white', linewidths=2)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_title('Gradient Magnitude')
    plt.colorbar(im, ax=ax)

    # Plot 6: Gradient alignment analysis
    ax = axes[1, 2]
    # Sample points near boundaries
    boundary_points = []
    alignments = []

    for i in range(100):
        # Random point
        p = np.random.randn(2) * 1.5
        z = torch.tensor(p, dtype=torch.float32, requires_grad=True).unsqueeze(0)

        # Get distances to two nearest codes
        with torch.no_grad():
            dists = torch.cdist(z, model.codebook, p=2).squeeze()
            sorted_dists, _ = torch.sort(dists)

        # If close to boundary (similar distances to top 2 codes)
        if sorted_dists[1] - sorted_dists[0] < 0.5:
            boundary_points.append(p)

            # Compute estimated gradient
            z_new = torch.tensor(p, dtype=torch.float32, requires_grad=True).unsqueeze(0)
            q_grad, _, q = model(z_new)
            loss = torch.sum((q_grad - z_new) ** 2)
            loss.backward()

            if z_new.grad is not None:
                est_grad = z_new.grad.squeeze().detach().numpy()
            else:
                continue

            # Compute true gradient
            true_grad = compute_finite_diff_gradient(model, p)

            # Compute alignment (cosine similarity)
            if np.linalg.norm(est_grad) > 1e-6 and np.linalg.norm(true_grad) > 1e-6:
                alignment = np.dot(est_grad, true_grad) / (np.linalg.norm(est_grad) * np.linalg.norm(true_grad))
                alignments.append(alignment)

    if alignments:
        ax.hist(alignments, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(alignments), color='red', linestyle='--',
                   label=f'Mean: {np.mean(alignments):.3f}')
        ax.set_xlabel('Gradient Alignment (Cosine Similarity)')
        ax.set_ylabel('Count')
        ax.set_title(f'Gradient Alignment at Boundaries ({model.estimator.upper()})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{title} - Estimator: {model.estimator.upper()}', fontsize=16)
    plt.tight_layout()

    return fig

def run_2d_comparison():
    """Run side-by-side comparison of STE vs Rotation"""
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate data
    data, labels = make_2d_mixture(N=500, K=4, sigma=0.15)

    # Create models
    model_ste = VQ2D(K=4, estimator='ste')
    model_rot = VQ2D(K=4, estimator='rotation')

    # Initialize with same codebook
    init_codes = torch.tensor([
        [-0.7, -0.7], [0.7, -0.7], [0.7, 0.7], [-0.7, 0.7]
    ], dtype=torch.float32)
    model_ste.codebook.data = init_codes.clone()
    model_rot.codebook.data = init_codes.clone()

    # Training loop
    optimizer_ste = torch.optim.Adam(model_ste.parameters(), lr=0.01)
    optimizer_rot = torch.optim.Adam(model_rot.parameters(), lr=0.01)

    data_tensor = torch.from_numpy(data)

    losses_ste = []
    losses_rot = []

    for epoch in range(100):
        # Train STE
        optimizer_ste.zero_grad()
        q_ste, idx_ste, q_detach_ste = model_ste(data_tensor)
        loss_ste = F.mse_loss(q_ste, data_tensor) + 0.25 * F.mse_loss(data_tensor, q_detach_ste)
        loss_ste.backward()
        optimizer_ste.step()
        losses_ste.append(loss_ste.item())

        # Train Rotation
        optimizer_rot.zero_grad()
        q_rot, idx_rot, q_detach_rot = model_rot(data_tensor)
        loss_rot = F.mse_loss(q_rot, data_tensor) + 0.25 * F.mse_loss(data_tensor, q_detach_rot)
        loss_rot.backward()
        optimizer_rot.step()
        losses_rot.append(loss_rot.item())

    # Create visualizations
    exp_dir = Path("runs") / datetime.now().strftime("%Y%m%d") / "2d_visualization"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Plot gradient fields
    fig_ste = plot_voronoi_and_gradients(model_ste, data, "STE Gradient Analysis")
    fig_ste.savefig(exp_dir / "ste_gradients.png", dpi=150, bbox_inches='tight')
    plt.close()

    fig_rot = plot_voronoi_and_gradients(model_rot, data, "Rotation Gradient Analysis")
    fig_rot.savefig(exp_dir / "rotation_gradients.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot training curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(losses_ste, label='STE', linewidth=2)
    ax.plot(losses_rot, label='Rotation', linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(exp_dir / "training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Compute final metrics
    with torch.no_grad():
        # Code usage for STE
        _, idx_ste, _ = model_ste(data_tensor)
        counts_ste = torch.bincount(idx_ste, minlength=4).float()
        entropy_ste = -(counts_ste/counts_ste.sum() * (counts_ste/counts_ste.sum() + 1e-8).log()).sum().item()
        dead_ste = (counts_ste == 0).sum().item()

        # Code usage for Rotation
        _, idx_rot, _ = model_rot(data_tensor)
        counts_rot = torch.bincount(idx_rot, minlength=4).float()
        entropy_rot = -(counts_rot/counts_rot.sum() * (counts_rot/counts_rot.sum() + 1e-8).log()).sum().item()
        dead_rot = (counts_rot == 0).sum().item()

    print(f"\n2D Visualization Results saved to: {exp_dir}")
    print("\nFinal Metrics:")
    print(f"STE     - Loss: {losses_ste[-1]:.4f}, Entropy: {entropy_ste:.3f}, Dead codes: {dead_ste}")
    print(f"Rotation - Loss: {losses_rot[-1]:.4f}, Entropy: {entropy_rot:.3f}, Dead codes: {dead_rot}")

    return {
        'ste': {'loss': losses_ste[-1], 'entropy': entropy_ste, 'dead': dead_ste},
        'rotation': {'loss': losses_rot[-1], 'entropy': entropy_rot, 'dead': dead_rot}
    }

if __name__ == "__main__":
    results = run_2d_comparison()