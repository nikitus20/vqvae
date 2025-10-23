#!/bin/bash
# Archive old experiment files that are now superseded by the unified framework

echo "Creating archive directory..."
mkdir -p archive

echo "Moving old experiment files to archive..."
mv vq_experiments_three_variants.py archive/ 2>/dev/null || echo "  (already moved or not found)"
mv vq_ste_rotation_lab.py archive/ 2>/dev/null || echo "  (already moved or not found)"
mv vq_comprehensive_experiments.py archive/ 2>/dev/null || echo "  (already moved or not found)"
mv vq_2d_visualization.py archive/ 2>/dev/null || echo "  (already moved or not found)"
mv vq_multimodal_experiment.py archive/ 2>/dev/null || echo "  (already moved or not found)"

echo ""
echo "✓ Old files archived"
echo "✓ Unified framework is in vqvae/"
echo "✓ Run experiments with: python run_comprehensive_experiments.py"
