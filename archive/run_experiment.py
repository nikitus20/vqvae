#!/usr/bin/env python3
"""
Main entry point for VQ-VAE experiments.

This script compares three methods for training VQ-VAE:
1. PCA + Lloyd Max (baseline)
2. Autograd STE (PyTorch automatic differentiation)
3. Manual STE (hand-derived gradients)
"""

from vqvae.experiment import main

if __name__ == "__main__":
    main()
