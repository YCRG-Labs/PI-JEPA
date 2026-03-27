#!/usr/bin/env python
"""
Generate Darcy Flow Dataset at 64x64 Resolution

Solves the steady-state Darcy flow equation:
    -∇·(K(x)∇p(x)) = f(x)
    
where:
    K(x) = permeability/coefficient field (input)
    p(x) = pressure field (output/solution)
    f(x) = source term (constant or specified)

Uses finite differences with direct solve for efficiency.
Generates random log-normal permeability fields with spatial correlation.
"""

import os
import argparse
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter


def generate_permeability_field(
    resolution: int = 64,
    length_scale: float = 0.1,
    variance: float = 1.0,
    seed: int = None
) -> np.ndarray:
    """
    Generate a random log-normal permeability field with spatial correlation.
    
    Args:
        resolution: Grid resolution (NxN)
        length_scale: Correlation length scale (fraction of domain)
        variance: Log-variance of the field
        seed: Random seed
        
    Returns:
        K: (resolution, resolution) permeability field
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate white noise
    noise = np.random.randn(resolution, resolution)
    
    # Apply Gaussian smoothing for spatial correlation
    sigma = length_scale * resolution
    smoothed = gaussian_filter(noise, sigma=sigma, mode='wrap')
    
    # Normalize and apply log-normal transform
    smoothed = (smoothed - smoothed.mean()) / (smoothed.std() + 1e-8)
    K = np.exp(variance * smoothed)
    
    # Ensure positive and bounded
    K = np.clip(K, 0.1, 10.0)
    
    return K


def solve_darcy_fd(
    K: np.ndarray,
    f: np.ndarray = None,
    bc_type: str = 'dirichlet'
) -> np.ndarray:
    """
    Solve Darcy flow using finite differences.
    
    -∇·(K∇p) = f with boundary conditions
    
    Args:
        K: (N, N) permeability field
        f: (N, N) source term (default: constant 1.0)
        bc_type: 'dirichlet' (p=0 on boundary) or 'mixed'
        
    Returns:
        p: (N, N) pressure solution
    """
    N = K.shape[0]
    h = 1.0 / (N - 1)  # Grid spacing
    
    if f is None:
        f = np.ones((N, N))
    
    # Number of unknowns (interior points for Dirichlet)
    if bc_type == 'dirichlet':
        n_interior = (N - 2) ** 2
        
        # Build sparse matrix for interior points
        # Using 5-point stencil with variable coefficients
        
        def idx(i, j):
            """Map 2D interior index to 1D."""
            return (i - 1) * (N - 2) + (j - 1)
        
        rows, cols, vals = [], [], []
        rhs = np.zeros(n_interior)
        
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                k = idx(i, j)
                
                # Harmonic mean for interface permeability
                K_e = 2 * K[i, j] * K[i, j+1] / (K[i, j] + K[i, j+1] + 1e-10)  # East
                K_w = 2 * K[i, j] * K[i, j-1] / (K[i, j] + K[i, j-1] + 1e-10)  # West
                K_n = 2 * K[i, j] * K[i-1, j] / (K[i, j] + K[i-1, j] + 1e-10)  # North
                K_s = 2 * K[i, j] * K[i+1, j] / (K[i, j] + K[i+1, j] + 1e-10)  # South
                
                # Diagonal (center)
                diag = (K_e + K_w + K_n + K_s) / h**2
                rows.append(k)
                cols.append(k)
                vals.append(diag)
                
                # Off-diagonals (neighbors)
                # East (j+1)
                if j + 1 < N - 1:
                    rows.append(k)
                    cols.append(idx(i, j+1))
                    vals.append(-K_e / h**2)
                
                # West (j-1)
                if j - 1 > 0:
                    rows.append(k)
                    cols.append(idx(i, j-1))
                    vals.append(-K_w / h**2)
                
                # North (i-1)
                if i - 1 > 0:
                    rows.append(k)
                    cols.append(idx(i-1, j))
                    vals.append(-K_n / h**2)
                
                # South (i+1)
                if i + 1 < N - 1:
                    rows.append(k)
                    cols.append(idx(i+1, j))
                    vals.append(-K_s / h**2)
                
                # RHS
                rhs[k] = f[i, j]
        
        # Build sparse matrix and solve
        A = sparse.csr_matrix((vals, (rows, cols)), shape=(n_interior, n_interior))
        p_interior = spsolve(A, rhs)
        
        # Reconstruct full solution with boundary conditions
        p = np.zeros((N, N))
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                p[i, j] = p_interior[idx(i, j)]
        
        return p
    
    else:
        raise NotImplementedError(f"BC type {bc_type} not implemented")


def generate_dataset(
    n_samples: int = 1000,
    resolution: int = 64,
    length_scale: float = 0.1,
    variance: float = 1.0,
    seed: int = 42,
    verbose: bool = True
) -> tuple:
    """
    Generate Darcy flow dataset.
    
    Args:
        n_samples: Number of samples to generate
        resolution: Grid resolution
        length_scale: Permeability correlation length
        variance: Permeability log-variance
        seed: Random seed
        verbose: Print progress
        
    Returns:
        K: (n_samples, resolution, resolution) permeability fields
        p: (n_samples, resolution, resolution) pressure solutions
    """
    np.random.seed(seed)
    
    K_all = np.zeros((n_samples, resolution, resolution))
    p_all = np.zeros((n_samples, resolution, resolution))
    
    for i in range(n_samples):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Generating sample {i + 1}/{n_samples}")
        
        # Generate random permeability
        K = generate_permeability_field(
            resolution=resolution,
            length_scale=length_scale,
            variance=variance,
            seed=seed + i
        )
        
        # Solve Darcy equation
        p = solve_darcy_fd(K)
        
        K_all[i] = K
        p_all[i] = p
    
    return K_all, p_all


def normalize_data(K: np.ndarray, p: np.ndarray) -> tuple:
    """
    Normalize data to zero mean, unit variance.
    
    Returns normalized data and normalization stats.
    """
    K_mean, K_std = K.mean(), K.std()
    p_mean, p_std = p.mean(), p.std()
    
    K_norm = (K - K_mean) / (K_std + 1e-8)
    p_norm = (p - p_mean) / (p_std + 1e-8)
    
    stats = {
        'K_mean': float(K_mean),
        'K_std': float(K_std),
        'p_mean': float(p_mean),
        'p_std': float(p_std)
    }
    
    return K_norm, p_norm, stats


def main():
    parser = argparse.ArgumentParser(description="Generate Darcy flow dataset")
    parser.add_argument("--n-train", type=int, default=1000, help="Training samples")
    parser.add_argument("--n-test", type=int, default=200, help="Test samples")
    parser.add_argument("--resolution", type=int, default=64, help="Grid resolution")
    parser.add_argument("--length-scale", type=float, default=0.1, help="Permeability correlation length")
    parser.add_argument("--variance", type=float, default=1.0, help="Permeability log-variance")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="data/darcy", help="Output directory")
    parser.add_argument("--normalize", action="store_true", help="Normalize data")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Darcy Flow Dataset Generation")
    print("=" * 60)
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Training samples: {args.n_train}")
    print(f"Test samples: {args.n_test}")
    print(f"Length scale: {args.length_scale}")
    print(f"Variance: {args.variance}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Generate training data
    print("Generating training data...")
    K_train, p_train = generate_dataset(
        n_samples=args.n_train,
        resolution=args.resolution,
        length_scale=args.length_scale,
        variance=args.variance,
        seed=args.seed
    )
    
    # Generate test data (different seed)
    print("\nGenerating test data...")
    K_test, p_test = generate_dataset(
        n_samples=args.n_test,
        resolution=args.resolution,
        length_scale=args.length_scale,
        variance=args.variance,
        seed=args.seed + 10000
    )
    
    # Normalize if requested
    stats = None
    if args.normalize:
        print("\nNormalizing data...")
        # Compute stats from training data only
        K_mean, K_std = K_train.mean(), K_train.std()
        p_mean, p_std = p_train.mean(), p_train.std()
        
        K_train = (K_train - K_mean) / (K_std + 1e-8)
        p_train = (p_train - p_mean) / (p_std + 1e-8)
        K_test = (K_test - K_mean) / (K_std + 1e-8)
        p_test = (p_test - p_mean) / (p_std + 1e-8)
        
        stats = {
            'K_mean': float(K_mean),
            'K_std': float(K_std),
            'p_mean': float(p_mean),
            'p_std': float(p_std)
        }
    
    # Convert to torch tensors
    K_train = torch.from_numpy(K_train).float().unsqueeze(1)  # (N, 1, H, W)
    p_train = torch.from_numpy(p_train).float().unsqueeze(1)
    K_test = torch.from_numpy(K_test).float().unsqueeze(1)
    p_test = torch.from_numpy(p_test).float().unsqueeze(1)
    
    # Save
    print("\nSaving data...")
    train_path = os.path.join(args.output_dir, "darcy_train.pt")
    test_path = os.path.join(args.output_dir, "darcy_test.pt")
    
    torch.save({
        'x': K_train,  # Coefficient/permeability
        'y': p_train,  # Solution/pressure
        'resolution': args.resolution,
        'n_samples': args.n_train,
        'stats': stats
    }, train_path)
    
    torch.save({
        'x': K_test,
        'y': p_test,
        'resolution': args.resolution,
        'n_samples': args.n_test,
        'stats': stats
    }, test_path)
    
    print(f"\nSaved training data to: {train_path}")
    print(f"Saved test data to: {test_path}")
    print(f"\nData shapes:")
    print(f"  K_train: {K_train.shape}")
    print(f"  p_train: {p_train.shape}")
    print(f"  K_test: {K_test.shape}")
    print(f"  p_test: {p_test.shape}")
    
    if stats:
        print(f"\nNormalization stats:")
        print(f"  K: mean={stats['K_mean']:.4f}, std={stats['K_std']:.4f}")
        print(f"  p: mean={stats['p_mean']:.4f}, std={stats['p_std']:.4f}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
