#!/usr/bin/env python
"""Compare PI-JEPA vs benchmarks with limited training data."""

import os
import sys
import json
import torch
import random
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PI-JEPA"))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_limited_loader(n_samples, batch_size=32):
    """Get train/test loaders with limited training samples."""
    from neuralop.data.datasets import load_darcy_flow_small
    
    train_loader, test_loaders, _ = load_darcy_flow_small(
        n_train=n_samples,
        n_tests=[200],
        batch_size=min(batch_size, n_samples),
        test_batch_sizes=[32]
    )
    test_loader = list(test_loaders.values())[0]
    return train_loader, test_loader


def fix_shape(x):
    if x.ndim == 3:
        x = x.unsqueeze(1)
    return x


def compute_l2(pred, target):
    return torch.norm(pred - target) / torch.norm(target)


def evaluate(model, loader, device):
    total_error = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            k = batch["x"].to(device).float()
            u = batch["y"].to(device).float()
            k, u = fix_shape(k), fix_shape(u)
            pred = fix_shape(model.predict(k))
            error = compute_l2(pred, u)
            total_error += error.item()
            count += 1
    return total_error / max(count, 1)


def train_model(model, loader, epochs, lr, device):
    if hasattr(model, "train_model"):
        model.train_model(loader, epochs=epochs, lr=lr)
    elif hasattr(model, "train_step"):
        if hasattr(model, "model"):
            model.model.train()
        for _ in range(epochs):
            for batch in loader:
                k = batch["x"].to(device).float()
                u = batch["y"].to(device).float()
                k, u = fix_shape(k), fix_shape(u)
                model.train_step(k, u, lr)


def get_model(name, device):
    if name == "fno":
        from benchmarks import FNOWrapper
        return FNOWrapper(device=device)
    elif name == "geo_fno":
        from benchmarks import get_geo_fno_wrapper
        return get_geo_fno_wrapper()(device=device)
    elif name == "deeponet":
        from benchmarks import DeepONetWrapper
        return DeepONetWrapper(device=device)
    elif name == "pinn":
        from benchmarks import PINNWrapper
        return PINNWrapper(device=device)
    else:
        raise ValueError(f"Unknown model: {name}")


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    
    n_labeled_values = [10, 25, 50, 100, 250, 500]
    models = ["fno", "geo_fno", "deeponet"]  # Skip PINN (slow)
    epochs = 100
    lr = 1e-3
    
    results = {model: {} for model in models}
    
    # Get test loader (same for all)
    _, test_loader = get_limited_loader(500)
    
    for n_labeled in n_labeled_values:
        print(f"\n{'='*60}")
        print(f"N_labeled = {n_labeled}")
        print('='*60)
        
        train_loader, _ = get_limited_loader(n_labeled)
        
        for model_name in models:
            print(f"\n  Training {model_name}...")
            
            try:
                model = get_model(model_name, device)
                train_model(model, train_loader, epochs, lr, device)
                error = evaluate(model, test_loader, device)
                results[model_name][n_labeled] = error
                print(f"    {model_name}: {error:.4f} ({error*100:.2f}%)")
            except Exception as e:
                print(f"    {model_name}: FAILED - {e}")
                results[model_name][n_labeled] = None
    
    # Save results
    output_dir = "outputs/data_efficiency"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "benchmark_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary table
    print("\n" + "="*60)
    print("DATA EFFICIENCY COMPARISON (Relative L2 Error)")
    print("="*60)
    print(f"{'N_labeled':<12}", end="")
    for model in models:
        print(f"{model:<15}", end="")
    print()
    print("-"*60)
    
    for n in n_labeled_values:
        print(f"{n:<12}", end="")
        for model in models:
            val = results[model].get(n)
            if val is not None:
                print(f"{val:.4f} ({val*100:.1f}%)", end="  ")
            else:
                print(f"{'N/A':<15}", end="")
        print()
    
    print(f"\nResults saved to {output_dir}/benchmark_comparison.json")


if __name__ == "__main__":
    main()
