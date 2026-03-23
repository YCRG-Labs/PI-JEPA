#!/usr/bin/env python
"""Run benchmarks comparing PI-JEPA against baseline models."""

import os
import sys
import yaml
import torch
import random
import numpy as np
import pandas as pd

# Add PI-JEPA directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PI-JEPA"))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_str):
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_loaders(batch_size):
    from neuralop.data.datasets import load_darcy_flow_small

    train_loader, test_loaders, _ = load_darcy_flow_small(
        n_train=1000,
        n_tests=[200],
        batch_size=batch_size,
        test_batch_sizes=[batch_size]
    )

    test_loader = list(test_loaders.values())[0]

    return train_loader, test_loader


def get_model(name, device):
    if name == "fno":
        from benchmarks import FNOWrapper
        return FNOWrapper(device=device)

    elif name == "geo_fno":
        from benchmarks import get_geo_fno_wrapper
        GeoFNOWrapper = get_geo_fno_wrapper()
        return GeoFNOWrapper(device=device)

    elif name == "pino":
        from benchmarks import PINOWrapper
        return PINOWrapper(device=device)

    elif name == "deeponet":
        from benchmarks import DeepONetWrapper
        return DeepONetWrapper(device=device)

    elif name == "pinn":
        from benchmarks import PINNWrapper
        return PINNWrapper(device=device)

    elif name == "pi_latent_no":
        from benchmarks import PILatentNOWrapper
        return PILatentNOWrapper(device=device)

    else:
        raise ValueError(f"Unknown model: {name}")


def fix_shape(x):
    """Ensure tensor is (B, C, H, W)"""
    if x.ndim == 3:
        x = x.unsqueeze(1)
    elif x.ndim == 5:
        x = x.squeeze(-1)
    assert x.ndim == 4, f"Expected 4D tensor, got {x.shape}"
    return x


def compute_l2(pred, target):
    return torch.norm(pred - target) / torch.norm(target)


def evaluate(model, loader, device):
    model.eval()
    total_error = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            k = batch["x"].to(device).float()
            u = batch["y"].to(device).float()

            k = fix_shape(k)
            u = fix_shape(u)

            x = k

            pred = model.predict(x)

            pred = fix_shape(pred)

            error = compute_l2(pred, u)

            total_error += error.item()
            count += 1

    return total_error / max(count, 1)


def train_model(model, loader, epochs, lr, device):
    if hasattr(model, "train_model"):
        model.train_model(loader, epochs=epochs, lr=lr)
        return

    if hasattr(model, "train_step"):
        model.train()

        for epoch in range(epochs):
            for batch in loader:
                k = batch["x"].to(device).float()
                u = batch["y"].to(device).float()

                k = fix_shape(k)
                u = fix_shape(u)

                x = k

                model.train_step(x, u, lr)

        return

    raise RuntimeError("Model has no train_model or train_step method")


def main(config_path="configs/benchmark.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = get_device(config["device"])

    print(f"\n Running on device: {device}")

    batch_size = config["dataset"]["batch_size"]
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    model_names = config["models"]

    train_loader, test_loader = get_loaders(batch_size)

    results = []

    results_file = config["output"]["results_file"]
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    for model_name in model_names:
        print(f"\n===== Model: {model_name} =====")

        model = get_model(model_name, device)

        train_model(model, train_loader, epochs, lr, device)

        error = evaluate(model, test_loader, device)

        print(f"Result → Model: {model_name}, Error: {error:.6f}")

        results.append({
            "model": model_name,
            "error": error
        })

        pd.DataFrame(results).to_csv(results_file, index=False)

    print(f"Final results saved to {results_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)
