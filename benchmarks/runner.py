import os
import yaml
import torch
import random
import numpy as np
import pandas as pd


# =========================
# Utils
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_str):
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =========================
# Darcy Loader (NeuralOp)
# =========================
def get_loaders(batch_size):
    from neuralop.data.datasets import load_darcy_flow_small

    train_loader, test_loaders, _ = load_darcy_flow_small(
        n_train=1000,
        n_tests=[200],                 # REQUIRED
        batch_size=batch_size,
        test_batch_sizes=[batch_size]  # REQUIRED
    )

    # NeuralOp returns dict of test loaders → take first
    test_loader = list(test_loaders.values())[0]

    return train_loader, test_loader


# =========================
# Model Factory
# =========================
def get_model(name, device):
    if name == "fno":
        from benchmarks.models.fno import FNOWrapper
        return FNOWrapper(device=device)

    elif name == "geo_fno":
        from benchmarks.models.geo_fno import GeoFNOWrapper
        return GeoFNOWrapper(device=device)

    elif name == "pino":
        from benchmarks.models.pino import PINOWrapper
        return PINOWrapper(device=device)

    elif name == "deeponet":
        from benchmarks.models.deeponet import DeepONetWrapper
        return DeepONetWrapper(device=device)

    elif name == "pinn":
        from benchmarks.models.pinn import PINNWrapper
        return PINNWrapper(device=device)

    elif name == "pi_latent_no":
        from benchmarks.models.pi_latent_no import PILatentNOWrapper
        return PILatentNOWrapper(device=device)

    else:
        raise ValueError(f"Unknown model: {name}")


# =========================
# Metric
# =========================
def compute_l2(pred, target):
    return torch.norm(pred - target) / torch.norm(target)


# =========================
# Evaluation
# =========================
def evaluate(model, loader, device):
    model.eval()
    total_error = 0.0
    count = 0

    with torch.no_grad():
        for k, u in loader:  # IMPORTANT: NeuralOp format
            k = k.to(device)
            u = u.to(device)

            # Convert to (u, k) format your models expect
            x = torch.stack([u, k], dim=1)

            pred = model.predict(x).detach()

            error = compute_l2(pred, u)

            total_error += error.item()
            count += 1

    return total_error / max(count, 1)


# =========================
# Training
# =========================
def train_model(model, loader, epochs, lr, device):
    for epoch in range(epochs):
        for k, u in loader:
            k = k.to(device)
            u = u.to(device)

            x = torch.stack([u, k], dim=1)

            model.train_step(x, u, lr)


# =========================
# Main
# =========================
def main():
    with open("benchmarks/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = get_device(config["device"])

    print(f"\n🚀 Running on device: {device}")

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

    print(f"\n✅ Final results saved to {results_file}")


if __name__ == "__main__":
    main()