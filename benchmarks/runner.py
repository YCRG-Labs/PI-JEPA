import os
import yaml
import torch
import random
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader


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
# Synthetic Darcy Loader (NO FILES)
# =========================
def get_loader(split, batch_size):
    if split == "train":
        N = 50
    else:
        N = 20

    T = 5
    H = W = 64

    # mimic Darcy structure
    u = torch.randn(N, T, H, W)
    k = torch.randn(N, 1, H, W)

    samples = []

    for i in range(N):
        for t in range(T - 1):
            u_t = u[i, t]
            u_tp1 = u[i, t + 1]

            k_i = k[i][0]

            x_t = torch.stack([u_t, k_i], dim=0)
            x_tp1 = torch.stack([u_tp1, k_i], dim=0)

            samples.append((x_t, x_tp1))

    return DataLoader(
        samples,
        batch_size=batch_size,
        shuffle=(split == "train")
    )


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
# Simple L2 Metric (no dependency)
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
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model.predict(x)
            pred = pred.detach()

            error = compute_l2(pred, y)

            total_error += error.item()
            count += 1

    return total_error / max(count, 1)


# =========================
# Training Wrapper
# =========================
def train_model(model, loader, epochs, lr):
    model.train_model(
        loader,
        epochs=epochs,
        lr=lr
    )


# =========================
# Main Runner
# =========================
def main():
    with open("benchmarks/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = get_device(config["device"])

    print(f"\n🚀 Running on device: {device}")

    results = []

    batch_size = config["dataset"]["batch_size"]
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    model_names = config["models"]

    train_loader = get_loader("train", batch_size)
    test_loader = get_loader("test", batch_size)

    results_file = config["output"]["results_file"]
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    for model_name in model_names:
        print(f"\n===== Model: {model_name} =====")

        model = get_model(model_name, device)

        train_model(model, train_loader, epochs, lr)

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