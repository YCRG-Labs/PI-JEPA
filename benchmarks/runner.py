import os
import yaml
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from torch.utils.data import DataLoader

# Dataset
from data.dataset import DarcyDataset

# Metrics
from eval.metrics import compute_l2


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
# Dataset Loader (FIXED)
# =========================
def get_loader(split, config):
    dataset = DarcyDataset(
        path=config["data"]["path"],
        config=config,
        split=split
    )

    return DataLoader(
        dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=(split == "train"),
        num_workers=0
    )


# =========================
# Model Factory
# =========================
def get_model(name, device):
    if name == "pi_jepa":
        from benchmarks.wrappers.pi_jepa_wrapper import PIJEPAWrapper
        return PIJEPAWrapper(device=device)

    elif name == "unet":
        from benchmarks.models.unet import UNetWrapper
        return UNetWrapper(device=device)

    elif name == "fno":
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
def train_model(model, loader, config):
    model.train_model(
        loader,
        epochs=config["training"]["epochs"],
        lr=config["training"]["learning_rate"]
    )


# =========================
# Main Runner
# =========================
def main():
    # Load config
    with open("benchmarks/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Setup
    set_seed(config["seed"])
    device = get_device(config["device"])

    print(f"\n🚀 Running on device: {device}")

    results = []

    model_names = config["models"]

    # Load datasets ONCE
    train_loader = get_loader("train", config)
    test_loader = get_loader("test", config)

    # Output path
    results_file = config["output"]["results_file"]
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # =========================
    # Loop
    # =========================
    for model_name in model_names:
        print(f"\n===== Model: {model_name} =====")

        # Model
        model = get_model(model_name, device)

        # Train
        train_model(model, train_loader, config)

        # Evaluate
        error = evaluate(model, test_loader, device)

        print(f"Result → Model: {model_name}, Error: {error:.6f}")

        result_row = {
            "model": model_name,
            "error": error
        }

        results.append(result_row)

        # Save progressively
        pd.DataFrame(results).to_csv(results_file, index=False)

    print(f"Final results saved to {results_file}")


# =========================
# Entry
# =========================
if __name__ == "__main__":
    main()