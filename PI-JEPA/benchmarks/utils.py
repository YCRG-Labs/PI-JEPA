"""Benchmark training utilities for consistent model comparison."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark training."""
    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    
    # Data split parameters
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Physics parameters (for PINO/PINN)
    physics_weight: float = 0.1
    collocation_size: int = 32
    
    # PINN-specific
    pinn_epochs_per_instance: int = 1000
    
    def __post_init__(self):
        """Validate configuration."""
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_data_splits(
    dataset,
    config: BenchmarkConfig
) -> Tuple[Subset, Subset, Subset]:
    """Create train/val/test splits with fixed seed."""
    set_seed(config.seed)
    
    n_total = len(dataset)
    n_train = int(n_total * config.train_ratio)
    n_val = int(n_total * config.val_ratio)
    n_test = n_total - n_train - n_val
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(
    train_dataset,
    val_dataset,
    test_dataset,
    config: BenchmarkConfig
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders with consistent settings."""
    # Use fixed seed for worker initialization
    def seed_worker(worker_id):
        worker_seed = config.seed + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(config.seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader


class BenchmarkTrainer:
    """Unified trainer for all benchmark models."""
    
    def __init__(
        self,
        config: BenchmarkConfig
    ):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set seed for reproducibility
        set_seed(config.seed)
    
    def train_model(
        self,
        model_wrapper,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, List[float]]:
        """Train a benchmark model."""
        history = {
            "train_loss": [],
            "val_loss": []
        }
        
        # Train using the wrapper's train_model method
        model_wrapper.train_model(
            train_loader,
            epochs=self.config.epochs,
            lr=self.config.learning_rate
        )
        
        # Evaluate on validation set if provided
        if val_loader is not None:
            val_loss = self.evaluate(model_wrapper, val_loader)
            history["val_loss"].append(val_loss)
        
        return history
    
    def evaluate(
        self,
        model_wrapper,
        data_loader: DataLoader
    ) -> float:
        """Evaluate model on a dataset."""
        model_wrapper.eval()
        
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                x = batch["x"].to(self.device).float()
                y = batch["y"].to(self.device).float()
                
                # Ensure correct shape
                if x.dim() == 3:
                    x = x.unsqueeze(1)
                if y.dim() == 3:
                    y = y.unsqueeze(1)
                
                pred = model_wrapper.predict(x)
                loss = nn.functional.mse_loss(pred, y)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def compute_metrics(
        self,
        model_wrapper,
        data_loader: DataLoader
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        model_wrapper.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                x = batch["x"].to(self.device).float()
                y = batch["y"].to(self.device).float()
                
                if x.dim() == 3:
                    x = x.unsqueeze(1)
                if y.dim() == 3:
                    y = y.unsqueeze(1)
                
                pred = model_wrapper.predict(x)
                
                all_preds.append(pred.cpu())
                all_targets.append(y.cpu())
        
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        mse = nn.functional.mse_loss(preds, targets).item()
        
        # Relative L2 error
        rel_l2 = torch.norm(preds - targets) / (torch.norm(targets) + 1e-8)
        rel_l2 = rel_l2.item()
        
        # Per-channel MSE
        per_channel_mse = {}
        for c in range(preds.shape[1]):
            per_channel_mse[f"channel_{c}_mse"] = nn.functional.mse_loss(
                preds[:, c], targets[:, c]
            ).item()
        
        return {
            "mse": mse,
            "relative_l2": rel_l2,
            **per_channel_mse
        }


def run_benchmark_comparison(
    dataset,
    model_wrappers: Dict[str, Any],
    config: BenchmarkConfig
) -> Dict[str, Dict[str, float]]:
    """Run comparison across all benchmark models."""
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Create identical splits for all models
    train_dataset, val_dataset, test_dataset = create_data_splits(dataset, config)
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, config
    )
    
    # Train and evaluate each model
    trainer = BenchmarkTrainer(config)
    results = {}
    
    for name, wrapper in model_wrappers.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        print(f"{'='*50}")
        
        # Reset seed before each model for reproducibility
        set_seed(config.seed)
        
        # Train
        trainer.train_model(wrapper, train_loader, val_loader)
        
        # Evaluate
        metrics = trainer.compute_metrics(wrapper, test_loader)
        results[name] = metrics
        
        print(f"\n{name} Results:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.6f}")
    
    return results


def create_benchmark_suite(
    device: torch.device,
    config: Optional[BenchmarkConfig] = None
) -> Dict[str, Any]:
    """Create a suite of benchmark models with consistent configuration."""
    if config is None:
        config = BenchmarkConfig()
    
    from .fno import FNOWrapper
    from .pino import PINOWrapper
    from .ufno import UFNOWrapper
    from .deeponet import DeepONetWrapper, UDeepONetWrapper
    from .pinn import PINNWrapper
    
    return {
        "FNO": FNOWrapper(device),
        "PINO": PINOWrapper(
            device,
            physics_weight=config.physics_weight,
            collocation_size=config.collocation_size
        ),
        "U-FNO": UFNOWrapper(device),
        "DeepONet": DeepONetWrapper(device),
        "U-DeepONet": UDeepONetWrapper(device),
        "PINN": PINNWrapper(
            device,
            physics_weight=config.physics_weight
        )
    }


def save_benchmark_results(
    results: Dict[str, Dict[str, float]],
    path: str
):
    """Save benchmark results to file."""
    import json
    
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def load_benchmark_results(path: str) -> Dict[str, Dict[str, float]]:
    """Load benchmark results from file."""
    import json
    
    with open(path, 'r') as f:
        return json.load(f)
