#!/usr/bin/env python
"""
Supervised Finetuning Script for PI-JEPA.

This script implements the supervised finetuning phase that trains a prediction
head on a frozen (or optionally unfrozen) encoder to map coefficient embeddings
to solution fields using limited labeled data.

Paper specifications:
- 100 epochs finetuning with batch size 32
- Frozen encoder by default (only prediction head trained)
- Optional full finetuning with lower encoder learning rate
- Support for N_l ∈ {10, 25, 50, 100, 250, 500} labeled samples

Validates: Requirements 4, 5 (Finetuning with Frozen/Full Encoder)
- AC 4.1: Load pretrained encoder weights and freeze all encoder parameters
- AC 4.2: Initialize Prediction_Head that maps encoder output to solution field dimensions
- AC 4.3: Encode coefficient field x using frozen encoder
- AC 4.4: Predict solution field y from encoded coefficient representation
- AC 4.5: Compute MSE loss between predicted and ground truth solution y
- AC 4.6: Update only Prediction_Head parameters, NOT encoder parameters
- AC 4.7: Support configurable labeled sample counts N_l ∈ {10, 25, 50, 100, 250, 500}
- AC 5.1: Option to unfreeze encoder parameters
- AC 5.2: Update both encoder and Prediction_Head when full finetuning
- AC 5.3: Use lower learning rate for encoder than Prediction_Head when full finetuning
- AC 5.4: Log separate metrics for frozen vs full finetuning modes
"""

import os
import sys
import argparse
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# Add PI-JEPA directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PI-JEPA"))

from models import ViTEncoder, PredictionHead
from utils import load_config


# ============================================================================
# Supervised Finetuner
# ============================================================================

class SupervisedFinetuner:
    """
    Supervised finetuning pipeline for PI-JEPA.
    
    Implements the finetuning phase with frozen encoder and trainable
    prediction head to map coefficient embeddings to solution fields.
    
    Validates: Requirements 4, 5
    - AC 4.1: Load pretrained encoder weights and freeze all encoder parameters
    - AC 4.3: Encode coefficient field x using frozen encoder
    - AC 4.5: Compute MSE loss between predicted and ground truth solution y
    - AC 4.6: Update only Prediction_Head parameters, NOT encoder parameters
    - AC 5.1: Option to unfreeze encoder parameters
    - AC 5.2: Update both encoder and Prediction_Head when full finetuning
    - AC 5.3: Use lower learning rate for encoder than Prediction_Head
    """
    
    # Supported labeled sample counts per paper specification
    SUPPORTED_N_LABELED = [10, 25, 50, 100, 250, 500]
    
    def __init__(
        self,
        encoder: nn.Module,
        prediction_head: PredictionHead,
        config: Dict[str, Any],
        device: torch.device
    ):
        """
        Initialize supervised finetuner.
        
        Args:
            encoder: Pretrained encoder (will be frozen by default)
            prediction_head: Trainable prediction head
            config: Finetuning configuration
            device: Training device
        """
        self.encoder = encoder.to(device)
        self.prediction_head = prediction_head.to(device)
        self.config = config
        self.device = device
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self._encoder_frozen = False
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
    
    def freeze_encoder(self) -> None:
        """
        Freeze encoder parameters.
        
        Validates: AC 4.1 - Load pretrained encoder weights and freeze all encoder parameters
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
        self._encoder_frozen = True
        print("Encoder frozen - only prediction head will be trained")
    
    def unfreeze_encoder(self) -> None:
        """
        Unfreeze encoder parameters for full finetuning.
        
        Validates: AC 5.1 - Option to unfreeze encoder parameters
        """
        for param in self.encoder.parameters():
            param.requires_grad = True
        self._encoder_frozen = False
        print("Encoder unfrozen - full finetuning enabled")
    
    @property
    def encoder_frozen(self) -> bool:
        """Return whether encoder is frozen."""
        return self._encoder_frozen
    
    def _build_optimizer(
        self,
        lr: float,
        encoder_lr_multiplier: float = 0.1
    ) -> optim.Optimizer:
        """
        Build optimizer with optional differential learning rates.
        
        Validates: AC 5.3 - Use lower learning rate for encoder than Prediction_Head
        
        Args:
            lr: Base learning rate for prediction head
            encoder_lr_multiplier: Multiplier for encoder learning rate (default 0.1)
            
        Returns:
            AdamW optimizer
        """
        finetuning_cfg = self.config.get("finetuning", {})
        optim_cfg = finetuning_cfg.get("optim", {})
        weight_decay = float(optim_cfg.get("weight_decay", 1e-4))
        
        param_groups = [
            {
                'params': self.prediction_head.parameters(),
                'lr': lr,
                'name': 'prediction_head'
            }
        ]
        
        # Add encoder params with lower LR if not frozen
        if not self._encoder_frozen:
            encoder_lr = lr * encoder_lr_multiplier
            param_groups.append({
                'params': self.encoder.parameters(),
                'lr': encoder_lr,
                'name': 'encoder'
            })
            print(f"Optimizer: prediction_head lr={lr}, encoder lr={encoder_lr}")
        else:
            print(f"Optimizer: prediction_head lr={lr} (encoder frozen)")
        
        return optim.AdamW(param_groups, weight_decay=weight_decay)
    
    def _limit_dataset(
        self,
        data_loader: DataLoader,
        n_labeled: int
    ) -> DataLoader:
        """
        Limit dataset to N_l labeled samples.
        
        Validates: AC 4.7 - Support configurable labeled sample counts
        
        Args:
            data_loader: Original data loader
            n_labeled: Number of labeled samples to use
            
        Returns:
            New DataLoader with limited samples
        """
        dataset = data_loader.dataset
        n_samples = min(n_labeled, len(dataset))
        indices = list(range(n_samples))
        subset = Subset(dataset, indices)
        
        batch_size = data_loader.batch_size or 1
        return DataLoader(
            subset,
            batch_size=min(batch_size, n_samples),
            shuffle=True,
            num_workers=getattr(data_loader, "num_workers", 0),
            pin_memory=getattr(data_loader, "pin_memory", False),
        )
    
    def _prepare_batch(
        self,
        batch: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare batch for finetuning.
        
        Extracts coefficient field x and solution field y from batch.
        
        Args:
            batch: Batch from data loader
            
        Returns:
            Tuple of (x, y) tensors on device
        """
        if isinstance(batch, (tuple, list)):
            x = batch[0]
            y = batch[1] if len(batch) > 1 else batch[0]
        elif isinstance(batch, dict):
            x = batch.get("x", batch.get("input", batch.get("coefficient")))
            y = batch.get("y", batch.get("target", batch.get("solution")))
            if x is None:
                raise KeyError("Batch must contain 'x', 'input', or 'coefficient' key")
            if y is None:
                raise KeyError("Batch must contain 'y', 'target', or 'solution' key")
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")
        
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        
        # Ensure shape is (B, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if y.dim() == 3:
            y = y.unsqueeze(1)
        
        return x, y
    
    def finetune(
        self,
        train_loader: DataLoader,
        n_labeled: int,
        n_epochs: int = 100,
        freeze_encoder: bool = True,
        checkpoint_dir: str = "outputs/finetune"
    ) -> Dict[str, Any]:
        """
        Run supervised finetuning with limited labeled data.
        
        Validates: Requirements 4, 5
        - AC 4.1: Load pretrained encoder weights and freeze all encoder parameters
        - AC 4.3: Encode coefficient field x using frozen encoder
        - AC 4.4: Predict solution field y from encoded coefficient representation
        - AC 4.5: Compute MSE loss between predicted and ground truth solution y
        - AC 4.6: Update only Prediction_Head parameters, NOT encoder parameters
        - AC 4.7: Support configurable labeled sample counts
        - AC 5.1: Option to unfreeze encoder parameters
        - AC 5.2: Update both encoder and Prediction_Head when full finetuning
        - AC 5.4: Log separate metrics for frozen vs full finetuning modes
        
        Args:
            train_loader: Labeled (x, y) data loader
            n_labeled: Number of labeled samples to use (N_l)
            n_epochs: Number of finetuning epochs (default 100)
            freeze_encoder: Whether to freeze encoder (default True)
            checkpoint_dir: Directory for saving checkpoints
            
        Returns:
            Dict with training metrics and final checkpoint path
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # AC 4.1: Freeze encoder by default
        if freeze_encoder:
            self.freeze_encoder()
        else:
            # AC 5.1: Option to unfreeze encoder
            self.unfreeze_encoder()
        
        # AC 4.7: Limit dataset to N_l samples
        limited_loader = self._limit_dataset(train_loader, n_labeled)
        
        # Build optimizer
        finetuning_cfg = self.config.get("finetuning", {})
        optim_cfg = finetuning_cfg.get("optim", {})
        lr = float(optim_cfg.get("lr", 1e-3))
        
        # AC 5.3: Lower learning rate for encoder
        full_finetune_cfg = finetuning_cfg.get("full_finetune", {})
        encoder_lr_multiplier = float(full_finetune_cfg.get("encoder_lr_multiplier", 0.1))
        
        optimizer = self._build_optimizer(lr, encoder_lr_multiplier)
        
        # Training mode
        mode = "frozen_encoder" if freeze_encoder else "full_finetune"
        
        print(f"\nStarting supervised finetuning ({mode})")
        print(f"  Device: {self.device}")
        print(f"  N_labeled: {n_labeled}")
        print(f"  Epochs: {n_epochs}")
        print(f"  Batch size: {limited_loader.batch_size}")
        print(f"  Actual samples: {len(limited_loader.dataset)}")
        
        # Set training mode
        self.prediction_head.train()
        if freeze_encoder:
            self.encoder.eval()
        else:
            self.encoder.train()
        
        all_losses = []
        best_loss = float('inf')
        
        for epoch in range(n_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            n_batches = 0
            
            for batch in limited_loader:
                # Prepare batch
                x, y = self._prepare_batch(batch)
                
                optimizer.zero_grad()
                
                # AC 4.3: Encode coefficient field x
                if freeze_encoder:
                    with torch.no_grad():
                        z = self.encoder(x)
                else:
                    # AC 5.2: Update encoder when full finetuning
                    z = self.encoder(x)
                
                # AC 4.4: Predict solution field y
                y_pred = self.prediction_head(z)
                
                # AC 4.5: Compute MSE loss
                loss = F.mse_loss(y_pred, y)
                
                # Backward pass
                loss.backward()
                
                # AC 4.6: Update only prediction head (or both if full finetuning)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
                self.global_step += 1
            
            avg_loss = epoch_loss / max(n_batches, 1)
            all_losses.append(avg_loss)
            
            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.6f}")
            
            # Save best checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save_checkpoint(
                    os.path.join(checkpoint_dir, f"finetune_n{n_labeled}_best.pt"),
                    optimizer,
                    {'loss': avg_loss, 'n_labeled': n_labeled, 'mode': mode}
                )
        
        # Save final checkpoint
        final_path = os.path.join(checkpoint_dir, f"finetune_n{n_labeled}.pt")
        self._save_checkpoint(
            final_path,
            optimizer,
            {'loss': avg_loss, 'n_labeled': n_labeled, 'mode': mode}
        )
        print(f"Saved final checkpoint to {final_path}")
        
        # AC 5.4: Log separate metrics for frozen vs full finetuning modes
        return {
            'train_losses': all_losses,
            'final_loss': avg_loss,
            'best_loss': best_loss,
            'checkpoint_path': final_path,
            'n_labeled': n_labeled,
            'n_epochs': n_epochs,
            'mode': mode,
            'encoder_frozen': freeze_encoder
        }
    
    def evaluate(
        self,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate on test set.
        
        Args:
            test_loader: Test data loader with (x, y) pairs
            
        Returns:
            Dict with 'relative_l2_error' and other metrics
        """
        self.encoder.eval()
        self.prediction_head.eval()
        
        total_l2_error = 0.0
        total_l2_norm = 0.0
        total_mse = 0.0
        n_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                x, y = self._prepare_batch(batch)
                
                # Encode and predict
                z = self.encoder(x)
                y_pred = self.prediction_head(z)
                
                # Compute errors
                diff = y_pred - y
                l2_error = torch.sqrt((diff ** 2).sum(dim=(1, 2, 3)))
                l2_norm = torch.sqrt((y ** 2).sum(dim=(1, 2, 3)))
                
                total_l2_error += l2_error.sum().item()
                total_l2_norm += l2_norm.sum().item()
                total_mse += F.mse_loss(y_pred, y, reduction='sum').item()
                n_samples += x.shape[0]
        
        # Compute metrics
        relative_l2_error = total_l2_error / (total_l2_norm + 1e-8)
        mse = total_mse / (n_samples * y.shape[1] * y.shape[2] * y.shape[3])
        
        return {
            'relative_l2_error': relative_l2_error,
            'mse': mse,
            'n_samples': n_samples
        }
    
    def _save_checkpoint(
        self,
        path: str,
        optimizer: optim.Optimizer,
        metrics: Dict[str, Any]
    ) -> None:
        """
        Save finetuning checkpoint.
        
        Validates: AC 8.3 - Finetuning checkpoints contain Prediction_Head weights and metadata
        """
        checkpoint = {
            'checkpoint_type': 'finetuning',
            'prediction_head_state_dict': self.prediction_head.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'encoder_frozen': self._encoder_frozen,
            'config': self.config,
            'metrics': metrics
        }
        
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(checkpoint, path)


# ============================================================================
# Model Building
# ============================================================================

def load_pretrained_encoder(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: torch.device
) -> ViTEncoder:
    """
    Load pretrained encoder from checkpoint.
    
    Validates: AC 4.1 - Load pretrained encoder weights
    Validates: AC 8.5 - Verify encoder architecture compatibility
    
    Args:
        checkpoint_path: Path to pretraining checkpoint
        config: Configuration dictionary
        device: Device to load model on
        
    Returns:
        Loaded ViTEncoder
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Verify checkpoint type
    checkpoint_type = checkpoint.get('checkpoint_type', 'unknown')
    if checkpoint_type != 'pretraining':
        print(f"Warning: Expected pretraining checkpoint, got '{checkpoint_type}'")
    
    # Build encoder with single channel (matching pretraining)
    encoder = ViTEncoder(config, in_channels=1).to(device)
    
    # Load weights
    if 'encoder_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        print(f"Loaded encoder weights from {checkpoint_path}")
    else:
        raise KeyError("Checkpoint does not contain 'encoder_state_dict'")
    
    return encoder


def build_prediction_head(
    config: Dict[str, Any],
    device: torch.device
) -> PredictionHead:
    """
    Build prediction head from config.
    
    Validates: AC 4.2 - Initialize Prediction_Head that maps encoder output to solution field
    
    Args:
        config: Configuration dictionary
        device: Device to build model on
        
    Returns:
        PredictionHead instance
    """
    encoder_cfg = config.get("model", {}).get("encoder", {})
    finetuning_cfg = config.get("finetuning", {})
    head_cfg = finetuning_cfg.get("prediction_head", {})
    
    embed_dim = encoder_cfg.get("embed_dim", 384)
    hidden_dim = head_cfg.get("hidden_dim", 512)
    output_channels = head_cfg.get("output_channels", 1)
    image_size = encoder_cfg.get("image_size", 64)
    patch_size = encoder_cfg.get("patch_size", 8)
    
    prediction_head = PredictionHead(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_channels=output_channels,
        image_size=image_size,
        patch_size=patch_size
    ).to(device)
    
    print(f"Built PredictionHead: embed_dim={embed_dim}, hidden_dim={hidden_dim}, "
          f"output_channels={output_channels}")
    
    return prediction_head


def build_labeled_dataloader(
    config: Dict[str, Any],
    split: str = "train"
) -> DataLoader:
    """
    Build data loader for labeled (x, y) pairs.
    
    Args:
        config: Configuration dictionary
        split: Data split ('train', 'test')
        
    Returns:
        DataLoader for labeled data
    """
    from data.loaders import DatasetFactory
    
    data_cfg = config.get("data", {})
    finetuning_cfg = config.get("finetuning", {})
    
    # Try to load Darcy dataset with labels
    dataset_config = {
        'path': data_cfg.get("path", ""),
        'n_samples': finetuning_cfg.get("n_labeled", 500),
        'resolution': data_cfg.get("grid_size", 64),
        'normalize': data_cfg.get("normalize", True)
    }
    
    # Use darcy_unlabeled but we need labeled data
    # For now, create a simple labeled dataset wrapper
    try:
        try:
            from neuralop.data.datasets import load_darcy_flow_small
        except ImportError:
            from neuralop.datasets import load_darcy_flow_small
        
        train_loader, test_loader, _ = load_darcy_flow_small(
            n_train=1000,
            n_tests=200,
            batch_size=finetuning_cfg.get("batch_size", 32),
            test_batch_sizes=[finetuning_cfg.get("batch_size", 32)],
            data_root=data_cfg.get("path") if data_cfg.get("path") else None,
        )
        
        if split == "train":
            return train_loader
        else:
            return test_loader
            
    except ImportError:
        # Fallback: create synthetic data for testing
        print("Warning: neuralop not available, using synthetic data")
        from torch.utils.data import TensorDataset
        
        n_samples = 1000 if split == "train" else 200
        resolution = data_cfg.get("grid_size", 64)
        
        x = torch.randn(n_samples, 1, resolution, resolution)
        y = torch.randn(n_samples, 1, resolution, resolution)
        
        dataset = TensorDataset(x, y)
        batch_size = finetuning_cfg.get("batch_size", 32)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0
        )


# ============================================================================
# Main Entry Point
# ============================================================================

def finetune(
    pretrain_checkpoint: str,
    config_path: str = "configs/darcy.yaml",
    n_labeled: int = 100,
    output_dir: str = "outputs/finetune",
    freeze_encoder: bool = True
) -> str:
    """
    Run supervised finetuning.
    
    Args:
        pretrain_checkpoint: Path to pretraining checkpoint
        config_path: Path to YAML configuration file
        n_labeled: Number of labeled samples to use
        output_dir: Directory to save checkpoints
        freeze_encoder: Whether to freeze encoder (default True)
        
    Returns:
        Path to the saved final checkpoint
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set device
    if config["experiment"].get("device") is not None:
        device = torch.device(config["experiment"]["device"])
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    
    # Set seed
    seed = config["experiment"].get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Load pretrained encoder
    encoder = load_pretrained_encoder(pretrain_checkpoint, config, device)
    
    # Build prediction head
    prediction_head = build_prediction_head(config, device)
    
    # Build data loader
    train_loader = build_labeled_dataloader(config, split="train")
    
    # Build finetuner
    finetuner = SupervisedFinetuner(
        encoder=encoder,
        prediction_head=prediction_head,
        config=config,
        device=device
    )
    
    # Get number of epochs
    finetuning_cfg = config.get("finetuning", {})
    n_epochs = finetuning_cfg.get("epochs", 100)
    
    # Run finetuning
    results = finetuner.finetune(
        train_loader=train_loader,
        n_labeled=n_labeled,
        n_epochs=n_epochs,
        freeze_encoder=freeze_encoder,
        checkpoint_dir=output_dir
    )
    
    # Evaluate on test set
    test_loader = build_labeled_dataloader(config, split="test")
    eval_results = finetuner.evaluate(test_loader)
    
    print(f"\nTest Results:")
    print(f"  Relative L2 Error: {eval_results['relative_l2_error']:.6f}")
    print(f"  MSE: {eval_results['mse']:.6f}")
    
    return results['checkpoint_path']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised finetuning for PI-JEPA")
    parser.add_argument(
        "--pretrain-checkpoint",
        required=True,
        help="Path to pretraining checkpoint"
    )
    parser.add_argument(
        "--config",
        default="configs/darcy.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--n-labeled",
        type=int,
        default=100,
        help="Number of labeled samples to use (default: 100)"
    )
    parser.add_argument(
        "--output",
        default="outputs/finetune",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        default=True,
        help="Freeze encoder during finetuning (default: True)"
    )
    parser.add_argument(
        "--full-finetune",
        action="store_true",
        help="Enable full finetuning (unfreeze encoder)"
    )
    args = parser.parse_args()
    
    # Determine freeze_encoder based on flags
    freeze_encoder = not args.full_finetune
    
    checkpoint_path = finetune(
        pretrain_checkpoint=args.pretrain_checkpoint,
        config_path=args.config,
        n_labeled=args.n_labeled,
        output_dir=args.output,
        freeze_encoder=freeze_encoder
    )
    print(f"\nFinetuning complete. Checkpoint saved to: {checkpoint_path}")
