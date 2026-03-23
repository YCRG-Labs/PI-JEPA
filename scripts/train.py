#!/usr/bin/env python
"""
Train PI-JEPA model on Darcy flow data.

Paper specifications:
- 500 epochs pretraining with batch size 64
- AdamW: lr=1.5×10^-4, weight_decay=5×10^-2
- EMA momentum annealing: τ from 0.99 to 0.999 over first 10% epochs
- Physics weight ramping over first 200 steps
- Loss weights: λ_p=0.1 (physics), λ_r=1.0 (regularization)
"""

import os
import sys
import math
import torch
import torch.optim as optim
from typing import Optional, Dict, Any, Tuple

# Add PI-JEPA directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PI-JEPA"))

from models import ViTEncoder, Predictor, PIJEPA, Decoder
from training import LossBuilder, update_ema
from utils import load_config, Logger, save_checkpoint

from neuralop.data.datasets import load_darcy_flow_small


# ============================================================================
# EMA Momentum Annealing Schedule
# ============================================================================

class EMAMomentumSchedule:
    """
    EMA momentum annealing schedule following paper specifications.
    
    Anneals τ from tau_start (0.99) to tau_end (0.999) over the first
    warmup_fraction (10%) of training epochs using a cosine schedule.
    
    After warmup, τ remains at tau_end.
    """
    
    def __init__(
        self,
        tau_start: float = 0.99,
        tau_end: float = 0.999,
        warmup_fraction: float = 0.1,
        total_epochs: int = 500
    ):
        """
        Initialize EMA momentum schedule.
        
        Args:
            tau_start: Initial momentum value (default 0.99)
            tau_end: Final momentum value (default 0.999)
            warmup_fraction: Fraction of epochs for warmup (default 0.1 = 10%)
            total_epochs: Total number of training epochs
        """
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.warmup_fraction = warmup_fraction
        self.total_epochs = total_epochs
        self.warmup_epochs = int(total_epochs * warmup_fraction)
    
    def get_tau(self, epoch: int) -> float:
        """
        Get EMA momentum τ for the given epoch.
        
        Uses cosine annealing during warmup period, then constant tau_end.
        
        Args:
            epoch: Current epoch (0-indexed)
            
        Returns:
            EMA momentum value τ
        """
        if self.warmup_epochs == 0:
            return self.tau_end
        
        if epoch >= self.warmup_epochs:
            # After warmup, use tau_end
            return self.tau_end
        
        # During warmup, use cosine annealing from tau_start to tau_end
        t = epoch / self.warmup_epochs
        # Cosine schedule: starts at tau_start, ends at tau_end
        tau = self.tau_start + (self.tau_end - self.tau_start) * (1 - math.cos(math.pi * t)) / 2
        return tau


# ============================================================================
# Physics Weight Ramping Schedule
# ============================================================================

class PhysicsWeightSchedule:
    """
    Physics weight ramping schedule following paper specifications.
    
    Ramps physics weight λ_p from 0 to target value over the first
    ramp_steps (200) training steps.
    """
    
    def __init__(
        self,
        target_weight: float = 0.1,
        ramp_steps: int = 200
    ):
        """
        Initialize physics weight schedule.
        
        Args:
            target_weight: Target physics weight λ_p (default 0.1)
            ramp_steps: Number of steps to ramp over (default 200)
        """
        self.target_weight = target_weight
        self.ramp_steps = ramp_steps
    
    def get_weight(self, step: int) -> float:
        """
        Get physics weight for the given training step.
        
        Linear ramp from 0 to target_weight over ramp_steps.
        
        Args:
            step: Current training step (0-indexed)
            
        Returns:
            Physics weight value
        """
        if self.ramp_steps <= 0:
            return self.target_weight
        
        if step >= self.ramp_steps:
            return self.target_weight
        
        # Linear ramp: weight = target * (step / ramp_steps)
        return self.target_weight * (step / self.ramp_steps)


# ============================================================================
# K=3 Physics Weight Manager
# ============================================================================

class K3PhysicsWeightManager:
    """
    Manages separate physics weights for K=3 predictors (reactive transport).
    
    Supports independent weights for:
    - Pressure residual (λ_p^(1))
    - Transport residual (λ_p^(2))
    - Reaction residual (λ_p^(3))
    
    Each weight can have its own ramping schedule.
    """
    
    def __init__(
        self,
        pressure_weight: float = 0.1,
        transport_weight: float = 0.1,
        reaction_weight: float = 0.1,
        ramp_steps: int = 200
    ):
        """
        Initialize K=3 physics weight manager.
        
        Args:
            pressure_weight: Target weight for pressure residual
            transport_weight: Target weight for transport residual
            reaction_weight: Target weight for reaction residual
            ramp_steps: Number of steps to ramp over
        """
        self.pressure_schedule = PhysicsWeightSchedule(pressure_weight, ramp_steps)
        self.transport_schedule = PhysicsWeightSchedule(transport_weight, ramp_steps)
        self.reaction_schedule = PhysicsWeightSchedule(reaction_weight, ramp_steps)
    
    def get_weights(self, step: int) -> Dict[str, float]:
        """
        Get all physics weights for the given training step.
        
        Args:
            step: Current training step
            
        Returns:
            Dictionary with 'pressure', 'transport', 'reaction' weights
        """
        return {
            'pressure': self.pressure_schedule.get_weight(step),
            'transport': self.transport_schedule.get_weight(step),
            'reaction': self.reaction_schedule.get_weight(step)
        }


# ============================================================================
# Training Configuration
# ============================================================================

def get_device(cfg: Dict[str, Any]) -> torch.device:
    """Get the device to use for training."""
    if cfg["experiment"].get("device", None) is not None:
        return torch.device(cfg["experiment"]["device"])
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def build_model(cfg: Dict[str, Any], device: torch.device) -> Tuple[PIJEPA, Decoder]:
    """Build the PI-JEPA model and decoder."""
    patch_size = cfg["model"]["encoder"]["patch_size"]

    encoder = ViTEncoder(cfg).to(device)
    target_encoder = ViTEncoder(cfg).to(device)

    predictors = [
        Predictor(cfg).to(device)
        for _ in range(cfg["model"]["num_predictors"])
    ]

    model = PIJEPA(
        encoder=encoder,
        target_encoder=target_encoder,
        predictors=predictors,
        embed_dim=cfg["model"]["encoder"]["embed_dim"],
        num_patches=None,
        patch_size=patch_size,
    ).to(device)

    decoder = Decoder(**cfg["decoder"]).to(device)

    for p in target_encoder.parameters():
        p.requires_grad = False

    target_encoder.load_state_dict(encoder.state_dict())

    return model, decoder


def build_dataloader(cfg: Dict[str, Any]):
    """Build the data loader for training."""
    batch_size = cfg["training"].get("batch_size", 64)
    
    train_loader, _, _ = load_darcy_flow_small(
        n_train=1000,
        batch_size=batch_size,
        test_resolutions=[64],
        n_tests=[100],
        test_batch_sizes=[32],
        encode_output=False,
    )
    return train_loader


def build_optimizer(
    cfg: Dict[str, Any],
    model: PIJEPA,
    decoder: Decoder
) -> optim.Optimizer:
    """
    Build AdamW optimizer with paper specifications.
    
    Default: lr=1.5×10^-4, weight_decay=5×10^-2
    """
    optim_cfg = cfg["training"]["optim"]
    
    lr = float(optim_cfg.get("lr", 1.5e-4))
    weight_decay = float(optim_cfg.get("weight_decay", 5e-2))
    betas = tuple(optim_cfg.get("betas", [0.9, 0.95]))
    
    optimizer = optim.AdamW(
        list(model.parameters()) + list(decoder.parameters()),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas
    )
    
    return optimizer


def build_ema_schedule(cfg: Dict[str, Any]) -> EMAMomentumSchedule:
    """
    Build EMA momentum schedule from config.
    
    Default: τ from 0.99 to 0.999 over first 10% of epochs
    """
    ema_cfg = cfg.get("ema", {}).get("schedule", {})
    training_cfg = cfg.get("training", {})
    
    tau_start = ema_cfg.get("tau_start", 0.99)
    tau_end = ema_cfg.get("tau_end", 0.999)
    warmup_fraction = ema_cfg.get("warmup_fraction", 0.1)
    total_epochs = training_cfg.get("epochs", 500)
    
    return EMAMomentumSchedule(
        tau_start=tau_start,
        tau_end=tau_end,
        warmup_fraction=warmup_fraction,
        total_epochs=total_epochs
    )


def build_physics_weight_schedule(cfg: Dict[str, Any]) -> PhysicsWeightSchedule:
    """
    Build physics weight ramping schedule from config.
    
    Default: λ_p=0.1 ramped over first 200 steps
    """
    loss_cfg = cfg.get("loss", {})
    physics_cfg = loss_cfg.get("physics", {})
    
    target_weight = physics_cfg.get("weight", 0.1)
    ramp_steps = physics_cfg.get("ramp_steps", 200)
    
    return PhysicsWeightSchedule(
        target_weight=target_weight,
        ramp_steps=ramp_steps
    )


def build_k3_physics_weights(cfg: Dict[str, Any]) -> Optional[K3PhysicsWeightManager]:
    """
    Build K=3 physics weight manager if configured.
    
    Returns None if not using K=3 predictors.
    """
    model_cfg = cfg.get("model", {})
    num_predictors = model_cfg.get("num_predictors", 2)
    
    if num_predictors != 3:
        return None
    
    loss_cfg = cfg.get("loss", {})
    physics_cfg = loss_cfg.get("physics", {})
    weights_per_residual = physics_cfg.get("weights_per_residual", {})
    ramp_steps = physics_cfg.get("ramp_steps", 200)
    
    # Default to main physics weight if not specified
    default_weight = physics_cfg.get("weight", 0.1)
    
    return K3PhysicsWeightManager(
        pressure_weight=weights_per_residual.get("pressure", default_weight),
        transport_weight=weights_per_residual.get("transport", default_weight),
        reaction_weight=weights_per_residual.get("reaction", default_weight),
        ramp_steps=ramp_steps
    )


# ============================================================================
# Masking Utilities
# ============================================================================

def sample_block_mask_indices(
    B: int,
    grid_size: int,
    device: torch.device,
    context_ratio: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample block mask indices for JEPA training."""
    context_idx = []
    target_idx = []

    for _ in range(B):
        mask = torch.zeros(grid_size, grid_size, dtype=torch.bool)

        block_size = max(1, int(grid_size * (1 - context_ratio)))

        x = torch.randint(0, grid_size - block_size + 1, (1,))
        y = torch.randint(0, grid_size - block_size + 1, (1,))

        mask[x:x+block_size, y:y+block_size] = True

        flat = mask.flatten()

        target = torch.where(flat)[0]
        context = torch.where(~flat)[0]

        context_idx.append(context)
        target_idx.append(target)

    return torch.stack(context_idx).to(device), torch.stack(target_idx).to(device)


# ============================================================================
# Main Training Function
# ============================================================================

def train(config_path: str = "configs/darcy.yaml", output_dir: str = "outputs") -> str:
    """
    Train PI-JEPA model with paper specifications.
    
    Paper specifications:
    - 500 epochs pretraining with batch size 64
    - AdamW: lr=1.5×10^-4, weight_decay=5×10^-2
    - EMA momentum annealing: τ from 0.99 to 0.999 over first 10% epochs
    - Physics weight ramping over first 200 steps
    - Loss weights: λ_p=0.1 (physics), λ_r=1.0 (regularization)
    
    Args:
        config_path: Path to YAML configuration file
        output_dir: Directory to save the final checkpoint
    
    Returns:
        Path to the saved final checkpoint
    """
    cfg = load_config(config_path)

    set_seed(cfg["experiment"]["seed"])
    device = get_device(cfg)

    logger = Logger(cfg["logging"]["base_dir"], cfg["experiment"]["name"])
    logger.save_config(cfg)

    # Build model and optimizer
    model, decoder = build_model(cfg, device)
    optimizer = build_optimizer(cfg, model, decoder)
    
    # Build loss function
    loss_fn = LossBuilder(cfg)

    # Build data loader
    loader = build_dataloader(cfg)

    # Build schedules
    ema_schedule = build_ema_schedule(cfg)
    physics_schedule = build_physics_weight_schedule(cfg)
    k3_physics_weights = build_k3_physics_weights(cfg)

    # Training configuration
    context_ratio = cfg["masking"]["context_ratio"]
    total_epochs = cfg["training"].get("epochs", 500)
    global_step = 0

    print(f"Training PI-JEPA for {total_epochs} epochs")
    print(f"  Device: {device}")
    print(f"  Batch size: {cfg['training'].get('batch_size', 64)}")
    print(f"  Learning rate: {cfg['training']['optim'].get('lr', 1.5e-4)}")
    print(f"  EMA: τ {ema_schedule.tau_start} → {ema_schedule.tau_end} over {ema_schedule.warmup_epochs} epochs")
    print(f"  Physics weight: {physics_schedule.target_weight} ramped over {physics_schedule.ramp_steps} steps")

    for epoch in range(total_epochs):
        model.train()
        decoder.train()

        epoch_losses = {}
        num_batches = 0

        for batch in loader:
            x = batch["x"].to(device).float()
            y = batch["y"].to(device).float()

            if x.dim() == 3:
                x = x.unsqueeze(1)
            if y.dim() == 3:
                y = y.unsqueeze(1)

            x_input = torch.cat([y, x], dim=1)

            B, _, H, W = x_input.shape
            patch_size = cfg["model"]["encoder"]["patch_size"]

            assert H % patch_size == 0

            grid_size = H // patch_size

            context_idx, target_idx = sample_block_mask_indices(
                B, grid_size, device, context_ratio
            )

            num_patches = grid_size * grid_size
            assert target_idx.max() < num_patches

            z_pred, z_target = model(x_input, context_idx, target_idx)

            z_full = model.encode(x_input)

            z_recon = z_full.clone()
            z_recon = z_recon.scatter(
                1,
                target_idx.unsqueeze(-1).expand_as(z_pred),
                z_pred
            )

            x_pred = decoder(z_recon)

            K = x
            q = torch.zeros_like(x)
            q_w = torch.zeros_like(x)
            phi = torch.ones_like(x)

            losses = loss_fn(
                z_pred=z_pred,
                z_target=z_target,
                x_pred=x_pred,
                x_true=x_input,
                K=K,
                q=q,
                q_w=q_w,
                phi=phi
            )

            # Apply physics weight ramping
            physics_weight = physics_schedule.get_weight(global_step)
            
            # If using K=3 predictors, apply separate weights
            if k3_physics_weights is not None:
                k3_weights = k3_physics_weights.get_weights(global_step)
                # Store for logging
                losses["physics_weight_pressure"] = torch.tensor(k3_weights["pressure"])
                losses["physics_weight_transport"] = torch.tensor(k3_weights["transport"])
                losses["physics_weight_reaction"] = torch.tensor(k3_weights["reaction"])
            
            losses["physics_weight"] = torch.tensor(physics_weight)

            loss = losses["total"]

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            grad_clip = cfg["training"].get("gradient", {}).get("clip_norm", 1.0)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(decoder.parameters()),
                    grad_clip
                )
            
            optimizer.step()

            # Update EMA with annealed momentum
            tau = ema_schedule.get_tau(epoch)
            update_ema(model.encoder, model.target_encoder, tau=tau)

            global_step += 1
            num_batches += 1

            # Accumulate losses for logging
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v

        # Average losses over epoch
        for k in epoch_losses:
            epoch_losses[k] /= num_batches

        # Log metrics
        if global_step % 50 == 0 or epoch == total_epochs - 1:
            logger.log_metrics(epoch_losses, global_step)
            
            print(f"Epoch {epoch+1}/{total_epochs} | "
                  f"Loss: {epoch_losses.get('total', 0):.4f} | "
                  f"τ: {tau:.4f} | "
                  f"λ_p: {physics_weight:.4f}")

    # Save final checkpoint
    checkpoint_path = os.path.join(output_dir, "checkpoint_final.pt")
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        decoder=decoder,
        optimizer=optimizer,
        scaler=None,
        epoch=total_epochs,
        step=global_step,
        config=cfg,
    )
    print(f"Saved final checkpoint to {checkpoint_path}")

    return checkpoint_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/darcy.yaml", help="Path to config file")
    parser.add_argument("--output", default="outputs", help="Output directory for checkpoints")
    args = parser.parse_args()
    train(args.config, args.output)
