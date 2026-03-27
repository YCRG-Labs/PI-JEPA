"""
Checkpoint utilities for PI-JEPA pretraining and finetuning.

This module provides functions for saving and loading checkpoints with proper
metadata to distinguish between pretraining and finetuning phases.

Validates: Requirement 8 (Checkpoint Compatibility)
- AC 8.1: Pretraining checkpoints contain encoder, target encoder, predictor, decoder weights
- AC 8.2: Finetuning loads pretrained checkpoints and initializes Prediction_Head from scratch
- AC 8.3: Finetuning checkpoints contain Prediction_Head weights and metadata
- AC 8.4: Checkpoint format includes metadata indicating pretraining vs finetuning
- AC 8.5: Verify encoder architecture compatibility when loading
"""

import os
from typing import Dict, Any, Optional, List, Union

import torch
import torch.nn as nn
import random
import numpy as np


# ============================================================================
# Checkpoint Type Constants
# ============================================================================

CHECKPOINT_TYPE_PRETRAINING = "pretraining"
CHECKPOINT_TYPE_FINETUNING = "finetuning"
CHECKPOINT_TYPE_LEGACY = "legacy"


# ============================================================================
# Legacy Checkpoint Functions (Backward Compatibility)
# ============================================================================

def save_checkpoint(
    path,
    model,
    decoder,
    optimizer,
    scaler,
    epoch,
    step,
    config,
    extra=None
):
    """
    Save checkpoint in legacy format (backward compatible).
    
    This function is maintained for backward compatibility with existing code.
    For new code, use save_pretraining_checkpoint or save_finetuning_checkpoint.
    """
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    checkpoint = {
        "checkpoint_type": CHECKPOINT_TYPE_LEGACY,
        "student_encoder": model.encoder.state_dict(),
        "target_encoder": model.target_encoder.state_dict(),
        "predictors": [p.state_dict() for p in model.predictors],
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "config": config.as_dict() if hasattr(config, "as_dict") else config,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "random": random.getstate()
        }
    }

    if extra is not None:
        checkpoint["extra"] = extra

    tmp_path = path + ".tmp"

    try:
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e


def load_checkpoint(
    path,
    model,
    decoder,
    optimizer=None,
    scaler=None,
    map_location="cpu"
):
    """
    Load checkpoint in legacy format (backward compatible).
    
    This function is maintained for backward compatibility with existing code.
    For new code, use load_pretrained_encoder or load_finetuning_checkpoint.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model.encoder.load_state_dict(checkpoint["student_encoder"])
    model.target_encoder.load_state_dict(checkpoint["target_encoder"])

    for p, state in zip(model.predictors, checkpoint["predictors"]):
        p.load_state_dict(state)

    decoder.load_state_dict(checkpoint["decoder"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scaler is not None and checkpoint["scaler"] is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    torch.set_rng_state(checkpoint["rng_state"]["torch"])

    if torch.cuda.is_available() and checkpoint["rng_state"]["cuda"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["rng_state"]["cuda"])

    np.random.set_state(checkpoint["rng_state"]["numpy"])
    random.setstate(checkpoint["rng_state"]["random"])

    return checkpoint["epoch"], checkpoint["step"], checkpoint.get("extra", None)


# ============================================================================
# Pretraining Checkpoint Functions
# ============================================================================

def save_pretraining_checkpoint(
    path: str,
    model: nn.Module,
    decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    config: Dict[str, Any],
    n_unlabeled: int = 0,
    ema_tau: float = 0.99,
    metrics: Optional[Dict[str, float]] = None,
    scaler: Optional[Any] = None
) -> None:
    """
    Save pretraining checkpoint with proper metadata.
    
    Validates: AC 8.1 - Pretraining checkpoints contain encoder, target encoder,
                        predictor, decoder weights
    Validates: AC 8.4 - Checkpoint format includes metadata indicating pretraining
    
    Args:
        path: Path to save checkpoint
        model: PI-JEPA model with encoder, target_encoder, predictors
        decoder: Decoder module
        optimizer: Optimizer state
        epoch: Current epoch
        step: Current global step
        config: Training configuration
        n_unlabeled: Number of unlabeled samples used for pretraining
        ema_tau: Current EMA momentum value
        metrics: Optional training metrics
        scaler: Optional gradient scaler for mixed precision
    """
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    
    # Extract encoder architecture info for compatibility checking
    encoder_config = _extract_encoder_config(model.encoder)
    
    checkpoint = {
        # Metadata (AC 8.4)
        "checkpoint_type": CHECKPOINT_TYPE_PRETRAINING,
        "version": "1.0",
        
        # Model weights (AC 8.1)
        "encoder_state_dict": model.encoder.state_dict(),
        "target_encoder_state_dict": model.target_encoder.state_dict(),
        "predictor_state_dicts": [p.state_dict() for p in model.predictors],
        "decoder_state_dict": decoder.state_dict(),
        
        # Optimizer state
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        
        # Training state
        "epoch": epoch,
        "step": step,
        "ema_tau": ema_tau,
        
        # Configuration
        "config": config.as_dict() if hasattr(config, "as_dict") else config,
        
        # Pretraining metadata
        "n_unlabeled": n_unlabeled,
        "metrics": metrics or {},
        
        # Encoder architecture for compatibility checking (AC 8.5)
        "encoder_config": encoder_config,
        
        # RNG state for reproducibility
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "random": random.getstate()
        }
    }
    
    _safe_save(checkpoint, path)


def load_pretrained_encoder(
    checkpoint_path: str,
    encoder: nn.Module,
    strict: bool = False,
    map_location: str = "cpu"
) -> nn.Module:
    """
    Load pretrained encoder from checkpoint.
    
    Validates: AC 8.2 - Finetuning loads pretrained checkpoints
    Validates: AC 8.5 - Verify encoder architecture compatibility when loading
    
    Args:
        checkpoint_path: Path to pretraining checkpoint
        encoder: Encoder module to load weights into
        strict: If True, raise error on architecture mismatch
        map_location: Device to load checkpoint to
        
    Returns:
        Encoder with loaded weights
        
    Raises:
        ValueError: If checkpoint type is not pretraining (when strict=True)
        ValueError: If encoder architecture is incompatible (when strict=True)
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    
    # Validate checkpoint type (AC 8.4)
    checkpoint_type = checkpoint.get("checkpoint_type", CHECKPOINT_TYPE_LEGACY)
    
    if checkpoint_type not in [CHECKPOINT_TYPE_PRETRAINING, CHECKPOINT_TYPE_LEGACY]:
        if strict:
            raise ValueError(
                f"Expected pretraining checkpoint, got '{checkpoint_type}'. "
                "Use load_finetuning_checkpoint for finetuning checkpoints."
            )
        else:
            print(f"Warning: Loading from '{checkpoint_type}' checkpoint, expected 'pretraining'")
    
    # Get encoder state dict (handle both new and legacy formats)
    if "encoder_state_dict" in checkpoint:
        encoder_state_dict = checkpoint["encoder_state_dict"]
    elif "student_encoder" in checkpoint:
        # Legacy format
        encoder_state_dict = checkpoint["student_encoder"]
    else:
        raise KeyError("Checkpoint does not contain encoder weights")
    
    # Verify architecture compatibility (AC 8.5)
    if "encoder_config" in checkpoint:
        _verify_encoder_compatibility(
            checkpoint["encoder_config"],
            encoder,
            strict=strict
        )
    
    # Load weights with potential channel adaptation
    if hasattr(encoder, "load_pretrained_weights"):
        encoder.load_pretrained_weights(encoder_state_dict, strict=strict)
    else:
        encoder.load_state_dict(encoder_state_dict, strict=strict)
    
    return encoder


def load_pretraining_checkpoint(
    path: str,
    model: nn.Module,
    decoder: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    map_location: str = "cpu",
    restore_rng: bool = True
) -> Dict[str, Any]:
    """
    Load full pretraining checkpoint including all components.
    
    Args:
        path: Path to checkpoint
        model: PI-JEPA model to load weights into
        decoder: Decoder module to load weights into
        optimizer: Optional optimizer to restore state
        scaler: Optional gradient scaler to restore state
        map_location: Device to load checkpoint to
        restore_rng: Whether to restore RNG state
        
    Returns:
        Dict with checkpoint metadata (epoch, step, config, metrics)
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    
    # Validate checkpoint type
    checkpoint_type = checkpoint.get("checkpoint_type", CHECKPOINT_TYPE_LEGACY)
    if checkpoint_type not in [CHECKPOINT_TYPE_PRETRAINING, CHECKPOINT_TYPE_LEGACY]:
        print(f"Warning: Loading '{checkpoint_type}' checkpoint as pretraining checkpoint")
    
    # Load model weights
    if "encoder_state_dict" in checkpoint:
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        model.target_encoder.load_state_dict(checkpoint["target_encoder_state_dict"])
        for p, state in zip(model.predictors, checkpoint["predictor_state_dicts"]):
            p.load_state_dict(state)
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
    else:
        # Legacy format
        model.encoder.load_state_dict(checkpoint["student_encoder"])
        model.target_encoder.load_state_dict(checkpoint["target_encoder"])
        for p, state in zip(model.predictors, checkpoint["predictors"]):
            p.load_state_dict(state)
        decoder.load_state_dict(checkpoint["decoder"])
    
    # Load optimizer state
    if optimizer is not None:
        optimizer_key = "optimizer_state_dict" if "optimizer_state_dict" in checkpoint else "optimizer"
        optimizer.load_state_dict(checkpoint[optimizer_key])
    
    # Load scaler state
    if scaler is not None:
        scaler_key = "scaler_state_dict" if "scaler_state_dict" in checkpoint else "scaler"
        if checkpoint.get(scaler_key) is not None:
            scaler.load_state_dict(checkpoint[scaler_key])
    
    # Restore RNG state
    if restore_rng and "rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["rng_state"]["torch"])
        if torch.cuda.is_available() and checkpoint["rng_state"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["rng_state"]["cuda"])
        np.random.set_state(checkpoint["rng_state"]["numpy"])
        random.setstate(checkpoint["rng_state"]["random"])
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "ema_tau": checkpoint.get("ema_tau", 0.99),
        "config": checkpoint.get("config", {}),
        "metrics": checkpoint.get("metrics", {}),
        "n_unlabeled": checkpoint.get("n_unlabeled", 0),
        "checkpoint_type": checkpoint_type
    }


# ============================================================================
# Finetuning Checkpoint Functions
# ============================================================================

def save_finetuning_checkpoint(
    path: str,
    encoder: nn.Module,
    prediction_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    config: Dict[str, Any],
    n_labeled: int,
    pretrain_checkpoint_path: Optional[str] = None,
    encoder_frozen: bool = True,
    metrics: Optional[Dict[str, float]] = None,
    test_error: Optional[float] = None
) -> None:
    """
    Save finetuning checkpoint with proper metadata.
    
    Validates: AC 8.3 - Finetuning checkpoints contain Prediction_Head weights and metadata
    Validates: AC 8.4 - Checkpoint format includes metadata indicating finetuning
    
    Args:
        path: Path to save checkpoint
        encoder: Encoder module (may be frozen)
        prediction_head: Prediction head module
        optimizer: Optimizer state
        epoch: Current epoch
        step: Current global step
        config: Training configuration
        n_labeled: Number of labeled samples used for finetuning
        pretrain_checkpoint_path: Path to pretraining checkpoint used
        encoder_frozen: Whether encoder was frozen during finetuning
        metrics: Optional training metrics
        test_error: Optional test set error
    """
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    
    # Extract encoder architecture info
    encoder_config = _extract_encoder_config(encoder)
    
    checkpoint = {
        # Metadata (AC 8.4)
        "checkpoint_type": CHECKPOINT_TYPE_FINETUNING,
        "version": "1.0",
        
        # Model weights (AC 8.3)
        "prediction_head_state_dict": prediction_head.state_dict(),
        "encoder_state_dict": encoder.state_dict(),
        
        # Optimizer state
        "optimizer_state_dict": optimizer.state_dict(),
        
        # Training state
        "epoch": epoch,
        "step": step,
        
        # Configuration
        "config": config.as_dict() if hasattr(config, "as_dict") else config,
        
        # Finetuning metadata (AC 8.3)
        "n_labeled": n_labeled,
        "pretrain_checkpoint_path": pretrain_checkpoint_path,
        "encoder_frozen": encoder_frozen,
        "metrics": metrics or {},
        "test_error": test_error,
        
        # Encoder architecture for compatibility checking (AC 8.5)
        "encoder_config": encoder_config,
        
        # RNG state
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "random": random.getstate()
        }
    }
    
    _safe_save(checkpoint, path)


def load_finetuning_checkpoint(
    path: str,
    encoder: nn.Module,
    prediction_head: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
    strict: bool = False,
    restore_rng: bool = True
) -> Dict[str, Any]:
    """
    Load finetuning checkpoint.
    
    Args:
        path: Path to checkpoint
        encoder: Encoder module to load weights into
        prediction_head: Prediction head module to load weights into
        optimizer: Optional optimizer to restore state
        map_location: Device to load checkpoint to
        strict: If True, raise error on architecture mismatch
        restore_rng: Whether to restore RNG state
        
    Returns:
        Dict with checkpoint metadata
        
    Raises:
        ValueError: If checkpoint type is not finetuning (when strict=True)
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    
    # Validate checkpoint type
    checkpoint_type = checkpoint.get("checkpoint_type", CHECKPOINT_TYPE_LEGACY)
    if checkpoint_type != CHECKPOINT_TYPE_FINETUNING:
        if strict:
            raise ValueError(
                f"Expected finetuning checkpoint, got '{checkpoint_type}'"
            )
        else:
            print(f"Warning: Loading '{checkpoint_type}' checkpoint as finetuning checkpoint")
    
    # Verify encoder compatibility (AC 8.5)
    if "encoder_config" in checkpoint:
        _verify_encoder_compatibility(
            checkpoint["encoder_config"],
            encoder,
            strict=strict
        )
    
    # Load encoder weights
    if hasattr(encoder, "load_pretrained_weights"):
        encoder.load_pretrained_weights(checkpoint["encoder_state_dict"], strict=strict)
    else:
        encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=strict)
    
    # Load prediction head weights
    prediction_head.load_state_dict(checkpoint["prediction_head_state_dict"])
    
    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Restore RNG state
    if restore_rng and "rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["rng_state"]["torch"])
        if torch.cuda.is_available() and checkpoint["rng_state"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["rng_state"]["cuda"])
        np.random.set_state(checkpoint["rng_state"]["numpy"])
        random.setstate(checkpoint["rng_state"]["random"])
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "config": checkpoint.get("config", {}),
        "n_labeled": checkpoint.get("n_labeled", 0),
        "pretrain_checkpoint_path": checkpoint.get("pretrain_checkpoint_path"),
        "encoder_frozen": checkpoint.get("encoder_frozen", True),
        "metrics": checkpoint.get("metrics", {}),
        "test_error": checkpoint.get("test_error"),
        "checkpoint_type": checkpoint_type
    }


# ============================================================================
# Checkpoint Validation Functions
# ============================================================================

def validate_checkpoint_type(
    checkpoint_path: str,
    expected_type: str,
    map_location: str = "cpu"
) -> bool:
    """
    Validate that a checkpoint is of the expected type.
    
    Validates: AC 8.4 - Checkpoint format includes metadata indicating type
    
    Args:
        checkpoint_path: Path to checkpoint
        expected_type: Expected checkpoint type ('pretraining' or 'finetuning')
        map_location: Device to load checkpoint to
        
    Returns:
        True if checkpoint type matches expected type
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    checkpoint_type = checkpoint.get("checkpoint_type", CHECKPOINT_TYPE_LEGACY)
    
    # Legacy checkpoints are treated as pretraining checkpoints
    if expected_type == CHECKPOINT_TYPE_PRETRAINING and checkpoint_type == CHECKPOINT_TYPE_LEGACY:
        return True
    
    return checkpoint_type == expected_type


def get_checkpoint_info(
    checkpoint_path: str,
    map_location: str = "cpu"
) -> Dict[str, Any]:
    """
    Get metadata information from a checkpoint without loading weights.
    
    Args:
        checkpoint_path: Path to checkpoint
        map_location: Device to load checkpoint to
        
    Returns:
        Dict with checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    
    info = {
        "checkpoint_type": checkpoint.get("checkpoint_type", CHECKPOINT_TYPE_LEGACY),
        "version": checkpoint.get("version", "unknown"),
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
    }
    
    # Add type-specific info
    if info["checkpoint_type"] == CHECKPOINT_TYPE_PRETRAINING:
        info["n_unlabeled"] = checkpoint.get("n_unlabeled", 0)
        info["ema_tau"] = checkpoint.get("ema_tau", 0.99)
    elif info["checkpoint_type"] == CHECKPOINT_TYPE_FINETUNING:
        info["n_labeled"] = checkpoint.get("n_labeled", 0)
        info["encoder_frozen"] = checkpoint.get("encoder_frozen", True)
        info["pretrain_checkpoint_path"] = checkpoint.get("pretrain_checkpoint_path")
        info["test_error"] = checkpoint.get("test_error")
    
    # Add encoder config if available
    if "encoder_config" in checkpoint:
        info["encoder_config"] = checkpoint["encoder_config"]
    
    # Add metrics if available
    if "metrics" in checkpoint:
        info["metrics"] = checkpoint["metrics"]
    
    return info


# ============================================================================
# Helper Functions
# ============================================================================

def _extract_encoder_config(encoder: nn.Module) -> Dict[str, Any]:
    """
    Extract encoder configuration for compatibility checking.
    
    Args:
        encoder: Encoder module
        
    Returns:
        Dict with encoder configuration
    """
    config = {}
    
    # Extract common attributes
    if hasattr(encoder, "embed_dim"):
        config["embed_dim"] = encoder.embed_dim
    if hasattr(encoder, "patch_size"):
        config["patch_size"] = encoder.patch_size
    if hasattr(encoder, "in_channels"):
        config["in_channels"] = encoder.in_channels
    if hasattr(encoder, "blocks"):
        config["depth"] = len(encoder.blocks)
    
    # Extract from patch embedding if available
    if hasattr(encoder, "patch_embed") and hasattr(encoder.patch_embed, "proj"):
        proj = encoder.patch_embed.proj
        config["in_channels"] = proj.in_channels
        config["embed_dim"] = proj.out_channels
        config["patch_size"] = proj.kernel_size[0] if isinstance(proj.kernel_size, tuple) else proj.kernel_size
    
    return config


def _verify_encoder_compatibility(
    checkpoint_config: Dict[str, Any],
    encoder: nn.Module,
    strict: bool = False
) -> None:
    """
    Verify encoder architecture compatibility with checkpoint.
    
    Validates: AC 8.5 - Verify encoder architecture compatibility when loading
    
    Args:
        checkpoint_config: Encoder config from checkpoint
        encoder: Encoder module to verify against
        strict: If True, raise error on mismatch
        
    Raises:
        ValueError: If architectures are incompatible (when strict=True)
    """
    model_config = _extract_encoder_config(encoder)
    
    # Check critical parameters
    critical_params = ["embed_dim", "patch_size", "depth"]
    
    for param in critical_params:
        if param in checkpoint_config and param in model_config:
            if checkpoint_config[param] != model_config[param]:
                msg = (
                    f"Encoder architecture mismatch: {param} is {checkpoint_config[param]} "
                    f"in checkpoint but {model_config[param]} in model"
                )
                if strict:
                    raise ValueError(msg)
                else:
                    print(f"Warning: {msg}")
    
    # Check in_channels (can be adapted)
    if "in_channels" in checkpoint_config and "in_channels" in model_config:
        if checkpoint_config["in_channels"] != model_config["in_channels"]:
            msg = (
                f"Input channel mismatch: checkpoint has {checkpoint_config['in_channels']} channels, "
                f"model expects {model_config['in_channels']} channels. "
                "Weights will be adapted if possible."
            )
            print(f"Info: {msg}")


def _safe_save(checkpoint: Dict[str, Any], path: str) -> None:
    """
    Safely save checkpoint with atomic write.
    
    Args:
        checkpoint: Checkpoint dictionary
        path: Path to save checkpoint
    """
    tmp_path = path + ".tmp"
    
    try:
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e
