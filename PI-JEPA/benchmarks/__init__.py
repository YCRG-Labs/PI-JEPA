"""
Benchmark Models for PI-JEPA Comparison

This module provides implementations of baseline models for comparing
against PI-JEPA, including:
- FNO: Fourier Neural Operator
- PINO: Physics-Informed Neural Operator
- U-FNO: U-Net-augmented FNO
- DeepONet: Deep Operator Network
- U-DeepONet: U-Net enhanced DeepONet
- PINN: Physics-Informed Neural Network
- GeoFNO: Geometry-aware FNO (requires neuralop)
- UNet: Standard U-Net
- PILatentNO: Physics-Informed Latent Neural Operator

Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6
"""

from .fno import FNOWrapper
from .pino import PINOWrapper
from .deeponet import DeepONetWrapper, UDeepONetWrapper
from .pinn import PINNWrapper
from .pi_latent_no import PILatentNOWrapper
from .unet import UNetWrapper
from .ufno import UFNOWrapper, UFNO
from .utils import (
    BenchmarkConfig,
    BenchmarkTrainer,
    set_seed,
    create_data_splits,
    create_data_loaders,
    run_benchmark_comparison,
    create_benchmark_suite,
    save_benchmark_results,
    load_benchmark_results,
)

# Lazy import for GeoFNO (requires neuralop)
def get_geo_fno_wrapper():
    from .geo_fno import GeoFNOWrapper
    return GeoFNOWrapper

__all__ = [
    # Model wrappers
    "FNOWrapper",
    "get_geo_fno_wrapper",
    "PINOWrapper",
    "DeepONetWrapper",
    "UDeepONetWrapper",
    "PINNWrapper",
    "PILatentNOWrapper",
    "UNetWrapper",
    "UFNOWrapper",
    "UFNO",
    # Utilities
    "BenchmarkConfig",
    "BenchmarkTrainer",
    "set_seed",
    "create_data_splits",
    "create_data_loaders",
    "run_benchmark_comparison",
    "create_benchmark_suite",
    "save_benchmark_results",
    "load_benchmark_results",
]
