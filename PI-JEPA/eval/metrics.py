"""Evaluation metrics for PI-JEPA."""

import torch
from typing import Dict, List, Optional, Union, Any


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error."""
    return torch.mean((pred - target) ** 2)


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root mean squared error."""
    return torch.sqrt(mse(pred, target))


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error."""
    return torch.mean(torch.abs(pred - target))


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Relative L2 error: ||û - u||_2 / ||u||_2."""
    num = torch.sum((pred - target) ** 2)
    denom = torch.sum(target ** 2) + eps
    return torch.sqrt(num / denom)


def relative_l2_per_field(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    eps: float = 1e-8
) -> Dict[str, torch.Tensor]:
    """Relative L2 error per channel."""
    # Get number of channels
    C = pred.shape[1]
    
    # Default channel names
    if C == 2:
        channel_names = ['pressure', 'saturation']
    else:
        channel_names = [f'field_{i}' for i in range(C)]
    
    result = {}
    for i, name in enumerate(channel_names):
        pred_field = pred[:, i]
        target_field = target[:, i]
        
        # Flatten spatial dimensions
        pred_flat = pred_field.reshape(pred_field.shape[0], -1)
        target_flat = target_field.reshape(target_field.shape[0], -1)
        
        # Compute per-sample relative L2, then average
        num = torch.sum((pred_flat - target_flat) ** 2, dim=1)
        denom = torch.sum(target_flat ** 2, dim=1) + eps
        rel_l2 = torch.sqrt(num / denom)
        
        result[name] = torch.mean(rel_l2)
    
    return result


def relative_l1(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Relative L1 error."""
    num = torch.sum(torch.abs(pred - target))
    denom = torch.sum(torch.abs(target)) + eps
    return num / denom


def max_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Maximum absolute error."""
    return torch.max(torch.abs(pred - target))


def per_channel_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE per channel."""
    B, C = pred.shape[:2]
    pred = pred.view(B, C, -1)
    target = target.view(B, C, -1)
    return torch.mean((pred - target) ** 2, dim=2)


def per_channel_mse_named(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    channel_names: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """MSE per channel with named outputs."""
    B, C = pred.shape[:2]
    
    # Default channel names
    if channel_names is None:
        if C == 2:
            channel_names = ['pressure', 'saturation']
        else:
            channel_names = [f'channel_{i}' for i in range(C)]
    
    if len(channel_names) != C:
        raise ValueError(f"Number of channel names ({len(channel_names)}) must match number of channels ({C})")
    
    result = {}
    for i, name in enumerate(channel_names):
        pred_channel = pred[:, i]
        target_channel = target[:, i]
        result[name] = torch.mean((pred_channel - target_channel) ** 2)
    
    return result


def rollout_mse(pred_seq: torch.Tensor, target_seq: torch.Tensor) -> torch.Tensor:
    """MSE per timestep in rollout."""
    return torch.mean((pred_seq - target_seq) ** 2, dim=(2, 3, 4))


def rollout_rmse(pred_seq: torch.Tensor, target_seq: torch.Tensor) -> torch.Tensor:
    """RMSE per timestep in rollout."""
    return torch.sqrt(rollout_mse(pred_seq, target_seq))


def rollout_mae(pred_seq: torch.Tensor, target_seq: torch.Tensor) -> torch.Tensor:
    """MAE per timestep in rollout."""
    return torch.mean(torch.abs(pred_seq - target_seq), dim=(2, 3, 4))


def rollout_relative_l2(
    pred_seq: torch.Tensor, 
    target_seq: torch.Tensor, 
    eps: float = 1e-8
) -> torch.Tensor:
    """Relative L2 error per timestep in rollout."""
    num = torch.sum((pred_seq - target_seq) ** 2, dim=(2, 3, 4))
    denom = torch.sum(target_seq ** 2, dim=(2, 3, 4)) + eps
    return torch.sqrt(num / denom)


def rollout_max_error(pred_seq: torch.Tensor, target_seq: torch.Tensor) -> torch.Tensor:
    """Maximum error per timestep in rollout."""
    return torch.amax(torch.abs(pred_seq - target_seq), dim=(2, 3, 4))


def rollout_cumulative_error(errors_per_horizon: Dict[int, float]) -> Dict[str, Any]:
    """Cumulative error across rollout horizons."""
    if not errors_per_horizon:
        return {
            'horizons': [],
            'errors': [],
            'cumulative': [],
            'mean_error': 0.0,
            'final_error': 0.0,
            'error_growth_rate': 0.0
        }
    
    # Sort by horizon
    sorted_horizons = sorted(errors_per_horizon.keys())
    errors = [errors_per_horizon[h] for h in sorted_horizons]
    
    # Compute cumulative errors
    cumulative = []
    running_sum = 0.0
    for e in errors:
        running_sum += e
        cumulative.append(running_sum)
    
    # Compute statistics
    mean_error = sum(errors) / len(errors)
    final_error = errors[-1]
    
    # Error growth rate (average change per horizon step)
    if len(sorted_horizons) > 1:
        total_horizon_span = sorted_horizons[-1] - sorted_horizons[0]
        error_span = errors[-1] - errors[0]
        error_growth_rate = error_span / total_horizon_span if total_horizon_span > 0 else 0.0
    else:
        error_growth_rate = 0.0
    
    return {
        'horizons': sorted_horizons,
        'errors': errors,
        'cumulative': cumulative,
        'mean_error': mean_error,
        'final_error': final_error,
        'error_growth_rate': error_growth_rate
    }


def temporal_consistency(pred_seq: torch.Tensor) -> torch.Tensor:
    """Temporal consistency (smoothness) of predictions."""
    diff = pred_seq[:, 1:] - pred_seq[:, :-1]
    return torch.mean(diff ** 2)


def energy(pred: torch.Tensor) -> torch.Tensor:
    """Energy (L2 norm squared) of predictions."""
    return torch.sum(pred ** 2, dim=(1, 2, 3))


def rollout_energy_drift(pred_seq: torch.Tensor) -> torch.Tensor:
    """Energy drift over rollout."""
    e0 = energy(pred_seq[:, 0])
    et = energy(pred_seq[:, -1])
    return torch.mean(torch.abs(et - e0) / (e0 + 1e-8))


def physics_residual_metric(residual: torch.Tensor) -> torch.Tensor:
    """Mean squared physics residual."""
    return torch.mean(residual ** 2)


def pde_residual_mse(
    pred: torch.Tensor,
    physics_module: Any,
    dx: float = 1.0,
    dy: float = 1.0,
    dt: float = 1.0,
    **kwargs
) -> torch.Tensor:
    """MSE of PDE residual on predicted fields."""
    # Handle sequence input (B, T, C, H, W) -> compute residual for each timestep
    if pred.dim() == 5:
        B, T, C, H, W = pred.shape
        residuals = []
        for t in range(T):
            if hasattr(physics_module, 'compute_residual'):
                res = physics_module.compute_residual(pred[:, t], dx=dx, dy=dy, dt=dt, **kwargs)
            elif hasattr(physics_module, 'total_residual'):
                res = physics_module.total_residual(pred[:, t], dx=dx, dy=dy, dt=dt, **kwargs)
            else:
                raise AttributeError(
                    "physics_module must have 'compute_residual' or 'total_residual' method"
                )
            residuals.append(res)
        residual = torch.stack(residuals, dim=1)
    else:
        # Single frame input (B, C, H, W)
        if hasattr(physics_module, 'compute_residual'):
            residual = physics_module.compute_residual(pred, dx=dx, dy=dy, dt=dt, **kwargs)
        elif hasattr(physics_module, 'total_residual'):
            residual = physics_module.total_residual(pred, dx=dx, dy=dy, dt=dt, **kwargs)
        else:
            raise AttributeError(
                "physics_module must have 'compute_residual' or 'total_residual' method"
            )
    
    return torch.mean(residual ** 2)


def compute_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relative L2 norm."""
    return torch.norm(pred - target) / torch.norm(target)


def ood_relative_l2(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    eps: float = 1e-8
) -> torch.Tensor:
    """Relative L2 error for OOD evaluation."""
    num = torch.sum((pred - target) ** 2)
    denom = torch.sum(target ** 2) + eps
    return torch.sqrt(num / denom)


def data_efficiency_curve(errors_by_n_labeled: Dict[int, float]) -> Dict[str, Any]:
    """Data efficiency curve for error vs N_l plots."""
    import math
    
    if not errors_by_n_labeled:
        return {
            'n_labeled': [],
            'errors': [],
            'log_n_labeled': [],
            'improvement_rate': 0.0,
            'min_error': 0.0,
            'max_error': 0.0,
            'efficiency_score': 0.0
        }
    
    # Sort by number of labeled samples
    sorted_n = sorted(errors_by_n_labeled.keys())
    errors = [errors_by_n_labeled[n] for n in sorted_n]
    
    # Compute log-scaled values for plotting
    log_n = [math.log10(n) if n > 0 else 0.0 for n in sorted_n]
    
    # Compute improvement rate (error reduction per doubling of data)
    # Using linear regression on log-log scale
    if len(sorted_n) > 1:
        log_errors = [math.log10(e) if e > 0 else -10.0 for e in errors]
        
        # Simple linear regression: log(error) = a * log(n) + b
        n_points = len(sorted_n)
        sum_x = sum(log_n)
        sum_y = sum(log_errors)
        sum_xy = sum(x * y for x, y in zip(log_n, log_errors))
        sum_x2 = sum(x ** 2 for x in log_n)
        
        denom = n_points * sum_x2 - sum_x ** 2
        if abs(denom) > 1e-10:
            slope = (n_points * sum_xy - sum_x * sum_y) / denom
            # Improvement rate: how much error decreases when data doubles
            # If error = c * n^slope, then doubling n gives error * 2^slope
            improvement_rate = 1.0 - (2.0 ** slope)
        else:
            improvement_rate = 0.0
    else:
        improvement_rate = 0.0
    
    min_error = min(errors)
    max_error = max(errors)
    
    # Efficiency score: how much error reduction per unit of data
    # Higher is better (more efficient use of labeled data)
    if max_error > 0 and len(sorted_n) > 1:
        error_reduction = (max_error - min_error) / max_error
        data_ratio = sorted_n[-1] / sorted_n[0]
        efficiency_score = error_reduction / math.log10(data_ratio) if data_ratio > 1 else 0.0
    else:
        efficiency_score = 0.0
    
    return {
        'n_labeled': sorted_n,
        'errors': errors,
        'log_n_labeled': log_n,
        'improvement_rate': improvement_rate,
        'min_error': min_error,
        'max_error': max_error,
        'efficiency_score': efficiency_score
    }
