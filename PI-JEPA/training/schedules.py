"""Training schedules for PI-JEPA."""

import math
from typing import Dict, Optional


class EMAMomentumSchedule:
    """EMA momentum annealing schedule."""
    
    def __init__(
        self,
        tau_start: float = 0.99,
        tau_end: float = 0.999,
        warmup_fraction: float = 0.1,
        total_epochs: int = 500
    ):
        if tau_start < 0 or tau_start > 1:
            raise ValueError(f"tau_start must be in [0, 1], got {tau_start}")
        if tau_end < 0 or tau_end > 1:
            raise ValueError(f"tau_end must be in [0, 1], got {tau_end}")
        if warmup_fraction < 0 or warmup_fraction > 1:
            raise ValueError(f"warmup_fraction must be in [0, 1], got {warmup_fraction}")
        if total_epochs < 1:
            raise ValueError(f"total_epochs must be >= 1, got {total_epochs}")
        
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.warmup_fraction = warmup_fraction
        self.total_epochs = total_epochs
        self.warmup_epochs = int(total_epochs * warmup_fraction)
    
    def get_tau(self, epoch: int) -> float:
        """Get EMA momentum τ for the given epoch."""
        if epoch < 0:
            raise ValueError(f"epoch must be >= 0, got {epoch}")
        
        if self.warmup_epochs == 0:
            return self.tau_end
        
        if epoch >= self.warmup_epochs:
            return self.tau_end
        
        t = epoch / self.warmup_epochs
        tau = self.tau_start + (self.tau_end - self.tau_start) * (1 - math.cos(math.pi * t)) / 2
        return tau


class PhysicsWeightSchedule:
    """Physics weight ramping schedule."""
    
    def __init__(self, target_weight: float = 0.1, ramp_steps: int = 200):
        if target_weight < 0:
            raise ValueError(f"target_weight must be >= 0, got {target_weight}")
        if ramp_steps < 0:
            raise ValueError(f"ramp_steps must be >= 0, got {ramp_steps}")
        
        self.target_weight = target_weight
        self.ramp_steps = ramp_steps
    
    def get_weight(self, step: int) -> float:
        """Get physics weight for the given training step."""
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
        
        if self.ramp_steps <= 0:
            return self.target_weight
        
        if step >= self.ramp_steps:
            return self.target_weight
        
        return self.target_weight * (step / self.ramp_steps)


class K3PhysicsWeightManager:
    """Manages separate physics weights for K=3 predictors."""
    
    def __init__(
        self,
        pressure_weight: float = 0.1,
        transport_weight: float = 0.1,
        reaction_weight: float = 0.1,
        ramp_steps: int = 200
    ):
        self.pressure_schedule = PhysicsWeightSchedule(pressure_weight, ramp_steps)
        self.transport_schedule = PhysicsWeightSchedule(transport_weight, ramp_steps)
        self.reaction_schedule = PhysicsWeightSchedule(reaction_weight, ramp_steps)
        
        self.pressure_weight = pressure_weight
        self.transport_weight = transport_weight
        self.reaction_weight = reaction_weight
        self.ramp_steps = ramp_steps
    
    def get_weights(self, step: int) -> Dict[str, float]:
        return {
            'pressure': self.pressure_schedule.get_weight(step),
            'transport': self.transport_schedule.get_weight(step),
            'reaction': self.reaction_schedule.get_weight(step)
        }
    
    def get_pressure_weight(self, step: int) -> float:
        return self.pressure_schedule.get_weight(step)
    
    def get_transport_weight(self, step: int) -> float:
        return self.transport_schedule.get_weight(step)
    
    def get_reaction_weight(self, step: int) -> float:
        return self.reaction_schedule.get_weight(step)


def build_ema_schedule(config: dict) -> EMAMomentumSchedule:
    """Build EMA momentum schedule from config."""
    ema_cfg = config.get("ema", {}).get("schedule", {})
    training_cfg = config.get("training", {})
    
    return EMAMomentumSchedule(
        tau_start=ema_cfg.get("tau_start", 0.99),
        tau_end=ema_cfg.get("tau_end", 0.999),
        warmup_fraction=ema_cfg.get("warmup_fraction", 0.1),
        total_epochs=training_cfg.get("epochs", 500)
    )


def build_physics_weight_schedule(config: dict) -> PhysicsWeightSchedule:
    """Build physics weight ramping schedule from config."""
    physics_cfg = config.get("loss", {}).get("physics", {})
    
    return PhysicsWeightSchedule(
        target_weight=physics_cfg.get("weight", 0.1),
        ramp_steps=physics_cfg.get("ramp_steps", 200)
    )


def build_k3_physics_weights(config: dict) -> Optional[K3PhysicsWeightManager]:
    """Build K=3 physics weight manager if configured."""
    model_cfg = config.get("model", {})
    num_predictors = model_cfg.get("num_predictors", 2)
    
    if num_predictors != 3:
        return None
    
    physics_cfg = config.get("loss", {}).get("physics", {})
    weights_per_residual = physics_cfg.get("weights_per_residual", {})
    ramp_steps = physics_cfg.get("ramp_steps", 200)
    default_weight = physics_cfg.get("weight", 0.1)
    
    return K3PhysicsWeightManager(
        pressure_weight=weights_per_residual.get("pressure", default_weight),
        transport_weight=weights_per_residual.get("transport", default_weight),
        reaction_weight=weights_per_residual.get("reaction", default_weight),
        ramp_steps=ramp_steps
    )
