"""
Physics-Informed JEPA Pretraining Objectives

This module implements novel pretraining objectives that combine:
1. Standard JEPA (predict masked patches in latent space)
2. Physics-informed objectives (learn PDE structure)

Key innovations:
- Operator learning objective: predict solution patches from coefficient patches
- PDE residual minimization in latent space
- Multi-task pretraining combining reconstruction and physics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class PhysicsInformedJEPALoss(nn.Module):
    """
    Combined JEPA + Physics loss for self-supervised pretraining.
    
    This loss combines:
    1. JEPA loss: predict masked coefficient patches from visible ones
    2. Operator loss: predict solution from coefficient (when solutions available)
    3. Physics residual: enforce PDE constraints in latent space
    
    The key insight is that we can use UNLABELED coefficient fields for JEPA,
    but also leverage any available (coefficient, solution) pairs for operator learning.
    """
    
    def __init__(
        self,
        jepa_weight: float = 1.0,
        operator_weight: float = 1.0,
        physics_weight: float = 0.1,
        normalize_embeddings: bool = True
    ):
        super().__init__()
        self.jepa_weight = jepa_weight
        self.operator_weight = operator_weight
        self.physics_weight = physics_weight
        self.normalize_embeddings = normalize_embeddings
    
    def jepa_loss(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Standard JEPA loss: predict masked patches in latent space.
        
        Args:
            z_pred: (B, N_target, D) predicted embeddings for masked patches
            z_target: (B, N_target, D) target embeddings (from EMA encoder)
        """
        if self.normalize_embeddings:
            z_pred = F.normalize(z_pred, dim=-1)
            z_target = F.normalize(z_target, dim=-1)
        
        return F.mse_loss(z_pred, z_target.detach())
    
    def operator_loss(
        self,
        z_coeff: torch.Tensor,
        z_solution: torch.Tensor,
        operator_head: nn.Module
    ) -> torch.Tensor:
        """
        Operator learning loss: predict solution embeddings from coefficient embeddings.
        
        This teaches the encoder to capture the coefficient -> solution mapping.
        
        Args:
            z_coeff: (B, N, D) coefficient field embeddings
            z_solution: (B, N, D) solution field embeddings (target)
            operator_head: Module that maps coefficient embeddings to solution embeddings
        """
        z_pred = operator_head(z_coeff)
        
        if self.normalize_embeddings:
            z_pred = F.normalize(z_pred, dim=-1)
            z_solution = F.normalize(z_solution, dim=-1)
        
        return F.mse_loss(z_pred, z_solution.detach())
    
    def forward(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
        z_coeff: Optional[torch.Tensor] = None,
        z_solution: Optional[torch.Tensor] = None,
        operator_head: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            z_pred: Predicted masked patch embeddings
            z_target: Target masked patch embeddings
            z_coeff: Full coefficient embeddings (optional, for operator loss)
            z_solution: Full solution embeddings (optional, for operator loss)
            operator_head: Operator mapping module (optional)
        
        Returns:
            Dict with individual losses and total
        """
        losses = {}
        
        # JEPA loss (always computed)
        losses['jepa'] = self.jepa_loss(z_pred, z_target)
        
        total = self.jepa_weight * losses['jepa']
        
        # Operator loss (if solution embeddings available)
        if z_coeff is not None and z_solution is not None and operator_head is not None:
            losses['operator'] = self.operator_loss(z_coeff, z_solution, operator_head)
            total = total + self.operator_weight * losses['operator']
        
        losses['total'] = total
        return losses


class OperatorHead(nn.Module):
    """
    Lightweight head that maps coefficient embeddings to solution embeddings.
    
    This is used during pretraining to learn the operator mapping in latent space.
    """
    
    def __init__(
        self,
        embed_dim: int = 384,
        hidden_dim: int = 512,
        n_layers: int = 2
    ):
        super().__init__()
        
        layers = []
        in_dim = embed_dim
        
        for i in range(n_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, embed_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, N, D) coefficient embeddings
        Returns:
            (B, N, D) predicted solution embeddings
        """
        return self.net(z)


class DarcyLatentPhysics(nn.Module):
    """
    Darcy flow physics constraints in latent space.
    
    Instead of computing PDE residuals in physical space, we learn
    physics-consistent relationships between latent patch embeddings.
    
    For Darcy flow: -∇·(K∇p) = f
    
    In latent space, neighboring patches should have consistent gradients
    based on the permeability structure.
    """
    
    def __init__(
        self,
        embed_dim: int = 384,
        grid_size: int = 8
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        
        # Learn latent gradient operator
        self.grad_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Learn latent divergence operator
        self.div_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1)
        )
    
    def forward(
        self,
        z_coeff: torch.Tensor,
        z_solution: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute latent physics residual.
        
        Args:
            z_coeff: (B, N, D) coefficient embeddings
            z_solution: (B, N, D) solution embeddings
            
        Returns:
            Scalar physics loss
        """
        B, N, D = z_coeff.shape
        G = self.grid_size
        
        # Reshape to grid
        z_coeff = z_coeff.view(B, G, G, D)
        z_solution = z_solution.view(B, G, G, D)
        
        # Compute latent gradients (finite differences in embedding space)
        # x-direction
        z_coeff_x = torch.cat([z_coeff[:, :, 1:], z_coeff[:, :, -1:]], dim=2)
        z_sol_x = torch.cat([z_solution[:, :, 1:], z_solution[:, :, -1:]], dim=2)
        
        grad_x_input = torch.cat([z_coeff, z_sol_x - z_solution], dim=-1)
        flux_x = self.grad_net(grad_x_input.view(B, -1, D * 2))
        
        # y-direction
        z_coeff_y = torch.cat([z_coeff[:, 1:], z_coeff[:, -1:]], dim=1)
        z_sol_y = torch.cat([z_solution[:, 1:], z_solution[:, -1:]], dim=1)
        
        grad_y_input = torch.cat([z_coeff, z_sol_y - z_solution], dim=-1)
        flux_y = self.grad_net(grad_y_input.view(B, -1, D * 2))
        
        # Divergence should be approximately constant (source term)
        div_input = torch.cat([flux_x, flux_y], dim=-1)
        divergence = self.div_net(div_input)
        
        # Physics loss: divergence should be consistent across domain
        div_mean = divergence.mean(dim=1, keepdim=True)
        physics_loss = F.mse_loss(divergence, div_mean.expand_as(divergence))
        
        return physics_loss


class ContrastiveOperatorLoss(nn.Module):
    """
    Contrastive loss for operator learning.
    
    Positive pairs: (coefficient_i, solution_i) from same sample
    Negative pairs: (coefficient_i, solution_j) from different samples
    
    This encourages the encoder to learn representations where
    matching coefficient-solution pairs are close in latent space.
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        embed_dim: int = 384
    ):
        super().__init__()
        self.temperature = temperature
        
        # Projection heads for contrastive learning
        self.coeff_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 128)
        )
        self.solution_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 128)
        )
    
    def forward(
        self,
        z_coeff: torch.Tensor,
        z_solution: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            z_coeff: (B, N, D) coefficient embeddings
            z_solution: (B, N, D) solution embeddings
            
        Returns:
            Contrastive loss
        """
        B = z_coeff.shape[0]
        
        # Global pool to get sample-level representations
        z_coeff_global = z_coeff.mean(dim=1)  # (B, D)
        z_solution_global = z_solution.mean(dim=1)  # (B, D)
        
        # Project
        z_c = F.normalize(self.coeff_proj(z_coeff_global), dim=-1)
        z_s = F.normalize(self.solution_proj(z_solution_global), dim=-1)
        
        # Compute similarity matrix
        sim = torch.mm(z_c, z_s.t()) / self.temperature  # (B, B)
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(B, device=z_coeff.device)
        
        # Cross-entropy loss (both directions)
        loss_c2s = F.cross_entropy(sim, labels)
        loss_s2c = F.cross_entropy(sim.t(), labels)
        
        return (loss_c2s + loss_s2c) / 2
