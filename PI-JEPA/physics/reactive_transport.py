"""Reactive transport physics module (K=3 operator splitting)."""

import torch
import torch.nn.functional as F
from typing import Optional

from .darcy import grad_x, grad_y, divergence


class ReactiveTransportPhysics:
    """K=3 operator splitting for advection-diffusion-reaction systems."""
    
    def __init__(
        self,
        n_species: int,
        stoichiometry: torch.Tensor,
        peclet: float = 1.0,
        damkohler: float = 1.0,
        dx: float = 1.0,
        dy: float = 1.0,
        dt: float = 1.0
    ):
        if n_species <= 0:
            raise ValueError(f"n_species must be positive, got {n_species}")
        if stoichiometry.dim() != 2:
            raise ValueError(f"stoichiometry must be 2D tensor, got {stoichiometry.dim()}D")
        if stoichiometry.shape[1] != n_species:
            raise ValueError(f"stoichiometry columns ({stoichiometry.shape[1]}) must match n_species ({n_species})")
        
        self.n_species = n_species
        self.stoichiometry = stoichiometry
        self.n_reactions = stoichiometry.shape[0]
        self.peclet = peclet
        self.damkohler = damkohler
        self.dx = dx
        self.dy = dy
        self.dt = dt
    
    def reaction_residual(
        self,
        c_pred: torch.Tensor,
        c_true: torch.Tensor,
        reaction_rates: torch.Tensor
    ) -> torch.Tensor:
        """R_3: stoichiometric mass conservation residual."""
        B, n_sp, H, W = c_pred.shape
        
        if n_sp != self.n_species:
            raise ValueError(
                f"c_pred has {n_sp} species channels, expected {self.n_species}"
            )
        
        if c_true.shape != c_pred.shape:
            raise ValueError(
                f"c_true shape {c_true.shape} must match c_pred shape {c_pred.shape}"
            )
        
        if reaction_rates.shape[1] != self.n_reactions:
            raise ValueError(
                f"reaction_rates has {reaction_rates.shape[1]} reactions, "
                f"expected {self.n_reactions}"
            )
        
        # Move stoichiometry to same device as input
        stoich = self.stoichiometry.to(c_pred.device)
        
        # Compute concentration change: dc = c_pred - c_true
        dc = c_pred - c_true  # (B, n_species, H, W)
        
        # Part 1: Stoichiometric mass conservation constraint
        # For each reaction j: Σ_i ν_ij · c_i = 0
        # This enforces that the weighted sum of concentrations (by stoichiometry)
        # is conserved. This is the algebraic consistency of the reaction network.
        # stoich: (n_reactions, n_species), c_pred: (B, n_species, H, W)
        # mass_conservation: (B, n_reactions, H, W)
        mass_conservation = torch.einsum(
            'rs,bsxy->brxy',
            stoich,  # (n_reactions, n_species)
            c_pred  # (B, n_species, H, W)
        )
        
        # Part 2: Reaction rate consistency
        # The change in concentration should follow stoichiometry:
        # dc_i = Σ_j ν_ij * r_j * dt * Da
        # Compute expected concentration change from reactions
        # stoich.T: (n_species, n_reactions)
        # reaction_rates: (B, n_reactions, H, W)
        # expected_dc: (B, n_species, H, W)
        expected_dc = torch.einsum(
            'sr,brxy->bsxy', 
            stoich.T,  # (n_species, n_reactions)
            reaction_rates  # (B, n_reactions, H, W)
        ) * self.damkohler * self.dt
        
        # Residual is the difference between actual and expected change
        rate_residual = dc - expected_dc  # (B, n_species, H, W)
        
        # Total residual: sum of mass conservation violation and rate consistency
        # The mass conservation term enforces Σ_j |Σ_i ν_ij · c_i| = 0
        mass_loss = (mass_conservation.abs()).sum(dim=1).mean()  # Sum over reactions, mean over batch/space
        rate_loss = (rate_residual ** 2).mean()
        
        return mass_loss + rate_loss
    
    def transport_residual(
        self,
        c_pred: torch.Tensor,
        c_true: torch.Tensor,
        velocity: torch.Tensor,
        diffusivity: torch.Tensor
    ) -> torch.Tensor:
        """Advection-diffusion residual with Péclet scaling."""
        B, n_sp, H, W = c_pred.shape
        
        if n_sp != self.n_species:
            raise ValueError(
                f"c_pred has {n_sp} species channels, expected {self.n_species}"
            )
        
        # Extract velocity components
        vx = velocity[:, 0:1, :, :]  # (B, 1, H, W)
        vy = velocity[:, 1:2, :, :]  # (B, 1, H, W)
        
        # Broadcast diffusivity if needed
        if diffusivity.shape[1] == 1:
            D = diffusivity.expand(-1, n_sp, -1, -1)  # (B, n_species, H, W)
        else:
            D = diffusivity
        
        total_residual = 0.0
        
        for i in range(self.n_species):
            c_i_pred = c_pred[:, i:i+1, :, :]  # (B, 1, H, W)
            c_i_true = c_true[:, i:i+1, :, :]  # (B, 1, H, W)
            D_i = D[:, i:i+1, :, :]  # (B, 1, H, W)
            
            # Time derivative (backward difference)
            dc_dt = (c_i_pred - c_i_true) / (self.dt + 1e-8)
            
            # Advection term: Pe * (v · ∇c)
            dc_dx = grad_x(c_i_pred, self.dx)
            dc_dy = grad_y(c_i_pred, self.dy)
            advection = self.peclet * (vx * dc_dx + vy * dc_dy)
            
            # Diffusion term: ∇·(D∇c) = D∇²c (assuming constant D)
            # Using second-order central differences for Laplacian
            d2c_dx2 = self._laplacian_x(c_i_pred)
            d2c_dy2 = self._laplacian_y(c_i_pred)
            diffusion = D_i * (d2c_dx2 + d2c_dy2)
            
            # Transport residual: ∂c/∂t + Pe*(v·∇c) - ∇²c = 0
            residual = dc_dt + advection - diffusion
            
            total_residual = total_residual + (residual ** 2).mean()
        
        return total_residual / self.n_species
    
    def pressure_residual(
        self,
        p_pred: torch.Tensor,
        permeability: torch.Tensor,
        source: torch.Tensor
    ) -> torch.Tensor:
        """Elliptic pressure residual: -∇·(K∇p) = q."""
        # Compute pressure gradients
        dp_dx = grad_x(p_pred, self.dx)
        dp_dy = grad_y(p_pred, self.dy)
        
        # Compute flux: -K∇p
        flux_x = -permeability * dp_dx
        flux_y = -permeability * dp_dy
        
        # Compute divergence of flux: -∇·(K∇p)
        div_flux = divergence(flux_x, flux_y, self.dx, self.dy)
        
        # Residual: -∇·(K∇p) - q = 0
        residual = div_flux - source
        
        return (residual ** 2).mean()
    
    def _laplacian_x(self, u: torch.Tensor) -> torch.Tensor:
        """Second derivative in x direction."""
        u_pad = F.pad(u, (1, 1, 0, 0), mode="reflect")
        d2u_dx2 = (u_pad[:, :, :, 2:] - 2 * u_pad[:, :, :, 1:-1] + u_pad[:, :, :, :-2]) / (self.dx ** 2 + 1e-8)
        return d2u_dx2
    
    def _laplacian_y(self, u: torch.Tensor) -> torch.Tensor:
        """Second derivative in y direction."""
        u_pad = F.pad(u, (0, 0, 1, 1), mode="reflect")
        d2u_dy2 = (u_pad[:, :, 2:, :] - 2 * u_pad[:, :, 1:-1, :] + u_pad[:, :, :-2, :]) / (self.dy ** 2 + 1e-8)
        return d2u_dy2
