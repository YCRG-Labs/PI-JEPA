"""PINN (Physics-Informed Neural Network) baseline with per-instance training."""

import torch
import torch.nn as nn
from typing import Optional, Callable, Tuple


class PINN(nn.Module):
    """Physics-Informed Neural Network mapping coordinates to field values."""
    
    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        hidden_layers: Optional[list] = None,
        activation: str = "tanh"
    ):
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = [64, 64, 64]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass. Coords (N, input_dim) -> field (N, output_dim)."""
        return self.net(coords)


class PINNWrapper:
    """PINN wrapper with per-instance training."""
    
    def __init__(
        self,
        device: torch.device,
        hidden_layers: Optional[list] = None,
        physics_weight: float = 1.0,
        n_collocation: int = 1024,
        activation: str = "tanh"
    ):
        self.device = device
        self.hidden_layers = hidden_layers if hidden_layers is not None else [64, 64, 64]
        self.physics_weight = physics_weight
        self.n_collocation = n_collocation
        self.activation = activation
        
        # Current model (created per instance)
        self.model = None
        self.optimizer = None
    
    def _create_model(self, output_dim: int = 1) -> PINN:
        """Create a new PINN model."""
        return PINN(
            input_dim=2,
            output_dim=output_dim,
            hidden_layers=self.hidden_layers,
            activation=self.activation
        ).to(self.device)
    
    def _compute_pde_residual(
        self,
        model: PINN,
        coords: torch.Tensor,
        permeability_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        """Compute PDE residual using automatic differentiation."""
        coords = coords.requires_grad_(True)
        
        # Forward pass
        u = model(coords)  # (N, 1)
        
        # Compute gradients using autodiff
        # du/dx and du/dy
        grad_u = torch.autograd.grad(
            u, coords,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]  # (N, 2)
        
        du_dx = grad_u[:, 0:1]  # (N, 1)
        du_dy = grad_u[:, 1:2]  # (N, 1)
        
        # Get permeability (default to 1.0)
        if permeability_fn is not None:
            K = permeability_fn(coords)
        else:
            K = torch.ones_like(du_dx)
        
        # Compute flux: -K * grad(u)
        flux_x = -K * du_dx
        flux_y = -K * du_dy
        
        # Compute divergence: d(flux_x)/dx + d(flux_y)/dy
        div_flux_x = torch.autograd.grad(
            flux_x, coords,
            grad_outputs=torch.ones_like(flux_x),
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]
        
        div_flux_y = torch.autograd.grad(
            flux_y, coords,
            grad_outputs=torch.ones_like(flux_y),
            create_graph=True,
            retain_graph=True
        )[0][:, 1:2]
        
        # PDE residual: -∇·(K∇u) - f = 0 (assuming f=0)
        residual = div_flux_x + div_flux_y
        
        return residual
    
    def train_instance(
        self,
        x_boundary: torch.Tensor,
        u_boundary: torch.Tensor,
        x_collocation: torch.Tensor,
        epochs: int = 1000,
        lr: float = 1e-3,
        permeability_fn: Optional[Callable] = None,
        verbose: bool = False
    ) -> dict:
        """Per-instance physics-informed training."""
        # Determine output dimension from boundary data
        output_dim = u_boundary.shape[-1] if u_boundary.dim() > 1 else 1
        
        # Create new model for this instance
        self.model = self._create_model(output_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Move data to device
        x_boundary = x_boundary.to(self.device).float()
        u_boundary = u_boundary.to(self.device).float()
        x_collocation = x_collocation.to(self.device).float()
        
        if u_boundary.dim() == 1:
            u_boundary = u_boundary.unsqueeze(-1)
        
        history = {
            "total_loss": [],
            "boundary_loss": [],
            "physics_loss": []
        }
        
        self.model.train()
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Boundary loss
            u_pred_boundary = self.model(x_boundary)
            boundary_loss = nn.functional.mse_loss(u_pred_boundary, u_boundary)
            
            # Physics residual loss
            residual = self._compute_pde_residual(
                self.model, x_collocation, permeability_fn
            )
            physics_loss = (residual ** 2).mean()
            
            # Total loss
            total_loss = boundary_loss + self.physics_weight * physics_loss
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            # Record history
            history["total_loss"].append(total_loss.item())
            history["boundary_loss"].append(boundary_loss.item())
            history["physics_loss"].append(physics_loss.item())
            
            if verbose and (epoch + 1) % 100 == 0:
                print(
                    f"[PINN] Epoch {epoch+1}/{epochs} "
                    f"Loss: {total_loss.item():.6f} "
                    f"(Boundary: {boundary_loss.item():.6f}, "
                    f"Physics: {physics_loss.item():.6f})"
                )
        
        return history
    
    def train_model(
        self,
        loader,
        epochs: int,
        lr: float
    ):
        """Train on dataset (for compatibility with other wrappers)."""
        # For compatibility, we train a model that maps (x, y, K) -> u
        # This is not the standard PINN approach but allows batch training
        
        # Determine dimensions from first batch
        first_batch = next(iter(loader))
        x_sample = first_batch["x"]
        y_sample = first_batch["y"]
        
        B, C_in, H, W = x_sample.shape if x_sample.dim() == 4 else (x_sample.shape[0], 1, x_sample.shape[1], x_sample.shape[2])
        C_out = y_sample.shape[1] if y_sample.dim() == 4 else 1
        
        # Create model with coordinate + permeability input
        self.model = PINN(
            input_dim=2 + 1,  # (x, y, K)
            output_dim=C_out,
            hidden_layers=self.hidden_layers,
            activation=self.activation
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for batch in loader:
                x = batch["x"].to(self.device).float()
                y = batch["y"].to(self.device).float()
                
                if x.dim() == 3:
                    x = x.unsqueeze(1)
                if y.dim() == 3:
                    y = y.unsqueeze(1)
                
                B, C, H, W = x.shape
                
                # Create coordinate grid
                coords_y = torch.linspace(0, 1, H, device=self.device)
                coords_x = torch.linspace(0, 1, W, device=self.device)
                yy, xx = torch.meshgrid(coords_y, coords_x, indexing='ij')
                coords = torch.stack([xx, yy], dim=-1)  # (H, W, 2)
                
                # Flatten and expand for batch
                coords_flat = coords.view(-1, 2)  # (H*W, 2)
                coords_batch = coords_flat.unsqueeze(0).expand(B, -1, -1)  # (B, H*W, 2)
                
                # Get permeability at each point
                K_flat = x[:, 0].view(B, -1, 1)  # (B, H*W, 1)
                
                # Concatenate coordinates and permeability
                inputs = torch.cat([coords_batch, K_flat], dim=-1)  # (B, H*W, 3)
                inputs = inputs.view(-1, 3)  # (B*H*W, 3)
                
                # Forward pass
                pred = self.model(inputs)  # (B*H*W, C_out)
                pred = pred.view(B, H, W, -1).permute(0, 3, 1, 2)  # (B, C_out, H, W)
                
                # Data loss
                loss = nn.functional.mse_loss(pred, y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            print(f"[PINN] Epoch {epoch+1}/{epochs} Loss: {total_loss:.6f}")
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make prediction with trained model."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train_instance or train_model first.")
        
        self.model.eval()
        
        with torch.no_grad():
            if x.dim() == 2 and x.shape[-1] == 2:
                # Direct coordinate input
                return self.model(x.to(self.device))
            else:
                # Grid input - need to create coordinates
                if x.dim() == 3:
                    x = x.unsqueeze(1)
                
                B, C, H, W = x.shape
                x = x.to(self.device).float()
                
                # Create coordinate grid
                coords_y = torch.linspace(0, 1, H, device=self.device)
                coords_x = torch.linspace(0, 1, W, device=self.device)
                yy, xx = torch.meshgrid(coords_y, coords_x, indexing='ij')
                coords = torch.stack([xx, yy], dim=-1).view(-1, 2)
                
                # Expand for batch
                coords_batch = coords.unsqueeze(0).expand(B, -1, -1)
                K_flat = x[:, 0].view(B, -1, 1)
                
                inputs = torch.cat([coords_batch, K_flat], dim=-1).view(-1, 3)
                
                pred = self.model(inputs)
                pred = pred.view(B, H, W, -1).permute(0, 3, 1, 2)
                
                return pred
    
    def predict_instance(
        self,
        H: int,
        W: int
    ) -> torch.Tensor:
        """Predict on a grid after per-instance training."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train_instance first.")
        
        self.model.eval()
        
        with torch.no_grad():
            # Create coordinate grid
            coords_y = torch.linspace(0, 1, H, device=self.device)
            coords_x = torch.linspace(0, 1, W, device=self.device)
            yy, xx = torch.meshgrid(coords_y, coords_x, indexing='ij')
            coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # (H*W, 2)
            
            # Predict
            pred = self.model(coords)  # (H*W, output_dim)
            pred = pred.view(H, W, -1).permute(2, 0, 1).unsqueeze(0)  # (1, output_dim, H, W)
            
            return pred
    
    def eval(self):
        """Set model to evaluation mode."""
        if self.model is not None:
            self.model.eval()
