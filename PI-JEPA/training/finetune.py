"""Fine-tuning pipeline for PI-JEPA model."""

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProbe(nn.Module):
    """Linear probe layer for fine-tuning pretrained embeddings."""
    
    def __init__(self, embed_dim: int, out_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(embed_dim, out_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class FineTuningPipeline:
    """Fine-tuning pipeline with frozen encoder support."""
    
    SUPPORTED_N_LABELED = [10, 25, 50, 100, 250, 500]
    
    def __init__(
        self,
        model: nn.Module,
        decoder: nn.Module,
        config: Optional[dict] = None
    ) -> None:
        self.model = model
        self.decoder = decoder
        self.config = config or {}
        
        self.lr = self.config.get('lr', 1.5e-4)
        self.weight_decay = self.config.get('weight_decay', 0.05)
        self.device = self.config.get('device', 'cpu')
        
        self.linear_probe: Optional[LinearProbe] = None
        self._encoder_frozen = False
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.epoch = 0
        self.step = 0
        
        if hasattr(self.model, 'to'):
            self.model = self.model.to(self.device)
        if hasattr(self.decoder, 'to'):
            self.decoder = self.decoder.to(self.device)
    
    def freeze_encoder(self) -> None:
        """Freeze the Context_Encoder weights."""
        if hasattr(self.model, 'encoder'):
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            self._encoder_frozen = True
        else:
            raise AttributeError("Model does not have an 'encoder' attribute")
    
    def unfreeze_encoder(self) -> None:
        """Unfreeze the Context_Encoder weights."""
        if hasattr(self.model, 'encoder'):
            for param in self.model.encoder.parameters():
                param.requires_grad = True
            self._encoder_frozen = False
        else:
            raise AttributeError("Model does not have an 'encoder' attribute")
    
    @property
    def encoder_frozen(self) -> bool:
        return self._encoder_frozen
    
    def setup_linear_probe(self, embed_dim: int, out_dim: int) -> LinearProbe:
        """Initialize and return a linear probe layer."""
        self.linear_probe = LinearProbe(embed_dim=embed_dim, out_dim=out_dim)
        self.linear_probe = self.linear_probe.to(self.device)
        return self.linear_probe
    
    def _get_trainable_parameters(self) -> list:
        params = []
        if self.linear_probe is not None:
            params.extend(self.linear_probe.parameters())
        params.extend(self.decoder.parameters())
        if not self._encoder_frozen and hasattr(self.model, 'encoder'):
            params.extend(self.model.encoder.parameters())
        if hasattr(self.model, 'predictors'):
            for predictor in self.model.predictors:
                params.extend(predictor.parameters())
        return params
    
    def _setup_optimizer(self, lr: Optional[float] = None) -> torch.optim.AdamW:
        lr = lr if lr is not None else self.lr
        params = self._get_trainable_parameters()
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=self.weight_decay)
        return self.optimizer
    
    def _limit_dataset(
        self,
        train_loader: torch.utils.data.DataLoader,
        n_labeled: int
    ) -> torch.utils.data.DataLoader:
        dataset = train_loader.dataset
        n_samples = min(n_labeled, len(dataset))
        indices = list(range(n_samples))
        subset = torch.utils.data.Subset(dataset, indices)
        batch_size = train_loader.batch_size or 1
        return torch.utils.data.DataLoader(
            subset,
            batch_size=min(batch_size, n_samples),
            shuffle=True,
            num_workers=getattr(train_loader, 'num_workers', 0),
            pin_memory=getattr(train_loader, 'pin_memory', False)
        )
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        n_labeled: int,
        n_epochs: int = 100,
        lr: Optional[float] = None
    ) -> dict:
        """Train the linear probe and decoder on N_l labeled trajectories."""
        if self.linear_probe is None:
            raise RuntimeError("Linear probe not set up. Call setup_linear_probe first.")
        
        # Ensure encoder is frozen for linear probe training
        if not self._encoder_frozen:
            self.freeze_encoder()
        
        # Limit dataset to n_labeled samples
        limited_loader = self._limit_dataset(train_loader, n_labeled)
        
        # Set up optimizer
        lr = lr if lr is not None else self.lr
        self._setup_optimizer(lr=lr)
        
        # Set models to training mode
        self.linear_probe.train()
        self.decoder.train()
        self.model.eval()  # Encoder in eval mode since frozen
        
        # Training metrics
        train_losses = []
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch in limited_loader:
                # Handle different batch formats
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                    target = batch[1] if len(batch) > 1 else batch[0]
                elif isinstance(batch, dict):
                    x = batch.get('input', batch.get('x'))
                    target = batch.get('target', batch.get('y', x))
                else:
                    x = batch
                    target = batch
                
                x = x.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                # Encode with frozen encoder
                with torch.no_grad():
                    z = self.model.encode(x)
                
                # Apply linear probe
                z_probed = self.linear_probe(z)
                
                # Decode to field
                pred = self.decoder(z_probed)
                
                # Compute loss (MSE)
                loss = F.mse_loss(pred, target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
                self.step += 1
            
            avg_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_loss)
            self.epoch = epoch + 1
        
        return {
            'train_losses': train_losses,
            'final_loss': train_losses[-1] if train_losses else 0.0,
            'n_labeled': n_labeled,
            'n_epochs': n_epochs
        }
    
    def full_finetune(
        self,
        train_loader: torch.utils.data.DataLoader,
        n_epochs: int = 100,
        lr: Optional[float] = None
    ) -> dict:
        """Full fine-tuning with all parameters unfrozen."""
        # Unfreeze encoder for full fine-tuning
        self.unfreeze_encoder()
        
        # Set up optimizer with all parameters
        lr = lr if lr is not None else self.lr
        self._setup_optimizer(lr=lr)
        
        # Set all models to training mode
        if self.linear_probe is not None:
            self.linear_probe.train()
        self.decoder.train()
        self.model.train()
        
        # Training metrics
        train_losses = []
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch in train_loader:
                # Handle different batch formats
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                    target = batch[1] if len(batch) > 1 else batch[0]
                elif isinstance(batch, dict):
                    x = batch.get('input', batch.get('x'))
                    target = batch.get('target', batch.get('y', x))
                else:
                    x = batch
                    target = batch
                
                x = x.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                # Encode (gradients flow through encoder)
                z = self.model.encode(x)
                
                # Apply linear probe if available
                if self.linear_probe is not None:
                    z = self.linear_probe(z)
                
                # Decode to field
                pred = self.decoder(z)
                
                # Compute loss (MSE)
                loss = F.mse_loss(pred, target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
                self.step += 1
            
            avg_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_loss)
            self.epoch = epoch + 1
        
        return {
            'train_losses': train_losses,
            'final_loss': train_losses[-1] if train_losses else 0.0,
            'n_epochs': n_epochs,
            'mode': 'full_finetune'
        }
    
    def save_checkpoint(self, path: str, metrics: Optional[dict] = None) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'config': self.config,
            'encoder_frozen': self._encoder_frozen,
        }
        
        # Save model state
        if hasattr(self.model, 'state_dict'):
            checkpoint['model_state_dict'] = self.model.state_dict()
        
        # Save encoder state separately for convenience
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'state_dict'):
            checkpoint['encoder_state_dict'] = self.model.encoder.state_dict()
        
        # Save decoder state
        if hasattr(self.decoder, 'state_dict'):
            checkpoint['decoder_state_dict'] = self.decoder.state_dict()
        
        # Save linear probe state if available
        if self.linear_probe is not None and hasattr(self.linear_probe, 'state_dict'):
            checkpoint['linear_probe_state_dict'] = self.linear_probe.state_dict()
        
        # Save optimizer state if available
        if self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        # Include metrics if provided
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> dict:
        """Load model checkpoint and restore training state."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Restore training state
        self.epoch = checkpoint.get('epoch', 0)
        self.step = checkpoint.get('step', 0)
        self._encoder_frozen = checkpoint.get('encoder_frozen', False)
        
        # Restore model state
        if 'model_state_dict' in checkpoint and hasattr(self.model, 'load_state_dict'):
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore decoder state
        if 'decoder_state_dict' in checkpoint and hasattr(self.decoder, 'load_state_dict'):
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        # Restore linear probe state if available
        if 'linear_probe_state_dict' in checkpoint and self.linear_probe is not None:
            self.linear_probe.load_state_dict(checkpoint['linear_probe_state_dict'])
        
        # Restore optimizer state if available
        if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Apply encoder frozen state
        if self._encoder_frozen:
            self.freeze_encoder()
        else:
            self.unfreeze_encoder()
        
        return checkpoint
