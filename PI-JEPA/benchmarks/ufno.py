"""U-FNO (U-Net-augmented Fourier Neural Operator) baseline."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class SpectralConv2d(nn.Module):
    """2D Spectral Convolution layer for FNO."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
    
    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication in Fourier space."""
        # (B, in_channels, H, W), (in_channels, out_channels, H, W) -> (B, out_channels, H, W)
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input (B, C, H, W) -> output (B, out_channels, H, W)."""
        B, C, H, W = x.shape
        
        # Compute FFT
        x_ft = torch.fft.rfft2(x)
        
        # Adapt modes to actual spatial dimensions
        modes1 = min(self.modes1, H // 2)
        modes2 = min(self.modes2, W // 2 + 1)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        # Use only the modes that fit
        out_ft[:, :, :modes1, :modes2] = self.compl_mul2d(
            x_ft[:, :, :modes1, :modes2], 
            self.weights1[:, :, :modes1, :modes2]
        )
        out_ft[:, :, -modes1:, :modes2] = self.compl_mul2d(
            x_ft[:, :, -modes1:, :modes2], 
            self.weights2[:, :, :modes1, :modes2]
        )
        
        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(H, W))
        
        return x


class FNOBlock(nn.Module):
    """FNO block with spectral convolution and skip connection."""
    
    def __init__(
        self,
        channels: int,
        modes1: int,
        modes2: int
    ):
        super().__init__()
        self.spectral_conv = SpectralConv2d(channels, channels, modes1, modes2)
        self.conv = nn.Conv2d(channels, channels, 1)
        self.norm = nn.InstanceNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input (B, C, H, W) -> output (B, C, H, W)."""
        x1 = self.spectral_conv(x)
        x2 = self.conv(x)
        out = self.norm(x1 + x2)
        return F.gelu(out)


class UNetEncoder(nn.Module):
    """U-Net encoder with downsampling blocks."""
    
    def __init__(
        self,
        in_channels: int,
        channels: List[int]
    ):
        super().__init__()
        self.channels = channels
        
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        prev_channels = in_channels
        for ch in channels:
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels, ch, 3, padding=1),
                    nn.InstanceNorm2d(ch),
                    nn.GELU(),
                    nn.Conv2d(ch, ch, 3, padding=1),
                    nn.InstanceNorm2d(ch),
                    nn.GELU()
                )
            )
            self.pools.append(nn.MaxPool2d(2))
            prev_channels = ch
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass. Returns (encoded features, skip connections)."""
        skips = []
        
        for block, pool in zip(self.blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)
        
        return x, skips


class UNetDecoder(nn.Module):
    """U-Net decoder with upsampling blocks and skip connections."""
    
    def __init__(
        self,
        channels: List[int],
        out_channels: int
    ):
        super().__init__()
        self.channels = channels
        
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        
        for i, ch in enumerate(channels):
            if i == 0:
                in_ch = ch
            else:
                in_ch = channels[i - 1]
            
            self.ups.append(
                nn.ConvTranspose2d(in_ch, ch, 2, stride=2)
            )
            # After concatenation with skip, input channels double
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(ch * 2, ch, 3, padding=1),
                    nn.InstanceNorm2d(ch),
                    nn.GELU(),
                    nn.Conv2d(ch, ch, 3, padding=1),
                    nn.InstanceNorm2d(ch),
                    nn.GELU()
                )
            )
        
        self.final = nn.Conv2d(channels[-1], out_channels, 1)
    
    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass. Combines bottleneck with skip connections."""
        # Reverse skips to match decoder order
        skips = skips[::-1]
        
        for i, (up, block) in enumerate(zip(self.ups, self.blocks)):
            x = up(x)
            
            # Handle size mismatch
            skip = skips[i]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        
        return self.final(x)


class UFNO(nn.Module):
    """U-Net-augmented Fourier Neural Operator."""
    
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        modes: Tuple[int, int] = (16, 16),
        width: int = 64,
        encoder_channels: Optional[List[int]] = None,
        n_fno_layers: int = 4
    ):
        super().__init__()
        
        if encoder_channels is None:
            encoder_channels = [32, 64, 128]
        
        self.encoder = UNetEncoder(in_channels, encoder_channels)
        
        # Projection to FNO width
        self.proj_in = nn.Conv2d(encoder_channels[-1], width, 1)
        
        # FNO layers
        self.fno_layers = nn.ModuleList([
            FNOBlock(width, modes[0], modes[1])
            for _ in range(n_fno_layers)
        ])
        
        # Projection back to decoder channels
        self.proj_out = nn.Conv2d(width, encoder_channels[-1], 1)
        
        # Decoder with reversed channels
        decoder_channels = encoder_channels[::-1]
        self.decoder = UNetDecoder(decoder_channels, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input (B, in_channels, H, W) -> output (B, out_channels, H, W)."""
        # U-Net encoder
        x_enc, skips = self.encoder(x)
        
        # Project to FNO width
        x_fno = self.proj_in(x_enc)
        
        # FNO layers
        for fno_layer in self.fno_layers:
            x_fno = fno_layer(x_fno)
        
        # Project back
        x_dec = self.proj_out(x_fno)
        
        # U-Net decoder with skip connections
        out = self.decoder(x_dec, skips)
        
        return out


def fix_shape(x: torch.Tensor) -> torch.Tensor:
    """Ensure input tensor has shape (B, C, H, W)."""
    if x.ndim == 3:
        x = x.unsqueeze(1)
    elif x.ndim == 4 and x.shape[-1] == 1:
        x = x.permute(0, 3, 1, 2)
    elif x.ndim == 4 and x.shape[1] in [1, 2]:
        pass
    else:
        # Assume it's already in correct format
        pass
    return x.contiguous()


class UFNOWrapper:
    """U-FNO wrapper for training and prediction."""
    
    def __init__(
        self,
        device: torch.device,
        in_channels: int = 2,
        out_channels: int = 2,
        modes: Tuple[int, int] = (16, 16),
        width: int = 64,
        encoder_channels: Optional[List[int]] = None,
        n_fno_layers: int = 4
    ):
        self.device = device
        
        self.model = UFNO(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            width=width,
            encoder_channels=encoder_channels,
            n_fno_layers=n_fno_layers
        ).to(device)
        
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
    def train_model(
        self,
        loader,
        epochs: int,
        lr: float
    ):
        """Train U-FNO model."""
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for batch in loader:
                x = fix_shape(batch["x"].to(self.device).float())
                y = fix_shape(batch["y"].to(self.device).float())
                
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            print(f"[U-FNO] Epoch {epoch+1}/{epochs} Loss: {total_loss:.6f}")
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make prediction with trained model."""
        self.model.eval()
        x = fix_shape(x)
        with torch.no_grad():
            return self.model(x)
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
