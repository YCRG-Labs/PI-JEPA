import torch
import torch.nn as nn
from neuralop.models import FNO


def get_grid(batchsize, size_x, size_y, device):
    x = torch.linspace(0, 1, size_x, device=device)
    y = torch.linspace(0, 1, size_y, device=device)
    gridx, gridy = torch.meshgrid(x, y, indexing='ij')

    grid = torch.stack((gridx, gridy), dim=0)  # (2, H, W)
    grid = grid.unsqueeze(0).repeat(batchsize, 1, 1, 1)  # (B, 2, H, W)

    return grid


class GeoFNO(nn.Module):
    """
    GeoFNO-style model:
    - Concatenate spatial coordinates to input
    - Learn operator in augmented space
    """

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.fno = FNO(
            n_modes=(16, 16),
            hidden_channels=64,
            in_channels=in_channels + 2,  # +2 for (x, y)
            out_channels=out_channels
        )

    def forward(self, x):
        B, C, H, W = x.shape
        grid = get_grid(B, H, W, x.device)

        x = torch.cat([x, grid], dim=1)  # (B, C+2, H, W)
        return self.fno(x)


class GeoFNOWrapper:
    def __init__(self, device):
        self.device = device
        self.model = GeoFNO().to(device)

    def train_model(self, loader, epochs, lr):
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)

                pred = self.model(x)
                loss = loss_fn(pred, y)

                opt.zero_grad()
                loss.backward()
                opt.step()

    def predict(self, x):
        return self.model(x)

    def eval(self):
        self.model.eval()