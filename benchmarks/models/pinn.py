import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 3)
        )

    def forward(self, coords):
        return self.net(coords)


class PINNWrapper:
    def __init__(self, device):
        self.device = device
        self.model = PINN().to(device)

    def train_model(self, loader, epochs, lr):
       for batch in loader:
        x = batch["x"].to(self.device)  # permeability (k)
        y = batch["y"].to(self.device)  # solution (u)

        # Convert to (u, k) format
        x = torch.stack([y, x], dim=1)

        pred = self.model(x)

        loss = self.loss_fn(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def predict(self, x):
        B, C, H, W = x.shape
        coords = torch.rand(B * H * W, 2).to(self.device)
        out = self.model(coords)
        return out.view(B, 3, H, W)

    def eval(self):
        self.model.eval()