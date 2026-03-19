import torch
import torch.nn as nn


class DeepONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch = nn.Sequential(nn.Linear(64*64, 256), nn.ReLU(), nn.Linear(256, 128))
        self.trunk = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 128))
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, -1)

        branch = self.branch(x_flat)

        coords = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, H),
            torch.linspace(0, 1, W),
            indexing='ij'
        ), dim=-1).view(-1, 2).to(x.device)

        trunk = self.trunk(coords)
        out = torch.einsum("bi,ni->bn", branch, trunk)
        return out.view(B, 1, H, W)


class DeepONetWrapper:
    def __init__(self, device):
        self.device = device
        self.model = DeepONet().to(device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train_model(self, loader, epochs, lr):
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for _ in range(epochs):
            for batch in loader:
                k = fix_shape(batch["x"].to(self.device).float())
                u = fix_shape(batch["y"].to(self.device).float())

                pred = self.model(k)
                loss = self.loss_fn(pred, u)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, x):
        self.model.eval()
        x = fix_shape(x)
        with torch.no_grad():
            return self.model(x)