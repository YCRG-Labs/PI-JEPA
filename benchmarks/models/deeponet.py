import torch
import torch.nn as nn


class DeepONet(nn.Module):
    def __init__(self, in_dim=3*64*64, out_dim=3*64*64):
        super().__init__()

        self.branch = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        self.trunk = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        self.final = nn.Linear(256, out_dim)

    def forward(self, x):
        B = x.shape[0]
        x_flat = x.view(B, -1)

        b = self.branch(x_flat)

        coords = torch.rand(B, 2, device=x.device)
        t = self.trunk(coords)

        return self.final(b * t).view(B, 3, 64, 64)


class DeepONetWrapper:
    def __init__(self, device):
        self.device = device
        self.model = DeepONet().to(device)

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