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
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        for _ in range(epochs):
            for x, y in loader:
                B, C, H, W = x.shape

                coords = torch.rand(B * H * W, 2).to(self.device)
                pred = self.model(coords)

                loss = pred.pow(2).mean()

                opt.zero_grad()
                loss.backward()
                opt.step()

    def predict(self, x):
        B, C, H, W = x.shape
        coords = torch.rand(B * H * W, 2).to(self.device)
        out = self.model(coords)
        return out.view(B, 3, H, W)

    def eval(self):
        self.model.eval()