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
            nn.Linear(128, 1)
        )

    def forward(self, coords):
        return self.net(coords)


class PINNWrapper:
    def __init__(self, device):
        self.device = device
        self.model = PINN().to(device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train_model(self, loader, epochs, lr):
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for _ in range(epochs):
            for batch in loader:
                u = batch["y"].to(self.device).float()
                B, _, H, W = u.shape

                coords = torch.rand(B*H*W, 2).to(self.device)
                target = u.view(-1, 1)

                pred = self.model(coords)
                loss = self.loss_fn(pred, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, x):
        return x  # placeholder