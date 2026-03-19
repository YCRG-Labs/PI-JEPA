import torch
import torch.nn as nn


def fix_shape(x):
    if x.ndim == 3:
        x = x.unsqueeze(1)
    elif x.ndim == 4 and x.shape[-1] == 1:
        x = x.permute(0, 3, 1, 2)
    elif x.ndim == 4 and x.shape[1] == 1:
        pass
    else:
        raise ValueError(f"Invalid shape: {x.shape}")
    return x.contiguous()


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)


class UNetWrapper:
    def __init__(self, device):
        self.device = device
        self.model = UNet().to(device)
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