from neuralop.models import FNO
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


class GeoFNOWrapper:
    def __init__(self, device):
        self.device = device
        self.model = FNO(12, 12, 48, in_channels=1, out_channels=1).to(device)

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