from neuralop.models import FNO
import torch
import torch.nn as nn


class PINOWrapper:
    def __init__(self, device):
        self.device = device
        self.model = FNO(
            n_modes=(16, 16),
            hidden_channels=64,
            in_channels=1,
            out_channels=1
        ).to(device)

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