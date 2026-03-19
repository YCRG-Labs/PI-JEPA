import torch
import torch.nn as nn
from neuralop.models import FNO


class FNOWrapper:
    def __init__(self, device):
        self.device = device

        self.model = FNO(
            n_modes=(16, 16),
            hidden_channels=64,
            in_channels=3,
            out_channels=3
        ).to(device)

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
        return self.model(x)

    def eval(self):
        self.model.eval()