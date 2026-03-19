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