import torch
import torch.nn as nn


class PILatentNO(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(1, 32, 3, padding=1)
        self.decoder = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        z = torch.relu(self.encoder(x))
        return self.decoder(z)


class PILatentNOWrapper:
    def __init__(self, device):
        self.device = device
        self.model = PILatentNO().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

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