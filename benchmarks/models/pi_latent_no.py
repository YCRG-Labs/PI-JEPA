import torch
import torch.nn as nn


class PILatentNO(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Conv2d(3, 32, 3, padding=1)
        self.operator = nn.Conv2d(32, 32, 3, padding=1)
        self.decoder = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, x):
        z = torch.relu(self.encoder(x))
        z = torch.relu(self.operator(z))
        return self.decoder(z)


class PILatentNOWrapper:
    def __init__(self, device):
        self.device = device
        self.model = PILatentNO().to(device)

    def train_model(self, loader, epochs, lr):
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

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