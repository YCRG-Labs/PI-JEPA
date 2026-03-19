import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU()
            )

        self.enc1 = block(in_channels, 32)
        self.pool = nn.MaxPool2d(2)

        self.enc2 = block(32, 64)
        self.enc3 = block(64, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = block(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = block(64, 32)

        self.out = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d1 = self.up1(e3)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        return self.out(d2)


class UNetWrapper:
    def __init__(self, device):
        self.device = device
        self.model = UNet().to(device)

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