import torch
import torch.nn as nn


class PINNWrapper:
    def __init__(self, device):
        self.device = device

    def train_model(self, loader, epochs, lr):
        pass

    def predict(self, x):
        return x