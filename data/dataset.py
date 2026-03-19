import os
import torch
import numpy as np
from torch.utils.data import Dataset


class DarcyDataset(Dataset):
    """
    Simplified dataset aligned with training loop:
    returns (x_t, x_t+1)
    """

    def __init__(self, path, config, split="train"):
        file = os.path.join(path, f"{split}.npz")

        if not os.path.exists(file):
            raise FileNotFoundError(f"Dataset not found: {file}")

        raw = np.load(file)

        self.u = torch.tensor(raw["u"]).float()  # (N, T, H, W)
        self.k = torch.tensor(raw["k"]).float()  # (N, 1, H, W) or (N, T, H, W)

        self.N, self.T, self.H, self.W = self.u.shape

        # ✅ FIX: correct config path
        self.patch_size = config["model"]["encoder"]["patch_size"]
        self.num_patches = (self.H // self.patch_size) ** 2

        # optional normalization
        if config["data"].get("normalize", False):
            self._normalize()

    def _normalize(self):
        # simple global normalization
        u_mean = self.u.mean()
        u_std = self.u.std() + 1e-6

        self.u = (self.u - u_mean) / u_std

    def __len__(self):
        return self.N * (self.T - 1)

    def __getitem__(self, idx):
        i = idx // (self.T - 1)
        t = idx % (self.T - 1)

        u_t = self.u[i, t]         # (H, W)
        u_tp1 = self.u[i, t + 1]   # (H, W)

        k = self.k[i]

        # ensure k shape is (H, W)
        if k.ndim == 3:
            k = k[0]

        # ✅ FIX: consistent 2-channel input
        x_t = torch.stack([u_t, k], dim=0)       # (2, H, W)
        x_tp1 = torch.stack([u_tp1, k], dim=0)

        return x_t, x_tp1


# Optional sequence dataset (for rollout training)
class SequenceDarcyDataset(Dataset):
    def __init__(self, path, config, split="train"):
        file = os.path.join(path, f"{split}.npz")

        raw = np.load(file)

        self.u = torch.tensor(raw["u"]).float()
        self.k = torch.tensor(raw["k"]).float()

        self.N, self.T, self.H, self.W = self.u.shape

        self.steps = config["training"]["rollout_training"]["steps"]

    def __len__(self):
        return self.N * (self.T - self.steps - 1)

    def __getitem__(self, idx):
        i = idx // (self.T - self.steps - 1)
        t = idx % (self.T - self.steps - 1)

        traj = []

        for s in range(self.steps):
            u_t = self.u[i, t + s]
            u_tp1 = self.u[i, t + s + 1]

            k = self.k[i]
            if k.ndim == 3:
                k = k[0]

            x_t = torch.stack([u_t, k], dim=0)
            x_tp1 = torch.stack([u_tp1, k], dim=0)

            traj.append((x_t, x_tp1))

        return traj


def build_dataloader(dataset, config, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=0, 
        pin_memory=True,
        drop_last=True
    )