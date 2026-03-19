import os
import torch
import numpy as np
from torch.utils.data import Dataset


class FieldNormalizer:
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = data.mean()
        self.std = data.std()

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean


class MultiFieldNormalizer:
    def __init__(self):
        self.stats = {}

    def fit(self, data_dict):
        for k, v in data_dict.items():
            norm = FieldNormalizer()
            norm.fit(v)
            self.stats[k] = norm

    def encode(self, data_dict):
        return {k: self.stats[k].encode(v) for k, v in data_dict.items()}

    def decode(self, data_dict):
        return {k: self.stats[k].decode(v) for k, v in data_dict.items()}


class PDEBaseDataset(Dataset):
    def __init__(
        self,
        path,
        split="train",
        sequence_length=2,
        stride=1,
        normalize=True,
        preload=True
    ):
        self.path = path
        self.split = split
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize

        self.data = self._load_data(preload)

        self.indices = self._build_indices()

        if self.normalize:
            self.normalizer = MultiFieldNormalizer()
            self._fit_normalizer()

    def _load_data(self, preload):
        file = os.path.join(self.path, f"{self.split}.npz")
        raw = np.load(file)

        data = {
            "u": raw["u"],     
            "k": raw["k"]      
        }

        if preload:
            data = {k: torch.tensor(v).float() for k, v in data.items()}

        return data

    def _fit_normalizer(self):
        sample = {k: v[:100] for k, v in self.data.items()}
        self.normalizer.fit(sample)

        self.data = self.normalizer.encode(self.data)

    def _build_indices(self):
        T = self.data["u"].shape[1]
        idx = []

        for i in range(0, T - self.sequence_length, self.stride):
            idx.append(i)

        return idx

    def __len__(self):
        return len(self.indices)

    def _get_sequence(self, idx):
        t0 = self.indices[idx]

        seq = {}
        for k, v in self.data.items():
            seq[k] = v[:, t0:t0 + self.sequence_length]

        return seq

    def __getitem__(self, idx):
        raise NotImplementedError


class DarcyDataset(PDEBaseDataset):
    def __init__(self, path, split="train", config=None):
        super().__init__(
            path=path,
            split=split,
            sequence_length=config["data"]["sequence_length"],
            stride=config["data"]["stride"],
            normalize=config["data"]["normalize"]
        )

        self.return_coeff = config["data"]["include_coeff"]

    def _merge_fields(self, u, k):
        if self.return_coeff:
            return torch.cat([u, k], dim=1)
        return u

    def __getitem__(self, idx):
        seq = self._get_sequence(idx)

        u = seq["u"]      
        k = seq["k"]      

        x_t = self._merge_fields(u[:, 0], k[:, 0])
        x_tp1 = self._merge_fields(u[:, 1], k[:, 1])

        return x_t, x_tp1


class SequenceDarcyDataset(PDEBaseDataset):
    def __init__(self, path, split="train", config=None):
        super().__init__(
            path=path,
            split=split,
            sequence_length=config["training"]["rollout_training"]["steps"] + 1,
            stride=config["data"]["stride"],
            normalize=config["data"]["normalize"]
        )

        self.return_coeff = config["data"]["include_coeff"]

    def _merge_fields(self, u, k):
        if self.return_coeff:
            return torch.cat([u, k], dim=1)
        return u

    def __getitem__(self, idx):
        seq = self._get_sequence(idx)

        u = seq["u"]
        k = seq["k"]

        inputs = []
        for t in range(len(u[0]) - 1):
            xt = self._merge_fields(u[:, t], k[:, t])
            xtp1 = self._merge_fields(u[:, t+1], k[:, t+1])
            inputs.append((xt, xtp1))

        return inputs


def build_dataloader(dataset, config, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        drop_last=True
    )
