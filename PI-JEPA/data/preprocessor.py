import os
import numpy as np
import torch
import torch.nn.functional as F


class Preprocessor:
    def __init__(
        self,
        input_path,
        output_path,
        grid_size=None,
        normalize=True,
        eps=1e-6
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.grid_size = grid_size
        self.normalize = normalize
        self.eps = eps

        os.makedirs(self.output_path, exist_ok=True)

        self.stats = {}

    def _load_raw(self, split):
        file = os.path.join(self.input_path, f"{split}.npz")
        data = np.load(file)

        u = torch.tensor(data["u"]).float()
        k = torch.tensor(data["k"]).float()

        return {"u": u, "k": k}

    def _resize(self, x):
        if self.grid_size is None:
            return x

        B, T, H, W = x.shape
        x = x.view(B * T, 1, H, W)

        x = F.interpolate(
            x,
            size=(self.grid_size, self.grid_size),
            mode="bilinear",
            align_corners=False
        )

        x = x.view(B, T, self.grid_size, self.grid_size)

        return x

    def _compute_stats(self, data):
        stats = {}

        for k, v in data.items():
            mean = v.mean()
            std = v.std()

            stats[k] = {
                "mean": mean,
                "std": std
            }

        return stats

    def _normalize(self, data, stats):
        out = {}

        for k, v in data.items():
            mean = stats[k]["mean"]
            std = stats[k]["std"]

            out[k] = (v - mean) / (std + self.eps)

        return out

    def _save(self, split, data):
        file = os.path.join(self.output_path, f"{split}.npz")

        np.savez(
            file,
            u=data["u"].cpu().numpy(),
            k=data["k"].cpu().numpy()
        )

    def _save_stats(self):
        file = os.path.join(self.output_path, "stats.pt")
        torch.save(self.stats, file)

    def process_split(self, split):
        data = self._load_raw(split)

        data["u"] = self._resize(data["u"])
        data["k"] = self._resize(data["k"])

        if self.normalize:
            if split == "train":
                self.stats = self._compute_stats(data)

            data = self._normalize(data, self.stats)

        self._save(split, data)

    def run(self):
        for split in ["train", "val", "test"]:
            self.process_split(split)

        if self.normalize:
            self._save_stats()
