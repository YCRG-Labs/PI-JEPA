import torch
import torch.nn as nn
import math


class Decoder(nn.Module):
    def __init__(self, embed_dim, out_channels, image_size, patch_size):
        super().__init__()

        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.image_size = image_size
        self.patch_size = patch_size

        self.proj = nn.Linear(
            embed_dim,
            out_channels * patch_size * patch_size
        )

    def forward(self, z_full):
        """
        z_full: (B, N, D)
        """

        B, N, D = z_full.shape

        n = int(math.sqrt(N))

        if n * n != N:
            raise ValueError(f"Cannot reshape {N} tokens into square grid")

        P = self.patch_size
        C = self.out_channels

        x = self.proj(z_full)

        x = x.view(B, n, n, C, P, P)
        x = x.permute(0, 3, 1, 4, 2, 5)

        x = x.contiguous().view(
            B,
            C,
            n * P,
            n * P
        )

        return x