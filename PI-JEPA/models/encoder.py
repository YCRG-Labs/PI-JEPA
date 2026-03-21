import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


def build_2d_sincos(embed_dim, grid_size, device):
    coords = torch.arange(grid_size, dtype=torch.float32, device=device)
    grid = torch.meshgrid(coords, coords, indexing="ij")
    grid = torch.stack(grid, dim=0).reshape(2, -1)

    half = embed_dim // 2
    freq = torch.arange(half // 2, dtype=torch.float32, device=device)
    freq = 1.0 / (10000 ** (freq / (half // 2)))

    out_h = torch.einsum("n,d->nd", grid[0], freq)
    out_w = torch.einsum("n,d->nd", grid[1], freq)

    emb_h = torch.cat([torch.sin(out_h), torch.cos(out_h)], dim=1)
    emb_w = torch.cat([torch.sin(out_w), torch.cos(out_w)], dim=1)

    pos = torch.cat([emb_h, emb_w], dim=1)

    return pos.unsqueeze(0)


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        assert dim % heads == 0

        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.drop(x)

        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio, dropout):
        super().__init__()
        hidden = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, dropout):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        enc_cfg = config["model"]["encoder"]

        self.embed_dim = enc_cfg["embed_dim"]
        self.patch_size = enc_cfg["patch_size"]

        self.patch_embed = PatchEmbed(
            in_channels=2,
            embed_dim=self.embed_dim,
            patch_size=self.patch_size
        )

        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim,
                heads=enc_cfg["heads"],
                mlp_ratio=enc_cfg["mlp_ratio"],
                dropout=enc_cfg["dropout"]
            )
            for _ in range(enc_cfg["depth"])
        ])

        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)

        B, N, D = x.shape

        grid_size = int(N ** 0.5)
        assert grid_size * grid_size == N, "Non-square patch grid"

        pos_embed = build_2d_sincos(
            embed_dim=D,
            grid_size=grid_size,
            device=x.device
        )

        x = x + pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x


class TargetEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            return self.encoder(x)


@torch.no_grad()
def update_ema(student, teacher, tau):
    for p, tp in zip(student.parameters(), teacher.parameters()):
        tp.data.mul_(tau).add_(p.data, alpha=1.0 - tau)
