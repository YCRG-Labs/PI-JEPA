import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
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


def _grid(grid_size, device):
    coords = torch.arange(grid_size, dtype=torch.float32, device=device)
    grid = torch.meshgrid(coords, coords, indexing='ij')
    grid = torch.stack(grid, dim=0).reshape(2, -1)
    return grid


def _sincos(dim, pos):
    half = dim // 2
    freq = torch.arange(half, dtype=torch.float32, device=pos.device)
    freq = 1.0 / (10000 ** (freq / half))
    out = torch.einsum("n,d->nd", pos, freq)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def build_2d_sincos(embed_dim, grid_size, device):
    g = _grid(grid_size, device)
    emb_h = _sincos(embed_dim // 2, g[0])
    emb_w = _sincos(embed_dim // 2, g[1])
    pos = torch.cat([emb_h, emb_w], dim=1)
    return pos.unsqueeze(0)


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = torch.rand(shape, device=x.device, dtype=x.dtype)
        mask = (rand < keep).float()
        return x * mask / keep


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
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
    def __init__(self, dim, heads, mlp_ratio, dropout, drop_path, res_scale):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)
        self.drop_path2 = DropPath(drop_path)

        self.res_scale = res_scale

    def forward(self, x):
        x = x + self.res_scale * self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.res_scale * self.drop_path2(self.mlp(self.norm2(x)))
        return x


class FeatureNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class ViTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        data_cfg = config["data"]
        model_cfg = config["model"]
        enc_cfg = model_cfg["encoder"]

        self.embed_dim = model_cfg["latent_dim"]
        self.patch_size = data_cfg["patch"]["size"]
        self.grid_size = data_cfg["grid_size"] // self.patch_size
        self.num_patches = self.grid_size * self.grid_size

        in_channels = len(data_cfg["channels"])

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dim=self.embed_dim,
            patch_size=self.patch_size
        )

        self.register_tokens = nn.Parameter(
            torch.zeros(1, enc_cfg.get("num_register_tokens", 4), self.embed_dim)
        )

        depth = enc_cfg["depth"]
        heads = enc_cfg["heads"]
        mlp_ratio = enc_cfg["mlp_ratio"]
        dropout = enc_cfg["dropout"]

        drop_path_rate = enc_cfg.get("drop_path", 0.1)
        res_scale = enc_cfg.get("residual_scale", 1.0)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()

        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim,
                heads=heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=dpr[i],
                res_scale=res_scale
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(self.embed_dim)
        self.feature_norm = FeatureNorm(self.embed_dim)

        self.pos_embed = None

    def _pos(self, device):
        if self.pos_embed is None or self.pos_embed.device != device:
            self.pos_embed = build_2d_sincos(self.embed_dim, self.grid_size, device)
        return self.pos_embed

    def forward(self, x, mask=None, return_registers=False):
        x = self.patch_embed(x)
        pos = self._pos(x.device)
        x = x + pos

        B = x.shape[0]
        reg = self.register_tokens.expand(B, -1, -1)
        x = torch.cat([reg, x], dim=1)

        if mask is not None:
            x = x[mask].view(B, -1, self.embed_dim)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.feature_norm(x)

        if return_registers:
            return x[:, :reg.shape[1]], x[:, reg.shape[1]:]

        return x


class TargetEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, mask=None):
        with torch.no_grad():
            return self.encoder(x, mask=mask)


@torch.no_grad()
def update_ema(student, teacher, tau):
    for p, tp in zip(student.parameters(), teacher.parameters()):
        tp.data.mul_(tau).add_(p.data * (1.0 - tau))
