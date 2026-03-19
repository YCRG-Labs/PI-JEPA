import torch
import torch.nn as nn


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
        # residual latent evolution (paper: latent dynamics formulation)
        x = x + self.res_scale * self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.res_scale * self.drop_path2(self.mlp(self.norm2(x)))
        return x


class Stage(nn.Module):
    def __init__(self, dim, depth, heads, mlp_ratio, dropout, drop_path, res_scale):
        super().__init__()

        dpr = torch.linspace(0, drop_path, depth).tolist()

        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                heads=heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=dpr[i],
                res_scale=res_scale
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # stage = one operator in splitting (paper: pressure / transport operators)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


class Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()

        model_cfg = config["model"]
        pred_cfg = model_cfg["predictor"]

        dim = model_cfg["latent_dim"]

        self.stage_names = []
        self.stages = nn.ModuleList()

        # builds ordered operators (paper: operator splitting π₁ ∘ π₂ ...)
        for stage_cfg in pred_cfg["stages"]:
            name = stage_cfg["name"]

            stage = Stage(
                dim=dim,
                depth=stage_cfg["depth"],
                heads=stage_cfg["heads"],
                mlp_ratio=stage_cfg["hidden_dim"] / dim,
                dropout=0.1,
                drop_path=pred_cfg.get("drop_path", 0.1),
                res_scale=1.0
            )

            self.stage_names.append(name)
            self.stages.append(stage)

        self.sequential = pred_cfg.get("sequential", True)
        self.skip = pred_cfg.get("skip_connections", True)

    def forward(self, z_context):
        # z_context = encoder output (paper: z_t)
        z = z_context
        outputs = {}

        # sequential latent evolution (paper: z_{t+1} = π₂(π₁(z_t)))
        for name, stage in zip(self.stage_names, self.stages):
            z_stage = stage(z)

            # residual update (paper: stable latent time stepping)
            if self.skip:
                z = z + z_stage
            else:
                z = z_stage

            # stage outputs for supervision (paper: stagewise JEPA loss)
            outputs[name] = z

        # final latent prediction (paper: ẑ_{t+1})
        return z, outputs

    def rollout(self, z0, steps, detach=True):
        z = z0
        traj = []

        for _ in range(steps):
            z, _ = self.forward(z)
            traj.append(z)

            # prevents gradient explosion (paper: rollout stabilization)
            if detach:
                z = z.detach()

        return traj


class MultiStepPredictor(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, z0, steps):
        # returns full trajectory + intermediate operator states
        z = z0
        outputs = []

        for _ in range(steps):
            z, stage_out = self.predictor(z)

            outputs.append({
                "z": z,
                "stages": stage_out
            })

        return outputs
