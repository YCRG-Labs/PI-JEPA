import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)

        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, q_input, kv_input):
        B, Nt, D = q_input.shape
        Nc = kv_input.shape[1]

        q = self.q(q_input).reshape(B, Nt, self.heads, self.head_dim).transpose(1, 2)
        kv = self.kv(kv_input).reshape(B, Nc, 2, self.heads, self.head_dim)

        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nt, D)
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
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class PredictorBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, dropout):
        super().__init__()

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.cross_attn = CrossAttention(dim, heads, dropout)

        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(self, z_target, z_context):
        z = z_target + self.cross_attn(
            self.norm_q(z_target),
            self.norm_kv(z_context)
        )

        # feedforward
        z = z + self.mlp(self.norm_mlp(z))

        return z


class Stage(nn.Module):
    def __init__(self, dim, depth, heads, mlp_ratio, dropout):
        super().__init__()

        self.blocks = nn.ModuleList([
            PredictorBlock(dim, heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

    def forward(self, z_target, z_context):
        for blk in self.blocks:
            z_target = blk(z_target, z_context)
        return z_target


# =========================
# MAIN PREDICTOR (FIXED)
# =========================
class Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()

        model_cfg = config["model"]
        pred_cfg = model_cfg["predictor"]

        dim = model_cfg["latent_dim"]

        self.stages = nn.ModuleList([
            Stage(
                dim=dim,
                depth=stage_cfg["depth"],
                heads=stage_cfg["heads"],
                mlp_ratio=stage_cfg.get("mlp_ratio", 4.0),
                dropout=stage_cfg.get("dropout", 0.1)
            )
            for stage_cfg in pred_cfg["stages"]
        ])

        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, z_full, context_idx, target_idx):
        B, N, D = z_full.shape

        z_context = torch.gather(
            z_full,
            1,
            context_idx.unsqueeze(-1).expand(-1, -1, D)
        )

        z_target = torch.gather(
            z_full,
            1,
            target_idx.unsqueeze(-1).expand(-1, -1, D)
        )

        z_target = self.mask_token.expand_as(z_target)

        stage_outputs = {}

        for i, stage in enumerate(self.stages):
            z_delta = stage(z_target, z_context)

            # residual update (VERY IMPORTANT)
            z_target = z_target + z_delta

            stage_outputs[f"stage_{i}"] = z_target

        return z_target, stage_outputs

    def rollout(self, z0, steps):
        traj = []
        z = z0

        B, N, D = z.shape
        idx = torch.arange(N, device=z.device).unsqueeze(0).repeat(B, 1)

        for _ in range(steps):
            z_pred, _ = self.forward(z, idx, idx)

            z = z.clone()
            z.scatter_(
                1,
                idx.unsqueeze(-1).expand(-1, -1, D),
                z_pred
            )

            traj.append(z)

        return torch.stack(traj, dim=1)

class MultiStepPredictor(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, z0, context_idx, target_idx, steps):
        z = z0
        outputs = []

        for _ in range(steps):
            z_pred, stage_out = self.predictor(z, context_idx, target_idx)

            z = z.clone()
            z.scatter_(
                1,
                target_idx.unsqueeze(-1).expand_as(z_pred),
                z_pred
            )

            outputs.append({
                "z": z,
                "stages": stage_out
            })

        return outputs