import torch
import torch.nn as nn


class PredictionLoss(nn.Module):
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def forward(self, z_pred, z_target):
        loss = (z_pred - z_target).pow(2)

        if self.normalize:
            loss = loss / (z_target.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

        return loss.mean()


class StagewiseLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, stage_outputs, stage_targets):
        total = 0.0

        for name in stage_outputs:
            if name not in stage_targets:
                continue

            w = self.weights.get(name, 1.0)
            loss = (stage_outputs[name] - stage_targets[name]).pow(2).mean()
            total += w * loss

        return total


class VarianceLoss(nn.Module):
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, z):
        std = torch.sqrt(z.var(dim=0) + 1e-6)
        loss = torch.relu(self.gamma - std).mean()
        return loss


class CovarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        B, N, D = z.shape
        z = z.reshape(B * N, D)

        z = z - z.mean(dim=0)

        cov = (z.T @ z) / (z.shape[0] - 1)
        off_diag = cov - torch.diag(torch.diag(cov))

        return (off_diag.pow(2).sum()) / D


class PhysicsResidual(nn.Module):
    def __init__(self, config):
        super().__init__()

        phys_cfg = config["physics"]["numerical"]

        self.dx = 1.0
        self.eps = phys_cfg.get("epsilon", 1e-6)

    def gradient(self, u):
        dx = self.dx

        grad_x = (u[:, :, 2:, 1:-1] - u[:, :, :-2, 1:-1]) / (2 * dx)
        grad_y = (u[:, :, 1:-1, 2:] - u[:, :, 1:-1, :-2]) / (2 * dx)

        return grad_x, grad_y

    def divergence(self, fx, fy):
        dx = self.dx

        div_x = (fx[:, :, 2:, :] - fx[:, :, :-2, :]) / (2 * dx)
        div_y = (fy[:, :, :, 2:] - fy[:, :, :, :-2]) / (2 * dx)

        return div_x[:, :, :, 1:-1] + div_y[:, :, 1:-1, :]

    def forward_pressure(self, p, K):
        gx, gy = self.gradient(p)
        fx = K[:, :, 1:-1, 1:-1] * gx
        fy = K[:, :, 1:-1, 1:-1] * gy
        div = self.divergence(fx, fy)
        return div

    def forward_transport(self, s, vx, vy):
        flux_x = vx * s[:, :, 1:-1, 1:-1]
        flux_y = vy * s[:, :, 1:-1, 1:-1]

        div = self.divergence(flux_x, flux_y)
        return div


class PhysicsLoss(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.residual = PhysicsResidual(config)

        self.weight = config["loss"]["physics"]["weight"]
        self.stage_cfg = config["loss"]["physics"]["stages"]

    def forward(self, preds):
        total = 0.0

        if "pressure_step" in preds:
            p = preds["pressure_step"]["pressure"]
            K = preds["pressure_step"]["permeability"]

            res = self.residual.forward_pressure(p, K)
            total += self.stage_cfg["pressure_step"]["weight"] * res.pow(2).mean()

        if "transport_step" in preds:
            s = preds["transport_step"]["saturation"]
            vx = preds["transport_step"]["vx"]
            vy = preds["transport_step"]["vy"]

            res = self.residual.forward_transport(s, vx, vy)
            total += self.stage_cfg["transport_step"]["weight"] * res.pow(2).mean()

        return self.weight * total


class RolloutLoss(nn.Module):
    def __init__(self, decay=0.9):
        super().__init__()
        self.decay = decay

    def forward(self, traj_pred, traj_target):
        total = 0.0

        for i, (zp, zt) in enumerate(zip(traj_pred, traj_target)):
            w = self.decay ** i
            total += w * (zp - zt).pow(2).mean()

        return total


class FullLoss(nn.Module):
    def __init__(self, config):
        super().__init__()

        loss_cfg = config["loss"]

        self.pred_loss = PredictionLoss(
            normalize=loss_cfg["prediction"]["normalize"]
        )

        self.stage_loss = StagewiseLoss(
            weights=loss_cfg["stagewise_prediction"]["weights"]
        ) if loss_cfg["stagewise_prediction"]["enabled"] else None

        self.var_loss = VarianceLoss(
            gamma=loss_cfg["regularization"]["variance"]["gamma"]
        )

        self.cov_loss = CovarianceLoss()

        self.physics_loss = PhysicsLoss(config) if loss_cfg["physics"]["enabled"] else None

        self.rollout_loss = RolloutLoss(
            decay=config["training"]["rollout_training"]["weight_decay_per_step"]
        )

        self.weights = {
            "pred": loss_cfg["prediction"]["weight"],
            "var": loss_cfg["regularization"]["variance"]["weight"],
            "cov": loss_cfg["regularization"]["covariance"]["weight"],
        }

        self.physics_weight = loss_cfg["physics"]["weight"] if loss_cfg["physics"]["enabled"] else 0.0

    def forward(
        self,
        z_pred,
        z_target,
        stage_outputs=None,
        stage_targets=None,
        physics_preds=None,
        rollout_pred=None,
        rollout_target=None
    ):
        losses = {}

        losses["prediction"] = self.pred_loss(z_pred, z_target)

        if self.stage_loss is not None:
            losses["stagewise"] = self.stage_loss(stage_outputs, stage_targets)

        losses["variance"] = self.var_loss(z_pred)
        losses["covariance"] = self.cov_loss(z_pred)

        if self.physics_loss is not None and physics_preds is not None:
            losses["physics"] = self.physics_loss(physics_preds)

        if rollout_pred is not None and rollout_target is not None:
            losses["rollout"] = self.rollout_loss(rollout_pred, rollout_target)

        total = 0.0

        total += self.weights["pred"] * losses["prediction"]

        if "stagewise" in losses:
            total += losses["stagewise"]

        total += self.weights["var"] * losses["variance"]
        total += self.weights["cov"] * losses["covariance"]

        if "physics" in losses:
            total += self.physics_weight * losses["physics"]

        if "rollout" in losses:
            total += losses["rollout"]

        losses["total"] = total

        return losses
