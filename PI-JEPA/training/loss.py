import torch
import torch.nn as nn


class JEPAAlignmentLoss(nn.Module):
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def forward(self, z_pred, z_target):
        diff = z_pred - z_target

        if self.normalize:
            scale = z_target.pow(2).mean(dim=-1, keepdim=True) + 1e-6
            diff = diff / scale

        return diff.pow(2).mean()


class StagewiseAlignmentLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, stage_outputs, z_target):
        total = 0.0

        for name, z_stage in stage_outputs.items():
            w = self.weights.get(name, 1.0)

            diff = z_stage - z_target
            total += w * diff.pow(2).mean()

        return total


class VarianceRegularization(nn.Module):
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, z):
        std = torch.sqrt(z.var(dim=0) + 1e-6)
        return torch.relu(self.gamma - std).mean()


class CovarianceRegularization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        B, N, D = z.shape
        z = z.reshape(B * N, D)

        z = z - z.mean(dim=0)

        cov = (z.T @ z) / (z.shape[0] - 1)
        off_diag = cov - torch.diag(torch.diag(cov))

        return off_diag.pow(2).sum() / D


class RolloutConsistencyLoss(nn.Module):
    def __init__(self, decay=0.9):
        super().__init__()
        self.decay = decay

    def forward(self, traj_pred, traj_target):
        total = 0.0

        for t, (zp, zt) in enumerate(zip(traj_pred, traj_target)):
            w = self.decay ** t
            total += w * (zp - zt).pow(2).mean()

        return total


class LossBuilder(nn.Module):
    def __init__(self, config):
        super().__init__()

        loss_cfg = config["loss"]

        self.pred_loss = JEPAAlignmentLoss(
            normalize=loss_cfg["prediction"]["normalize"]
        )

        self.stage_loss = None
        if loss_cfg["stagewise_prediction"]["enabled"]:
            self.stage_loss = StagewiseAlignmentLoss(
                weights=loss_cfg["stagewise_prediction"]["weights"]
            )

        self.var_reg = VarianceRegularization(
            gamma=loss_cfg["regularization"]["variance"]["gamma"]
        )

        self.cov_reg = CovarianceRegularization()

        self.rollout_loss = RolloutConsistencyLoss(
            decay=config["training"]["rollout_training"]["weight_decay_per_step"]
        )

        self.weights = {
            "pred": loss_cfg["prediction"]["weight"],
            "var": loss_cfg["regularization"]["variance"]["weight"],
            "cov": loss_cfg["regularization"]["covariance"]["weight"],
        }

        self.physics_weight = loss_cfg["physics"]["weight"]
        self.use_physics = loss_cfg["physics"]["enabled"]

    def align_tokens(self, z_pred, z_target):
        if z_pred.shape[1] == z_target.shape[1]:
            return z_pred, z_target

        Np = z_pred.shape[1]
        return z_pred, z_target[:, :Np]

    def forward(
        self,
        z_pred,
        z_target,
        stage_outputs=None,
        rollout_pred=None,
        rollout_target=None,
        physics_loss=None
    ):
        losses = {}

        z_pred, z_target = self.align_tokens(z_pred, z_target)

        losses["jepa"] = self.pred_loss(z_pred, z_target)

        if self.stage_loss is not None and stage_outputs is not None:
            losses["stagewise"] = self.stage_loss(stage_outputs, z_target)

        losses["variance"] = self.var_reg(z_pred)
        losses["covariance"] = self.cov_reg(z_pred)

        if rollout_pred is not None and rollout_target is not None:
            losses["rollout"] = self.rollout_loss(rollout_pred, rollout_target)

        if self.use_physics and physics_loss is not None:
            losses["physics"] = physics_loss

        total = 0.0

        total += self.weights["pred"] * losses["jepa"]

        if "stagewise" in losses:
            total += losses["stagewise"]

        total += self.weights["var"] * losses["variance"]
        total += self.weights["cov"] * losses["covariance"]

        if "rollout" in losses:
            total += losses["rollout"]

        if "physics" in losses:
            total += self.physics_weight * losses["physics"]

        losses["total"] = total

        return losses
