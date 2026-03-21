import torch


def mse(pred, target):
    return torch.mean((pred - target) ** 2)


def rmse(pred, target):
    return torch.sqrt(mse(pred, target))


def mae(pred, target):
    return torch.mean(torch.abs(pred - target))


def relative_l2(pred, target, eps=1e-8):
    num = torch.sum((pred - target) ** 2)
    denom = torch.sum(target ** 2) + eps
    return torch.sqrt(num / denom)


def relative_l1(pred, target, eps=1e-8):
    num = torch.sum(torch.abs(pred - target))
    denom = torch.sum(torch.abs(target)) + eps
    return num / denom


def max_error(pred, target):
    return torch.max(torch.abs(pred - target))


def per_channel_mse(pred, target):
    B, C = pred.shape[:2]
    pred = pred.view(B, C, -1)
    target = target.view(B, C, -1)
    return torch.mean((pred - target) ** 2, dim=2)


def rollout_mse(pred_seq, target_seq):
    return torch.mean((pred_seq - target_seq) ** 2, dim=(2, 3, 4))


def rollout_rmse(pred_seq, target_seq):
    return torch.sqrt(rollout_mse(pred_seq, target_seq))


def rollout_mae(pred_seq, target_seq):
    return torch.mean(torch.abs(pred_seq - target_seq), dim=(2, 3, 4))


def rollout_relative_l2(pred_seq, target_seq, eps=1e-8):
    num = torch.sum((pred_seq - target_seq) ** 2, dim=(2, 3, 4))
    denom = torch.sum(target_seq ** 2, dim=(2, 3, 4)) + eps
    return torch.sqrt(num / denom)


def rollout_max_error(pred_seq, target_seq):
    return torch.amax(torch.abs(pred_seq - target_seq), dim=(2, 3, 4))


def temporal_consistency(pred_seq):
    diff = pred_seq[:, 1:] - pred_seq[:, :-1]
    return torch.mean(diff ** 2)


def energy(pred):
    return torch.sum(pred ** 2, dim=(1, 2, 3))


def rollout_energy_drift(pred_seq):
    e0 = energy(pred_seq[:, 0])
    et = energy(pred_seq[:, -1])
    return torch.mean(torch.abs(et - e0) / (e0 + 1e-8))


def physics_residual_metric(residual):
    return torch.mean(residual ** 2)


def compute_l2(pred, target):
    return torch.norm(pred - target) / torch.norm(target)
