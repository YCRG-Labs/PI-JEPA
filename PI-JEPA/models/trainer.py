import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from models.encoder import ViTEncoder, TargetEncoder, update_ema
from models.predictor import Predictor
from models.losses import FullLoss


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config["experiment"]["device"]

        self.student_encoder = ViTEncoder(config).to(self.device)
        self.teacher_encoder = TargetEncoder(ViTEncoder(config)).to(self.device)

        self.predictor = Predictor(config).to(self.device)

        self.loss_fn = FullLoss(config)

        self._init_teacher()

        self.optimizer = torch.optim.AdamW(
            list(self.student_encoder.parameters()) +
            list(self.predictor.parameters()),
            lr=config["training"]["optim"]["lr"],
            weight_decay=config["training"]["optim"]["weight_decay"],
            betas=tuple(config["training"]["optim"]["betas"])
        )

        self.scaler = GradScaler(enabled=(config["experiment"]["precision"] == "fp16"))

        self.tau_start = config["ema"]["schedule"]["tau_start"]
        self.tau_end = config["ema"]["schedule"]["tau_end"]
        self.total_epochs = config["training"]["epochs"]

    def _init_teacher(self):
        for s, t in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            t.data.copy_(s.data)
            t.requires_grad = False

    def _ema_tau(self, epoch):
        t = epoch / self.total_epochs
        tau = self.tau_end - (self.tau_end - self.tau_start) * (1 - t)
        return tau

    def _generate_mask(self, B, N, device):
        ratio = self.config["masking"]["context_ratio"]
        num_keep = int(N * ratio)

        mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        for i in range(B):
            idx = torch.randperm(N, device=device)[:num_keep]
            mask[i, idx] = True

        return mask

    def train_step(self, batch, epoch):
        self.student_encoder.train()
        self.predictor.train()

        x_t, x_tp1 = batch
        x_t = x_t.to(self.device)
        x_tp1 = x_tp1.to(self.device)

        with autocast(enabled=(self.config["experiment"]["precision"] == "fp16")):

            z_target = self.teacher_encoder(x_tp1)

            z_context_full = self.student_encoder(x_t)

            B, N, D = z_context_full.shape
            mask = self._generate_mask(B, N, x_t.device)

            z_context = z_context_full[mask].view(B, -1, D)

            z_pred, stage_outputs = self.predictor(z_context)

            losses = self.loss_fn(
                z_pred=z_pred,
                z_target=z_target,
                stage_outputs=stage_outputs,
                stage_targets=None
            )

            total_loss = losses["total"]

        self.optimizer.zero_grad()

        self.scaler.scale(total_loss).backward()

        if self.config["training"]["gradient"]["clip_norm"] is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.student_encoder.parameters()) +
                list(self.predictor.parameters()),
                self.config["training"]["gradient"]["clip_norm"]
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        tau = self._ema_tau(epoch)
        update_ema(self.student_encoder, self.teacher_encoder, tau)

        return losses

    def train_epoch(self, dataloader, epoch):
        logs = {}

        for batch in dataloader:
            losses = self.train_step(batch, epoch)

            for k, v in losses.items():
                logs[k] = logs.get(k, 0.0) + v.item()

        for k in logs:
            logs[k] /= len(dataloader)

        return logs

    @torch.no_grad()
    def evaluate_rollout(self, x0, steps):
        self.student_encoder.eval()
        self.predictor.eval()

        x0 = x0.to(self.device)

        z = self.student_encoder(x0)

        traj = self.predictor.rollout(z, steps)

        return traj

    def save(self, path):
        torch.save({
            "encoder": self.student_encoder.state_dict(),
            "predictor": self.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)

        self.student_encoder.load_state_dict(ckpt["encoder"])
        self.predictor.load_state_dict(ckpt["predictor"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
