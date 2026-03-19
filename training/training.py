import torch
import math
from torch.cuda.amp import autocast, GradScaler

from training.loss import LossBuilder
from training.ema import update_ema


class Engine:
    def __init__(
        self,
        config,
        model,             
        optimizer,
        train_loader,
        val_loader=None
    ):
        self.cfg = config
        self.device = config["experiment"]["device"]

        self.model = model.to(self.device)

        self.optimizer = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.loss_builder = LossBuilder(config)

        self.scaler = GradScaler(
            enabled=(config["experiment"]["precision"] == "fp16")
        )

        self.epochs = config["training"]["epochs"]

        # EMA
        self.tau_start = config["ema"]["schedule"]["tau_start"]
        self.tau_end = config["ema"]["schedule"]["tau_end"]

    # -------------------------
    # EMA SCHEDULE
    # -------------------------
    def _ema_tau(self, epoch):
        t = epoch / self.epochs
        return self.tau_start + (self.tau_end - self.tau_start) * (
            (1 - math.cos(math.pi * t)) / 2
        )

    # -------------------------
    # RANDOM MASK SAMPLING (PATCH SPACE)
    # -------------------------
    def _sample_indices(self, B, N, device):
        ratio = self.cfg["masking"]["context_ratio"]

        num_context = int(N * ratio)

        all_idx = torch.arange(N, device=device).unsqueeze(0).repeat(B, 1)

        context_indices = []
        target_indices = []

        for b in range(B):
            perm = torch.randperm(N, device=device)

            context = perm[:num_context]
            target = perm[num_context:]

            context_indices.append(context)
            target_indices.append(target)

        context_indices = torch.stack(context_indices, dim=0)
        target_indices = torch.stack(target_indices, dim=0)

        return context_indices, target_indices

    # -------------------------
    # TRAIN STEP (FIXED)
    # -------------------------
    def train_one_step(self, batch, epoch):
        self.model.train()

        x, _ = batch   # assume (x_t, x_tp1) but JEPA uses only x
        x = x.to(self.device)

        with autocast(enabled=(self.cfg["experiment"]["precision"] == "fp16")):

            with torch.no_grad():
                dummy = self.model.encode(x)
                B, N, _ = dummy.shape

            context_indices, target_indices = self._sample_indices(
                B, N, x.device
            )

            z_pred, z_target = self.model(
                x,
                context_indices,
                target_indices
            )

            losses = self.loss_builder(
                z_pred=z_pred,
                z_target=z_target
            )

            total_loss = losses["total"]

        self.optimizer.zero_grad()

        self.scaler.scale(total_loss).backward()

        if self.cfg["training"]["gradient"]["clip_norm"] is not None:
            self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg["training"]["gradient"]["clip_norm"]
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        tau = self._ema_tau(epoch)
        update_ema(
            self.model.encoder,
            self.model.target_encoder,
            tau
        )

        return losses

    def train_one_epoch(self, epoch):
        logs = {}

        for batch in self.train_loader:
            losses = self.train_one_step(batch, epoch)

            for k, v in losses.items():
                logs[k] = logs.get(k, 0.0) + v.item()

        for k in logs:
            logs[k] /= len(self.train_loader)

        return logs

    @torch.no_grad()
    def validate(self):
        if self.val_loader is None:
            return {}

        self.model.eval()

        logs = {}

        for batch in self.val_loader:
            x, _ = batch
            x = x.to(self.device)

            B = x.shape[0]

            dummy = self.model.encode(x)
            _, N, _ = dummy.shape

            context_indices, target_indices = self._sample_indices(
                B, N, x.device
            )

            z_pred, z_target = self.model(
                x,
                context_indices,
                target_indices
            )

            losses = self.loss_builder(
                z_pred=z_pred,
                z_target=z_target
            )

            for k, v in losses.items():
                logs[k] = logs.get(k, 0.0) + v.item()

        for k in logs:
            logs[k] /= len(self.val_loader)

        return logs
    
    def fit(self):
        history = []

        for epoch in range(self.epochs):
            train_logs = self.train_one_epoch(epoch)
            val_logs = self.validate()

            history.append({
                "epoch": epoch,
                "train": train_logs,
                "val": val_logs
            })

        return history