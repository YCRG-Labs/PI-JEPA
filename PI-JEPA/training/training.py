import torch
import math
from torch.cuda.amp import autocast, GradScaler

from training.loss import LossBuilder
from models.encoder import update_ema


class Engine:
    def __init__(
        self,
        config,
        student_encoder,
        teacher_encoder,
        predictor,
        optimizer,
        train_loader,
        val_loader=None
    ):
        self.cfg = config
        self.device = config["experiment"]["device"]

        self.student = student_encoder
        self.teacher = teacher_encoder
        self.predictor = predictor

        self.optimizer = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.loss_builder = LossBuilder(config)

        self.scaler = GradScaler(
            enabled=(config["experiment"]["precision"] == "fp16")
        )

        self.epochs = config["training"]["epochs"]

        self.tau_start = config["ema"]["schedule"]["tau_start"]
        self.tau_end = config["ema"]["schedule"]["tau_end"]

        self.rollout_steps = config["training"]["rollout_training"]["steps"]

        self.mask_cfg = config["masking"]

    def _ema_tau(self, epoch):
        t = epoch / self.epochs
        return self.tau_start + (self.tau_end - self.tau_start) * (1 - math.cos(math.pi * t)) / 2

    def _block_mask(self, B, H, W, device):
        ratio = self.mask_cfg["context_ratio"]
        block = self.mask_cfg["block_size"]

        total = H * W
        keep = int(total * ratio)

        mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)

        for b in range(B):
            filled = 0
            while filled < keep:
                i = torch.randint(0, H - block + 1, (1,), device=device)
                j = torch.randint(0, W - block + 1, (1,), device=device)

                mask[b, i:i+block, j:j+block] = True
                filled = mask[b].sum().item()

        return mask.view(B, -1)

    def _extract_context(self, z, mask):
        B, N, D = z.shape
        z_ctx = z[mask].view(B, -1, D)
        return z_ctx

    @torch.no_grad()
    def _teacher_targets(self, x_seq):
        targets = []
        for x in x_seq:
            targets.append(self.teacher(x))
        return targets

    def _build_rollout_targets(self, x_t, steps):
        x_seq = [x_t]
        for _ in range(steps):
            x_seq.append(x_seq[-1])
        return self._teacher_targets(x_seq[1:])

    def train_one_step(self, batch, epoch):
        self.student.train()
        self.predictor.train()

        x_t, x_tp1 = batch
        x_t = x_t.to(self.device)
        x_tp1 = x_tp1.to(self.device)

        with autocast(enabled=(self.cfg["experiment"]["precision"] == "fp16")):

            z_teacher_tp1 = self.teacher(x_tp1)

            z_student_full = self.student(x_t)

            B, N, D = z_student_full.shape
            H = W = int(N ** 0.5)

            mask = self._block_mask(B, H, W, x_t.device)

            z_context = self._extract_context(z_student_full, mask)

            z_pred, stage_outputs = self.predictor(z_context)

            rollout_pred = None
            rollout_target = None

            if self.rollout_steps > 0:
                z_roll = z_context
                rollout_pred = []

                for _ in range(self.rollout_steps):
                    z_roll, _ = self.predictor(z_roll)
                    rollout_pred.append(z_roll)

                rollout_target = self._build_rollout_targets(x_tp1, self.rollout_steps)

            losses = self.loss_builder(
                z_pred=z_pred,
                z_target=z_teacher_tp1,
                stage_outputs=stage_outputs,
                rollout_pred=rollout_pred,
                rollout_target=rollout_target,
                physics_loss=None
            )

            total_loss = losses["total"]

        self.optimizer.zero_grad()

        self.scaler.scale(total_loss).backward()

        if self.cfg["training"]["gradient"]["clip_norm"] is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.student.parameters()) +
                list(self.predictor.parameters()),
                self.cfg["training"]["gradient"]["clip_norm"]
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        tau = self._ema_tau(epoch)
        update_ema(self.student, self.teacher, tau)

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

        self.student.eval()
        self.predictor.eval()

        logs = {}

        for batch in self.val_loader:
            x_t, x_tp1 = batch
            x_t = x_t.to(self.device)
            x_tp1 = x_tp1.to(self.device)

            z_teacher_tp1 = self.teacher(x_tp1)
            z_student_full = self.student(x_t)

            B, N, D = z_student_full.shape
            H = W = int(N ** 0.5)

            mask = self._block_mask(B, H, W, x_t.device)
            z_context = self._extract_context(z_student_full, mask)

            z_pred, stage_outputs = self.predictor(z_context)

            losses = self.loss_builder(
                z_pred=z_pred,
                z_target=z_teacher_tp1,
                stage_outputs=stage_outputs
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
