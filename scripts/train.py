#!/usr/bin/env python
"""Train PI-JEPA model on Darcy flow data."""

import os
import sys
import math
import torch
import torch.optim as optim

# Add PI-JEPA directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PI-JEPA"))

from models import ViTEncoder, Predictor, PIJEPA, Decoder
from training import LossBuilder, update_ema
from utils import load_config, Logger, save_checkpoint

from neuralop.data.datasets import load_darcy_flow_small


def get_device(cfg):
    if cfg["experiment"].get("device", None) is not None:
        return torch.device(cfg["experiment"]["device"])
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def set_seed(seed, deterministic=False):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def build_model(cfg, device):
    patch_size = cfg["model"]["encoder"]["patch_size"]

    encoder = ViTEncoder(cfg).to(device)
    target_encoder = ViTEncoder(cfg).to(device)

    predictors = [
        Predictor(cfg).to(device)
        for _ in range(cfg["model"]["num_predictors"])
    ]

    model = PIJEPA(
        encoder=encoder,
        target_encoder=target_encoder,
        predictors=predictors,
        embed_dim=cfg["model"]["encoder"]["embed_dim"],
        num_patches=None,
        patch_size=patch_size,
    ).to(device)

    decoder = Decoder(**cfg["decoder"]).to(device)

    for p in target_encoder.parameters():
        p.requires_grad = False

    target_encoder.load_state_dict(encoder.state_dict())

    return model, decoder


def build_dataloader(cfg):
    train_loader, _, _ = load_darcy_flow_small(
        n_train=1000,
        batch_size=cfg["training"]["batch_size"],
        test_resolutions=[64],
        n_tests=[100],
        test_batch_sizes=[32],
        encode_output=False,
    )
    return train_loader


def get_ema_tau(cfg, epoch):
    tau_start = cfg["ema"]["schedule"]["tau_start"]
    tau_end = cfg["ema"]["schedule"]["tau_end"]
    total_epochs = cfg["training"]["epochs"]

    t = epoch / total_epochs
    tau = tau_end - (tau_end - tau_start) * (math.cos(math.pi * t) + 1) / 2
    return tau


def sample_block_mask_indices(B, grid_size, device, context_ratio):
    context_idx = []
    target_idx = []

    for _ in range(B):
        mask = torch.zeros(grid_size, grid_size, dtype=torch.bool)

        block_size = max(1, int(grid_size * (1 - context_ratio)))

        x = torch.randint(0, grid_size - block_size + 1, (1,))
        y = torch.randint(0, grid_size - block_size + 1, (1,))

        mask[x:x+block_size, y:y+block_size] = True

        flat = mask.flatten()

        target = torch.where(flat)[0]
        context = torch.where(~flat)[0]

        context_idx.append(context)
        target_idx.append(target)

    return torch.stack(context_idx).to(device), torch.stack(target_idx).to(device)


def train(config_path="configs/darcy.yaml"):
    cfg = load_config(config_path)

    set_seed(cfg["experiment"]["seed"])
    device = get_device(cfg)

    logger = Logger(cfg["logging"]["base_dir"], cfg["experiment"]["name"])
    logger.save_config(cfg)

    model, decoder = build_model(cfg, device)
    loss_fn = LossBuilder(cfg)

    loader = build_dataloader(cfg)

    optimizer = optim.AdamW(
        list(model.parameters()) + list(decoder.parameters()),
        lr=float(cfg["training"]["optim"]["lr"]),
        weight_decay=float(cfg["training"]["optim"]["weight_decay"]),
        betas=tuple(cfg["training"]["optim"]["betas"])
    )

    context_ratio = cfg["masking"]["context_ratio"]
    global_step = 0

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        decoder.train()

        for batch in loader:
            x = batch["x"].to(device).float()
            y = batch["y"].to(device).float()

            if x.dim() == 3:
                x = x.unsqueeze(1)
            if y.dim() == 3:
                y = y.unsqueeze(1)

            x_input = torch.cat([y, x], dim=1)

            B, _, H, W = x_input.shape
            patch_size = cfg["model"]["encoder"]["patch_size"]

            assert H % patch_size == 0

            grid_size = H // patch_size

            context_idx, target_idx = sample_block_mask_indices(
                B, grid_size, device, context_ratio
            )

            num_patches = grid_size * grid_size
            assert target_idx.max() < num_patches

            z_pred, z_target = model(x_input, context_idx, target_idx)

            z_full = model.encode(x_input)

            z_recon = z_full.clone()
            z_recon = z_recon.scatter(
                1,
                target_idx.unsqueeze(-1).expand_as(z_pred),
                z_pred
            )

            x_pred = decoder(z_recon)

            K = x
            q = torch.zeros_like(x)
            q_w = torch.zeros_like(x)
            phi = torch.ones_like(x)

            losses = loss_fn(
                z_pred=z_pred,
                z_target=z_target,
                x_pred=x_pred,
                x_true=x_input,
                K=K,
                q=q,
                q_w=q_w,
                phi=phi
            )

            loss = losses["total"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tau = get_ema_tau(cfg, epoch)
            update_ema(model.encoder, model.target_encoder, tau=tau)

            global_step += 1

            if global_step % 50 == 0:
                logger.log_metrics(
                    {k: v.item() for k, v in losses.items()},
                    global_step
                )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/darcy.yaml", help="Path to config file")
    args = parser.parse_args()
    train(args.config)
