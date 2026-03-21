import torch


def rollout(
    model,
    x_init,
    steps,
    decoder,
    noise_std=0.0,
    device="cuda"
):
    model.eval()
    x = x_init.to(device)
    outputs = []

    with torch.no_grad():
        for _ in range(steps):
            z_full = model.encoder(x)

            B, N, D = z_full.shape

            context_indices = torch.arange(N, device=device).unsqueeze(0).repeat(B, 1)
            target_indices = context_indices

            z_pred = model.predict_latent(
                z_full,
                context_indices,
                target_indices
            )

            if noise_std > 0.0:
                z_pred = z_pred + noise_std * torch.randn_like(z_pred)

            x = decoder(z_pred)

            outputs.append(x)

    return torch.stack(outputs, dim=1)


def rollout_with_metrics(
    model,
    x_init,
    ground_truth,
    steps,
    decoder,
    metric_fn,
    noise_std=0.0,
    device="cuda"
):
    model.eval()
    x = x_init.to(device)
    outputs = []
    metrics = []

    with torch.no_grad():
        for t in range(steps):
            z_full = model.encoder(x)

            B, N, D = z_full.shape

            context_indices = torch.arange(N, device=device).unsqueeze(0).repeat(B, 1)
            target_indices = context_indices

            z_pred = model.predict_latent(
                z_full,
                context_indices,
                target_indices
            )

            if noise_std > 0.0:
                z_pred = z_pred + noise_std * torch.randn_like(z_pred)

            x = decoder(z_pred)

            outputs.append(x)

            if ground_truth is not None:
                gt = ground_truth[:, t].to(device)
                metrics.append(metric_fn(x, gt))

    outputs = torch.stack(outputs, dim=1)

    if len(metrics) > 0:
        metrics = torch.stack(metrics, dim=1)
        return outputs, metrics

    return outputs


def single_step(
    model,
    x,
    decoder,
    noise_std=0.0,
    device="cuda"
):
    model.eval()

    with torch.no_grad():
        x = x.to(device)

        z_full = model.encoder(x)

        B, N, D = z_full.shape

        context_indices = torch.arange(N, device=device).unsqueeze(0).repeat(B, 1)
        target_indices = context_indices

        z_pred = model.predict_latent(
            z_full,
            context_indices,
            target_indices
        )

        if noise_std > 0.0:
            z_pred = z_pred + noise_std * torch.randn_like(z_pred)

        x_next = decoder(z_pred)

    return x_next
