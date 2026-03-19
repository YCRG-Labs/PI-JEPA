import torch
import torch.nn.functional as F


# -------------------------
# GRADIENTS
# -------------------------
def grad_x(u, dx):
    u_pad = F.pad(u, (1, 1, 0, 0), mode="reflect")
    return (u_pad[:, :, :, 2:] - u_pad[:, :, :, :-2]) / (2 * dx + 1e-6)


def grad_y(u, dy):
    u_pad = F.pad(u, (0, 0, 1, 1), mode="reflect")
    return (u_pad[:, :, 2:, :] - u_pad[:, :, :-2, :]) / (2 * dy + 1e-6)


def divergence(fx, fy, dx, dy):
    fx_pad = F.pad(fx, (1, 1, 0, 0), mode="reflect")
    fy_pad = F.pad(fy, (0, 0, 1, 1), mode="reflect")

    dfx_dx = (fx_pad[:, :, :, 2:] - fx_pad[:, :, :, :-2]) / (2 * dx + 1e-6)
    dfy_dy = (fy_pad[:, :, 2:, :] - fy_pad[:, :, :-2, :]) / (2 * dy + 1e-6)

    return dfx_dx + dfy_dy


# -------------------------
# HELPERS
# -------------------------
def effective_saturation(Sw, Swr=0.0, Snr=0.0):
    return (Sw - Swr) / (1.0 - Swr - Snr + 1e-6)


def rel_perm(Sw, krw_exp, kro_exp, Swr=0.0, Snr=0.0):
    Se = effective_saturation(Sw, Swr, Snr)
    Se = Se.clamp(1e-4, 1.0 - 1e-4)

    krw = Se ** krw_exp
    kro = (1.0 - Se) ** kro_exp

    return krw, kro


def mobility(Sw, mu_w, mu_o, krw_exp, kro_exp):
    krw, kro = rel_perm(Sw, krw_exp, kro_exp)

    lambda_w = krw / (mu_w + 1e-6)
    lambda_o = kro / (mu_o + 1e-6)

    lambda_t = lambda_w + lambda_o + 1e-6
    fw = lambda_w / lambda_t

    return lambda_t, fw


# -------------------------
# PRESSURE LOSS
# -------------------------
def physics_loss_pressure(
    p, Sw, K, q,
    mu_w, mu_o,
    krw_exp, kro_exp,
    dx=1.0, dy=1.0
):
    # constrain outputs
    p = torch.tanh(p) * 5.0
    Sw = torch.sigmoid(Sw)

    lambda_t, _ = mobility(Sw, mu_w, mu_o, krw_exp, kro_exp)

    dp_dx = grad_x(p, dx)
    dp_dy = grad_y(p, dy)

    vx = -K * lambda_t * dp_dx
    vy = -K * lambda_t * dp_dy

    div_term = divergence(vx, vy, dx, dy)

    residual = div_term - q

    # 🔥 NO NORMALIZATION
    loss = (residual ** 2).mean()

    # small gradient encouragement
    grad_penalty = dp_dx.abs().mean() + dp_dy.abs().mean()

    return loss + 0.05 * grad_penalty


# -------------------------
# SATURATION LOSS
# -------------------------
def physics_loss_saturation(
    Sw_pred, Sw_true, p, K, q_w, phi,
    mu_w, mu_o,
    krw_exp, kro_exp,
    dx=1.0, dy=1.0, dt=1.0
):
    p = torch.tanh(p) * 5.0
    Sw_pred = torch.sigmoid(Sw_pred)
    Sw_true = torch.sigmoid(Sw_true)

    lambda_t, fw = mobility(Sw_pred, mu_w, mu_o, krw_exp, kro_exp)

    dp_dx = grad_x(p, dx)
    dp_dy = grad_y(p, dy)

    vx = -K * lambda_t * dp_dx
    vy = -K * lambda_t * dp_dy

    fx = fw * vx
    fy = fw * vy

    div_term = divergence(fx, fy, dx, dy)

    dSw_dt = (Sw_pred - Sw_true) / (dt + 1e-6)
    q_term = q_w / (phi + 1e-6)

    residual = dSw_dt + div_term - q_term

    # 🔥 NO NORMALIZATION
    loss = (residual ** 2).mean()

    grad_penalty = dp_dx.abs().mean() + dp_dy.abs().mean()

    return loss + 0.05 * grad_penalty