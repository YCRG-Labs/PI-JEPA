import torch
import torch.nn.functional as F


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


def physics_loss_pressure(
    p, Sw, K, q,
    mu_w, mu_o,
    krw_exp, kro_exp,
    dx=1.0, dy=1.0
):
    p = torch.tanh(p) * 5.0
    Sw = torch.sigmoid(Sw)

    lambda_t, _ = mobility(Sw, mu_w, mu_o, krw_exp, kro_exp)

    dp_dx = grad_x(p, dx)
    dp_dy = grad_y(p, dy)

    vx = -K * lambda_t * dp_dx
    vy = -K * lambda_t * dp_dy

    div_term = divergence(vx, vy, dx, dy)

    residual = div_term - q

    loss = (residual ** 2).mean()

    grad_penalty = dp_dx.abs().mean() + dp_dy.abs().mean()

    return loss + 0.05 * grad_penalty


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

    loss = (residual ** 2).mean()

    grad_penalty = dp_dx.abs().mean() + dp_dy.abs().mean()

    return loss + 0.05 * grad_penalty


class BrooksCoreyModel:
    """Brooks-Corey relative permeability model."""

    def __init__(self, S_wr: float = 0.0, S_nr: float = 0.0, lambda_bc: float = 2.0):
        self.S_wr = S_wr
        self.S_nr = S_nr
        self.lambda_bc = lambda_bc

    def effective_saturation(self, S_w: torch.Tensor) -> torch.Tensor:
        """S_e = (S_w - S_wr) / (1 - S_wr - S_nr)."""
        denominator = 1.0 - self.S_wr - self.S_nr
        S_e = (S_w - self.S_wr) / (denominator + 1e-8)
        return S_e.clamp(0.0, 1.0)

    def relative_permeability_water(self, S_e: torch.Tensor) -> torch.Tensor:
        """k_rw = S_e^((2 + 3λ) / λ)."""
        S_e_safe = S_e.clamp(1e-8, 1.0)
        exponent = (2.0 + 3.0 * self.lambda_bc) / self.lambda_bc
        return S_e_safe ** exponent

    def relative_permeability_nonwetting(self, S_e: torch.Tensor) -> torch.Tensor:
        """k_rn = (1 - S_e)² · (1 - S_e^((2 + λ) / λ))."""
        S_e_safe = S_e.clamp(0.0, 1.0 - 1e-8)
        term1 = (1.0 - S_e_safe) ** 2
        exponent = (2.0 + self.lambda_bc) / self.lambda_bc
        term2 = 1.0 - S_e_safe ** exponent
        return term1 * term2

    def capillary_pressure(self, S_e: torch.Tensor, P_entry: float = 1.0) -> torch.Tensor:
        """P_c = P_entry · S_e^(-1/λ)."""
        S_e_safe = S_e.clamp(1e-8, 1.0)
        exponent = -1.0 / self.lambda_bc
        return P_entry * (S_e_safe ** exponent)


class TwoPhaseDarcyPhysics:
    """Two-phase Darcy flow with capillary pressure."""

    def __init__(
        self,
        brooks_corey: BrooksCoreyModel,
        mu_w: float = 1.0,
        mu_n: float = 1.0,
        collocation_size: int = 32,
        dx: float = 1.0,
        dy: float = 1.0
    ):
        self.brooks_corey = brooks_corey
        self.mu_w = mu_w
        self.mu_n = mu_n
        self.collocation_size = collocation_size
        self.dx = dx
        self.dy = dy

    def fractional_flow(self, S_w: torch.Tensor) -> torch.Tensor:
        """f_w = (k_rw/μ_w) / (k_rw/μ_w + k_rn/μ_n)."""
        S_e = self.brooks_corey.effective_saturation(S_w)
        k_rw = self.brooks_corey.relative_permeability_water(S_e)
        k_rn = self.brooks_corey.relative_permeability_nonwetting(S_e)
        lambda_w = k_rw / (self.mu_w + 1e-8)
        lambda_n = k_rn / (self.mu_n + 1e-8)
        lambda_total = lambda_w + lambda_n + 1e-8
        return lambda_w / lambda_total

    def _compute_mobilities(self, S_w: torch.Tensor):
        S_e = self.brooks_corey.effective_saturation(S_w)
        k_rw = self.brooks_corey.relative_permeability_water(S_e)
        k_rn = self.brooks_corey.relative_permeability_nonwetting(S_e)
        lambda_w = k_rw / (self.mu_w + 1e-8)
        lambda_n = k_rn / (self.mu_n + 1e-8)
        lambda_T = lambda_w + lambda_n + 1e-8
        return lambda_w, lambda_n, lambda_T

    def pressure_residual(
        self,
        p_w: torch.Tensor,
        S_w: torch.Tensor,
        K: torch.Tensor,
        q_T: torch.Tensor
    ) -> torch.Tensor:
        """R_1: -∇·(λ_T K ∇p_w) + ∇·(λ_n K ∇P_c(S_w)) = q_T."""
        # Ensure 4D tensors for consistent processing
        if p_w.dim() == 3:
            p_w = p_w.unsqueeze(1)
        if S_w.dim() == 3:
            S_w = S_w.unsqueeze(1)
        if K.dim() == 3:
            K = K.unsqueeze(1)
        if q_T.dim() == 3:
            q_T = q_T.unsqueeze(1)

        # Compute mobilities
        lambda_w, lambda_n, lambda_T = self._compute_mobilities(S_w)

        # Compute effective saturation for capillary pressure
        S_e = self.brooks_corey.effective_saturation(S_w)

        # Compute capillary pressure
        P_c = self.brooks_corey.capillary_pressure(S_e)

        # Term 1: -∇·(λ_T K ∇p_w)
        # Compute pressure gradient
        dp_dx = grad_x(p_w, self.dx)
        dp_dy = grad_y(p_w, self.dy)

        # Compute flux: -λ_T K ∇p_w
        flux_x_1 = -lambda_T * K * dp_dx
        flux_y_1 = -lambda_T * K * dp_dy

        # Compute divergence of flux
        div_term_1 = divergence(flux_x_1, flux_y_1, self.dx, self.dy)

        # Term 2: ∇·(λ_n K ∇P_c(S_w))
        # Compute capillary pressure gradient
        dPc_dx = grad_x(P_c, self.dx)
        dPc_dy = grad_y(P_c, self.dy)

        # Compute capillary flux: λ_n K ∇P_c
        flux_x_2 = lambda_n * K * dPc_dx
        flux_y_2 = lambda_n * K * dPc_dy

        # Compute divergence of capillary flux
        div_term_2 = divergence(flux_x_2, flux_y_2, self.dx, self.dy)

        # Compute residual: -∇·(λ_T K ∇p_w) + ∇·(λ_n K ∇P_c) - q_T
        # Note: div_term_1 already has the negative sign from the flux definition
        residual = -div_term_1 + div_term_2 - q_T

        return residual

    def saturation_residual(
        self,
        S_w_pred: torch.Tensor,
        S_w_true: torch.Tensor,
        p_w: torch.Tensor,
        K: torch.Tensor,
        phi: torch.Tensor,
        q_w: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """R_2: φ·∂S_w/∂t + ∇·(f_w · v_T) = q_w."""
        # Ensure 4D tensors for consistent processing
        if S_w_pred.dim() == 3:
            S_w_pred = S_w_pred.unsqueeze(1)
        if S_w_true.dim() == 3:
            S_w_true = S_w_true.unsqueeze(1)
        if p_w.dim() == 3:
            p_w = p_w.unsqueeze(1)
        if K.dim() == 3:
            K = K.unsqueeze(1)
        if phi.dim() == 3:
            phi = phi.unsqueeze(1)
        if q_w.dim() == 3:
            q_w = q_w.unsqueeze(1)

        # Compute time derivative: ∂S_w/∂t ≈ (S_w_pred - S_w_true) / dt
        dSw_dt = (S_w_pred - S_w_true) / (dt + 1e-8)

        # Compute mobilities using predicted saturation
        _, _, lambda_T = self._compute_mobilities(S_w_pred)

        # Compute fractional flow
        f_w = self.fractional_flow(S_w_pred)

        # Compute pressure gradient
        dp_dx = grad_x(p_w, self.dx)
        dp_dy = grad_y(p_w, self.dy)

        # Compute total Darcy velocity: v_T = -λ_T K ∇p_w
        v_T_x = -lambda_T * K * dp_dx
        v_T_y = -lambda_T * K * dp_dy

        # Compute water flux: f_w · v_T
        flux_w_x = f_w * v_T_x
        flux_w_y = f_w * v_T_y

        # Compute divergence of water flux
        div_flux_w = divergence(flux_w_x, flux_w_y, self.dx, self.dy)

        # Compute residual: φ·∂S_w/∂t + ∇·(f_w · v_T) - q_w
        residual = phi * dSw_dt + div_flux_w - q_w

        return residual