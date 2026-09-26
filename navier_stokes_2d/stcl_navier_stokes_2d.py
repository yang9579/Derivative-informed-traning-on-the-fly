"""sTCL loss for the state map U -> (Y, P).

The surrogate predicts (y1, y2, p).  The strong-form residual is

    R(u) = ( R_mom(y, p, u), R_div(y), R_bc(y) ),
    R_mom = -mu Delta y + (y.grad)y + grad p - u,
    R_div = div y,
    R_bc  = y on dOmega.

Forward-mode differentiation of R along an input direction s = delta u gives the
exact directional tangent residual DR(u)[s]; because the surrogate carries a
pressure channel, the momentum-tangent block can vanish jointly with the
continuity block (the incompressible tangent the DIFNO labels solve).  No
derivative labels and no tangent solve are used.

Loss conditioning (``precond``).  Following the paper's conditioning analysis
(Sec. 3): freezing the surrogate state, the raw tangent-residual loss
||A w + b||^2 has Hessian 2 A^T A, so its condition number is kappa(A^T A) =
kappa(A)^2.  For the momentum block A contains -mu*Delta, so raw L2 gives kappa^2
conditioning and leaves the low-frequency Jacobian unconstrained.  A PSD
preconditioner M turns the Hessian into 2 A^T M A with condition number
kappa(A^T M A) << kappa(A)^2.  The available metrics include inverse-Laplacian,
Dirichlet-DST, Leray-projected, and isotropic Oseen-symbol variants.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict

import numpy as np
import torch
from torch.func import jvp as func_jvp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime_profile import end as profile_end
from runtime_profile import start as profile_start

try:
    from .navier_stokes_2d import (
        boundary_values,
        convect,
        divergence,
        grad_scalar,
        interior,
        lap,
        make_grid,
        stream_velocity,
    )
except ImportError:  # Support direct script execution.
    from navier_stokes_2d import (
        boundary_values,
        convect,
        divergence,
        grad_scalar,
        interior,
        lap,
        make_grid,
        stream_velocity,
    )


def sample_directions(u: torch.Tensor, n_modes: int = 8) -> torch.Tensor:
    """Fresh smooth zero-boundary control directions (same prior as the DIFNO bank)."""
    batch, _, gy, gx = u.shape
    xs = torch.linspace(0.0, 1.0, gx, device=u.device, dtype=u.dtype)
    ys = torch.linspace(0.0, 1.0, gy, device=u.device, dtype=u.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    max_m = min(n_modes, gy - 2, gx - 2)
    direction = torch.zeros_like(u)
    for k in range(1, max_m + 1):
        for l in range(1, max_m + 1):
            basis = torch.sin(k * math.pi * xx) * torch.sin(l * math.pi * yy)
            scale = float((k * k + l * l) ** -1.5)
            coeff = torch.randn(batch, 2, 1, 1, device=u.device, dtype=u.dtype)
            direction = direction + scale * coeff * basis[None, None]
    norm = direction.flatten(1).norm(dim=1).reshape(-1, 1, 1, 1)
    return direction / (norm + 1e-12)


def sample_directions_solenoidal(u: torch.Tensor, stream_kmax: int = 6,
                                 stream_decay: float = 1.5) -> torch.Tensor:
    """Fresh solenoidal zero-boundary directions = curl of a random stream function.

    Matches the prior of the forward-solve control U (which is sampled the same way),
    so DIFNO/sTCL/eval all probe the input (control) distribution.  Unit-normalized.
    """
    batch, _, gy, gx = u.shape
    x, y = make_grid(gy, gx, device=u.device, dtype=u.dtype)
    k = torch.arange(1, stream_kmax + 1, device=u.device, dtype=u.dtype)
    kk, ll = torch.meshgrid(k, k, indexing="ij")
    scale = (kk ** 2 + ll ** 2) ** (-stream_decay)
    b = torch.randn(batch, stream_kmax, stream_kmax, device=u.device, dtype=u.dtype) * scale[None]
    d = stream_velocity(b, x, y)
    norm = d.flatten(1).norm(dim=1).reshape(-1, 1, 1, 1)
    return d / (norm + 1e-12)


def draw_directions(u, mode="sine", n_modes=8, stream_kmax=6, stream_decay=1.5):
    """Dispatch the online direction sampler by mode (sine = generic control-space,
    solenoidal = curl-of-stream, matched to a forward-solve U prior)."""
    if mode == "solenoidal":
        return sample_directions_solenoidal(u, stream_kmax, stream_decay)
    return sample_directions(u, n_modes=n_modes)


def _hinv_weight(ny: int, nx: int, hx: float, hy: float, tau: float, device, dtype):
    ky = torch.fft.fftfreq(ny, d=hy, device=device, dtype=dtype) * (2 * np.pi)
    kx = torch.fft.rfftfreq(nx, d=hx, device=device, dtype=dtype) * (2 * np.pi)
    sym = ky[:, None] ** 2 + kx[None, :] ** 2
    return 1.0 / (sym + tau * sym.max().clamp_min(1e-20))            # M in Fourier


_PRECOND_POWER = {"hinv": 1, "hinv2": 2}   # M = (-Delta + tau Lmax)^-power (FFT-diagonal)


def _hinv_energy(r: torch.Tensor, hx: float, hy: float, tau: float, power: int = 1) -> torch.Tensor:
    """<r, M r> with M = (-Delta + tau*Lmax)^-power, interior field r (B, ny, nx)."""
    ny, nx = r.shape[-2:]
    rh = torch.fft.rfft2(r, norm="ortho")
    w = _hinv_weight(ny, nx, hx, hy, tau, r.device, r.dtype) ** power
    return ((rh.real ** 2 + rh.imag ** 2) * w[None]).mean()


_DST_CACHE = {}


def _dst_mats(n: int, h: float, device, dtype):
    """Orthonormal DST-I matrix S (n,n) and Dirichlet-Laplacian eigenvalues lam (n,).

    S diagonalizes the 3-point Laplacian with homogeneous Dirichlet BC exactly (the
    correct basis for a boundary-vanishing tangent field, unlike periodic FFT)."""
    key = (n, round(h, 12), device, dtype)
    if key not in _DST_CACHE:
        j = torch.arange(1, n + 1, device=device, dtype=dtype)
        S = torch.sin(math.pi * j[:, None] * j[None, :] / (n + 1)) * math.sqrt(2.0 / (n + 1))
        lam = (2.0 - 2.0 * torch.cos(math.pi * j / (n + 1))) / h ** 2
        _DST_CACHE[key] = (S, lam)
    return _DST_CACHE[key]


def _dst_hinv_energy(r: torch.Tensor, hx: float, hy: float, tau: float, power: int = 1) -> torch.Tensor:
    """<r, M r> with M = (-Delta_Dirichlet + tau*Lmax)^-power via separable DST-I."""
    ny, nx = r.shape[-2:]
    Sy, lamy = _dst_mats(ny, hy, r.device, r.dtype)
    Sx, lamx = _dst_mats(nx, hx, r.device, r.dtype)
    rh = torch.einsum("ij,bjk,lk->bil", Sy, r, Sx)          # 2D DST-I (orthonormal)
    sym = lamy[:, None] + lamx[None, :]
    w = (1.0 / (sym + tau * sym.max().clamp_min(1e-20))) ** power
    return (rh ** 2 * w[None]).mean()


def _oseen_energy(r: torch.Tensor, hx: float, hy: float, mu: float, b0: float,
                  tau: float, power: float = 1.0) -> torch.Tensor:
    """<r, M r> with M = (|Oseen symbol| + tau*max)^-power, FFT-diagonal Oseen:
    |A(k)| = sqrt( (mu|k|^2)^2 + (b0|k|)^2 )  (isotropic advection speed b0).

    power=1: residual-loss Hessian symbol |A|^2 * |A|^-1 = |A|  -> kappa = kappa(A).
    power=2: |A|^2 * |A|^-2 = 1 -> kappa = 1, i.e. the residual loss is preconditioned
             to behave like DIFNO's direct JVP-error loss (identity Hessian).
    """
    ny, nx = r.shape[-2:]
    ky = torch.fft.fftfreq(ny, d=hy, device=r.device, dtype=r.dtype) * (2 * np.pi)
    kx = torch.fft.rfftfreq(nx, d=hx, device=r.device, dtype=r.dtype) * (2 * np.pi)
    k2 = ky[:, None] ** 2 + kx[None, :] ** 2
    sym = torch.sqrt((mu * k2) ** 2 + (b0 ** 2) * k2)                # |Oseen symbol|
    w = (1.0 / (sym + tau * sym.max().clamp_min(1e-20))) ** power
    rh = torch.fft.rfft2(r, norm="ortho")
    return ((rh.real ** 2 + rh.imag ** 2) * w[None]).mean()


def _leray_project(r: torch.Tensor, hx: float, hy: float):
    """Solenoidal (divergence-free) part of a 2-component interior field r (B,2,ny,nx).

    Removes the curl-free (pressure-gradient) component of the momentum residual by
    FFT Helmholtz projection, so the score focuses on the velocity-tangent balance.
    Returns the two projected components (B, ny, nx).
    """
    ny, nx = r.shape[-2:]
    rx = torch.fft.rfft2(r[:, 0], norm="ortho")
    ry = torch.fft.rfft2(r[:, 1], norm="ortho")
    ky = (torch.fft.fftfreq(ny, d=hy, device=r.device, dtype=r.dtype) * (2 * np.pi))[:, None]
    kx = (torch.fft.rfftfreq(nx, d=hx, device=r.device, dtype=r.dtype) * (2 * np.pi))[None, :]
    k2 = (kx ** 2 + ky ** 2).clamp_min(1e-20)
    dot = kx * rx + ky * ry
    px = rx - kx * dot / k2
    py = ry - ky * dot / k2
    return (torch.fft.irfft2(px, s=(ny, nx), norm="ortho"),
            torch.fft.irfft2(py, s=(ny, nx), norm="ortho"))


def stcl_loss(
    model,
    u: torch.Tensor,
    *,
    n_sketch: int = 4,
    n_modes: int = 8,
    direction_mode: str = "sine",
    stream_kmax: int = 6,
    stream_decay: float = 1.5,
    directions: torch.Tensor | None = None,
    residual_weights: Dict[str, float] | None = None,
    precond: str = "leray_oseen",
    precond_div: str = "none",
    precond_tau: float = 0.005,
    oseen_speed: float = 3.0,
    oseen_power: float = 1.0,
    return_parts: bool = False,
):
    """Sketched tangent-consistency loss; optionally returns the per-block breakdown.

    directions   : optional (B, q, 2, gy, gx) precomputed sketch directions (the shared
                   DIFNO/sTCL bank).  If given, these q directions are used instead of
                   freshly sampling n_sketch smooth directions -- so sTCL and DIFNO see
                   identical directions.
    precond      : momentum-block residual metric; see README for all choices
    precond_div  : continuity-block conditioning (hinv / hinv2 / none)
    """
    weights = residual_weights or {}
    w_mom = float(weights.get("state", 1.0))
    w_div = float(weights.get("div_y", 1.0))
    w_bc = float(weights.get("boundary", 1.0))
    hx, hy = model.hx, model.hy

    def residual(uu: torch.Tensor):
        y1, y2, p = model.state_grid(uu)
        y_pred = torch.stack([y1, y2], dim=1)
        r_mom = (-model.mu * lap(y_pred, hx, hy) + convect(y_pred, y_pred, hx, hy)
                 + grad_scalar(p, hx, hy) - interior(uu))
        r_div = divergence(y_pred, hx, hy)
        r_bc = boundary_values(y_pred)
        return r_mom, r_div, r_bc

    def blocks(dres):
        d_mom, d_div, d_bc = dres
        _precond_pt = profile_start('preconditioner_krylov')
        if precond == "leray":                                   # solenoidal-projected + H^-1
            px, py = _leray_project(d_mom, hx, hy)
            mom = (_hinv_energy(px, hx, hy, precond_tau, 1)
                   + _hinv_energy(py, hx, hy, precond_tau, 1))
        elif precond == "leray_raw":                             # solenoidal-projected + raw L2
            px, py = _leray_project(d_mom, hx, hy)
            mom = (px ** 2).mean() + (py ** 2).mean()
        elif precond == "leray_oseen":                           # solenoidal-projected + Oseen^-1
            px, py = _leray_project(d_mom, hx, hy)
            mom = (_oseen_energy(px, hx, hy, model.mu, oseen_speed, precond_tau, oseen_power)
                   + _oseen_energy(py, hx, hy, model.mu, oseen_speed, precond_tau, oseen_power))
        elif precond == "oseen":                                 # Oseen^-1, NO Leray projection
            mom = (_oseen_energy(d_mom[:, 0], hx, hy, model.mu, oseen_speed, precond_tau, oseen_power)
                   + _oseen_energy(d_mom[:, 1], hx, hy, model.mu, oseen_speed, precond_tau, oseen_power))
        elif precond == "dst":                                   # DST-I (Dirichlet) H^-1
            mom = (_dst_hinv_energy(d_mom[:, 0], hx, hy, precond_tau, 1)
                   + _dst_hinv_energy(d_mom[:, 1], hx, hy, precond_tau, 1))
        elif precond == "leray_dst":                             # solenoidal-projected + DST H^-1
            px, py = _leray_project(d_mom, hx, hy)
            mom = (_dst_hinv_energy(px, hx, hy, precond_tau, 1)
                   + _dst_hinv_energy(py, hx, hy, precond_tau, 1))
        elif precond in _PRECOND_POWER:
            p = _PRECOND_POWER[precond]
            mom = (_hinv_energy(d_mom[:, 0], hx, hy, precond_tau, p)
                   + _hinv_energy(d_mom[:, 1], hx, hy, precond_tau, p))
        else:
            mom = (d_mom ** 2).mean()
        profile_end(_precond_pt)
        if precond_div in _PRECOND_POWER:
            dv = _hinv_energy(d_div, hx, hy, precond_tau, _PRECOND_POWER[precond_div])
        else:
            dv = (d_div ** 2).mean()
        return w_mom * mom, w_div * dv, w_bc * (d_bc ** 2).mean()

    mom_t = u.new_zeros(())
    div_t = u.new_zeros(())
    bc_t = u.new_zeros(())
    n = directions.shape[1] if directions is not None else n_sketch
    for s in range(n):
        direction = (directions[:, s] if directions is not None
                     else draw_directions(u, direction_mode, n_modes, stream_kmax, stream_decay))
        _pt = profile_start('fused_jvp_pde_residual')
        _, dres = func_jvp(residual, (u,), (direction,))
        profile_end(_pt)
        m, d, b = blocks(dres)
        mom_t = mom_t + m
        div_t = div_t + d
        bc_t = bc_t + b
    mom_t, div_t, bc_t = mom_t / n, div_t / n, bc_t / n
    total = mom_t + div_t + bc_t
    if return_parts:
        return total, {"mom": mom_t, "div": div_t, "bc": bc_t}
    return total
