"""Navier--Stokes 2D control-to-state operators and residual helpers.

The benchmark uses

    -mu Delta y + (y . grad)y + grad p = u + f,
    div y = 0,
    y = 0 on the boundary.

The fixed forcing is ``f = 0`` and viscosity is stored with each dataset. The
production generator samples a smooth solenoidal control and solves the PDE
forward. The surrogate predicts ``(y1, y2, p)``, and the strong-form residual
includes ``grad p``. This module provides the finite-difference operators,
smooth-field helpers, and residual components shared by data generation and
training.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

MU_FIXED = 1.0
FORCE_ZERO = True
Y_NAMES = ("y1", "y2")
U_NAMES = ("u1", "u2")

# Default smooth-field configuration used by helper routines.
STREAM_KMAX = 2          # stream-function modes k, l = 1..STREAM_KMAX
STREAM_DECAY = 1.5       # coefficient decay (k^2+l^2)^(-decay)
PRESSURE_MMAX = 2        # pressure cosine modes m, n = 0..PRESSURE_MMAX
PRESSURE_DECAY = 1.5
VEL_AMP_MIN = 2.5        # per-sample target ||y||_inf drawn uniformly in [min, max]
VEL_AMP_MAX = 3.5        # amp~3 => conv/diff(RMS)~0.1, Re~||y||L/nu~3 (clearly nonlinear, steady)
PRESSURE_AMP = 0.3       # per-sample target ||p||_inf


def make_grid(
    gy: int,
    gx: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = torch.linspace(0.0, 1.0, gx, device=device, dtype=dtype)
    ys = torch.linspace(0.0, 1.0, gy, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return xx.contiguous(), yy.contiguous()


# --------------------------------------------------------------------------- #
# Finite-difference stencils (centered, interior-only).                        #
# --------------------------------------------------------------------------- #
def interior(v: torch.Tensor) -> torch.Tensor:
    return v[..., 1:-1, 1:-1]


def ddx(v: torch.Tensor, hx: float) -> torch.Tensor:
    return (v[..., 1:-1, 2:] - v[..., 1:-1, :-2]) / (2.0 * hx)


def ddy(v: torch.Tensor, hy: float) -> torch.Tensor:
    return (v[..., 2:, 1:-1] - v[..., :-2, 1:-1]) / (2.0 * hy)


def lap(v: torch.Tensor, hx: float, hy: float) -> torch.Tensor:
    return (
        (v[..., 1:-1, 2:] - 2.0 * v[..., 1:-1, 1:-1] + v[..., 1:-1, :-2]) / hx**2
        + (v[..., 2:, 1:-1] - 2.0 * v[..., 1:-1, 1:-1] + v[..., :-2, 1:-1]) / hy**2
    )


def convect(a: torch.Tensor, b: torch.Tensor, hx: float, hy: float) -> torch.Tensor:
    ai = interior(a)
    bx = ddx(b, hx)
    by = ddy(b, hy)
    out1 = ai[:, 0] * bx[:, 0] + ai[:, 1] * by[:, 0]
    out2 = ai[:, 0] * bx[:, 1] + ai[:, 1] * by[:, 1]
    return torch.stack([out1, out2], dim=1)


def divergence(v: torch.Tensor, hx: float, hy: float) -> torch.Tensor:
    return ddx(v[:, 0], hx) + ddy(v[:, 1], hy)


def grad_scalar(p: torch.Tensor, hx: float, hy: float) -> torch.Tensor:
    """Interior gradient of a scalar field p (B, gy, gx) -> (B, 2, gy-2, gx-2)."""
    return torch.stack([ddx(p, hx), ddy(p, hy)], dim=1)


def boundary_values(y: torch.Tensor) -> torch.Tensor:
    bottom = y[:, :, 0, :]
    top = y[:, :, -1, :]
    left = y[:, :, 1:-1, 0]
    right = y[:, :, 1:-1, -1]
    return torch.cat(
        [
            bottom.reshape(y.shape[0], -1),
            top.reshape(y.shape[0], -1),
            left.reshape(y.shape[0], -1),
            right.reshape(y.shape[0], -1),
        ],
        dim=1,
    )


# --------------------------------------------------------------------------- #
# Smooth random-field helpers.                                                  #
# --------------------------------------------------------------------------- #
def stream_velocity(a: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Divergence-free no-slip velocity y = curl(psi) on the grid.

    a: (B, Kmax, Kmax) stream coefficients a_{kl} (1-indexed in k, l).
    Returns (B, 2, gy, gx).
    """
    pi = math.pi
    B, Kmax, _ = a.shape
    device, dtype = x.device, x.dtype
    y1 = torch.zeros((B,) + x.shape, device=device, dtype=dtype)
    y2 = torch.zeros((B,) + x.shape, device=device, dtype=dtype)
    for ki in range(Kmax):
        k = ki + 1
        sx2 = torch.sin(k * pi * x).pow(2)                 # sin^2(k pi x1)
        s2x = torch.sin(2 * k * pi * x)                    # sin(2k pi x1)
        for li in range(Kmax):
            l = li + 1
            akl = a[:, ki, li].reshape(-1, 1, 1)
            sy2 = torch.sin(l * pi * y).pow(2)
            s2y = torch.sin(2 * l * pi * y)
            # d psi/d x2 = sin^2(k x1) * l pi sin(2 l x2)
            y1 = y1 + akl * (sx2 * (l * pi * s2y))
            # -d psi/d x1 = -(k pi sin(2k x1)) * sin^2(l x2)
            y2 = y2 - akl * ((k * pi * s2x) * sy2)
    return torch.stack([y1, y2], dim=1)


def pressure_field(b: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Smooth mean-zero pressure p = sum b_{mn} cos(m pi x1) cos(n pi x2).

    b: (B, Mmax+1, Mmax+1); entry (0,0) is ignored. Returns (B, gy, gx).
    """
    pi = math.pi
    B, M1, _ = b.shape
    device, dtype = x.device, x.dtype
    p = torch.zeros((B,) + x.shape, device=device, dtype=dtype)
    for m in range(M1):
        cx = torch.cos(m * pi * x)
        for n in range(M1):
            if m == 0 and n == 0:
                continue
            bmn = b[:, m, n].reshape(-1, 1, 1)
            p = p + bmn * (cx * torch.cos(n * pi * y))
    p = p - p.mean(dim=(1, 2), keepdim=True)
    return p


def control_from_state(
    y: torch.Tensor, p: torch.Tensor, hx: float, hy: float
) -> torch.Tensor:
    """u = -mu Delta y + (y . grad)y + grad p on the interior; 0 on the boundary."""
    u = torch.zeros_like(y)
    u[:, :, 1:-1, 1:-1] = (
        -MU_FIXED * lap(y, hx, hy) + convect(y, y, hx, hy) + grad_scalar(p, hx, hy)
    )
    return u


def physical_pressure(state: torch.Tensor, hx: float, hy: float) -> torch.Tensor:
    """Incompressible pressure-Poisson solution p(y): Neumann-Laplacian p = -div((y.grad)y).

    Deterministic function of the velocity, so it is recoverable from u via y (the
    physical NS pressure, unlike an independent random field).  Solved by DCT-II.
    """
    import scipy.fft as sfft

    yn = state.detach().cpu().numpy()
    B, _, gy, gx = yn.shape
    dy1dx = np.gradient(yn[:, 0], hx, axis=2); dy1dy = np.gradient(yn[:, 0], hy, axis=1)
    dy2dx = np.gradient(yn[:, 1], hx, axis=2); dy2dy = np.gradient(yn[:, 1], hy, axis=1)
    c1 = yn[:, 0] * dy1dx + yn[:, 1] * dy1dy
    c2 = yn[:, 0] * dy2dx + yn[:, 1] * dy2dy
    rhs = -(np.gradient(c1, hx, axis=2) + np.gradient(c2, hy, axis=1))         # -div((y.grad)y)
    gh = sfft.dctn(rhs, type=2, axes=(1, 2), norm="ortho")
    ax = -(2.0 - 2.0 * np.cos(np.pi * np.arange(gx) / gx)) / hx ** 2           # Neumann DCT-II eigs
    ay = -(2.0 - 2.0 * np.cos(np.pi * np.arange(gy) / gy)) / hy ** 2
    lam = ay[:, None] + ax[None, :]
    lam[0, 0] = 1.0
    ph = gh / lam[None]
    ph[:, 0, 0] = 0.0
    p = sfft.idctn(ph, type=2, axes=(1, 2), norm="ortho")
    p = p - p.mean(axis=(1, 2), keepdims=True)
    return torch.from_numpy(p.astype(yn.dtype)).to(state.device)


def manufactured_triple(
    a: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    vel_amp: torch.Tensor | None = None,
    pressure_amp: float = PRESSURE_AMP,
    pressure_mode: str = "independent",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (u, y_field, p_field) for stream coeffs a and pressure coeffs b.

    vel_amp: optional (B,) per-sample target for ||y||_inf; if given the velocity
    field is rescaled so max|y| == vel_amp.
    pressure_mode: "independent" -> random cos field rescaled to pressure_amp;
                   "physical"    -> pressure-Poisson solution p(y) (learnable via y).
    """
    hx = 1.0 / (x.shape[1] - 1)
    hy = 1.0 / (x.shape[0] - 1)
    state = stream_velocity(a, x, y)
    if vel_amp is not None:
        cur = state.abs().amax(dim=(1, 2, 3)).clamp_min(1e-12)
        state = state * (vel_amp.to(state) / cur).reshape(-1, 1, 1, 1)
    if pressure_mode == "physical":
        press = physical_pressure(state, hx, hy)
    elif pressure_mode == "independent":
        press = pressure_field(b, x, y)
        if pressure_amp is not None:
            curp = press.abs().amax(dim=(1, 2)).clamp_min(1e-12)
            press = press * (pressure_amp / curp).reshape(-1, 1, 1)
    else:
        raise ValueError(pressure_mode)
    control = control_from_state(state, press, hx, hy)
    return control, state, press


# --------------------------------------------------------------------------- #
# Strong-form residual blocks (used by sTCL).                                   #
# --------------------------------------------------------------------------- #
def residual_blocks(
    y_pred: torch.Tensor,
    p_pred: torch.Tensor,
    u: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    hx: float,
    hy: float,
    *,
    weights: Dict[str, float] | None = None,
) -> Tuple[torch.Tensor, ...]:
    """Strong-form NS state residual blocks (momentum, continuity, boundary).

    momentum: -mu Delta y + (y.grad)y + grad p - u
    continuity: div y
    boundary: y on dOmega (the surrogate does not hard-enforce no-slip)
    """
    weights = weights or {}
    r_state = (
        -MU_FIXED * lap(y_pred, hx, hy)
        + convect(y_pred, y_pred, hx, hy)
        + grad_scalar(p_pred, hx, hy)
        - interior(u)
    )
    r_div = divergence(y_pred, hx, hy).unsqueeze(1)
    r_bc = boundary_values(y_pred)
    batch = y_pred.shape[0]
    blocks = {
        "state": r_state.reshape(batch, -1),
        "div_y": r_div.reshape(batch, -1),
        "boundary": r_bc,
    }
    return tuple(
        float(weights.get(name, 1.0)) * blocks[name]
        for name in ("state", "div_y", "boundary")
    )


def block_mse(blocks: Iterable[torch.Tensor]) -> torch.Tensor:
    vals = [(b * b).mean() for b in blocks]
    if not vals:
        raise ValueError("empty residual block list")
    return torch.stack(vals).sum()
