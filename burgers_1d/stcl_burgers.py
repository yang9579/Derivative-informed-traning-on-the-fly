#!/usr/bin/env python
"""
sTCL for Burgers 1D on (x, t) Grid — FNO-based
=================================================
Sketched Tangent-Consistency Loss for the PDE:
  u_t + u*u_x = nu*u_xx + f(t)
  u(0,t) = u(1,t) = 0  (Dirichlet BC)
  u(x,0) = 0           (zero IC)

Tangent equation (linearization in f direction v(t)):
  w_t + u*w_x + w*u_x - nu*w_xx = v(t)
  w(0,t) = w(1,t) = 0,  w(x,0) = 0

Here:
  w = JVP of the FNO output u wrt input f in direction v
  v(t) is a random perturbation of the control (space-uniform)
  u is the FNO prediction at the current input f

Grid: (Nx, Nt) on [0,1] x [0, T].
FNO input channels: (x, t, f(t)) -> 3 channels, output: u(x,t) -> 1 channel.
The perturbation v is applied to the f(t) channel (broadcast to all x).

LHS (applied to w): w_t + u*w_x + w*u_x - nu*w_xx
RHS: v(t)   (space-uniform, so same at all x for each t)

We enforce this at interior points only:
  x: indices 1..Nx-2  (Dirichlet: w=0 at x=0, x=1)
  t: indices 1..Nt-1  (IC: w=0 at t=0)
"""

import torch
from torch.func import jvp as _func_jvp
from runtime_profile import end as profile_end
from runtime_profile import start as profile_start

NU = 0.01


def _build_fd_operators_1d(N, dx, device):
    """
    Build 1D FD matrices for first and second derivatives.
    Interior-only: operates on points 0..N-1 of a grid of size N,
    but we apply them on the full grid and then extract interior.

    Returns sparse-like dense matrices for simplicity (small enough).
    """
    # First derivative: central differences (N, N)
    D1 = torch.zeros(N, N, device=device)
    for i in range(1, N - 1):
        D1[i, i + 1] = 1.0 / (2 * dx)
        D1[i, i - 1] = -1.0 / (2 * dx)
    # One-sided at boundaries
    D1[0, 0] = -1.0 / dx
    D1[0, 1] = 1.0 / dx
    D1[-1, -2] = -1.0 / dx
    D1[-1, -1] = 1.0 / dx

    # Second derivative (N, N)
    D2 = torch.zeros(N, N, device=device)
    for i in range(1, N - 1):
        D2[i, i - 1] = 1.0 / dx ** 2
        D2[i, i] = -2.0 / dx ** 2
        D2[i, i + 1] = 1.0 / dx ** 2
    # One-sided at boundaries (not used for interior, but fill for safety)
    D2[0, 0] = 1.0 / dx ** 2
    D2[0, 1] = -2.0 / dx ** 2
    if N > 2:
        D2[0, 2] = 1.0 / dx ** 2
    D2[-1, -1] = 1.0 / dx ** 2
    D2[-1, -2] = -2.0 / dx ** 2
    if N > 2:
        D2[-1, -3] = 1.0 / dx ** 2

    return D1, D2


_fd_cache = {}


def _get_fd_ops(Nx, Nt, T, device):
    """Cached FD operators for spatial and temporal derivatives."""
    key = (Nx, Nt, T, str(device))
    if key not in _fd_cache:
        dx = 1.0 / (Nx - 1)
        dt = T / (Nt - 1)
        Dx, Dxx = _build_fd_operators_1d(Nx, dx, device)
        Dt, _ = _build_fd_operators_1d(Nt, dt, device)
        _fd_cache[key] = (Dx, Dxx, Dt)
    return _fd_cache[key]


def _get_spectral_precond(Nx, device, gamma=0.1):
    """Spectral preconditioner M_k = 1/(1 + gamma*(k*pi)^2) for Dirichlet BCs.
    Damps high-frequency residual components in x via DST."""
    k = torch.arange(1, Nx - 1, device=device, dtype=torch.float32)
    lam_k = (k * 3.14159) ** 2
    M = 1.0 / (1.0 + gamma * lam_k)
    return M  # (Nx-2,)


def stcl_burgers(fno, f_batch, coord_grid, q=4, T=1.0, nu=NU, precond_gamma=0.0):
    """
    Compute sTCL loss for Burgers 1D.

    Args:
        fno: FNO2D model
        f_batch: (B, Nx, Nt, 1) input field (f(t) broadcast to all x)
        coord_grid: (Nx, Nt, 2) coordinate grid with (x, t) at each point
        q: number of random tangent directions
        T: final time
        nu: viscosity

    Returns: scalar sTCL loss (mean over batch and directions)
    """
    device = f_batch.device
    B, Nx, Nt, _ = f_batch.shape

    Dx, Dxx, Dt = _get_fd_ops(Nx, Nt, T, device)

    coords = coord_grid.unsqueeze(0).expand(B, -1, -1, -1)

    total_res = torch.tensor(0.0, device=device)

    for _ in range(q):
        # Random perturbation direction v(t): space-uniform
        # v is (B, 1, Nt, 1) broadcast to (B, Nx, Nt, 1) in the input
        v_t = torch.randn(B, 1, Nt, 1, device=device)
        v_t = v_t / (v_t.norm() / (B * Nt) ** 0.5 + 1e-8)
        delta_f = v_t.expand(B, Nx, Nt, 1)

        # FNO forward + JVP
        def fno_fn(f_field):
            inp = torch.cat([coords, f_field], dim=-1)
            return fno(inp)

        _pt = profile_start("online_jvp")
        u_pred, w_pred = _func_jvp(fno_fn, (f_batch,), (delta_f,))
        profile_end(_pt)

        # u_pred: (B, Nx, Nt, 1), w_pred: (B, Nx, Nt, 1) = JVP
        u = u_pred.squeeze(-1)  # (B, Nx, Nt)
        w = w_pred.squeeze(-1)  # (B, Nx, Nt)

        _pt = profile_start("pde_residual")
        # Compute spatial and temporal derivatives via FD matrices
        # Dx is (Nx, Nx), operates on dim 1; Dt is (Nt, Nt), operates on dim 2
        # w is (B, Nx, Nt)

        # w_t: time derivative — w @ Dt.T applies Dt along the last (time) dim
        w_t = torch.matmul(w, Dt.T)  # (B, Nx, Nt)

        # w_x: spatial derivative — Dx @ w[b] for each batch
        # w is (B, Nx, Nt), Dx is (Nx, Nx) -> use einsum
        w_x = torch.einsum('ij,bjk->bik', Dx, w)  # (B, Nx, Nt)

        # w_xx: second spatial derivative
        w_xx = torch.einsum('ij,bjk->bik', Dxx, w)  # (B, Nx, Nt)

        # u_x: spatial derivative of u
        u_x = torch.einsum('ij,bjk->bik', Dx, u)  # (B, Nx, Nt)

        # LHS of tangent equation: w_t + u*w_x + w*u_x - nu*w_xx
        lhs = w_t + u.detach() * w_x + w * u_x.detach() - nu * w_xx

        # RHS: v(t) — space-uniform, same for all x
        v_rhs = v_t.squeeze(-1).expand(B, Nx, Nt)  # (B, Nx, Nt)

        # Residual at interior points only:
        #   x: 1..Nx-2 (skip boundary)
        #   t: 1..Nt-1 (skip IC at t=0)
        residual = (lhs - v_rhs)[:, 1:-1, 1:]  # (B, Nx-2, Nt-1)
        profile_end(_pt)

        # Preconditioned residual norm
        if precond_gamma > 0:
            _pt = profile_start("preconditioner_krylov")
            M = _get_spectral_precond(Nx, device, precond_gamma)
            # Apply M along spatial dim via DST-I (ortho, involutive).
            # PyTorch has no native DST; build it via FFT antisymmetric extension.
            r = residual                    # (B, Nx-2, Nt-1)
            # Move spatial axis to last for DST
            r_perm = r.movedim(1, -1)        # (B, Nt-1, Nx-2)
            Nsp = r_perm.shape[-1]
            zero = torch.zeros(*r_perm.shape[:-1], 1, device=device, dtype=r_perm.dtype)
            ext = torch.cat([zero, r_perm, zero, -r_perm.flip(-1)], dim=-1)   # 2(Nsp+1)
            Xf = torch.fft.fft(ext, dim=-1)
            r_dst = (-Xf.imag)[..., 1:Nsp + 1] * 0.5 * ((2.0 / (Nsp + 1)) ** 0.5)
            Mr_dst = r_dst * M.unsqueeze(0).unsqueeze(0)
            # Inverse DST-I (involutive under ortho scaling)
            zero2 = torch.zeros(*Mr_dst.shape[:-1], 1, device=device, dtype=Mr_dst.dtype)
            ext2 = torch.cat([zero2, Mr_dst, zero2, -Mr_dst.flip(-1)], dim=-1)
            Xf2 = torch.fft.fft(ext2, dim=-1)
            Mr_perm = (-Xf2.imag)[..., 1:Nsp + 1] * 0.5 * ((2.0 / (Nsp + 1)) ** 0.5)
            Mr = Mr_perm.movedim(-1, 1)      # back to (B, Nx-2, Nt-1)
            res_sq = (residual * Mr).sum(dim=(-2, -1))
            profile_end(_pt)
        else:
            res_sq = residual.pow(2).sum(dim=(-2, -1))  # (B,)
        with torch.no_grad():
            rhs_sq = v_rhs[:, 1:-1, 1:].pow(2).sum(dim=(-2, -1)) + 1e-10  # (B,)
        total_res = total_res + (res_sq / rhs_sq).mean()

    return total_res / q
