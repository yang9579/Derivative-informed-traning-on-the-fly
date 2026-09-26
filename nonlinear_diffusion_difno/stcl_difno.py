#!/usr/bin/env python
"""
sTCL for Nonlinear Diffusion-Reaction — DIFNO Setup (all-Dirichlet BCs)
=========================================================================
Sketched Tangent-Consistency Loss for the PDE:
  -div(exp(a) * grad(u)) + u^3 = f
  u = 0 on all of dOmega  (homogeneous Dirichlet)

Tangent equation (delta_a -> delta_u, homogeneous Dirichlet for delta_u too):
  -div(D*grad(delta_u)) + 3*u^2*delta_u = div(D*delta_a*grad(u))

This is simpler than the mixed-BC DINO setup because there are NO boundary
contributions in either the LHS or RHS.

Grid: 65x65 on [0,1]^2, dx=dy=1/64.
Unknown DOFs: i=1..63, j=1..63 (interior only). N_unk = 63*63 = 3969.
"""

import torch
from torch.func import jvp as _func_jvp

def jvp(fn, primals, tangents, create_graph=True):
    return _func_jvp(fn, primals, tangents)


def build_fd_matrices_dirichlet_torch(nx, ny, device):
    """
    Build FD matrices on [0,1]^2 with homogeneous Dirichlet BCs.
    Operates on interior DOFs: i=1..nx-2, j=1..ny-2. N_unk = (nx-2)*(ny-2).
    Flat indexing: k = j_int*(nx-2) + i_int where i_int=i-1, j_int=j-1.

    Returns: L2d, Dx, Dy (all dense torch tensors on device)
    """
    import scipy.sparse as sp
    import numpy as np

    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    nx_int = nx - 2
    ny_int = ny - 2

    # 1D second derivative in x: Dirichlet
    Lx = sp.diags([np.ones(nx_int-1), -2*np.ones(nx_int), np.ones(nx_int-1)],
                   [-1, 0, 1], shape=(nx_int, nx_int)).tocsr() / dx**2

    # 1D second derivative in y: Dirichlet
    Ly = sp.diags([np.ones(ny_int-1), -2*np.ones(ny_int), np.ones(ny_int-1)],
                   [-1, 0, 1], shape=(ny_int, ny_int)).tocsr() / dy**2

    # 2D Laplacian
    L2d = sp.kron(sp.eye(ny_int), Lx) + sp.kron(Ly, sp.eye(nx_int))

    # 1D first derivative in x: central differences
    Dx1 = sp.diags([-np.ones(nx_int-1), np.ones(nx_int-1)],
                    [-1, 1], shape=(nx_int, nx_int)).tocsr() / (2*dx)

    # 1D first derivative in y: central differences
    Dy1 = sp.diags([-np.ones(ny_int-1), np.ones(ny_int-1)],
                    [-1, 1], shape=(ny_int, ny_int)).tocsr() / (2*dy)

    Dx = sp.kron(sp.eye(ny_int), Dx1)
    Dy = sp.kron(Dy1, sp.eye(nx_int))

    to_t = lambda M: torch.tensor(M.toarray(), dtype=torch.float32, device=device)
    return to_t(L2d), to_t(Dx), to_t(Dy)


_fd_cache = {}

def get_fd_matrices(nx, ny, device):
    """Cached FD matrices."""
    key = (nx, ny, str(device))
    if key not in _fd_cache:
        _fd_cache[key] = build_fd_matrices_dirichlet_torch(nx, ny, device)
    return _fd_cache[key]


_hinv_cache = {}

def get_hinv_weights(nx, ny, device):
    """
    Inverse Laplacian weights in DST basis for Dirichlet domain.
    For N=(nx-2)×(ny-2) interior grid, the discrete sine basis diagonalizes
    the FD Laplacian exactly:
      sin(i*pi*k/(nx-1)) * sin(j*pi*l/(ny-1))  for k=1..nx-2, l=1..ny-2
    Eigenvalues of -Lap (5-point stencil):
      lam_{kl} = (4/dx²)*sin²(pi*k/(2*(nx-1))) + (4/dy²)*sin²(pi*l/(2*(ny-1)))
    H^{-1} weight = 1/lam_{kl}
    Returns: (nx-2, ny-2) tensor of sqrt(1/lam_{kl}) weights.
    """
    key = (nx, ny, str(device))
    if key not in _hinv_cache:
        import numpy as np
        dx = 1.0 / (nx - 1)
        dy = 1.0 / (ny - 1)
        nx_int = nx - 2
        ny_int = ny - 2
        k = np.arange(1, nx_int + 1)
        l = np.arange(1, ny_int + 1)
        lam_x = (4.0 / dx**2) * np.sin(np.pi * k / (2 * (nx - 1))) ** 2
        lam_y = (4.0 / dy**2) * np.sin(np.pi * l / (2 * (ny - 1))) ** 2
        lam = lam_x[:, None] + lam_y[None, :]  # (nx_int, ny_int)
        weights = 1.0 / np.sqrt(lam)  # sqrt weights so ||weighted||² = H^{-1} norm²
        _hinv_cache[key] = torch.tensor(weights, dtype=torch.float32,
                                         device=device)
    return _hinv_cache[key]


def dst2(x):
    """
    2D Discrete Sine Transform (type-I) via FFT, using odd extension.
    x: (..., N, M) tensor. Returns same shape, DST-I coefficients.

    Implemented via FFT of the antisymmetric extension:
      ext[n] = [0, x[0], x[1], ..., x[N-1], 0, -x[N-1], ..., -x[0]]  length 2*(N+1)
    The DFT of this gives sine coefficients (imaginary part) / 2.

    For normalization: DST-I with orthonormal scaling is
      X[k] = sqrt(2/(N+1)) * sum_n x[n] * sin(pi*(n+1)*(k+1)/(N+1))
    """
    # Odd extension along last two axes
    *B, N, M = x.shape
    # Extension along N: [0, x[0..N-1], 0, -x[N-1..0]]
    zero_n = torch.zeros(*B, 1, M, device=x.device, dtype=x.dtype)
    ext_n = torch.cat([zero_n, x, zero_n, -x.flip(-2)], dim=-2)
    # FFT along N, take imaginary part * (-1) and first N modes
    # fft length = 2*(N+1)
    X_n = torch.fft.fft(ext_n, dim=-2)
    X_n = (-X_n.imag)[..., 1:N+1, :] * 0.5  # extract sine coefs

    # Same along M
    zero_m = torch.zeros(*B, N, 1, device=x.device, dtype=x.dtype)
    ext_m = torch.cat([zero_m, X_n, zero_m, -X_n.flip(-1)], dim=-1)
    X_nm = torch.fft.fft(ext_m, dim=-1)
    X_nm = (-X_nm.imag)[..., :, 1:M+1] * 0.5

    # Orthonormal scaling: DST-I has eigvec orthonormality
    # with scaling sqrt(2/(N+1))*sqrt(2/(M+1))
    scale = 2.0 / ((N + 1) ** 0.5 * (M + 1) ** 0.5)
    return X_nm * scale


def full_to_unk_torch(field_2d, nx, ny):
    """
    Extract interior DOFs from full-grid tensor.
    field_2d: (B, Nx, Ny) where Nx=nx, Ny=ny
    Returns: (B, (nx-2)*(ny-2)) with flat ordering k = j_int*(nx-2) + i_int
    """
    interior = field_2d[:, 1:-1, 1:-1]  # (B, nx-2, ny-2)
    return interior.permute(0, 2, 1).reshape(field_2d.shape[0], -1)


def compute_derivatives_full(field, nx, ny):
    """Central differences on full (B, Nx, Ny) grid. One-sided at boundaries."""
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)

    dfx = torch.zeros_like(field)
    dfy = torch.zeros_like(field)

    dfx[:, 1:-1, :] = (field[:, 2:, :] - field[:, :-2, :]) / (2*dx)
    dfx[:, 0, :]    = (field[:, 1, :]  - field[:, 0, :])  / dx
    dfx[:, -1, :]   = (field[:, -1, :] - field[:, -2, :]) / dx

    dfy[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / (2*dy)
    dfy[:, :, 0]    = (field[:, :, 1]  - field[:, :, 0])  / dy
    dfy[:, :, -1]   = (field[:, :, -1] - field[:, :, -2]) / dy

    return dfx, dfy


def stcl_nonlinear_diffusion_difno(fno, a_batch, coord_grid, q=4,
                                     normalize="rhs"):
    """
    Compute sTCL loss for nonlinear diffusion-reaction PDE (DIFNO setup).

    All-Dirichlet BCs: u=0 on boundary, delta_u=0 on boundary.
    No BC contributions needed in either LHS or RHS.

    Args:
        fno: FNO2D model
        a_batch: (B, Nx, Ny, 1) input log-diffusivity
        coord_grid: (Nx, Ny, 2) coordinate grid
        q: number of random tangent directions
        normalize: "none" | "rhs" | "lhs_rhs" | "const"
    Returns: scalar sTCL loss (mean over batch and directions)
    """
    device = a_batch.device
    B, Nx, Ny, _ = a_batch.shape

    L2d, Dx, Dy = get_fd_matrices(Nx, Ny, device)

    coords = coord_grid.unsqueeze(0).expand(B, -1, -1, -1)

    # Precompute a derivatives on full grid, extract at interior
    a_full = a_batch.squeeze(-1)  # (B, Nx, Ny)
    dax_full, day_full = compute_derivatives_full(a_full, Nx, Ny)
    dax = full_to_unk_torch(dax_full, Nx, Ny)  # (B, N_unk)
    day = full_to_unk_torch(day_full, Nx, Ny)

    # Diffusivity at interior
    D = torch.exp(full_to_unk_torch(a_full, Nx, Ny))  # (B, N_unk)

    total_res = torch.tensor(0.0, device=device)

    for _ in range(q):
        # Random perturbation direction
        delta_a = torch.randn(B, Nx, Ny, 1, device=device)
        delta_a = delta_a / (delta_a.norm() / (B * Nx * Ny) ** 0.5 + 1e-8)

        # FNO forward and JVP
        def fno_fn(a_field):
            inp = torch.cat([coords, a_field], dim=-1)
            return fno(inp)

        u_pred, du_pred = jvp(fno_fn, (a_batch,), (delta_a,),
                              create_graph=True)

        # FNO JVP output (this is what we train)
        du_unk = full_to_unk_torch(du_pred.squeeze(-1), Nx, Ny)

        # Linearization point: current FNO prediction. Detach the coefficient
        # state so the sTCL term trains the JVP rather than adding a second
        # solution-value residual through A_y(u_theta) and F_a(u_theta).
        u_lin_full = u_pred.detach().squeeze(-1)
        u_unk = full_to_unk_torch(u_lin_full, Nx, Ny)

        # delta_a derivatives
        da_full = delta_a.squeeze(-1)
        d_dax_full, d_day_full = compute_derivatives_full(da_full, Nx, Ny)
        da = full_to_unk_torch(da_full, Nx, Ny)
        d_da_x = full_to_unk_torch(d_dax_full, Nx, Ny)
        d_da_y = full_to_unk_torch(d_day_full, Nx, Ny)

        # ── LHS: -div(D*grad(delta_u)) + 3*u^2*delta_u ──
        # No BC contributions (homogeneous Dirichlet for delta_u)
        Lap_du = (L2d @ du_unk.T).T
        dx_du  = (Dx  @ du_unk.T).T
        dy_du  = (Dy  @ du_unk.T).T

        lhs = (-D * Lap_du
               - D * dax * dx_du
               - D * day * dy_du
               + 3 * u_unk**2 * du_unk)

        # ── RHS: div(D * delta_a * grad(u)) ──
        # No BC contributions (homogeneous Dirichlet for u)
        Lap_u = (L2d @ u_unk.T).T
        ux    = (Dx  @ u_unk.T).T
        uy    = (Dy  @ u_unk.T).T

        rhs = (D * da * Lap_u
               + D * (dax * da + d_da_x) * ux
               + D * (day * da + d_da_y) * uy)

        residual = lhs - rhs

        if normalize == "none":
            total_res = total_res + residual.pow(2).mean()
        elif normalize == "rhs":
            # Detached normalizer: fixed per step, avoids grad-of-ratio issues.
            res_sq = residual.pow(2).sum(dim=-1)
            with torch.no_grad():
                rhs_sq = rhs.pow(2).sum(dim=-1) + 1e-10
            total_res = total_res + (res_sq / rhs_sq).mean()
        elif normalize == "lhs_rhs":
            res_sq = residual.pow(2).sum(dim=-1)
            with torch.no_grad():
                denom_sq = lhs.pow(2).sum(dim=-1) + rhs.pow(2).sum(dim=-1) + 1e-10
            total_res = total_res + (res_sq / denom_sq).mean()
        elif normalize == "const":
            total_res = total_res + residual.pow(2).mean() / 1e-4
        elif normalize == "hinv":
            # H^{-1} preconditioning via DST (exact for Dirichlet Laplacian)
            # ||residual||²_{H^{-1}} = residual^T * (-Lap)^{-1} * residual
            # In DST basis, (-Lap)^{-1} is diagonal with eigenvalues 1/lam_{kl}
            nx_int = Nx - 2
            ny_int = Ny - 2
            # Reshape (B, N_unk) -> (B, ny_int, nx_int)
            # Recall full_to_unk_torch flattens as: k = j_int*nx_int + i_int
            res_2d = residual.view(B, ny_int, nx_int)
            # Apply DST along both spatial dims
            res_dst = dst2(res_2d)  # (B, ny_int, nx_int)
            # Inverse Laplacian weights (sqrt form, so squaring gives H^{-1} norm²)
            w = get_hinv_weights(Nx, Ny, device)  # (nx_int, ny_int)
            # w is (nx_int, ny_int), res_dst is (B, ny_int, nx_int)
            w_reshaped = w.T.unsqueeze(0)  # (1, ny_int, nx_int)
            weighted = res_dst * w_reshaped
            # Sum of squares = H^{-1} norm squared
            hinv_norm_sq = weighted.pow(2).sum(dim=(-2, -1))  # (B,)
            total_res = total_res + hinv_norm_sq.mean()
        else:
            raise ValueError(f"Unknown normalize: {normalize}")

    return total_res / q
