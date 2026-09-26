"""Finite-difference helpers for nonlinear-diffusion direct preconditioned loss.

This file intentionally contains only the operators needed by the clean
preconditioned residual loss

    L_stcl = r^T M r,   M = D^{-1/2} (-Delta)^{-1} D^{-1/2}.

It does not build a tangent-solve target.
"""

import numpy as np
import scipy.sparse as sp
import torch


def build_fd_matrices_dirichlet_torch(nx, ny, device):
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    nx_int = nx - 2
    ny_int = ny - 2

    Lx = sp.diags(
        [np.ones(nx_int - 1), -2 * np.ones(nx_int), np.ones(nx_int - 1)],
        [-1, 0, 1],
        shape=(nx_int, nx_int),
    ).tocsr() / dx**2
    Ly = sp.diags(
        [np.ones(ny_int - 1), -2 * np.ones(ny_int), np.ones(ny_int - 1)],
        [-1, 0, 1],
        shape=(ny_int, ny_int),
    ).tocsr() / dy**2
    L2d = sp.kron(sp.eye(ny_int), Lx) + sp.kron(Ly, sp.eye(nx_int))

    Dx1 = sp.diags(
        [-np.ones(nx_int - 1), np.ones(nx_int - 1)],
        [-1, 1],
        shape=(nx_int, nx_int),
    ).tocsr() / (2 * dx)
    Dy1 = sp.diags(
        [-np.ones(ny_int - 1), np.ones(ny_int - 1)],
        [-1, 1],
        shape=(ny_int, ny_int),
    ).tocsr() / (2 * dy)
    Dx = sp.kron(sp.eye(ny_int), Dx1)
    Dy = sp.kron(Dy1, sp.eye(nx_int))

    def to_torch(mat):
        # Dense matrices reproduce the published 65x65 implementation.
        # At 128x128 the same matrices would occupy about 3 GiB and turn
        # stencil applications into dense O(N^2) products, so use the exact
        # same SciPy stencils in sparse COO form for larger grids.
        if max(nx, ny) <= 65:
            return torch.tensor(
                mat.toarray(), dtype=torch.float32, device=device)
        coo = mat.tocoo()
        indices = torch.tensor(
            np.vstack((coo.row, coo.col)), dtype=torch.long,
            device=device)
        values = torch.tensor(
            coo.data, dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(
            indices, values, size=coo.shape,
            dtype=torch.float32, device=device).coalesce()

    return to_torch(L2d), to_torch(Dx), to_torch(Dy)


_fd_cache = {}


def get_fd_matrices(nx, ny, device):
    key = (nx, ny, str(device))
    if key not in _fd_cache:
        _fd_cache[key] = build_fd_matrices_dirichlet_torch(nx, ny, device)
    return _fd_cache[key]


def dst2(x):
    """Orthonormal 2-D DST-I via odd extension and FFT."""
    *batch, n, m = x.shape

    zero_n = torch.zeros(*batch, 1, m, device=x.device, dtype=x.dtype)
    ext_n = torch.cat([zero_n, x, zero_n, -x.flip(-2)], dim=-2)
    x_n = torch.fft.fft(ext_n, dim=-2)
    x_n = (-x_n.imag)[..., 1 : n + 1, :] * 0.5

    zero_m = torch.zeros(*batch, n, 1, device=x.device, dtype=x.dtype)
    ext_m = torch.cat([zero_m, x_n, zero_m, -x_n.flip(-1)], dim=-1)
    x_nm = torch.fft.fft(ext_m, dim=-1)
    x_nm = (-x_nm.imag)[..., :, 1 : m + 1] * 0.5

    scale = 2.0 / ((n + 1) ** 0.5 * (m + 1) ** 0.5)
    return x_nm * scale


def full_to_unk_torch(field_2d, nx, ny):
    interior = field_2d[:, 1:-1, 1:-1]
    return interior.permute(0, 2, 1).reshape(field_2d.shape[0], -1)


def compute_derivatives_full(field, nx, ny):
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    dfx = torch.zeros_like(field)
    dfy = torch.zeros_like(field)

    dfx[:, 1:-1, :] = (field[:, 2:, :] - field[:, :-2, :]) / (2 * dx)
    dfx[:, 0, :] = (field[:, 1, :] - field[:, 0, :]) / dx
    dfx[:, -1, :] = (field[:, -1, :] - field[:, -2, :]) / dx

    dfy[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / (2 * dy)
    dfy[:, :, 0] = (field[:, :, 1] - field[:, :, 0]) / dy
    dfy[:, :, -1] = (field[:, :, -1] - field[:, :, -2]) / dy
    return dfx, dfy


def apply_Ay_batch(w_unk, D, dax, day, u_unk, L2d, Dx, Dy):
    """Apply A_y(u) w = -div(exp(a) grad w) + 3 u^2 w."""
    lap_w = (L2d @ w_unk.T).T
    dx_w = (Dx @ w_unk.T).T
    dy_w = (Dy @ w_unk.T).T
    return -D * lap_w - D * dax * dx_w - D * day * dy_w + 3 * u_unk**2 * w_unk


def get_inv_lam(nx, ny, device):
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    nx_int = nx - 2
    ny_int = ny - 2
    k = np.arange(1, nx_int + 1)
    ell = np.arange(1, ny_int + 1)
    lam_x = (4.0 / dx**2) * np.sin(np.pi * k / (2 * (nx - 1))) ** 2
    lam_y = (4.0 / dy**2) * np.sin(np.pi * ell / (2 * (ny - 1))) ** 2
    lam = lam_x[:, None] + lam_y[None, :]
    return torch.tensor((1.0 / lam).T, dtype=torch.float32, device=device)


def apply_M(f_flat, D_sqrt_inv, nx, ny, inv_lam):
    f_scaled = D_sqrt_inv * f_flat
    f_2d = f_scaled.view(f_scaled.shape[0], ny - 2, nx - 2)
    f_dst = dst2(f_2d)
    f_inv = dst2(f_dst * inv_lam.unsqueeze(0))
    return D_sqrt_inv * f_inv.reshape(f_scaled.shape[0], -1)
