#!/usr/bin/env python
"""
Batched GPU MINRES for the Helmholtz tangent system (online sTCL targets).

The Helmholtz tangent operator

    A = -Delta - kappa^2 * exp(2a)

is symmetric but indefinite, so conjugate gradients does not apply.  sTCL
(``train_helmholtz.py --method fminres``) runs a few iterations of
preconditioned MINRES on A w = g for every sketch direction, with the SPD
shifted-Laplacian preconditioner

    M = (-Delta_h + shift_scale * kappa^2)^{-1}

applied exactly in the DST-I basis of the interior Dirichlet grid.
"""
import numpy as np
import torch

from generate_data_difno import KAPPA


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


def apply_A_batch(w_flat, D_batch, nx, ny, dx, dy):
    """Batched apply of A = -Delta - kappa^2 * exp(2a)  to w.
    w_flat: (B, Nint)   where Nint = (nx-2)*(ny-2)
    D_batch: (B, Nint)  holding exp(2a_i) on the interior
    Returns: (B, Nint)
    """
    B = w_flat.shape[0]
    nxi, nyi = nx - 2, ny - 2
    # Reshape to (B, nyi, nxi) in the full_to_unk convention
    w = w_flat.view(B, nyi, nxi)
    # -Delta via 5-point stencil (zero outside = homogeneous Dirichlet)
    lap = torch.zeros_like(w)
    lap[:, 1:-1, :] += -w[:, 2:, :] - w[:, :-2, :]
    lap[:, 0,    :] += -w[:, 1,  :]
    lap[:, -1,   :] += -w[:, -2, :]
    lap = lap * (1.0 / dy**2)
    lapx = torch.zeros_like(w)
    lapx[:, :, 1:-1] += -w[:, :, 2:] - w[:, :, :-2]
    lapx[:, :, 0]    += -w[:, :, 1]
    lapx[:, :, -1]   += -w[:, :, -2]
    lapx = lapx * (1.0 / dx**2)
    lap_total = lap + lapx + w * (2.0 / dx**2 + 2.0 / dy**2)   # -Delta = 4/h^2 center - ...
    # Helmholtz: -Delta w - kappa^2 D * w
    out = lap_total.reshape(B, -1) - (KAPPA**2) * D_batch * w_flat
    return out


def get_inv_shifted_lam(nx, ny, shift_scale, device):
    """Eigenvalues of (-Delta + shift_scale*kappa^2)^{-1} on interior Dirichlet DoFs
    via DST basis. shift_scale > 0 ensures SPD."""
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    nxi = nx - 2
    nyi = ny - 2
    k = np.arange(1, nxi + 1)
    l = np.arange(1, nyi + 1)
    lam_x = (4.0 / dx**2) * np.sin(np.pi * k / (2 * (nx - 1))) ** 2
    lam_y = (4.0 / dy**2) * np.sin(np.pi * l / (2 * (ny - 1))) ** 2
    lam = lam_x[:, None] + lam_y[None, :] + shift_scale * (KAPPA**2)
    return torch.tensor((1.0 / lam).T, dtype=torch.float32, device=device)


def shifted_fourier_solve(f_flat, nx, ny, inv_lam):
    B = f_flat.shape[0]
    f_2d = f_flat.view(B, ny - 2, nx - 2)
    f_dst = dst2(f_2d)
    f_dst = f_dst * inv_lam.unsqueeze(0)
    f_inv = dst2(f_dst)
    return f_inv.reshape(B, -1)


def minres_fourier(apply_A, b, M_inv_fn, n_iter=50):
    """Batched preconditioned MINRES.  M_inv_fn is SPD."""
    # Same structure as above but M^{-1} is a function
    x = torch.zeros_like(b)
    v_prev = torch.zeros_like(b)
    v_curr = b.clone()
    z_curr = M_inv_fn(v_curr)
    beta_curr = torch.sqrt(((v_curr * z_curr).sum(-1, keepdim=True)).clamp(min=1e-30))
    v_curr = v_curr / beta_curr
    z_curr = z_curr / beta_curr
    eta = beta_curr
    s_prev = torch.zeros_like(eta)
    s_curr = torch.zeros_like(eta)
    c_prev = torch.ones_like(eta)
    c_curr = torch.ones_like(eta)
    w_prev = torch.zeros_like(b)
    w_curr = torch.zeros_like(b)

    for _ in range(n_iter):
        Az = apply_A(z_curr)
        alpha = (z_curr * Az).sum(-1, keepdim=True)
        v_next = Az - alpha * v_curr - beta_curr * v_prev
        z_next = M_inv_fn(v_next)
        beta_next = torch.sqrt(((v_next * z_next).sum(-1, keepdim=True)).clamp(min=1e-30))
        v_next = v_next / beta_next
        z_next = z_next / beta_next
        delta1 = c_curr * alpha - c_prev * s_curr * beta_curr
        gamma1 = torch.sqrt(delta1 ** 2 + beta_next ** 2)
        epsilon = s_prev * beta_curr
        delta2 = s_curr * alpha + c_prev * c_curr * beta_curr
        c_new = delta1 / gamma1
        s_new = beta_next / gamma1
        w_new = (z_curr - delta2 * w_curr - epsilon * w_prev) / gamma1
        x = x + c_new * eta * w_new
        eta = -s_new * eta
        v_prev, v_curr = v_curr, v_next
        z_curr = z_next
        beta_curr = beta_next
        c_prev, c_curr = c_curr, c_new
        s_prev, s_curr = s_curr, s_new
        w_prev, w_curr = w_curr, w_new
    return x
