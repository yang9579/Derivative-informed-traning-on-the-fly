"""Differentiable 1D unforced Allen-Cahn initial-to-trajectory solver (time-dependent case).

Parabolic space-time sibling of the Burgers benchmark.  Canonical phase-field map:

    u_t = eps^2 u_xx + u - u^3,    x in [0, 1] (periodic),   t in [0, T],
    u(x, 0) = a(x).

Forward map  S : a(x) -> u(x, t)  (the whole space-time trajectory).  No forcing -- the
dynamics are driven entirely by the initial field a, which sharpens into +/-1 domains and
slowly coarsens.  The operator INPUT is the initial condition a(x) (a smooth random field),
not a scalar parameter.

Tangent (forward sensitivity) for an IC perturbation v = delta_a, with w = delta_u:

    w_t = eps^2 w_xx + (1 - 3 u^2) w,    w(x, 0) = v(x).

A homogeneous, time-dependent, variable-coefficient PARABOLIC equation driven by the IC
perturbation (the reaction coefficient 1 - 3u^2 flips sign with the phase).  The sTCL
residual is the raw L^2 space-time residual r = w_t - eps^2 w_xx - (1 - 3u^2) w (no
preconditioner -- parabolic, same class as Burgers and the 2D Allen-Cahn).

Discretization: 1D periodic Fourier spectral in space, semi-implicit (IMEX) Euler in time
(stiff diffusion implicit via FFT, reaction explicit).  The whole time loop is torch and is
differentiable in a, so the exact sensitivity du/da . v comes from torch.func.jvp through the
solver -- no separate tangent solver (the "autograd through the implicit forward solver"
approach also used for Burgers).  Run `python ac1d_time_solver.py` for the validity check.
"""
import numpy as np
import torch

L = 1.0            # periodic domain [0, L]
EPS = 0.03         # interface parameter (coefficient eps^2); sharp interfaces -> hard Jacobian
T_FINAL = 5.0      # coarsening regime: domain survival is IC-sensitive -> harder field prediction
NX = 128           # spatial grid points
NT = 40            # stored frames including t=0
N_SUBSTEP = 125    # internal IMEX substeps between frames (~ dt_internal 1e-3 at T=5)
IC_SCALE = 1.0     # IC amplitude

ELL = 0.07         # IC correlation length (initial domain size); rich initial domain network
TAU = 4.0          # IC Matern smoothness exponent (nu = tau - d/2 = 3.5 in 1D, very smooth).
                   # High tau is essential: a rough IC injects high-k content that the stiff
                   # eps^2 k^2 tangent term over-amplifies, raising the raw sTCL residual floor.
                   # A smooth GRF (literature-standard operator input) keeps the tangent residual
                   # well-conditioned with no preconditioner.


def _k(n, device, dtype):
    """Angular wavenumbers for domain length L; u_xx <-> -(k^2) in Fourier."""
    return 2.0 * np.pi * torch.fft.fftfreq(n, d=L / n, device=device).to(dtype)


def solve(a, eps=EPS, nt=NT, T=T_FINAL, n_substep=N_SUBSTEP, dtype=torch.float64):
    """Solve unforced 1D Allen-Cahn from IC field a (B, nx).  Returns u (B, nt, nx).
    Differentiable in a (used for exact du/da via torch.func.jvp)."""
    a = torch.as_tensor(a, dtype=dtype)
    if a.ndim == 1:
        a = a.unsqueeze(0)
    device = a.device
    n = a.shape[-1]
    k2 = _k(n, device, dtype) ** 2                                # (nx,), -Laplace symbol
    dt = T / ((nt - 1) * n_substep)
    denom = 1.0 + dt * eps ** 2 * k2                              # implicit diffusion
    u = a
    uh = torch.fft.fft(u, dim=-1)
    frames = [u]
    for _ in range(nt - 1):
        for _ in range(n_substep):
            reaction = u - u ** 3                                 # explicit (+u - u^3)
            uh = (uh + dt * torch.fft.fft(reaction, dim=-1)) / denom
            u = torch.fft.ifft(uh, dim=-1).real
        frames.append(u)
    return torch.stack(frames, dim=1)                            # (B, nt, nx)


def true_jvp(a, v, **kw):
    """Exact du/da . v via forward-mode AD through the solver.  a, v: (B, nx)."""
    from torch.func import jvp as func_jvp
    a = torch.as_tensor(a, dtype=kw.get('dtype', torch.float64))
    v = torch.as_tensor(v, dtype=a.dtype)
    return func_jvp(lambda aa: solve(aa, **kw), (a,), (v,))


def grf_ic(B, n, gen, ell=ELL, sigma=IC_SCALE, tau=TAU, tanh=False,
           device='cpu', dtype=torch.float64):
    """Principled IC: zero-mean periodic Gaussian random field with a Matern power spectrum.

    Spectrum P(k) = (k^2 + ell^{-2})^{-tau} -> correlation length `ell` (initial domain size)
    and smoothness `tau`; sampled by coloring white noise in Fourier and inverse FFT (exact on
    the periodic grid).  Normalized to per-sample std `sigma`.  `tanh=True` maps it into the
    (-1,1) phase-field range (otherwise the AC dynamics snap |u|>1 to the wells).  Defined in
    physical wavenumbers, so subsampling a fine-grid field gives the same continuous IC."""
    kf = _k(n, 'cpu', dtype)
    S = (kf ** 2 + ell ** -2) ** (-tau)
    S = S.clone(); S[0] = 0.0                                    # zero mean
    Sr = torch.sqrt(S)
    re = torch.randn(B, n, generator=gen, dtype=dtype)
    im = torch.randn(B, n, generator=gen, dtype=dtype)
    a = torch.fft.ifft((re + 1j * im) * Sr).real
    a = a / a.std(dim=1, keepdim=True)                          # unit std
    a = torch.tanh(sigma * a) if tanh else sigma * a
    return a.to(device)


def random_ic(B, n, gen, device='cpu', dtype=torch.float64, scale=IC_SCALE):
    """Recommended IC = bounded Matern GRF phase field, tanh into (-1,1) (see grf_ic)."""
    return grf_ic(B, n, gen, sigma=scale, tanh=True, device=device, dtype=dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    a = random_ic(4, NX, gen).to(dev)
    print(f"1D unforced Allen-Cahn  u_t = eps^2 u_xx + u - u^3 | eps={EPS} nx={NX} nt={NT} T={T_FINAL}\n")
    u = solve(a)
    print(f"trajectory u shape={tuple(u.shape)}")
    print(f"  IC a range=[{a.min():.3f},{a.max():.3f}]  (input phase field)")
    print(f"  u(.,T) range=[{u[:,-1].min():.4f},{u[:,-1].max():.4f}]  (should approach wells +/-1)")
    frac = (u[:, -1].abs() > 0.9).float().mean().item()
    print(f"  fraction of final field in wells (|u|>0.9): {100*frac:.1f}%")
    print(f"  any NaN/Inf: {bool(torch.isnan(u).any() or torch.isinf(u).any())}")
    uT = u[:, -1]
    tv = uT.diff(dim=-1).abs().mean().item()
    print(f"  final-field mean |grad| (interface density proxy): {tv:.3f}")
    # exact tangent vs central-FD JVP in a random IC direction
    v = random_ic(4, NX, gen).to(dev)
    _, w = true_jvp(a, v)
    h = 1e-6
    wfd = (solve(a + h * v) - solve(a - h * v)) / (2 * h)
    rel = (w - wfd).norm() / wfd.norm()
    print(f"  tangent vs central-FD JVP: rel err = {rel:.2e}")
