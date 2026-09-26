"""Online sTCL loss for the 1D Allen-Cahn initial->trajectory map.

For a smooth-GRF direction field v (matched to the input prior, unit-norm), the tangent
residual is the exact forward-mode JVP of the strong-form residual map a -> R(G_theta(a)):

    r_theta(a,v) = w_t - eps^2 w_xx,theta - (1 - 3 u_theta^2) w_theta,   w_theta = DS_theta(a)[v],

produced together by torch.func.jvp of model.phys (which uses the SPECTRAL Laplacian, matching
the solver -- an FD Laplacian gives a spurious ~20% floor).  Raw L^2 (parabolic, no
preconditioner -- same class as Burgers and the 2D Allen-Cahn).  Online and label-free: only
forward(-mode) passes of the operator, no tangent solves and no stored derivative labels.
"""
import numpy as np
import torch
from torch.func import jvp as func_jvp

from ac1d_time_solver import L, ELL, TAU
from runtime_profile import end as profile_end
from runtime_profile import start as profile_start

_Sr = {}


def _spectral_sqrt(nx, dev, dtype, ell=ELL, tau=TAU):
    key = (nx, str(dev), dtype, ell, tau)
    if key not in _Sr:
        kf = 2.0 * np.pi * torch.fft.fftfreq(nx, d=L / nx, device=dev).to(dtype)
        S = (kf ** 2 + ell ** -2) ** (-tau)
        S = S.clone(); S[0] = 0.0
        _Sr[key] = torch.sqrt(S)
    return _Sr[key]


def sample_dirs(B, nx, dev, dtype, ell=ELL, tau=TAU):
    """Smooth-GRF (Matern) unit-norm direction fields on device, matched to the input prior.
    Pass the DATASET's ell/tau (not the module defaults) so the sketch distribution matches a
    tuned dataset's input distribution."""
    Sr = _spectral_sqrt(nx, dev, dtype, ell=ell, tau=tau)
    w = (torch.randn(B, nx, device=dev, dtype=dtype)
         + 1j * torch.randn(B, nx, device=dev, dtype=dtype)) * Sr
    v = torch.fft.ifft(w).real
    return v / (v.norm(dim=1, keepdim=True) + 1e-12)


def stcl_loss(model, a, n_sketch=4, ell=ELL, tau=TAU):
    """ell/tau set the sketch-direction prior; pass the dataset's values (read from its metadata)
    so sTCL and DIFNO sample from the same input distribution as the data."""
    def residual(aa):
        return model.phys(aa)
    tot = a.new_zeros(())
    for _ in range(n_sketch):
        v = sample_dirs(a.shape[0], a.shape[-1], a.device, a.dtype, ell=ell, tau=tau)
        _pt = profile_start("fused_jvp_pde_residual")
        _, r = func_jvp(residual, (a,), (v,))
        profile_end(_pt)
        tot = tot + (r ** 2).mean()
    return tot / n_sketch
