"""FNO operator with hard-IC ansatz for the 1D Allen-Cahn initial->trajectory map.

G_theta : a(x) (B, nx) -> u(x, t) (B, nt, nx).  A 2D FNO over (t, x), matching the
Burgers-style space-time protocol.
The IC enters as a broadcast input channel; the hard-IC ansatz

    u_theta(a)(t, x) = a(x) + (t / T) * N_theta([a, t/T, x])

makes u_theta(., 0) = a exactly, and the spectral convolution makes the field periodic in x.
So the network only fills in the interior dynamics.

`.phys(a)` evaluates the strong-form residual

    R = u_t - eps^2 u_xx - u + u^3

with centered FD in time and the SPECTRAL (FFT) Laplacian in space (matching the spectral
solver -- an FD Laplacian here gives a spurious ~20% inconsistency floor, as found in the 2D
case).  Forward-mode JVP of `phys` is exactly the forward-sensitivity (tangent) residual used
by sTCL.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ac1d_time_solver import L, EPS, T_FINAL, NX, NT, _k


class SpectralConv2d(nn.Module):
    """2D FFT spectral convolution over (t, x); keeps (mt, mx) modes.  x is the rfft axis."""
    def __init__(self, cin, cout, mt, mx):
        super().__init__()
        self.cout, self.mt, self.mx = cout, mt, mx
        s = 1.0 / (cin * cout)
        # two corner blocks: (+mt, mx) and (-mt, mx)
        self.w1 = nn.Parameter(s * torch.rand(cin, cout, mt, mx, 2))
        self.w2 = nn.Parameter(s * torch.rand(cin, cout, mt, mx, 2))

    def forward(self, x):                                          # (B, cin, T, X)
        B, _, T, X = x.shape
        xft = torch.fft.rfft2(x, norm='ortho')                    # (B, cin, T, X//2+1)
        out = torch.zeros(B, self.cout, T, X // 2 + 1, dtype=torch.cfloat, device=x.device)
        mt = min(self.mt, T // 2); mx = min(self.mx, X // 2 + 1)
        w1 = torch.view_as_complex(self.w1)[:, :, :mt, :mx]
        w2 = torch.view_as_complex(self.w2)[:, :, :mt, :mx]
        out[:, :, :mt, :mx] = torch.einsum('bitx,iotx->botx', xft[:, :, :mt, :mx], w1)
        out[:, :, -mt:, :mx] = torch.einsum('bitx,iotx->botx', xft[:, :, -mt:, :mx], w2)
        return torch.fft.irfft2(out, s=(T, X), norm='ortho')


class FNO(nn.Module):
    def __init__(self, nt=NT, nx=NX, width=32, modes=(12, 12), nlayers=4, eps=EPS, T=T_FINAL):
        super().__init__()
        if isinstance(modes, int):
            modes = (modes, modes)
        self.nt, self.nx, self.eps, self.T = nt, nx, eps, T
        ts = torch.linspace(0.0, T, nt)
        xs = torch.arange(nx) * (L / nx)
        TG, XG = torch.meshgrid(ts, xs, indexing='ij')            # (nt, nx)
        self.register_buffer('TG', TG.contiguous())
        self.register_buffer('grid', torch.stack([TG / T, XG], -1).contiguous())  # (nt,nx,2)
        self.dt = float(ts[1] - ts[0])
        self.fc0 = nn.Linear(3, width)                            # channels: [a, t, x]
        self.sconv = nn.ModuleList([SpectralConv2d(width, width, *modes) for _ in range(nlayers)])
        self.ww = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(nlayers)])
        self.fc1 = nn.Linear(width, 128); self.fc2 = nn.Linear(128, 1)

    def field(self, a):                                           # a (B,nx) -> u (B,nt,nx)
        B = a.shape[0]
        abc = a[:, None, :, None].expand(B, self.nt, self.nx, 1)  # broadcast IC over time
        inp = torch.cat([abc, self.grid[None].expand(B, -1, -1, -1)], dim=-1)   # (B,nt,nx,3)
        h = self.fc0(inp).permute(0, 3, 1, 2)                     # (B,width,nt,nx)
        for sc, w in zip(self.sconv, self.ww):
            h = F.gelu(sc(h) + w(h))
        Nout = self.fc2(F.gelu(self.fc1(h.permute(0, 2, 3, 1))))[..., 0]        # (B,nt,nx)
        return a[:, None, :] + (self.TG[None] / self.T) * Nout    # hard-IC ansatz

    def phys(self, a):                                            # strong-form residual, t=1..nt-1
        u = self.field(a)                                         # (B,nt,nx)
        k2 = _k(self.nx, u.device, u.dtype) ** 2
        # time derivative on ALL frames t=1..nt-1 (INCLUDING the final frame t=T), matching the
        # Burgers protocol: 2nd-order centered FD on the interior, 2nd-order one-sided backward FD
        # at the last frame (1st-order backward leaves a ~8% floor exactly at t=T; 2nd-order -> ~1%).
        u_t_interior = (u[:, 2:] - u[:, :-2]) / (2 * self.dt)                 # (B,nt-2,nx): frames 1..nt-2
        u_t_last = ((3 * u[:, -1] - 4 * u[:, -2] + u[:, -3]) / (2 * self.dt))[:, None]  # (B,1,nx): frame nt-1
        u_t = torch.cat([u_t_interior, u_t_last], dim=1)                     # (B,nt-1,nx): frames 1..nt-1
        lap = torch.fft.ifft(-k2[None, None] * torch.fft.fft(u, dim=-1), dim=-1).real  # spectral u_xx
        ui, lapi = u[:, 1:], lap[:, 1:]                           # frames 1..nt-1 (skip t=0 IC, keep last)
        R = u_t - self.eps ** 2 * lapi - ui + ui ** 3
        return R.reshape(u.shape[0], -1)
