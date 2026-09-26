"""Thin wrapper around the repository's existing FNO2D model.

The actual architecture is ``fno2d.py::FNO2D`` in the repository root.  The wrapper prepares
the grid input

    [x1, x2, u1_norm, u2_norm]

(the control channels are standardized by dataset statistics, since the
manufactured control ``u = -mu Delta y + (y.grad)y + grad p`` has a much larger
magnitude than the O(1) state) and returns the predicted fields

    (y1, y2, p).

``forward`` returns the velocity ``(y1, y2)`` for JVP / evaluation; ``state_uvp``
returns ``(y1, y2, p)`` for the data loss; ``state_grid`` returns the three raw
fields used to build the strong-form residual in sTCL.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

try:
    from .navier_stokes_2d import make_grid, MU_FIXED
    from ..fno2d import FNO2D
except ImportError:  # Support ``python navier_stokes_2d/<script>.py``.
    sys.path.append(PARENT_DIR)
    sys.path.insert(0, SCRIPT_DIR)
    from navier_stokes_2d import make_grid, MU_FIXED
    from fno2d import FNO2D


class FNOAONN(nn.Module):
    def __init__(self, gy: int = 64, gx: int = 64, width: int = 32, modes: int = 12, layers: int = 4):
        super().__init__()
        self.gy = gy
        self.gx = gx
        x, y = make_grid(gy, gx)
        self.register_buffer("xgrid", x)
        self.register_buffer("ygrid", y)
        self.mu = MU_FIXED   # viscosity used by the sTCL residual; set from the dataset in training
        self.hx = 1.0 / (gx - 1)
        self.hy = 1.0 / (gy - 1)
        # Input standardization for the two control channels (set from train data).
        self.register_buffer("u_mean", torch.zeros(2))
        self.register_buffer("u_std", torch.ones(2))
        self.fno = FNO2D(
            in_channels=4,
            out_channels=3,
            width=width,
            modes1=modes,
            modes2=modes,
            n_layers=layers,
        )

    def set_input_norm(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.u_mean.copy_(mean.reshape(2).to(self.u_mean))
        self.u_std.copy_(std.reshape(2).clamp_min(1e-8).to(self.u_std))

    def input_grid(self, u: torch.Tensor) -> torch.Tensor:
        batch = u.shape[0]
        x = self.xgrid[None].expand(batch, -1, -1)
        y = self.ygrid[None].expand(batch, -1, -1)
        un = (u - self.u_mean.reshape(1, 2, 1, 1)) / self.u_std.reshape(1, 2, 1, 1)
        return torch.cat([x[..., None], y[..., None], un.permute(0, 2, 3, 1)], dim=-1)

    def state_grid(self, u: torch.Tensor):
        out = self.fno(self.input_grid(u)).permute(0, 3, 1, 2).contiguous()  # (B,3,gy,gx)
        return out[:, 0], out[:, 1], out[:, 2]

    def state_uvp(self, u: torch.Tensor) -> torch.Tensor:
        y1, y2, p = self.state_grid(u)
        return torch.stack([y1, y2, p], dim=1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        y1, y2, _ = self.state_grid(u)
        return torch.stack([y1, y2], dim=1).contiguous()

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
