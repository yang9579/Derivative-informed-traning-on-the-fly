#!/usr/bin/env python
"""
FNO2D — Fourier Neural Operator for 2D operator learning
=========================================================
Maps a 2D input field a(x,y) to a 2D output field u(x,y).

Architecture (following Li et al., 2021 and DIFNO paper):
  Input [B, Nx, Ny, C_in] -> Lifting FC -> [B, Nx, Ny, width]
    -> Permute -> [B, width, Nx, Ny]
    -> n_layers x [SpectralConv2D + Conv1x1 + GELU]
    -> Permute -> [B, Nx, Ny, width]
    -> FC(width -> 128) + GELU + FC(128 -> C_out)
    -> Output [B, Nx, Ny, C_out]

Default: width=32, modes=(12,12), n_layers=4, ~500K params.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2D(nn.Module):
    """FFT-based spectral convolution in 2D."""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))

    def forward(self, x):
        B = x.shape[0]
        # x: [B, C_in, Nx, Ny]
        x_ft = torch.fft.rfft2(x, norm="ortho")  # [B, C_in, Nx, Ny//2+1]

        Nx = x.shape[2]
        out_ft = torch.zeros(B, self.out_channels, Nx, x.shape[3] // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        w1 = torch.view_as_complex(self.weights1)  # [C_in, C_out, m1, m2]
        w2 = torch.view_as_complex(self.weights2)

        # Positive frequencies in dim -2, low frequencies in dim -1
        out_ft[:, :, :self.modes1, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy",
                         x_ft[:, :, :self.modes1, :self.modes2], w1)

        # Negative frequencies in dim -2
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy",
                         x_ft[:, :, -self.modes1:, :self.modes2], w2)

        return torch.fft.irfft2(out_ft, s=(x.shape[2], x.shape[3]), norm="ortho")


class FNO2D(nn.Module):
    """Fourier Neural Operator for 2D-to-2D operator learning."""

    def __init__(self, in_channels=3, out_channels=1,
                 width=32, modes1=12, modes2=12, n_layers=4):
        super().__init__()
        self.width = width
        self.n_layers = n_layers

        # Lifting
        self.fc0 = nn.Linear(in_channels, width)

        # Fourier layers
        self.convs = nn.ModuleList()
        self.ws = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(SpectralConv2D(width, width, modes1, modes2))
            self.ws.append(nn.Conv2d(width, width, 1))

        # Projection
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        """
        x: [B, Nx, Ny, C_in]  (C_in includes coordinates + input field)
        Returns: [B, Nx, Ny, C_out]
        """
        # Lifting
        x = self.fc0(x)  # [B, Nx, Ny, width]
        x = x.permute(0, 3, 1, 2)  # [B, width, Nx, Ny]

        # Fourier layers
        for conv, w in zip(self.convs, self.ws):
            x1 = conv(x)
            x2 = w(x)
            x = x1 + x2
            x = F.gelu(x)

        # Projection
        x = x.permute(0, 2, 3, 1)  # [B, Nx, Ny, width]
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
