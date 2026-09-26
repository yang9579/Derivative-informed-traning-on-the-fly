#!/usr/bin/env python
"""Forward steady-state Navier-Stokes solver for the control-to-state map U -> (Y, P).

Given a body force U, solve the incompressible steady state equation

    -mu Delta Y + (Y.grad)Y + grad P = U,   div Y = 0,   Y = 0 on dOmega,

by Newton's method.  The Newton Jacobian is exactly the Oseen saddle system already
assembled by ``generate_jvps.factor_tangent`` (so this reuses that machinery):

    Stokes initial guess  ->  Newton steps  J(Y) [dY; dP] = -F(Y, P)

with F the nonlinear residual and J the Newton/Oseen linearization.  A
Brezzi-Pitkaranta pressure stabilization (alpha) removes the collocated
checkerboard mode.  The final Newton LU factor is the Oseen tangent at the
converged state -- reused to produce the DIFNO/eval tangent labels for free.
"""
from __future__ import annotations

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from .generate_jvps import fd_ops, factor_tangent
except ImportError:  # Support direct script execution.
    from generate_jvps import fd_ops, factor_tangent


def _nonlinear_residual(Y1, Y2, P, U1, U2, mu, lap, Dx, Dy, hx, hy, alpha):
    """Discrete residual F = (F_mom1, F_mom2, F_cont) on interior unknowns."""
    conv1 = Y1 * (Dx @ Y1) + Y2 * (Dy @ Y1)          # (Y.grad)Y1
    conv2 = Y1 * (Dx @ Y2) + Y2 * (Dy @ Y2)          # (Y.grad)Y2
    Fm1 = -mu * (lap @ Y1) + conv1 + Dx @ P - U1
    Fm2 = -mu * (lap @ Y2) + conv2 + Dy @ P - U2
    Fc = Dx @ Y1 + Dy @ Y2 - alpha * hx * hy * (lap @ P)
    return np.concatenate([Fm1, Fm2, Fc])


def solve_steady_ns(U, mu, alpha=1e-3, tol=1e-11, maxit=50, verbose=False):
    """Solve steady NS for body force U (2, gy, gx). Returns (Y, P, info).

    Y: (2, gy, gx) with zero boundary; P: (gy, gx) mean-subtracted interior;
    info: dict(iters, res, lu, lap, Dx, Dy, hx, hy) -- lu is the Oseen factor at Y*
    (reusable for tangent solves).
    """
    _, gy, gx = U.shape
    lap, Dx, Dy, hx, hy = fd_ops(gy, gx)
    n = (gy - 2) * (gx - 2)
    U1 = U[0, 1:-1, 1:-1].ravel().astype(np.float64)
    U2 = U[1, 1:-1, 1:-1].ravel().astype(np.float64)
    rhs0 = np.concatenate([U1, U2, np.zeros(n)])

    # Stokes initial guess: Oseen Jacobian at Y=0 is the Stokes operator.
    lu = factor_tangent(np.zeros((2, gy, gx)), lap, Dx, Dy, hx, hy, mu, alpha)
    sol = lu.solve(rhs0)
    Y1, Y2, P = sol[:n].copy(), sol[n:2 * n].copy(), sol[2 * n:].copy()

    res = np.inf
    denom = np.linalg.norm(rhs0) + 1e-30
    it = 0
    for it in range(1, maxit + 1):
        F = _nonlinear_residual(Y1, Y2, P, U1, U2, mu, lap, Dx, Dy, hx, hy, alpha)
        res = np.linalg.norm(F) / denom
        if verbose:
            print(f"    newton {it}: res={res:.3e}")
        if res < tol:
            break
        Yfull = np.zeros((2, gy, gx))
        Yfull[0, 1:-1, 1:-1] = Y1.reshape(gy - 2, gx - 2)
        Yfull[1, 1:-1, 1:-1] = Y2.reshape(gy - 2, gx - 2)
        lu = factor_tangent(Yfull, lap, Dx, Dy, hx, hy, mu, alpha)
        d = lu.solve(-F)
        Y1 += d[:n]; Y2 += d[n:2 * n]; P += d[2 * n:]

    Y = np.zeros((2, gy, gx), dtype=np.float64)
    Y[0, 1:-1, 1:-1] = Y1.reshape(gy - 2, gx - 2)
    Y[1, 1:-1, 1:-1] = Y2.reshape(gy - 2, gx - 2)
    Pf = np.zeros((gy, gx), dtype=np.float64)
    Pf[1:-1, 1:-1] = P.reshape(gy - 2, gx - 2)
    Pf = Pf - Pf.mean()
    info = dict(iters=it, res=res, lu=lu, lap=lap, Dx=Dx, Dy=Dy, hx=hx, hy=hy)
    return Y, Pf, info
