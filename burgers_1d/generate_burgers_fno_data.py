#!/usr/bin/env python
"""
Data Generation for FNO-based Burgers 1D OCP
=============================================
Samples random RBF control functions f(t) and solves the Burgers PDE
using the implicit Picard FD solver from adjoint_burgers.py.

PDE:
  u_t + u*u_x = nu*u_xx + f(t),  x in [0,1], t in [0,1]
  u(0,t) = u(1,t) = 0  (Dirichlet)
  u(x,0) = 0           (zero IC)
  nu = 0.01

Control:
  f(t) = sum_i c_i * exp(-(t - t_i)^2 / (2*sigma^2))
  M=16 RBF centers, sigma=0.2, c_i ~ N(0, sigma_c^2)
  The paper/Drive dataset uses sigma_c=1.5 with seed=42.

Output:
  fno_data/burgers_fno_data.npz containing:
    train_f:  (N_train, Nx, Nt) — f(t) broadcast to all x
    train_u:  (N_train, Nx, Nt) — u(x,t) solution
    test_f:   (N_test, Nx, Nt)
    test_u:   (N_test, Nx, Nt)
    val_f:    (N_val, Nx, Nt)
    val_u:    (N_val, Nx, Nt)
    x_grid:   (Nx,)
    t_grid:   (Nt,)

Usage:
  python generate_burgers_fno_data.py [--N_train 2048] [--sigma_c 1.5] [--seed 42]
"""

import os
import sys
import argparse
import time

import numpy as np
import scipy.sparse as sp

# Import the FD solver infrastructure from adjoint_burgers
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from adjoint_burgers import (
    build_laplacian_1d, forward_solve_implicit,
    NU, T_FINAL, N_PICARD,
)


def generate_rbf_control(c, M, T, sigma, t_grid):
    """
    Evaluate f(t) = sum_i c_i * exp(-(t - t_i)^2 / (2*sigma^2)).
    c: (M,) coefficients
    t_grid: (Nt,) time points
    Returns: (Nt,)
    """
    t_centers = np.linspace(0, T, M)
    diffs = (t_grid[:, None] - t_centers[None, :]) ** 2  # (Nt, M)
    phi = np.exp(-diffs / (2 * sigma ** 2))               # (Nt, M)
    return phi @ c                                         # (Nt,)


def solve_one_sample(c, M, T, sigma, Nx_solve, Nt_solve, Nx_out, Nt_out):
    """
    Solve Burgers for control coefficients c on the fine FD grid
    (Nx_solve, Nt_solve), then interpolate to (Nx_out, Nt_out).

    Returns:
        f_grid: (Nx_out, Nt_out) — f(t) broadcast to all x
        u_grid: (Nx_out, Nt_out) — u(x,t) on the output grid
    """
    dt_solve = T / Nt_solve
    dx_solve = 1.0 / (Nx_solve + 1)

    # Build solver matrix
    L = build_laplacian_1d(Nx_solve)
    A_sp = sp.eye(Nx_solve, format="csc") - dt_solve * NU * L.tocsc()

    # Time grid for solver: t = 0, dt, ..., (Nt_solve-1)*dt
    t_solve = np.arange(Nt_solve) * dt_solve
    f_vals = generate_rbf_control(c, M, T, sigma, t_solve)

    # Forward solve: U is (Nx_solve, Nt_solve+1), includes t=0
    U = forward_solve_implicit(A_sp, f_vals, Nx_solve, Nt_solve, dt_solve, dx_solve)
    # U[:, 0] = 0 (IC), U[:, n+1] = solution at t = (n+1)*dt

    # Solver spatial grid (interior only): x_i = i*dx, i=1..Nx_solve
    x_solve = np.arange(1, Nx_solve + 1) * dx_solve
    # Solver temporal grid: t = 0, dt, ..., T
    t_solve_full = np.arange(Nt_solve + 1) * dt_solve

    # Output grid (including boundaries for FNO)
    x_out = np.linspace(0, 1, Nx_out)
    t_out = np.linspace(0, T, Nt_out)

    # Interpolate U from (Nx_solve interior, Nt_solve+1) to (Nx_out, Nt_out)
    from scipy.interpolate import RegularGridInterpolator

    # Add boundary values (u=0 at x=0 and x=1)
    x_full = np.concatenate([[0], x_solve, [1]])
    U_full = np.zeros((Nx_solve + 2, Nt_solve + 1))
    U_full[1:-1, :] = U

    interp = RegularGridInterpolator(
        (x_full, t_solve_full), U_full,
        method='linear', bounds_error=False, fill_value=0.0
    )

    XX, TT = np.meshgrid(x_out, t_out, indexing='ij')
    pts = np.stack([XX.ravel(), TT.ravel()], axis=-1)
    u_grid = interp(pts).reshape(Nx_out, Nt_out)

    # f(t) on the output time grid, broadcast to all x
    f_out = generate_rbf_control(c, M, T, sigma, t_out)
    f_grid = np.broadcast_to(f_out[None, :], (Nx_out, Nt_out)).copy()

    return f_grid, u_grid


def validate_sample(f_grid, u_grid, max_abs_u):
    """Return float32 arrays only if the solver output is numerically usable."""
    if not np.isfinite(f_grid).all():
        raise ValueError("Non-finite control values")
    if not np.isfinite(u_grid).all():
        raise ValueError("Non-finite solution")
    max_u = float(np.max(np.abs(u_grid)))
    if max_u > max_abs_u:
        raise ValueError(f"Unstable solution: max |u|={max_u:.3e} > {max_abs_u:.3e}")

    f32 = f_grid.astype(np.float32)
    u32 = u_grid.astype(np.float32)
    if not np.isfinite(f32).all() or not np.isfinite(u32).all():
        raise ValueError("Non-finite values after float32 cast")
    return f32, u32


def main():
    parser = argparse.ArgumentParser(description="Generate Burgers FNO training data")
    parser.add_argument("--N_train", type=int, default=2048)
    parser.add_argument("--N_test", type=int, default=128)
    parser.add_argument("--N_val", type=int, default=128)
    parser.add_argument("--Nx_out", type=int, default=64,
                        help="Spatial resolution for FNO")
    parser.add_argument("--Nt_out", type=int, default=100,
                        help="Temporal resolution for FNO")
    parser.add_argument("--Nx_solve", type=int, default=128,
                        help="Spatial resolution for FD solver (interior points)")
    parser.add_argument("--Nt_solve", type=int, default=200,
                        help="Temporal resolution for FD solver")
    parser.add_argument("--M", type=int, default=16,
                        help="Number of RBF centers")
    parser.add_argument("--sigma_rbf", type=float, default=0.2,
                        help="RBF width")
    parser.add_argument("--sigma_c", type=float, default=1.5,
                        help="Std of RBF coefficients c_i ~ N(0, sigma_c^2); paper/Drive value is 1.5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join(SCRIPT_DIR, "fno_data"))
    parser.add_argument("--max_retries", type=int, default=100,
                        help="Maximum same-distribution resampling attempts per sample.")
    parser.add_argument("--max_abs_u", type=float, default=20.0,
                        help="Reject numerically unstable samples with max |u| above this value.")
    parser.add_argument("--save_diagnostics", action="store_true",
                        help="Store rejection diagnostics in the npz. Disabled by default to keep paper-style keys.")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    N_total = args.N_train + args.N_test + args.N_val

    print("=" * 60)
    print("  Burgers 1D — FNO Data Generation")
    print("=" * 60)
    print(f"  N_total={N_total} (train={args.N_train}, test={args.N_test}, val={args.N_val})")
    print(f"  Solver grid: Nx={args.Nx_solve}, Nt={args.Nt_solve}")
    print(f"  Output grid: Nx={args.Nx_out}, Nt={args.Nt_out}")
    print(f"  RBF: M={args.M}, sigma={args.sigma_rbf}, sigma_c={args.sigma_c}")
    print(f"  Rejection check: finite u and max |u| <= {args.max_abs_u}")
    print(f"  Seed: {args.seed}")
    print(f"  Save: {args.save_dir}")
    print()

    all_f = np.zeros((N_total, args.Nx_out, args.Nt_out), dtype=np.float32)
    all_u = np.zeros((N_total, args.Nx_out, args.Nt_out), dtype=np.float32)
    all_c = np.zeros((N_total, args.M), dtype=np.float32)

    t0 = time.time()
    n_rejected = 0
    attempts_per_sample = np.zeros(N_total, dtype=np.int32)

    for i in range(N_total):
        last_error = None
        for attempt in range(1, args.max_retries + 1):
            c = rng.randn(args.M).astype(np.float64) * args.sigma_c
            try:
                f_grid, u_grid = solve_one_sample(
                    c, args.M, T_FINAL, args.sigma_rbf,
                    args.Nx_solve, args.Nt_solve,
                    args.Nx_out, args.Nt_out,
                )
                f32, u32 = validate_sample(f_grid, u_grid, args.max_abs_u)
                all_f[i] = f32
                all_u[i] = u32
                all_c[i] = c.astype(np.float32)
                attempts_per_sample[i] = attempt
                n_rejected += attempt - 1
                break
            except Exception as e:
                last_error = e
        else:
            raise RuntimeError(
                f"Could not generate a valid sample after {args.max_retries} "
                f"attempts; last error: {last_error}"
            )

        if (i + 1) % 100 == 0 or (i + 1) <= 5:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (N_total - i - 1)
            print(f"  [{i+1:5d}/{N_total}]  "
                  f"|u|_max={np.abs(all_u[i]).max():.2f}  "
                  f"|f|_max={np.abs(all_f[i]).max():.2f}  "
                  f"t={elapsed:.0f}s  ETA={eta:.0f}s  "
                  f"attempts={attempts_per_sample[i]}  "
                  f"rejects={n_rejected}", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s  ({n_rejected} rejected draws)")

    # Split into train/test/val
    idx = np.arange(N_total)
    rng.shuffle(idx)

    train_idx = idx[:args.N_train]
    test_idx = idx[args.N_train:args.N_train + args.N_test]
    val_idx = idx[args.N_train + args.N_test:]

    save_path = os.path.join(args.save_dir, "burgers_fno_data.npz")
    save_kwargs = dict(
        train_f=all_f[train_idx],
        train_u=all_u[train_idx],
        test_f=all_f[test_idx],
        test_u=all_u[test_idx],
        val_f=all_f[val_idx],
        val_u=all_u[val_idx],
        train_c=all_c[train_idx],
        test_c=all_c[test_idx],
        val_c=all_c[val_idx],
        x_grid=np.linspace(0, 1, args.Nx_out).astype(np.float32),
        t_grid=np.linspace(0, T_FINAL, args.Nt_out).astype(np.float32),
    )
    if args.save_diagnostics:
        save_kwargs.update(
            attempts_per_sample=attempts_per_sample,
            max_abs_u=np.array(args.max_abs_u, dtype=np.float32),
        )
    np.savez(save_path, **save_kwargs)
    fsize = os.path.getsize(save_path) / 1e6
    print(f"\n  Saved: {save_path} ({fsize:.1f} MB)")

    # Print statistics
    print(f"\n  Train: f range [{all_f[train_idx].min():.2f}, {all_f[train_idx].max():.2f}]"
          f"  u range [{all_u[train_idx].min():.2f}, {all_u[train_idx].max():.2f}]")
    print(f"  Test:  f range [{all_f[test_idx].min():.2f}, {all_f[test_idx].max():.2f}]"
          f"  u range [{all_u[test_idx].min():.2f}, {all_u[test_idx].max():.2f}]")
    print(f"  Val:   f range [{all_f[val_idx].min():.2f}, {all_f[val_idx].max():.2f}]"
          f"  u range [{all_u[val_idx].min():.2f}, {all_u[val_idx].max():.2f}]")


if __name__ == "__main__":
    main()
