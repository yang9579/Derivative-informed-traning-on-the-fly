#!/usr/bin/env python
"""
Nonlinear Diffusion-Reaction Data Generator — DIFNO Setup (Section 6.2)
========================================================================
Replicates the exact setup from:
  Yao et al., "Derivative-Informed Fourier Neural Operator", arXiv:2512.14086

PDE:
  -div(exp(a(x)) * grad(u)) + u^3 = f(x),   x in (0,1)^2
  u(x) = 0,                                   x in dOmega

Input distribution:
  a ~ GP with Matern covariance C_X = (omega*I - rho*Delta)^{-tau}
  omega = 10/3, rho = 1/30, tau = 2, mean = 0
  (Neumann BCs on covariance operator)

Source term:
  f = sum of 4 Gaussian bumps at corners of [0.25, 0.75]^2

Grid: 65x65 uniform on [0,1]^2 (dx = dy = 1/64)

Usage:
  python generate_data_difno.py --n_train 4096 --n_test 128 --jvps
"""

import os
import sys
import time
import argparse
import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ── config ────────────────────────────────────────────────────────────────────

NX = 65
NY = 65
OMEGA_COV = 10.0 / 3.0   # ~3.333
RHO_COV = 1.0 / 30.0     # ~0.0333
TAU_COV = 2
MEAN_FIELD = 0.0          # centered Gaussian

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")


# ── GP sampling via KLE (Matern with Neumann BCs on covariance) ─────────────

def sample_gp_matern(n_samples, nx, ny, omega=OMEGA_COV, rho=RHO_COV,
                      tau=TAU_COV, mean=MEAN_FIELD, n_modes=40, seed=0):
    """
    Sample from GP with Matern covariance C = (omega*I - rho*Delta)^{-tau}
    on [0,1]^2 with Neumann BCs on the covariance operator.

    Eigenfunctions (Neumann): phi_{kl}(x,y) = c_k*cos(k*pi*x) * c_l*cos(l*pi*y)
      where c_0 = 1, c_k = sqrt(2) for k > 0 (L2-orthonormal on [0,1])
    Eigenvalues of -Delta: lambda_{kl} = (k^2 + l^2) * pi^2
    Eigenvalues of C: sigma_{kl} = (omega + rho*lambda_{kl})^{-tau}

    Returns: (n_samples, nx, ny) numpy array
    """
    rng = np.random.default_rng(seed)

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)

    cos_x = np.zeros((n_modes, nx))
    cos_y = np.zeros((n_modes, ny))
    for k in range(n_modes):
        cos_x[k] = np.cos(k * np.pi * x)
        cos_y[k] = np.cos(k * np.pi * y)

    # Compute eigenvalues and normalization constants
    sigmas = np.zeros((n_modes, n_modes))
    norms = np.zeros((n_modes, n_modes))
    for k in range(n_modes):
        for l in range(n_modes):
            lam_kl = (k**2 + l**2) * np.pi**2
            sigmas[k, l] = (omega + rho * lam_kl) ** (-tau)
            ck = 1.0 if k == 0 else math.sqrt(2.0)
            cl = 1.0 if l == 0 else math.sqrt(2.0)
            norms[k, l] = ck * cl

    # Sort modes by eigenvalue magnitude (largest variance first)
    flat_idx = np.argsort(sigmas.ravel())[::-1]
    n_keep = min(n_modes * n_modes, 800)

    samples = np.zeros((n_samples, nx, ny), dtype=np.float32)

    for s in range(n_samples):
        xi = rng.standard_normal(n_keep)
        a = np.zeros((nx, ny))
        for idx_i, flat in enumerate(flat_idx[:n_keep]):
            k = flat // n_modes
            l = flat % n_modes
            coeff = math.sqrt(sigmas[k, l]) * xi[idx_i] * norms[k, l]
            a += coeff * np.outer(cos_x[k], cos_y[l])
        samples[s] = (mean + a).astype(np.float32)

    return samples


# ── Source term: 4 Gaussian bumps (DIFNO setup) ───────────────────────────

def source_term_4bumps(nx, ny):
    """
    4 Gaussian bumps at corners of [0.25, 0.75]^2.
    Matches DIFNO paper Figure 3 (peak ~16, positive bumps).

    f(x) = sum_{i=1}^{4} A * exp(-||x - c_i||^2 / (2*sigma^2))
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
    sigma = 0.1
    A = 10.0  # amplitude — tuned to match Figure 3 colorbar (~16 peak)

    f = np.zeros((nx, ny))
    for cx, cy in centers:
        f += A * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))

    return f


# ── FD operators for homogeneous Dirichlet BCs ──────────────────────────────

def build_fd_operators_dirichlet(nx, ny):
    """
    Build FD matrices on [0,1]^2 with u=0 on all boundaries.

    Unknowns: interior points (i,j) with i=1..nx-2, j=1..ny-2
    N_unk = (nx-2) * (ny-2)
    Flat indexing: k = j_int*(nx-2) + i_int  where i_int = i-1, j_int = j-1

    Returns:
      L2d: Laplacian on unknowns (N_unk x N_unk)
      Dx:  du/dx on unknowns (central differences)
      Dy:  du/dy on unknowns (central differences)
    """
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    nx_int = nx - 2
    ny_int = ny - 2
    N = nx_int * ny_int

    # 1D second derivative in x: Dirichlet BCs (boundary values = 0)
    Lx = sp.diags([np.ones(nx_int-1), -2*np.ones(nx_int), np.ones(nx_int-1)],
                   [-1, 0, 1], shape=(nx_int, nx_int)).tocsr() / dx**2

    # 1D second derivative in y: Dirichlet BCs
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

    # 2D first derivatives
    Dx = sp.kron(sp.eye(ny_int), Dx1)
    Dy = sp.kron(Dy1, sp.eye(nx_int))

    return L2d.tocsr(), Dx.tocsr(), Dy.tocsr()


# ── Grid helper functions ──────────────────────────────────────────────────

def full_to_unk(field, nx, ny):
    """Extract interior DOF values from full (nx, ny) grid.
    Unknowns: i=1..nx-2, j=1..ny-2. Flat: k = j_int*(nx-2) + i_int.
    """
    return field[1:-1, 1:-1].T.ravel()  # (nx_int*ny_int,)


def unk_to_full(u_unk, nx, ny):
    """Embed interior DOFs into full (nx, ny) grid with u=0 on boundary."""
    nx_int = nx - 2
    ny_int = ny - 2
    u_full = np.zeros((nx, ny))
    u_full[1:-1, 1:-1] = u_unk.reshape(ny_int, nx_int).T
    return u_full


def compute_a_derivatives(a_field, nx, ny):
    """Compute da/dx, da/dy on full grid using central differences."""
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)

    dax = np.zeros_like(a_field)
    day = np.zeros_like(a_field)

    dax[1:-1, :] = (a_field[2:, :] - a_field[:-2, :]) / (2*dx)
    dax[0, :] = (a_field[1, :] - a_field[0, :]) / dx
    dax[-1, :] = (a_field[-1, :] - a_field[-2, :]) / dx

    day[:, 1:-1] = (a_field[:, 2:] - a_field[:, :-2]) / (2*dy)
    day[:, 0] = (a_field[:, 1] - a_field[:, 0]) / dy
    day[:, -1] = (a_field[:, -1] - a_field[:, -2]) / dy

    return dax, day


# ── Newton solver ────────────────────────────────────────────────────────────

def solve_nonlinear_diffusion(a_field, f_field, L2d, Dx, Dy, nx, ny,
                              max_newton=50, tol=1e-10):
    """
    Solve: -div(exp(a)*grad(u)) + u^3 = f on (0,1)^2
    BC: u = 0 on all of dOmega (homogeneous Dirichlet)

    Returns: u on full (nx, ny) grid including boundary values.
    """
    nx_int = nx - 2
    ny_int = ny - 2
    N = nx_int * ny_int

    # Diffusivity and its derivatives at interior points
    D_full = np.exp(a_field)
    D = full_to_unk(D_full, nx, ny)

    dax_full, day_full = compute_a_derivatives(a_field, nx, ny)
    dax = full_to_unk(dax_full, nx, ny)
    day = full_to_unk(day_full, nx, ny)

    f_unk = full_to_unk(f_field, nx, ny)

    # Diffusion operator: A = -(D*Lap + D*dax*Dx + D*day*Dy)
    # From -div(exp(a)*grad(u)) = -exp(a)*(Lap(u) + da/dx*du/dx + da/dy*du/dy)
    D_diag = sp.diags(D)
    A_diff = -(D_diag @ L2d) - sp.diags(D * dax) @ Dx - sp.diags(D * day) @ Dy

    # No BC contributions since u=0 on all boundaries
    rhs = f_unk

    # Initial guess: linear solve
    u = spla.spsolve(A_diff, rhs)

    for it in range(max_newton):
        R = A_diff @ u + u**3 - rhs
        res_norm = np.linalg.norm(R) / (np.linalg.norm(rhs) + 1e-15)

        if res_norm < tol:
            break

        J = A_diff + sp.diags(3.0 * u**2)
        delta_u = spla.spsolve(J, -R)
        u = u + delta_u

    if res_norm > 1e-6:
        print(f"    Newton warning: res={res_norm:.2e} after {it+1} iters")

    return unk_to_full(u, nx, ny)


def solve_tangent_equation(a_field, u_full, delta_a, L2d, Dx, Dy, nx, ny):
    """
    Solve tangent equation for delta_a -> delta_u:
      -div(D*grad(delta_u)) + 3u^2*delta_u = div(D*delta_a*grad(u))

    Both u and delta_u have homogeneous Dirichlet BCs (u=0, delta_u=0).
    No BC contributions needed.

    Returns: delta_u on full (nx, ny) grid.
    """
    nx_int = nx - 2
    ny_int = ny - 2

    D_full = np.exp(a_field)
    D = full_to_unk(D_full, nx, ny)

    dax_full, day_full = compute_a_derivatives(a_field, nx, ny)
    dax = full_to_unk(dax_full, nx, ny)
    day = full_to_unk(day_full, nx, ny)

    u_unk = full_to_unk(u_full, nx, ny)

    # Jacobian of PDE operator
    D_diag = sp.diags(D)
    A_diff = -(D_diag @ L2d) - sp.diags(D * dax) @ Dx - sp.diags(D * day) @ Dy
    J = A_diff + sp.diags(3.0 * u_unk**2)

    # RHS: div(D * delta_a * grad(u))
    da = full_to_unk(delta_a, nx, ny)
    ddax_full, dday_full = compute_a_derivatives(delta_a, nx, ny)
    d_da_x = full_to_unk(ddax_full, nx, ny)
    d_da_y = full_to_unk(dday_full, nx, ny)

    # u derivatives — no BC contributions since u=0 on boundary
    Lap_u = L2d @ u_unk
    ux = Dx @ u_unk
    uy = Dy @ u_unk

    rhs = (D * da * Lap_u
           + D * (dax * da + d_da_x) * ux
           + D * (day * da + d_da_y) * uy)

    delta_u = spla.spsolve(J, rhs)
    return unk_to_full(delta_u, nx, ny)


# ── dataset generation ────────────────────────────────────────────────────────

def generate_dataset(n_samples, nx, ny, seed=0, compute_jvps=False, n_jvp_dirs=5):
    """Generate (a, u) pairs and optionally true JVPs."""
    print(f"\n  Sampling {n_samples} GP fields (Matern, omega={OMEGA_COV:.3f}, "
          f"rho={RHO_COV:.4f}) ...", flush=True)
    a_all = sample_gp_matern(n_samples, nx, ny, n_modes=40, seed=seed)

    print(f"  Building FD operators (nx={nx}, ny={ny}, all-Dirichlet) ...",
          flush=True)
    L2d, Dx, Dy = build_fd_operators_dirichlet(nx, ny)
    f = source_term_4bumps(nx, ny)

    u_all = np.zeros((n_samples, nx, ny), dtype=np.float32)

    delta_dirs = None
    jvp_all = None
    if compute_jvps:
        # Sample perturbation directions from same GP (mean=0)
        delta_dirs = sample_gp_matern(n_jvp_dirs, nx, ny, n_modes=40,
                                       mean=0.0, seed=seed + 999999)
        for d in range(n_jvp_dirs):
            delta_dirs[d] /= np.linalg.norm(delta_dirs[d]) + 1e-10
        jvp_all = np.zeros((n_samples, n_jvp_dirs, nx, ny), dtype=np.float32)

    t0 = time.time()
    for i in range(n_samples):
        u_all[i] = solve_nonlinear_diffusion(
            a_all[i], f, L2d, Dx, Dy, nx, ny)

        if compute_jvps:
            for d in range(n_jvp_dirs):
                jvp_all[i, d] = solve_tangent_equation(
                    a_all[i], u_all[i], delta_dirs[d],
                    L2d, Dx, Dy, nx, ny)

        if (i + 1) % max(1, n_samples // 10) == 0 or (i + 1) == n_samples:
            elapsed = time.time() - t0
            print(f"    [{i+1:4d}/{n_samples}]  "
                  f"a range=[{a_all[i].min():.2f}, {a_all[i].max():.2f}]  "
                  f"u range=[{u_all[i].min():.3f}, {u_all[i].max():.3f}]  "
                  f"t={elapsed:.1f}s  ({elapsed/(i+1):.2f}s/sample)",
                  flush=True)

    if compute_jvps:
        return a_all, u_all, jvp_all, delta_dirs
    return a_all, u_all


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=4096)
    ap.add_argument("--n_test",  type=int, default=128)
    ap.add_argument("--n_val",   type=int, default=128)
    ap.add_argument("--nx",      type=int, default=NX)
    ap.add_argument("--ny",      type=int, default=NY)
    ap.add_argument("--jvps",    action="store_true",
                    help="Compute true JVPs for test set")
    ap.add_argument("--val_jvps", action="store_true",
                    help="Compute an independent true-JVP bank for validation")
    ap.add_argument("--n_jvp_dirs", type=int, default=10)
    ap.add_argument("--save_dir", type=str, default=DATA_DIR)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 65)
    print("  Nonlinear Diffusion-Reaction — DIFNO Setup (Section 6.2)")
    print("=" * 65)
    print(f"  nx={args.nx}  ny={args.ny}")
    print(f"  PDE: -div(exp(a)*grad(u)) + u^3 = f")
    print(f"  BC:  u = 0 on all of dOmega (homogeneous Dirichlet)")
    print(f"  Prior: Matern C=(10I/3 - Delta/30)^{{-2}}, mean=0")
    print(f"  Source: 4 Gaussian bumps at corners of [0.25,0.75]^2")
    print(f"  N_train={args.n_train}  N_test={args.n_test}  N_val={args.n_val}")
    print(f"  Compute test JVPs: {args.jvps}")
    print(f"  Compute validation JVPs: {args.val_jvps}")

    # Training set
    print(f"\nGenerating {args.n_train} training samples ...", flush=True)
    a_train, u_train = generate_dataset(
        args.n_train, args.nx, args.ny, seed=0)
    train_path = os.path.join(args.save_dir, "train.npz")
    np.savez_compressed(train_path, a=a_train, u=u_train)
    print(f"  Saved {train_path} ({os.path.getsize(train_path)/1e6:.1f} MB)")

    # Validation set
    print(f"\nGenerating {args.n_val} validation samples ...", flush=True)
    val_path = os.path.join(args.save_dir, "val.npz")
    if args.val_jvps:
        a_val, u_val, jvp_val, delta_dirs_val = generate_dataset(
            args.n_val, args.nx, args.ny, seed=50000,
            compute_jvps=True, n_jvp_dirs=args.n_jvp_dirs)
        np.savez_compressed(
            val_path, a=a_val, u=u_val, jvps=jvp_val,
            delta_dirs=delta_dirs_val)
    else:
        a_val, u_val = generate_dataset(
            args.n_val, args.nx, args.ny, seed=50000)
        np.savez_compressed(val_path, a=a_val, u=u_val)
    print(f"  Saved {val_path} ({os.path.getsize(val_path)/1e6:.1f} MB)")

    # Test set
    print(f"\nGenerating {args.n_test} test samples ...", flush=True)
    if args.jvps:
        a_test, u_test, jvp_test, delta_dirs = generate_dataset(
            args.n_test, args.nx, args.ny, seed=100000,
            compute_jvps=True, n_jvp_dirs=args.n_jvp_dirs)
        test_path = os.path.join(args.save_dir, "test.npz")
        np.savez_compressed(test_path, a=a_test, u=u_test,
                           jvps=jvp_test, delta_dirs=delta_dirs)
    else:
        a_test, u_test = generate_dataset(
            args.n_test, args.nx, args.ny, seed=100000)
        test_path = os.path.join(args.save_dir, "test.npz")
        np.savez_compressed(test_path, a=a_test, u=u_test)

    print(f"  Saved {test_path} ({os.path.getsize(test_path)/1e6:.1f} MB)")

    # Print summary statistics
    print(f"\n{'='*65}")
    print(f"  Summary:")
    print(f"    a_train: mean={a_train.mean():.3f} std={a_train.std():.3f} "
          f"range=[{a_train.min():.2f}, {a_train.max():.2f}]")
    print(f"    u_train: mean={u_train.mean():.4f} std={u_train.std():.4f} "
          f"range=[{u_train.min():.4f}, {u_train.max():.4f}]")
    print(f"    a_test:  mean={a_test.mean():.3f} std={a_test.std():.3f}")
    print(f"    u_test:  mean={u_test.mean():.4f} std={u_test.std():.4f}")
    print(f"{'='*65}")
    print("\nDone.")


if __name__ == "__main__":
    main()
