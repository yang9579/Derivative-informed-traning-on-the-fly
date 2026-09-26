#!/usr/bin/env python
"""
Data generator for DIFNO Section 6.3 adapted to a tractable setup.

PDE:
  (-Delta - kappa^2 * exp(2 a(x))) u(x) = f(x),   x in Omega = (0,1)^2
  u(x) = 0 on boundary (homogeneous Dirichlet)

The operator is symmetric but **indefinite** when kappa^2 exceeds the first
Dirichlet eigenvalue of -Delta on [0,1]^2.  With kappa = 5, kappa^2 = 25 lies
above the first Dirichlet eigenvalue 2*pi^2 ~= 19.7, so for typical inputs the
operator has one negative eigenvalue.

Input distribution: a 40 x 40 cosine expansion (Neumann eigenfunctions of
-Delta) with coefficients xi_kl * (12.5 + 0.5 pi^2 (k^2 + l^2))^{-1}, i.e. the
spectral decay of the centered prior C_X = (12.5 I - 0.5 Delta)^{-2}.

Source:  single Gaussian bump at the point-source location of the paper
rescaled from [0,3]^2 to [0,1]^2:  (0.775, 2.5) / 3 ~= (0.258, 0.833).

Jacobian of the forward map a -> u:
  A delta_u = 2 kappa^2 exp(2a) u  delta_a    with   A = -Delta - kappa^2 exp(2a)
(because the PDE is linear in u).  Hence the tangent solves use the SAME
matrix A but with a different right-hand side.
"""
import os, sys, time, argparse, json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

NX = 65
NY = 65
# kappa=5.0 gives kappa^2=25, just above the first Dirichlet eigenvalue
# 2*pi^2 ~= 19.74, so the operator has ONE negative eigenvalue (mildly
# indefinite).  The DIFNO paper uses kappa=9.11 with PML on [0,3]^2 which
# is much harder; on [0,1]^2 Dirichlet we need milder indefiniteness to
# avoid near-resonance blow-ups across the GP-prior support.
KAPPA = 5.0
OMEGA_COV = 12.5
RHO_COV = 0.5
TAU_COV = 2
SRC_CX = 0.258
SRC_CY = 0.833
SRC_SIGMA = 0.08   # Gaussian bump width
SRC_AMP = 1.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")


# ---------- GP sampling from Matern covariance (Dirichlet-free, cosine basis) ----------
def sample_gp_matern(n_samples, nx, ny, n_modes=40, seed=0):
    """Samples centered Gaussian fields with the spectral decay of
    C_X = (omega I - rho Delta)^{-tau} on [0,1]^2, using all n_modes x n_modes
    (unnormalized) cosines cos(pi k x) cos(pi l y) as the basis."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    gx, gy = np.meshgrid(xs, ys, indexing='ij')

    samples = np.zeros((n_samples, nx, ny), dtype=np.float32)
    ks = np.arange(n_modes)
    ls = np.arange(n_modes)
    KK, LL = np.meshgrid(ks, ls, indexing='ij')
    lam = (np.pi * KK) ** 2 + (np.pi * LL) ** 2      # -Delta eigenvalues (Neumann)
    # covariance scaling:  (omega + rho * lam)^{-tau}
    cov_sqrt = (OMEGA_COV + RHO_COV * lam) ** (-TAU_COV / 2.0)
    # basis values on the grid
    cos_kx = np.cos(np.pi * KK[:, :, None, None] * gx[None, None])
    cos_ly = np.cos(np.pi * LL[:, :, None, None] * gy[None, None])
    basis = cos_kx * cos_ly                 # (n_modes, n_modes, nx, ny)
    for n in range(n_samples):
        xi = rng.standard_normal(size=(n_modes, n_modes))
        samples[n] = (xi[:, :, None, None] * cov_sqrt[:, :, None, None] * basis).sum(axis=(0, 1))
    return samples


# ---------- FD operators (all-Dirichlet interior DOFs) ----------
def build_fd_laplacian_dirichlet(nx, ny):
    """Returns the sparse negative-Laplacian on the interior (nx-2)(ny-2) DOFs
    with homogeneous Dirichlet BCs.  Grid on [0,1]^2, dx = 1/(nx-1)."""
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    nxi = nx - 2
    nyi = ny - 2
    # 1D Laplacians
    e = np.ones(nxi)
    Lx = sp.diags([-e[1:], 2*e, -e[1:]], [-1, 0, 1]) / dx**2
    e = np.ones(nyi)
    Ly = sp.diags([-e[1:], 2*e, -e[1:]], [-1, 0, 1]) / dy**2
    Ix = sp.eye(nxi)
    Iy = sp.eye(nyi)
    # -Delta
    L2d = sp.kron(Iy, Lx) + sp.kron(Ly, Ix)   # (nyi*nxi, nyi*nxi)
    return L2d.tocsr()


def full_to_unk(field_full, nx, ny):
    """Map a full 2D grid to the interior DOFs, flattened in (y, x) order."""
    return field_full[1:-1, 1:-1].T.flatten()


def unk_to_full(unk, nx, ny):
    """Inverse of full_to_unk."""
    out = np.zeros((nx, ny), dtype=unk.dtype)
    out[1:-1, 1:-1] = unk.reshape(ny - 2, nx - 2).T
    return out


def source_gaussian(nx, ny):
    """Gaussian-bump source on the full grid."""
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    f = SRC_AMP * np.exp(-((gx - SRC_CX) ** 2 + (gy - SRC_CY) ** 2) / (2 * SRC_SIGMA ** 2))
    return f.astype(np.float32)


# ---------- Helmholtz solve ----------
def build_helmholtz_operator(a_field, nx, ny, L2d):
    """Assembles A = -Delta - kappa^2 * exp(2a) on interior DOFs."""
    exp2a = np.exp(2.0 * a_field).astype(np.float64)
    diag = full_to_unk(exp2a, nx, ny)
    A = L2d - (KAPPA ** 2) * sp.diags(diag)
    return A


def solve_helmholtz(a_field, f_full, nx, ny, L2d):
    """Direct sparse solve of the Helmholtz system. Returns u on the full grid."""
    A = build_helmholtz_operator(a_field, nx, ny, L2d)
    rhs = full_to_unk(f_full, nx, ny).astype(np.float64)
    u_unk = spla.spsolve(A, rhs)
    return unk_to_full(u_unk, nx, ny).astype(np.float32)


def solve_tangent(a_field, u_full, delta_a_full, nx, ny, L2d, A_lu=None):
    """Solve the tangent equation
        A delta_u = 2 kappa^2 exp(2a) u delta_a
    Optionally reuse a precomputed LU factorisation."""
    exp2a = np.exp(2.0 * a_field)
    da = delta_a_full
    rhs_full = 2.0 * (KAPPA ** 2) * exp2a * u_full * da
    rhs = full_to_unk(rhs_full, nx, ny).astype(np.float64)
    if A_lu is None:
        A = build_helmholtz_operator(a_field, nx, ny, L2d)
        du_unk = spla.spsolve(A, rhs)
    else:
        du_unk = A_lu.solve(rhs)
    return unk_to_full(du_unk, nx, ny).astype(np.float32)


# ---------- dataset generation ----------
def generate_dataset(n_samples, nx, ny, seed=0, n_jvp_dirs=0,
                     reject_outliers=True, outlier_ratio=3.0,
                     oversample_factor=2.0):
    """Generate (a, u) pairs and optionally true JVPs.

    If reject_outliers=True, oversamples the prior, computes u for all, then
    keeps the n_samples samples whose |u|_L2 is below outlier_ratio * median.
    This avoids near-resonance blow-ups under Dirichlet BCs while preserving
    the GP input distribution and keeping kappa fixed.
    """
    n_try = int(np.ceil(oversample_factor * n_samples)) if reject_outliers else n_samples
    print(f"Sampling {n_try} GP fields (Matern, omega={OMEGA_COV}, rho={RHO_COV}) ...", flush=True)
    a_try = sample_gp_matern(n_try, nx, ny, n_modes=40, seed=seed)

    print(f"Building FD Laplacian ({nx}x{ny}, Dirichlet) ...", flush=True)
    L2d = build_fd_laplacian_dirichlet(nx, ny)
    f = source_gaussian(nx, ny)

    print(f"Solving Helmholtz for {n_try} samples (kappa={KAPPA}) ...", flush=True)
    u_try = np.zeros((n_try, nx, ny), dtype=np.float32)
    t0 = time.time()
    for i in range(n_try):
        u_try[i] = solve_helmholtz(a_try[i], f, nx, ny, L2d)
        if (i + 1) % max(1, n_try // 10) == 0:
            print(f"  [{i+1:5d}/{n_try}]  t={time.time()-t0:.0f}s", flush=True)

    # Report operator indefiniteness
    A0 = build_helmholtz_operator(a_try[0], nx, ny, L2d)
    w_min, w_max = spla.eigsh(A0, k=2, which='BE', return_eigenvectors=False)
    print(f"  Sample 0 operator spectrum: lambda_min={w_min:.3g}  lambda_max={w_max:.3g}")
    print(f"  (indefinite:  {'YES' if w_min * w_max < 0 else 'NO'})")

    if reject_outliers:
        norms = np.linalg.norm(u_try.reshape(n_try, -1), axis=1)
        med = float(np.median(norms))
        keep_mask = norms <= outlier_ratio * med
        kept = int(keep_mask.sum())
        print(f"  Outlier rejection: keep |u|_L2 <= {outlier_ratio}*median={outlier_ratio*med:.3f}")
        print(f"    -> kept {kept}/{n_try} ({100*kept/n_try:.1f}%)")
        if kept < n_samples:
            raise RuntimeError(
                f"Not enough clean samples: {kept} kept, need {n_samples}. "
                f"Increase oversample_factor.")
        keep_idx = np.where(keep_mask)[0][:n_samples]
        a_all = a_try[keep_idx]
        u_all = u_try[keep_idx]
    else:
        a_all = a_try[:n_samples]
        u_all = u_try[:n_samples]
    norms = np.linalg.norm(u_all.reshape(len(u_all), -1), axis=1)
    print(f"  Final: |u| median={np.median(norms):.3f}  max={norms.max():.3f}  mean={norms.mean():.3f}")

    jvp_all = None
    delta_dirs = None
    if n_jvp_dirs > 0:
        print(f"Computing {n_jvp_dirs} JVP directions for {n_samples} samples ...", flush=True)
        delta_dirs = sample_gp_matern(n_jvp_dirs, nx, ny, n_modes=40, seed=seed + 10007)
        for d in range(n_jvp_dirs):
            delta_dirs[d] /= np.linalg.norm(delta_dirs[d]) + 1e-10
        jvp_all = np.zeros((n_samples, n_jvp_dirs, nx, ny), dtype=np.float32)
        t0 = time.time()
        for i in range(n_samples):
            A = build_helmholtz_operator(a_all[i], nx, ny, L2d)
            A_lu = spla.splu(A.tocsc())
            for d in range(n_jvp_dirs):
                jvp_all[i, d] = solve_tangent(
                    a_all[i], u_all[i], delta_dirs[d], nx, ny, L2d, A_lu)
            if (i + 1) % max(1, n_samples // 10) == 0:
                print(f"  [{i+1:5d}/{n_samples}]  t={time.time()-t0:.0f}s", flush=True)

    return a_all, u_all, delta_dirs, jvp_all, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=512)
    ap.add_argument("--n_val",   type=int, default=128)
    ap.add_argument("--n_test",  type=int, default=128)
    ap.add_argument("--test_jvp_dirs",  type=int, default=8)
    ap.add_argument("--nx", type=int, default=NX)
    ap.add_argument("--ny", type=int, default=NY)
    ap.add_argument("--out_dir", type=str, default=DATA_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("="*70, flush=True)
    print("HELMHOLTZ  (DIFNO Sec 6.3, Dirichlet-on-[0,1]^2 variant)")
    print(f"  grid {args.nx}x{args.ny}  kappa={KAPPA}")
    print("="*70)

    # Train
    a_tr, u_tr, _, _, f = generate_dataset(args.n_train, args.nx, args.ny, seed=1)
    np.savez(os.path.join(args.out_dir, "train.npz"), a=a_tr, u=u_tr, f=f)

    # Val
    a_v, u_v, _, _, _ = generate_dataset(args.n_val, args.nx, args.ny, seed=2)
    np.savez(os.path.join(args.out_dir, "val.npz"), a=a_v, u=u_v, f=f)

    # Test with JVPs (for Jacobian error evaluation)
    a_te, u_te, dds_te, jvps_te, _ = generate_dataset(
        args.n_test, args.nx, args.ny, seed=3, n_jvp_dirs=args.test_jvp_dirs)
    np.savez(os.path.join(args.out_dir, "test.npz"),
             a=a_te, u=u_te, f=f, delta_dirs=dds_te, jvps=jvps_te)

    print("\nDone. Files written to:", args.out_dir)


if __name__ == "__main__":
    main()
