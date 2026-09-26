#!/usr/bin/env python
"""
Generate Burgers JVP data — v2 (autograd-through-forward-solver).

This replaces generate_burgers_jvps.py to fix two issues identified in the
audit:

(1) The previous tangent solver used SEMI-IMPLICIT (explicit convection)
    time stepping while the forward solver uses IMPLICIT PICARD (implicit
    convection).  These two schemes converge to the same continuous tangent
    as dt -> 0 but give different discrete Jacobians at finite dt.  DIFNO
    labels were therefore not the exact discrete Jacobian of the data-
    generating forward solver.  Here we fix this by computing J*v as
        true_jvp = torch.func.jvp(forward_solve_implicit_torch)(f, v)
    so the label is *by construction* the exact discrete Jacobian.

(2) train_jvps.npz and test_jvps.npz reused the SAME random directions
    (DIFNO trained on, evaluation tested on, the same 4 vectors).  We use
    independent seeds for train and test directions.

The forward operator is the same implicit-Picard solver as adjoint_burgers.py
(used by data generation), ported to torch.
"""
import os, sys, time, argparse, math
import numpy as np
import torch
from torch.func import jvp as func_jvp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from adjoint_burgers import NU, T_FINAL, N_PICARD
from generate_burgers_fno_data import generate_rbf_control

# Solver grid (must match data generation in adjoint_burgers / fno_data)
NX_SOLVE = 128
NT_SOLVE = 200
NX_OUT = 64
NT_OUT = 100


# ---------- torch-native implicit Picard forward solver ----------

def build_A_torch(Nx, dx, dt, nu, device, dtype=torch.float64):
    """A = I - dt*nu*L  with L the negative-diagonal 1D Laplacian
    (matching adjoint_burgers.build_laplacian_1d).
    L[i,i] = -2/dx^2, L[i, i±1] = 1/dx^2 ; Dirichlet (i.e., L is on the
    interior nodes only)."""
    L = torch.zeros(Nx, Nx, device=device, dtype=dtype)
    diag = torch.full((Nx,), -2.0 / dx**2, device=device, dtype=dtype)
    off  = torch.full((Nx-1,), 1.0 / dx**2, device=device, dtype=dtype)
    L = torch.diag(diag) + torch.diag(off, 1) + torch.diag(off, -1)
    A = torch.eye(Nx, device=device, dtype=dtype) - dt * nu * L
    return A


def convection_centered_torch(u, dx):
    """N_i(u) = u_i * (u_{i+1} - u_{i-1}) / (2*dx) ; Dirichlet ghost = 0."""
    # u: (Nx,) ; pad with zeros
    u_ext = torch.cat([torch.zeros_like(u[:1]), u, torch.zeros_like(u[:1])])
    return u * (u_ext[2:] - u_ext[:-2]) / (2.0 * dx)


def forward_solve_implicit_torch(f, A, Nx, Nt, dt, dx, n_picard=N_PICARD):
    """Implicit Picard forward solve, identical scheme to
    adjoint_burgers.forward_solve_implicit but in torch (so we can
    differentiate through it).

    f : (Nt,) torch tensor — control values f^n for n=0,..,Nt-1
    Returns U : (Nx, Nt+1) torch tensor with U[:, 0] = 0.
    """
    device, dtype = f.device, f.dtype
    ones = torch.ones(Nx, device=device, dtype=dtype)

    U_cols = [torch.zeros(Nx, device=device, dtype=dtype)]   # U[:, 0] = 0
    for n in range(Nt):
        u_n = U_cols[n]
        rhs0 = u_n + dt * f[n] * ones
        u_k = u_n
        for _ in range(n_picard):
            rhs = rhs0 - dt * convection_centered_torch(u_k, dx)
            u_k = torch.linalg.solve(A, rhs)
        U_cols.append(u_k)
    return torch.stack(U_cols, dim=1)   # (Nx, Nt+1)


# ---------- interpolation utilities ----------

def interpolate_to_fno_grid(W_hires, Nx_solve, Nt_solve, Nx_out, Nt_out, T):
    """W_hires: (Nx, Nt+1) numpy ; returns (Nx_out, Nt_out) numpy on the
    output grid x_out = linspace(0,1,Nx_out), t_out = linspace(0,T,Nt_out).
    Boundary values w=0 added at x=0 and x=1 (Dirichlet)."""
    from scipy.interpolate import RegularGridInterpolator
    dx_s = 1.0 / (Nx_solve + 1)
    x_solve = np.arange(1, Nx_solve + 1) * dx_s
    t_solve_full = np.arange(Nt_solve + 1) * (T / Nt_solve)
    x_full = np.concatenate([[0], x_solve, [1]])
    W_full = np.zeros((Nx_solve + 2, Nt_solve + 1))
    W_full[1:-1, :] = W_hires

    interp = RegularGridInterpolator(
        (x_full, t_solve_full), W_full,
        method='linear', bounds_error=False, fill_value=0.0)
    x_out = np.linspace(0, 1, Nx_out)
    t_out = np.linspace(0, T, Nt_out)
    XX, TT = np.meshgrid(x_out, t_out, indexing='ij')
    return interp(np.stack([XX.ravel(), TT.ravel()], axis=-1)).reshape(Nx_out, Nt_out)


# ---------- direction sampling ----------

def sample_directions_solver_grid(n_dirs, Nt_solve, seed):
    """Sample n_dirs random space-uniform directions on the solver time grid.
    Each direction is L2-normalised so ||v||_2 = 1 in R^Nt_solve."""
    rng = np.random.RandomState(seed)
    V = rng.randn(n_dirs, Nt_solve)
    for d in range(n_dirs):
        V[d] /= (np.linalg.norm(V[d]) + 1e-12)
    return V.astype(np.float64)


def project_dir_to_fno_grid(v_hires, Nt_solve, Nt_out, T):
    """v_hires: (Nt_solve,) ; returns (Nt_out,) on output time grid."""
    t_solve = np.arange(Nt_solve) * (T / Nt_solve)
    t_out = np.linspace(0, T, Nt_out)
    return np.interp(t_out, t_solve, v_hires)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str,
                    default=os.path.join(SCRIPT_DIR, "fno_data", "burgers_fno_data.npz"))
    ap.add_argument("--save_dir", type=str,
                    default=os.path.join(SCRIPT_DIR, "fno_data"))
    ap.add_argument("--n_dirs", type=int, default=4)
    ap.add_argument("--seed_train", type=int, default=12345)
    ap.add_argument("--seed_test",  type=int, default=98765)
    ap.add_argument("--Nx_solve", type=int, default=NX_SOLVE)
    ap.add_argument("--Nt_solve", type=int, default=NT_SOLVE)
    ap.add_argument("--Nx_out", type=int, default=NX_OUT)
    ap.add_argument("--Nt_out", type=int, default=NT_OUT)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--sigma_rbf", type=float, default=0.2)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="float64",
                    choices=["float32", "float64"])
    ap.add_argument("--save_suffix", type=str, default="_v2",
                    help="Suffix on output filenames so v2 doesn't overwrite v1")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    Nx, Nt = args.Nx_solve, args.Nt_solve
    dt = T_FINAL / Nt
    dx = 1.0 / (Nx + 1)
    A = build_A_torch(Nx, dx, dt, NU, device, dtype=dtype)

    print(f"Generator v2: autograd-through-implicit-Picard forward solver", flush=True)
    print(f"  device={device}, dtype={dtype}, Nx={Nx}, Nt={Nt}, dt={dt:.5f}, dx={dx:.5f}")
    print(f"  N_PICARD={N_PICARD}, NU={NU}, T={T_FINAL}", flush=True)

    # Load (f, u, c) data
    print(f"\nLoading {args.data_path} ...", flush=True)
    data = np.load(args.data_path)
    train_c = data["train_c"]
    test_c  = data["test_c"]
    N_train, N_test = train_c.shape[0], test_c.shape[0]
    print(f"  N_train={N_train}, N_test={N_test}, M={train_c.shape[1]}")

    # Sample directions on the solver grid
    V_train = sample_directions_solver_grid(args.n_dirs, Nt, args.seed_train)
    V_test  = sample_directions_solver_grid(args.n_dirs, Nt, args.seed_test)
    print(f"  Train dir seed={args.seed_train}; test dir seed={args.seed_test}; "
          f"|<v_train_d, v_test_d>|.max()={float(np.abs(V_train @ V_test.T).max()):.4f}",
          flush=True)

    # Project directions to FNO grid (for saving + evaluation later)
    delta_train_fno = np.zeros((args.n_dirs, args.Nx_out, args.Nt_out), dtype=np.float32)
    delta_test_fno  = np.zeros((args.n_dirs, args.Nx_out, args.Nt_out), dtype=np.float32)
    for d in range(args.n_dirs):
        v_t_train = project_dir_to_fno_grid(V_train[d], Nt, args.Nt_out, T_FINAL)
        v_t_test  = project_dir_to_fno_grid(V_test[d],  Nt, args.Nt_out, T_FINAL)
        delta_train_fno[d] = np.broadcast_to(v_t_train[None, :].astype(np.float32),
                                              (args.Nx_out, args.Nt_out))
        delta_test_fno[d]  = np.broadcast_to(v_t_test[None, :].astype(np.float32),
                                              (args.Nx_out, args.Nt_out))

    # Convert directions to torch tensors on device
    V_train_t = torch.tensor(V_train, dtype=dtype, device=device)
    V_test_t  = torch.tensor(V_test,  dtype=dtype, device=device)

    t_solve = np.arange(Nt) * dt

    def gen_jvps_for_split(c_split, V_dirs_t, label):
        """For each sample i in c_split, compute true J*v_d for d=0..n_dirs-1
        via torch.func.jvp through the implicit-Picard solver.
        Returns ndarray (N, n_dirs, Nx_out, Nt_out)."""
        N = c_split.shape[0]
        out = np.zeros((N, args.n_dirs, args.Nx_out, args.Nt_out), dtype=np.float32)
        t0 = time.time()
        for i in range(N):
            c_i = c_split[i].astype(np.float64)
            f_np = generate_rbf_control(c_i, args.M, T_FINAL, args.sigma_rbf, t_solve)
            f_t = torch.tensor(f_np, dtype=dtype, device=device)
            for d in range(args.n_dirs):
                v_t = V_dirs_t[d]
                # u, w = jvp(f -> U)(f, v) ; U is (Nx, Nt+1)
                def fwd(fff):
                    return forward_solve_implicit_torch(fff, A, Nx, Nt, dt, dx)
                _, W = func_jvp(fwd, (f_t,), (v_t,))   # (Nx, Nt+1)
                W_np = W.detach().cpu().numpy()
                # Interpolate to FNO grid
                w_fno = interpolate_to_fno_grid(W_np, Nx, Nt, args.Nx_out, args.Nt_out, T_FINAL)
                out[i, d] = w_fno.astype(np.float32)
            if (i + 1) % 100 == 0 or (i + 1) <= 3:
                el = time.time() - t0
                eta = el / (i + 1) * (N - i - 1)
                mw = float(np.abs(out[i]).max())
                print(f"  [{label}] [{i+1:4d}/{N}]  max|w|={mw:.4f}  "
                      f"t={el:.0f}s  eta={eta:.0f}s", flush=True)
        return out

    print(f"\n=== Generating TRAIN jvps ({N_train} samples × {args.n_dirs} dirs) ===")
    train_jvps = gen_jvps_for_split(train_c, V_train_t, "train")
    print(f"\n=== Generating TEST jvps ({N_test} samples × {args.n_dirs} dirs) ===")
    test_jvps  = gen_jvps_for_split(test_c,  V_test_t,  "test ")

    # Save with explicit suffix so v1 files are preserved
    os.makedirs(args.save_dir, exist_ok=True)
    p_train = os.path.join(args.save_dir, f"train_jvps{args.save_suffix}.npz")
    p_test  = os.path.join(args.save_dir, f"test_jvps{args.save_suffix}.npz")
    np.savez_compressed(p_train, jvps=train_jvps, delta_dirs=delta_train_fno,
                        seed=args.seed_train, source="autograd_through_implicit_picard_v2")
    np.savez_compressed(p_test,  jvps=test_jvps,  delta_dirs=delta_test_fno,
                        seed=args.seed_test,  source="autograd_through_implicit_picard_v2")
    print(f"\nSaved: {p_train}  ({os.path.getsize(p_train)/1e6:.1f} MB)")
    print(f"       {p_test}   ({os.path.getsize(p_test)/1e6:.1f} MB)")
    print(f"  jvps shapes: train={train_jvps.shape}, test={test_jvps.shape}")
    print("Done.")


if __name__ == "__main__":
    main()
