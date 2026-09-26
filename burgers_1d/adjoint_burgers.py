#!/usr/bin/env python
"""
Adjoint Method Ground Truth for 1D Burgers' OCP
================================================
Computes J* = best achievable J via classical Direct-Adjoint Looping (DAL)
on the exact Burgers' PDE (no surrogate).

Problem
-------
  min_{f(t)}  J = mean_{x,t} (u(x,t) - u_hat(x,t))^2

  s.t.  u_t + u*u_x = nu * u_xx + f(t),   x in [0,1], t in [0,1]
        u(0,t) = u(1,t) = 0  (Dirichlet BC)
        u(x,0) = 0           (zero IC)

  u_hat(x,t) = sin(pi*x) * sin(pi*t)
  nu = 0.01

  f(t) is space-uniform.

Discrete scheme (Fully Implicit Picard)
----------------------------------------
At each time step, solve by Picard iteration:
  A * u^{n+1,k+1} = u^n + dt*f^n*1 - dt*N(u^{n+1,k})
  N_i(u) = u_i * (u_{i+1} - u_{i-1}) / (2*dx)   (centered convection)

At convergence: A*u^{n+1} + dt*N(u^{n+1}) = u^n + dt*f^n*1  [implicit].
A = I - dt*nu*L  (L = 1-D Laplacian, implicit diffusion).

Fully Implicit Discrete Adjoint
---------------------------------
B_n = A + dt * dN/du^{n+1} (Newton Jacobian evaluated at converged u^{n+1}).

Adjoint backward recursion:
  B_{m-1}^T * p^m = scale*(u^m - u_hat^m) + p^{m+1},  m = Nt, ..., 1
  p^{Nt+1} = 0

(dN/du^m)^T : diagonal = (u_{j+1}-u_{j-1})/(2dx),
               super    = -u_{j+1}/(2dx),
               sub      =  u_{j-1}/(2dx)

Gradient:
  dJ/df^n = dt * sum_x p^{n+1}

Output
------
  results_adjoint/
    metrics.json
    f_optimal.npy
    history.csv
    plots/convergence.png, optimal_control.png, state_comparison.png
"""

import os
import json
import csv
import math
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import minimize

# ── configuration ─────────────────────────────────────────────────────────────
NU      = 0.01
T_FINAL = 1.0
NX      = 128
NT      = 200
MAX_ITER = 1000
N_PICARD = 3

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_adjoint")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")


# ── helpers ──────────────────────────────────────────────────────────────────

def build_laplacian_1d(Nx):
    """
    1-D FD Laplacian (Dirichlet BC) with NEGATIVE sign convention:
    L_{ii} = -2/dx^2,  L_{i,i±1} = 1/dx^2.
    Matches run_adjoint.py (heat 2D): A = I - dt*nu*L is positive definite
    since L has negative eigenvalues.
    """
    dx = 1.0 / (Nx + 1)
    d  = -2.0 * np.ones(Nx)    / dx**2   # negative diagonal
    o  =  1.0 * np.ones(Nx-1) / dx**2   # positive off-diagonal
    return sp.diags([o, d, o], [-1, 0, 1], (Nx, Nx), format="csr")


def build_target(Nx, Nt):
    dx = 1.0 / (Nx + 1)
    xs = np.arange(1, Nx + 1) * dx
    ts = np.arange(Nt + 1) * (T_FINAL / Nt)
    return (np.sin(math.pi * xs)[:, None]
            * np.sin(math.pi * ts)[None, :])


def convection_centered(u, dx):
    u_ext = np.concatenate([[0.0], u, [0.0]])
    return u * (u_ext[2:] - u_ext[:-2]) / (2.0 * dx)


def build_BT(u_m, A_sp, dt, dx):
    """
    B^T = (A + dt*dN/du^m)^T = A + dt*(dN/du^m)^T.

    (dN/du^m)^T elements:
      diagonal[j]       = (u_{j+1} - u_{j-1}) / (2*dx)
      superdiagonal[j]  = -u_{j+1} / (2*dx)   at position (j, j+1)
      subdiagonal[j]    =  u_{j-1} / (2*dx)   at position (j, j-1)
    with u_{-1} = u_N = 0.
    """
    Nx    = len(u_m)
    u_ext = np.concatenate([[0.0], u_m, [0.0]])
    diag  = (u_ext[2:] - u_ext[:-2]) / (2.0 * dx)
    sup   = -u_m[1:]  / (2.0 * dx)
    sub   =  u_m[:-1] / (2.0 * dx)
    dN_T  = sp.diags([sub, diag, sup], [-1, 0, 1], (Nx, Nx), format="csc")
    return A_sp.tocsc() + dt * dN_T


def forward_solve_implicit(A_sp, f, Nx, Nt, dt, dx):
    A_csc = A_sp.tocsc()
    ones  = np.ones(Nx)
    U     = np.zeros((Nx, Nt + 1))
    for n in range(Nt):
        u_k  = U[:, n].copy()
        rhs0 = U[:, n] + dt * f[n] * ones
        for _ in range(N_PICARD):
            u_k = spla.spsolve(A_csc, rhs0 - dt * convection_centered(u_k, dx))
        U[:, n + 1] = u_k
    return U


def adjoint_solve_implicit(U, U_hat, Nx, Nt, scale, dt, dx, A_sp):
    """
    B_{m-1}^T * p^m = scale*(u^m - u_hat^m) + p^{m+1},  p^{Nt+1}=0.
    """
    p = np.zeros((Nx, Nt + 1))
    B_T       = build_BT(U[:, Nt], A_sp, dt, dx)
    p[:, Nt]  = spla.spsolve(B_T, scale * (U[:, Nt] - U_hat[:, Nt]))
    for m in range(Nt - 1, 0, -1):
        B_T    = build_BT(U[:, m], A_sp, dt, dx)
        rhs    = scale * (U[:, m] - U_hat[:, m]) + p[:, m + 1]
        p[:, m] = spla.spsolve(B_T, rhs)
    return p


def compute_J_and_grad(f, A_sp, U_hat, Nx, Nt, dt, dx):
    scale = 2.0 / float(Nx * (Nt + 1))
    U     = forward_solve_implicit(A_sp, f, Nx, Nt, dt, dx)
    if not np.isfinite(U).all():
        return np.inf, np.zeros_like(f)
    J     = float(((U - U_hat) ** 2).mean())
    p     = adjoint_solve_implicit(U, U_hat, Nx, Nt, scale, dt, dx, A_sp)
    grad  = dt * p[:, 1:].sum(axis=0)
    return J, grad


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    dt = T_FINAL / NT
    dx = 1.0 / (NX + 1)

    print("=" * 60)
    print("  Adjoint GT — 1D Burgers' OCP  (fully implicit Picard)")
    print("=" * 60)
    print(f"  Nx={NX}  Nt={NT}  dt={dt:.5f}  dx={dx:.5f}  nu={NU}")

    L    = build_laplacian_1d(NX)
    A_sp = sp.eye(NX, format="csc") - dt * NU * L.tocsc()

    U_hat = build_target(NX, NT)

    # Warm start: spatial mean of linearised Burgers' exact forcing
    # f_lin(t) = 2*cos(pi*t) + 2*nu*pi*sin(pi*t)
    ts = np.arange(NT) * dt
    f0 = 2.0 * np.cos(math.pi * ts) + 2.0 * NU * math.pi * np.sin(math.pi * ts)

    J_zero, _ = compute_J_and_grad(np.zeros(NT), A_sp, U_hat, NX, NT, dt, dx)
    J0, _     = compute_J_and_grad(f0,            A_sp, U_hat, NX, NT, dt, dx)
    print(f"\n  J at f=0:        {J_zero:.6e}")
    print(f"  J at warm-start: {J0:.6e}")

    f_start = f0 if (np.isfinite(J0) and J0 < J_zero) else np.zeros(NT)
    print(f"  Using start: {'warm-start' if np.array_equal(f_start, f0) else 'f=0'}\n")

    history = []
    t_start = time.time()

    def objective(f):
        J, g = compute_J_and_grad(f, A_sp, U_hat, NX, NT, dt, dx)
        wall = time.time() - t_start
        it   = len(history)
        history.append({"iter": it, "J": float(J),
                         "grad_norm": float(np.linalg.norm(g)), "wall_s": wall})
        if it % 20 == 0 or it < 3:
            print(f"  [{it:4d}]  J={J:.6e}  |g|={np.linalg.norm(g):.4e}  "
                  f"t={wall:.1f}s", flush=True)
        return float(J), g

    print("Running L-BFGS-B ...", flush=True)
    result = minimize(objective, f_start, jac=True, method="L-BFGS-B",
                      options={"maxiter": MAX_ITER, "ftol": 1e-15, "gtol": 1e-10})

    f_opt      = result.x
    J_opt      = result.fun
    wall_total = time.time() - t_start
    best_J     = min((h["J"] for h in history if np.isfinite(h["J"])),
                     default=float(J_opt))

    print()
    print(f"  L-BFGS-B: {result.message}")
    print(f"  Iterations: {result.nit}")
    print(f"  Final J:    {J_opt:.6e}")
    print(f"  Best J:     {best_J:.6e}")
    print(f"  Wall time:  {wall_total:.1f}s")

    # ── save ─────────────────────────────────────────────────────────────────
    np.save(os.path.join(RESULTS_DIR, "f_optimal.npy"), f_opt)

    with open(os.path.join(RESULTS_DIR, "history.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=history[0].keys())
        w.writeheader(); w.writerows(history)

    metrics = {
        "method":      "adjoint_DAL_fully_implicit_LBFGSB",
        "Nx": NX, "Nt": NT, "nu": NU, "n_picard": N_PICARD,
        "n_iter":      result.nit,
        "J_at_zero":   float(J_zero),
        "J_warmstart": float(J0) if np.isfinite(J0) else None,
        "final_J":     float(J_opt),
        "best_J":      float(best_J),
        "wall_s":      float(wall_total),
        "converged":   bool(result.success),
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nSaved → {RESULTS_DIR}/metrics.json")

    # ── plots ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        good = [(h["iter"], h["J"]) for h in history if np.isfinite(h["J"])]
        if good:
            iters_g, Js_g = zip(*good)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.semilogy(iters_g, Js_g, "b-", lw=2)
            ax.axhline(best_J, color="r", ls="--", lw=1,
                       label=f"J* = {best_J:.4e}")
            ax.set_xlabel("L-BFGS-B Iteration", fontsize=12)
            ax.set_ylabel("Objective J (log scale)", fontsize=12)
            ax.set_title(f"Adjoint GT — 1D Burgers'\n(Nx={NX}, Nt={NT}, nu={NU})",
                         fontsize=10)
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, "convergence.png"),
                        dpi=150, bbox_inches="tight"); plt.close()

        ts_plot = np.linspace(0, T_FINAL, NT, endpoint=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ts_plot, f_start, "g--", lw=1.5, label="Start f(t)", alpha=0.7)
        ax.plot(ts_plot, f_opt,   "r-",  lw=2,
                label=f"Optimal f*(t)  [J*={best_J:.4e}]")
        ax.set_xlabel("t"); ax.set_ylabel("f(t)")
        ax.set_title("Optimal Control — 1D Burgers' (Adjoint GT)")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "optimal_control.png"),
                    dpi=150, bbox_inches="tight"); plt.close()

        U_opt   = forward_solve_implicit(A_sp, f_opt, NX, NT, dt, dx)
        ts_grid = np.arange(NT + 1) * dt
        xs_grid = np.arange(1, NX + 1) * dx
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax_i, (Z, ttl) in zip(axes,
                [(U_hat, "u_hat (target)"), (U_opt, "u_opt (adjoint)")]):
            im = ax_i.contourf(ts_grid, xs_grid, Z, 20, cmap="RdBu_r")
            plt.colorbar(im, ax=ax_i)
            ax_i.set_xlabel("t"); ax_i.set_ylabel("x"); ax_i.set_title(ttl)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "state_comparison.png"),
                    dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved plots → {PLOTS_DIR}")

    except ImportError:
        print("matplotlib not available, skipping plots")

    print("\nDone.")
    return best_J


if __name__ == "__main__":
    main()
