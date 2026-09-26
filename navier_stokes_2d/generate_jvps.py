#!/usr/bin/env python
"""DIFNO / evaluation JVP labels for the control-to-state map U -> Y.

Aligned with the other benchmarks (Burgers / Allen-Cahn / nonlinear diffusion):

  * Directions are sampled in the CONTROL (input) space with the SAME smooth
    zero-boundary prior that sTCL uses online -- we call `stcl_navier_stokes_2d.sample_directions`
    directly, so DIFNO/eval and sTCL draw from an identical direction distribution.
  * A shared bank of `r` directions delta_u is drawn once per split.
  * For each sample, the velocity tangent delta_y = J(U_i) delta_u is obtained by
    solving the linearized (Newton/Oseen) incompressible state equation:

        -mu Delta w + (Y_i.grad)w + (w.grad)Y_i + grad(pi) = delta_u,
        div w = 0,
        w = 0 on the boundary.

    The saddle system is LU-factored once per sample (at the stored state Y_i)
    and reused across all directions.  Collocated centered div/grad has a pressure
    checkerboard null mode; a Brezzi-Pitkaranta term -alpha*hx*hy*Lap(pi) stabilizes
    it without contaminating the velocity tangent w.

Only the divergence-free velocity tangent w is stored as the label (`jvps`); the
shared directions are stored once as `delta_dirs`.  sTCL does NOT read this file --
it samples delta_u online during training.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from . import navier_stokes_2d as A
    from .stcl_navier_stokes_2d import draw_directions
except ImportError:  # Support direct script execution.
    import navier_stokes_2d as A
    from stcl_navier_stokes_2d import draw_directions


def fd_ops(gy: int, gx: int):
    """Interior (homogeneous-Dirichlet) FD operators: Laplacian and centered d/dx, d/dy."""
    hx = 1.0 / (gx - 1)
    hy = 1.0 / (gy - 1)
    nx, ny = gx - 2, gy - 2
    Ix, Iy = sp.identity(nx), sp.identity(ny)
    Dxx = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(nx, nx)) / hx ** 2
    Dyy = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(ny, ny)) / hy ** 2
    lap = sp.kron(Iy, Dxx) + sp.kron(Dyy, Ix)
    Dx1 = sp.diags([-1.0, 1.0], [-1, 1], shape=(nx, nx)) / (2 * hx)
    Dy1 = sp.diags([-1.0, 1.0], [-1, 1], shape=(ny, ny)) / (2 * hy)
    Dx = sp.kron(Iy, Dx1)
    Dy = sp.kron(Dy1, Ix)
    return lap.tocsr(), Dx.tocsr(), Dy.tocsr(), hx, hy


def factor_tangent(state, lap, Dx, Dy, hx, hy, mu, alpha):
    """LU-factor the Newton-linearized saddle system at velocity `state` (2,gy,gx)."""
    Y1 = state[0, 1:-1, 1:-1].ravel()
    Y2 = state[1, 1:-1, 1:-1].ravel()
    conv = sp.diags(Y1) @ Dx + sp.diags(Y2) @ Dy          # (Y.grad)(.)
    A11 = -mu * lap + conv + sp.diags(Dx @ Y1)            # + dY1/dx
    A12 = sp.diags(Dy @ Y1)                               # dY1/dy  (couples w2 into row 1)
    A21 = sp.diags(Dx @ Y2)                               # dY2/dx  (couples w1 into row 2)
    A22 = -mu * lap + conv + sp.diags(Dy @ Y2)            # + dY2/dy
    S = -alpha * hx * hy * lap                            # Brezzi-Pitkaranta pressure stab
    K = sp.bmat([[A11, A12, Dx],
                 [A21, A22, Dy],
                 [Dx,  Dy,  S]], format="csc")
    return spla.splu(K)


def solve_tangent(lu, delta_u, gy, gx):
    """Solve the saddle system for direction delta_u (2,gy,gx) -> velocity tangent (2,gy,gx)."""
    n = (gy - 2) * (gx - 2)
    rhs = np.concatenate([delta_u[0, 1:-1, 1:-1].ravel(),
                          delta_u[1, 1:-1, 1:-1].ravel(),
                          np.zeros(n)])
    sol = lu.solve(rhs)
    w = np.zeros((2, gy, gx), dtype=np.float64)
    w[0, 1:-1, 1:-1] = sol[:n].reshape(gy - 2, gx - 2)
    w[1, 1:-1, 1:-1] = sol[n:2 * n].reshape(gy - 2, gx - 2)
    return w


def solve_tangent_bank(lu, delta_dirs, gy, gx, *, return_pressure=False):
    """Solve all shared directions in one SuperLU multi-RHS call.

    This is mathematically identical to repeated ``solve_tangent`` calls but is
    substantially faster for the 200-direction DIFNO training bank.
    """
    r = delta_dirs.shape[0]
    n = (gy - 2) * (gx - 2)
    rhs = np.zeros((3 * n, r), dtype=np.float64)
    rhs[:n] = delta_dirs[:, 0, 1:-1, 1:-1].reshape(r, n).T
    rhs[n : 2 * n] = delta_dirs[:, 1, 1:-1, 1:-1].reshape(r, n).T
    sol = lu.solve(rhs)
    w = np.zeros((r, 2, gy, gx), dtype=np.float64)
    w[:, 0, 1:-1, 1:-1] = sol[:n].T.reshape(r, gy - 2, gx - 2)
    w[:, 1, 1:-1, 1:-1] = sol[n : 2 * n].T.reshape(r, gy - 2, gx - 2)
    if not return_pressure:
        return w
    p = np.zeros((r, gy, gx), dtype=np.float64)
    p[:, 1:-1, 1:-1] = sol[2 * n :].T.reshape(r, gy - 2, gx - 2)
    return w, p


def split_seed(name: str, base: int) -> int:
    return base + {"train": 0, "val": 100_000, "test": 200_000}.get(name, 300_000)


def _residual_checks(state, delta_u, dy, mu, hx, hy):
    """Curl-form momentum residual (pressure eliminated) and relative divergence of dy."""
    y = torch.tensor(state)[None]
    du = torch.tensor(delta_u)[None]
    w = torch.tensor(dy)[None]
    M = -mu * A.lap(w, hx, hy) + A.convect(y, w, hx, hy) + A.convect(w, y, hx, hy)
    r = torch.zeros_like(du); r[:, :, 1:-1, 1:-1] = M
    r = r - du
    curl = lambda F: ((F[:, 1, 1:-1, 2:] - F[:, 1, 1:-1, :-2]) / (2 * hx)
                      - (F[:, 0, 2:, 1:-1] - F[:, 0, :-2, 1:-1]) / (2 * hy))
    mom = float(curl(r).norm() / (curl(du).norm() + 1e-30))
    dv = A.divergence(w, hx, hy)
    g = torch.stack([A.ddx(w[:, 0], hx), A.ddy(w[:, 0], hy),
                     A.ddx(w[:, 1], hx), A.ddy(w[:, 1], hy)])
    return mom, float(dv.norm() / (g.norm() + 1e-30))


def generate_split(npz_path, out_path, split, n_dirs, seed, n_modes, alpha,
                   sample_start=0, sample_stop=None):
    z = np.load(npz_path)
    all_state = z["Y"]
    source_n_samples = len(all_state)
    sample_stop = source_n_samples if sample_stop is None else sample_stop
    if not (0 <= sample_start < sample_stop <= source_n_samples):
        raise ValueError(
            f"invalid sample range [{sample_start}, {sample_stop}) for {source_n_samples} samples"
        )
    state = all_state[sample_start:sample_stop].astype(np.float64)
    gy, gx = int(z["gy"]), int(z["gx"])
    mu = float(z["mu_fixed"]) if "mu_fixed" in z.files else 1.0
    if alpha is None:
        alpha = float(z["alpha"]) if "alpha" in z.files else 1e-3
    # Direction prior: read from the dataset so DIFNO/eval match the input distribution
    # (sine = generic control-space; solenoidal = curl-of-stream, matched to a forward-solve U).
    mode = str(z["direction_mode"]) if "direction_mode" in z.files else "sine"
    skmax = int(z["stream_kmax"]) if "stream_kmax" in z.files else 6
    sdecay = float(z["stream_decay"]) if "stream_decay" in z.files else 1.5
    lap, Dx, Dy, hx, hy = fd_ops(gy, gx)

    # Shared direction bank -- the SAME sampler sTCL uses online.
    torch.manual_seed(split_seed(split, seed))
    delta_dirs = draw_directions(torch.zeros(n_dirs, 2, gy, gx), mode, n_modes,
                                 skmax, sdecay).numpy().astype(np.float64)

    N = len(state)
    jvps = np.zeros((N, n_dirs, 2, gy, gx), dtype=np.float32)
    mom_res, div_res = [], []
    for i in range(N):
        lu = factor_tangent(state[i], lap, Dx, Dy, hx, hy, mu, alpha)
        bank = solve_tangent_bank(lu, delta_dirs, gy, gx)
        jvps[i] = bank.astype(np.float32)
        for d in range(min(n_dirs, 16)):
            if i < 8:
                m, dv = _residual_checks(state[i], delta_dirs[d], bank[d], mu, hx, hy)
                mom_res.append(m); div_res.append(dv)
        if (i + 1) % max(1, N // 5) == 0:
            print(f"  [{split}] {i+1}/{N}", flush=True)

    if not np.isfinite(jvps).all():
        raise RuntimeError(f"{split}: non-finite tangent labels generated")
    np.savez(
        out_path,
        delta_dirs=delta_dirs.astype(np.float32),
        jvps=jvps,
        gy=np.int64(gy), gx=np.int64(gx),
        seed=np.int64(split_seed(split, seed)),
        n_dirs=np.int64(n_dirs), n_modes=np.int64(n_modes), alpha=np.float64(alpha),
        source_split_seed=np.int64(z["split_seed"]) if "split_seed" in z.files else np.int64(-1),
        source_n_samples=np.int64(source_n_samples),
        sample_start=np.int64(sample_start), sample_stop=np.int64(sample_stop),
        direction_mode=np.str_(mode),
    )
    print(f"  saved {out_path}: delta_dirs{delta_dirs.shape} jvps{jvps.shape} "
          f"tangent_curl_res={np.mean(mom_res):.2e} rel_div={np.mean(div_res)*100:.3f}%", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(SCRIPT_DIR, "data"))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--n_dirs", type=int, default=None,
                    help="Override direction count for every requested split; paper defaults are train=200, val=test=16")
    ap.add_argument("--n_modes", type=int, default=8, help="sine-prior modes (must match sTCL --sketch_modes)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--alpha", type=float, default=None,
                    help="Brezzi-Pitkaranta pressure stabilization; default reads dataset metadata")
    ap.add_argument("--start", type=int, default=0, help="first sample index for sharded generation")
    ap.add_argument("--stop", type=int, default=None, help="exclusive sample stop for sharded generation")
    ap.add_argument("--output", default=None, help="explicit output path; requires exactly one split")
    args = ap.parse_args()

    if args.output is not None and len(args.splits) != 1:
        raise ValueError("--output requires exactly one --splits entry")

    for split in args.splits:
        src = os.path.join(args.data_dir, f"{split}.npz")
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        out = args.output or os.path.join(args.data_dir, f"{split}_jvps.npz")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        n_dirs = args.n_dirs if args.n_dirs is not None else (200 if split == "train" else 16)
        generate_split(src, out, split, n_dirs, args.seed, args.n_modes, args.alpha,
                       sample_start=args.start, sample_stop=args.stop)


if __name__ == "__main__":
    main()
