#!/usr/bin/env python
"""Forward-solve data generation for the control-to-state map U -> (Y, P).

This samples the control first and solves the PDE forward, matching the convention
used by the other clean-code cases. Directions drawn from the U prior are therefore
on the input distribution.

  1. Sample a solenoidal, zero-boundary body force  U = A * curl(psi(b)),
     psi(b) = sum b_{kl} sin^2(k pi x1) sin^2(l pi x2)  (via stream_velocity).
     A body force's curl-free part is absorbed by the pressure and does not drive
     the flow, so sampling U solenoidal keeps U -> Y well-posed (no Jacobian null
     space) and physically meaningful.
  2. Solve steady NS forward (Newton, ns_forward.solve_steady_ns) for (Y, P).

The saved arrays contain U, Y, P, mu, gy, gx, direction_mode='solenoidal', and
the stream-prior metadata (stream_kmax, stream_decay), so generate_jvps.py and
train_navier_stokes_2d.py can consume them directly.
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from .navier_stokes_2d import FORCE_ZERO, U_NAMES, Y_NAMES, stream_velocity, make_grid
    from .ns_forward import solve_steady_ns
except ImportError:  # Support direct script execution.
    from navier_stokes_2d import FORCE_ZERO, U_NAMES, Y_NAMES, stream_velocity, make_grid
    from ns_forward import solve_steady_ns

_MU = 0.1
_ALPHA = 1e-3


def stream_scale(kmax, decay):
    k = np.arange(1, kmax + 1)
    kk, ll = np.meshgrid(k, k, indexing="ij")
    return (kk ** 2 + ll ** 2).astype(np.float64) ** (-decay)


def sample_controls(n, gy, gx, kmax, decay, amp_min, amp_max, seed):
    """Sample n solenoidal zero-boundary forces U (n, 2, gy, gx), per-sample |U|inf in [amp_min,amp_max]."""
    rng = np.random.RandomState(seed)
    x, y = make_grid(gy, gx)
    scale = stream_scale(kmax, decay)[None]
    b = torch.tensor((rng.randn(n, kmax, kmax) * scale).astype(np.float32))
    U = stream_velocity(b, x, y).numpy().astype(np.float64)            # (n,2,gy,gx) solenoidal
    amp = rng.uniform(amp_min, amp_max, size=n)
    U = U / (np.abs(U).reshape(n, -1).max(axis=1).reshape(n, 1, 1, 1) + 1e-12) * amp.reshape(n, 1, 1, 1)
    return U.astype(np.float32)


def _solve_one(U):
    Y, P, info = solve_steady_ns(U.astype(np.float64), _MU, alpha=_ALPHA, tol=1e-11, maxit=40)
    return Y.astype(np.float32), P.astype(np.float32), info["iters"], info["res"]


def generate_split(name, n, args, seed):
    U = sample_controls(n, args.gy, args.gx, args.stream_kmax, args.stream_decay,
                        args.vel_amp_min, args.vel_amp_max, seed)
    with Pool(args.workers) as pool:
        out = pool.map(_solve_one, list(U), chunksize=4)
    Y = np.stack([o[0] for o in out]); P = np.stack([o[1] for o in out])
    its = np.array([o[2] for o in out]); res = np.array([o[3] for o in out])
    if not (np.isfinite(U).all() and np.isfinite(Y).all() and np.isfinite(P).all() and np.isfinite(res).all()):
        raise RuntimeError(f"{name}: non-finite values found in generated data")
    if res.max() > args.max_accepted_residual:
        bad = np.flatnonzero(res > args.max_accepted_residual)
        raise RuntimeError(
            f"{name}: {len(bad)} Newton solves exceed --max_accepted_residual="
            f"{args.max_accepted_residual:.1e}; worst={res.max():.3e}, first_bad={bad[:10].tolist()}"
        )
    path = os.path.join(args.data_dir, f"{name}.npz")
    np.savez(path, U=U, Y=Y, P=P,
             F=np.zeros_like(U, dtype=np.float32),
             mu=np.full((n, 1), _MU, dtype=np.float32), mu_fixed=np.float32(_MU),
             alpha=np.float64(_ALPHA),
             gy=np.int64(args.gy), gx=np.int64(args.gx), split_seed=np.int64(seed),
             stream_kmax=np.int64(args.stream_kmax), stream_decay=np.float32(args.stream_decay),
             vel_amp_min=np.float32(args.vel_amp_min), vel_amp_max=np.float32(args.vel_amp_max),
             force_zero=np.bool_(FORCE_ZERO),
             input_names=np.array(U_NAMES), output_names=np.array(Y_NAMES),
             newton_iters=its.astype(np.int16), newton_rel_res=res.astype(np.float64),
             direction_mode=np.str_("solenoidal"))
    print(f"  saved {path}: U{U.shape} Y{Y.shape} P{P.shape} | "
          f"newton_iters~{its.mean():.1f} max_res={res.max():.1e} "
          f"|Y|max={np.abs(Y).max():.2f} Re~{np.abs(Y).max()/_MU:.0f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=2048)
    ap.add_argument("--n_val", type=int, default=128)
    ap.add_argument("--n_test", type=int, default=128)
    ap.add_argument("--gy", type=int, default=64)
    ap.add_argument("--gx", type=int, default=64)
    ap.add_argument("--mu", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=1e-3,
                    help="Brezzi-Pitkaranta pressure stabilization")
    ap.add_argument("--stream_kmax", type=int, default=6)
    ap.add_argument("--stream_decay", type=float, default=1.5)
    ap.add_argument("--vel_amp_min", type=float, default=16.0, help="per-sample |U|inf lower (Re~vel_amp/mu*0.14)")
    ap.add_argument("--vel_amp_max", type=float, default=24.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_accepted_residual", type=float, default=1e-9,
                    help="abort instead of saving if any Newton relative residual exceeds this")
    ap.add_argument("--data_dir", default=os.path.join(SCRIPT_DIR, "data"))
    args = ap.parse_args()
    global _MU, _ALPHA
    _MU = args.mu
    _ALPHA = args.alpha
    os.makedirs(args.data_dir, exist_ok=True)
    print(f"Forward-solve data: grid={args.gy}x{args.gx} mu={_MU} alpha={_ALPHA} solenoidal U | "
          f"stream_kmax={args.stream_kmax} |U|inf in [{args.vel_amp_min},{args.vel_amp_max}]", flush=True)
    for name, n, seed in [("train", args.n_train, 1), ("val", args.n_val, 2), ("test", args.n_test, 3)]:
        generate_split(name, n, args, seed)


if __name__ == "__main__":
    main()
