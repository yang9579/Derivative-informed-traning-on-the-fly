#!/usr/bin/env python
"""
Generate train_jvps_r{q}.npz for κ=5 Helmholtz Dirichlet setup at any N.
Parallel across samples; reuses LU factorisation for the q directions.

Usage:
    python generate_train_jvps.py --n_train 2048 --q 4 --n_workers 24
"""
import os, sys, time, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import scipy.sparse.linalg as spla

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from generate_data_difno import (
    sample_gp_matern, build_fd_laplacian_dirichlet, solve_tangent,
    NX, NY, KAPPA,
)


def _solve_one(args):
    """Reuses LU factorisation across the q tangent directions."""
    i, a, u, deltas, nx, ny = args
    L2d = build_fd_laplacian_dirichlet(nx, ny)
    K = deltas.shape[0]
    out = np.zeros((K, nx, ny), dtype=np.float32)
    # solve_tangent in the existing module accepts an optional A_lu cache;
    # build it once for all K directions on this sample
    from generate_data_difno import build_helmholtz_operator
    A = build_helmholtz_operator(a, nx, ny, L2d)
    A_lu = spla.splu(A.tocsc())
    for k in range(K):
        out[k] = solve_tangent(a, u, deltas[k], nx, ny, L2d, A_lu=A_lu)
    return i, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(SCRIPT_DIR, "data"))
    ap.add_argument("--n_train", type=int, default=2048)
    ap.add_argument("--q", "--n_dirs", dest="q", type=int, default=4)
    ap.add_argument("--n_workers", type=int, default=24)
    ap.add_argument("--seed",    type=int, default=10007)
    args = ap.parse_args()

    print(f"Loading train.npz from {args.data_dir} ...", flush=True)
    tr = np.load(os.path.join(args.data_dir, "train.npz"))
    a_tr = tr["a"][:args.n_train]                                # (N, nx, ny)
    u_tr = tr["u"][:args.n_train]                                # (N, nx, ny)
    N = a_tr.shape[0]
    nx, ny = a_tr.shape[1], a_tr.shape[2]
    print(f"  N={N}, grid {nx}×{ny}, q={args.q}")

    print(f"Sampling {args.q} GP directions ...", flush=True)
    deltas = sample_gp_matern(args.q, nx, ny, n_modes=40, seed=args.seed)
    for k in range(args.q):
        deltas[k] /= np.linalg.norm(deltas[k]) + 1e-12

    args_list = [(i, a_tr[i], u_tr[i], deltas, nx, ny) for i in range(N)]
    out = np.zeros((N, args.q, nx, ny), dtype=np.float32)

    print(f"Generating train JVPs on {args.n_workers} workers ...", flush=True)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futures = [ex.submit(_solve_one, a) for a in args_list]
        for fut in as_completed(futures):
            i, jvps = fut.result()
            out[i] = jvps
            done += 1
            if done % max(1, N // 20) == 0:
                eta = (time.time() - t0) * (N - done) / done
                print(f"  [{done:5d}/{N}]  elapsed={time.time()-t0:.0f}s  eta={eta:.0f}s",
                      flush=True)

    out_path = os.path.join(args.data_dir, f"train_jvps_r{args.q}.npz")
    print(f"\nSaving to {out_path} ...", flush=True)
    np.savez_compressed(out_path,
        jvps=out, delta_dirs=deltas,
        offline_seconds=np.array(time.time() - t0), n_dirs=np.array(args.q))
    print(f"  total wall = {time.time()-t0:.0f}s,  shape = {out.shape}")


if __name__ == "__main__":
    main()
