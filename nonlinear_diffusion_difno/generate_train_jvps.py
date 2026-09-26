#!/usr/bin/env python
"""
Generate JVP (derivative) training data for DIFNO-style training.
================================================================
Loads existing (a, u) training pairs and computes true JVPs
by solving the tangent equation for a shared set of random directions.

For large n_dirs (e.g., 289 matching DIFNO paper), data is saved in chunks
to avoid memory issues (~20GB for 4096 samples x 289 dirs x 65x65).

Usage:
  python generate_train_jvps.py --n_dirs 289
  python generate_train_jvps.py --n_dirs 10   # quick test
"""

import os
import sys
import time
import argparse

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

from generate_data_difno import (
    sample_gp_matern, build_fd_operators_dirichlet,
    solve_tangent_equation, NX, NY
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=DATA_DIR)
    ap.add_argument("--n_dirs",   type=int, default=289,
                    help="Number of JVP directions (DIFNO paper: 289)")
    ap.add_argument("--nx",       type=int, default=NX)
    ap.add_argument("--ny",       type=int, default=NY)
    ap.add_argument("--chunk_size", type=int, default=256,
                    help="Samples per chunk (for memory management)")
    args = ap.parse_args()

    # Load existing training data
    train_path = os.path.join(args.data_dir, "train.npz")
    print(f"Loading {train_path} ...", flush=True)
    train_data = np.load(train_path)
    a_train = train_data["a"]
    u_train = train_data["u"]
    n_train = len(a_train)
    print(f"  {n_train} samples, shape a={a_train.shape} u={u_train.shape}")

    # Build FD operators
    print(f"Building FD operators (nx={args.nx}, ny={args.ny}) ...", flush=True)
    L2d, Dx, Dy = build_fd_operators_dirichlet(args.nx, args.ny)

    # Sample random perturbation directions from GP prior (mean=0)
    print(f"Sampling {args.n_dirs} JVP directions from GP prior ...", flush=True)
    delta_dirs = sample_gp_matern(
        args.n_dirs, args.nx, args.ny, n_modes=40,
        mean=0.0, seed=777777)
    for d in range(args.n_dirs):
        delta_dirs[d] /= np.linalg.norm(delta_dirs[d]) + 1e-10

    # Save directions
    dirs_path = os.path.join(args.data_dir, "train_jvp_dirs.npz")
    np.savez_compressed(dirs_path, delta_dirs=delta_dirs)
    print(f"  Saved directions: {dirs_path}")

    # Estimate sizes
    total_solves = n_train * args.n_dirs
    mem_gb = n_train * args.n_dirs * args.nx * args.ny * 4 / 1e9
    print(f"\n  Total tangent solves: {total_solves:,}")
    print(f"  Estimated output: {mem_gb:.1f} GB")

    # Compute JVPs in chunks and save each chunk separately
    jvp_dir = os.path.join(args.data_dir, "train_jvps_chunks")
    os.makedirs(jvp_dir, exist_ok=True)

    t0 = time.time()
    chunk_id = 0

    for start in range(0, n_train, args.chunk_size):
        end = min(start + args.chunk_size, n_train)
        chunk_n = end - start

        # Check if chunk already exists (for resumability)
        chunk_path = os.path.join(jvp_dir, f"chunk_{chunk_id:04d}.npz")
        if os.path.exists(chunk_path):
            print(f"  [SKIP] Chunk {chunk_id} ({start}-{end}) already exists",
                  flush=True)
            chunk_id += 1
            continue

        jvp_chunk = np.zeros((chunk_n, args.n_dirs, args.nx, args.ny),
                              dtype=np.float32)

        for i_local in range(chunk_n):
            i_global = start + i_local
            for d in range(args.n_dirs):
                jvp_chunk[i_local, d] = solve_tangent_equation(
                    a_train[i_global], u_train[i_global], delta_dirs[d],
                    L2d, Dx, Dy, args.nx, args.ny)

            if (i_local + 1) % max(1, chunk_n // 5) == 0:
                elapsed = time.time() - t0
                total_done = start + i_local + 1
                rate = elapsed / total_done
                eta = rate * (n_train - total_done)
                print(f"    [{total_done:5d}/{n_train}]  "
                      f"t={elapsed:.0f}s  rate={rate:.3f}s/sample  "
                      f"ETA={eta:.0f}s ({eta/60:.0f}min)", flush=True)

        np.savez_compressed(chunk_path, jvps=jvp_chunk)
        chunk_size_mb = os.path.getsize(chunk_path) / 1e6
        print(f"  Saved chunk {chunk_id}: {chunk_path} ({chunk_size_mb:.0f} MB)",
              flush=True)
        chunk_id += 1

    # Also save a combined smaller version for backward compat (first 10 dirs only)
    print(f"\nCreating combined file with first 10 directions for quick tests...",
          flush=True)
    # Load chunks and extract first 10 dirs
    jvp_small = np.zeros((n_train, min(10, args.n_dirs), args.nx, args.ny),
                          dtype=np.float32)
    for ci in range(chunk_id):
        chunk_path = os.path.join(jvp_dir, f"chunk_{ci:04d}.npz")
        data = np.load(chunk_path)
        start = ci * args.chunk_size
        end = min(start + args.chunk_size, n_train)
        jvp_small[start:end] = data["jvps"][:, :min(10, args.n_dirs)]

    small_path = os.path.join(args.data_dir, "train_jvps.npz")
    np.savez_compressed(small_path,
                        jvps=jvp_small,
                        delta_dirs=delta_dirs[:min(10, args.n_dirs)])
    print(f"  Saved {small_path} ({os.path.getsize(small_path)/1e6:.0f} MB)")

    # Save metadata
    meta = {
        "n_train": n_train,
        "n_dirs": args.n_dirs,
        "nx": args.nx, "ny": args.ny,
        "n_chunks": chunk_id,
        "chunk_size": args.chunk_size,
        "total_time_s": time.time() - t0,
    }
    import json
    meta_path = os.path.join(jvp_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.0f}min)")
    print(f"Chunks saved in: {jvp_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
