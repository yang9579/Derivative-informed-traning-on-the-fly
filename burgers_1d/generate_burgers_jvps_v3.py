#!/usr/bin/env python
"""
Generate Burgers JVP data — v3 (vmap-batched directions).

Same numerical computation as v2 (autograd-through-implicit-Picard) but the
inner direction loop is replaced with a single vmap call. Probe at K=50
showed 35.85x speedup with machine-precision agreement vs sequential v2.

Usage:
  python generate_burgers_jvps_v3.py --n_dirs 200 --save_suffix _r200_v3
"""
import os, sys, time, argparse, math
import numpy as np
import torch
from torch.func import jvp as func_jvp, vmap as func_vmap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from adjoint_burgers import NU, T_FINAL, N_PICARD
from generate_burgers_fno_data import (
    generate_rbf_control, solve_one_sample, validate_sample,
)
from generate_burgers_jvps_v2 import (
    NX_SOLVE, NT_SOLVE, NX_OUT, NT_OUT,
    build_A_torch, forward_solve_implicit_torch,
    interpolate_to_fno_grid, sample_directions_solver_grid,
    project_dir_to_fno_grid,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str,
                    default=os.path.join(SCRIPT_DIR, "fno_data", "burgers_fno_data.npz"))
    ap.add_argument("--save_dir", type=str,
                    default=os.path.join(SCRIPT_DIR, "fno_data"))
    ap.add_argument("--n_dirs", type=int, default=200)
    ap.add_argument("--seed_train", type=int, default=12345)
    ap.add_argument("--seed_test",  type=int, default=98765)
    ap.add_argument("--Nx_solve", type=int, default=NX_SOLVE)
    ap.add_argument("--Nt_solve", type=int, default=NT_SOLVE)
    ap.add_argument("--Nx_out", type=int, default=NX_OUT)
    ap.add_argument("--Nt_out", type=int, default=NT_OUT)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--sigma_rbf", type=float, default=0.2)
    ap.add_argument("--sigma_c", type=float, default=1.5,
                    help="Std used only when resampling unstable samples; paper/Drive value is 1.5.")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="float64",
                    choices=["float32", "float64"])
    ap.add_argument("--save_suffix", type=str, default="_r200_v3")
    ap.add_argument("--max_abs_jvp", type=float, default=0.0,
                    help="Optional positive threshold for max |JVP|; 0 disables finite-value clipping.")
    ap.add_argument("--skip_test", action="store_true",
                    help="Only generate train_jvps; reuse existing test_jvps.")
    ap.add_argument("--resample_unstable", action="store_true",
                    help="Resample samples whose implicit-Picard JVPs are non-finite or overflow float32.")
    ap.add_argument("--resample_seed", type=int, default=24680)
    ap.add_argument("--max_retries", type=int, default=100,
                    help="Maximum same-distribution resampling attempts per unstable sample.")
    ap.add_argument("--max_abs_u", type=float, default=None,
                    help="Forward-solution rejection threshold for resampled samples; defaults to data max_abs_u or 20.")
    ap.add_argument("--clean_data_path", type=str, default=None,
                    help="Where to write cleaned burgers_fno_data.npz after resampling; default overwrites data_path.")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    Nx, Nt = args.Nx_solve, args.Nt_solve
    dt = T_FINAL / Nt
    dx = 1.0 / (Nx + 1)
    A = build_A_torch(Nx, dx, dt, NU, device, dtype=dtype)

    print(f"Generator v3 (vmap-batched directions)", flush=True)
    print(f"  device={device}, dtype={dtype}, Nx={Nx}, Nt={Nt}, dt={dt:.5f}, dx={dx:.5f}")
    print(f"  N_PICARD={N_PICARD}, NU={NU}, T={T_FINAL}", flush=True)

    print(f"\nLoading {args.data_path} ...", flush=True)
    data_npz = np.load(args.data_path)
    data = {k: data_npz[k].copy() for k in data_npz.files}
    data_npz.close()
    train_c = data["train_c"].copy()
    test_c  = data["test_c"].copy()
    N_train, N_test = train_c.shape[0], test_c.shape[0]
    print(f"  N_train={N_train}, N_test={N_test}, M={train_c.shape[1]}")
    max_abs_u = args.max_abs_u
    if max_abs_u is None:
        max_abs_u = float(data["max_abs_u"]) if "max_abs_u" in data else 20.0
    if args.resample_unstable:
        print(f"  Resampling unstable JVP samples with seed={args.resample_seed}, "
              f"max |u| <= {max_abs_u}", flush=True)
    if args.max_abs_jvp > 0:
        print(f"  JVP rejection check: max |JVP| <= {args.max_abs_jvp}", flush=True)

    V_train = sample_directions_solver_grid(args.n_dirs, Nt, args.seed_train)
    V_test  = sample_directions_solver_grid(args.n_dirs, Nt, args.seed_test)
    print(f"  Train dir seed={args.seed_train}; test dir seed={args.seed_test}; "
          f"|<v_train_d, v_test_d>|.max()={float(np.abs(V_train @ V_test.T).max()):.4f}",
          flush=True)

    delta_train_fno = np.zeros((args.n_dirs, args.Nx_out, args.Nt_out), dtype=np.float32)
    delta_test_fno  = np.zeros((args.n_dirs, args.Nx_out, args.Nt_out), dtype=np.float32)
    for d in range(args.n_dirs):
        v_t_train = project_dir_to_fno_grid(V_train[d], Nt, args.Nt_out, T_FINAL)
        v_t_test  = project_dir_to_fno_grid(V_test[d],  Nt, args.Nt_out, T_FINAL)
        delta_train_fno[d] = np.broadcast_to(v_t_train[None, :].astype(np.float32),
                                              (args.Nx_out, args.Nt_out))
        delta_test_fno[d]  = np.broadcast_to(v_t_test[None, :].astype(np.float32),
                                              (args.Nx_out, args.Nt_out))

    V_train_t = torch.tensor(V_train, dtype=dtype, device=device)
    V_test_t  = torch.tensor(V_test,  dtype=dtype, device=device)

    t_solve = np.arange(Nt) * dt

    rng_resample = np.random.RandomState(args.resample_seed)

    def compute_sample_jvps(c_i, V_dirs_t, label, sample_index):
        f_np = generate_rbf_control(c_i, args.M, T_FINAL, args.sigma_rbf, t_solve)
        f_t = torch.tensor(f_np, dtype=dtype, device=device)

        def fwd(fff):
            return forward_solve_implicit_torch(fff, A, Nx, Nt, dt, dx)
        def jvp_one(v):
            _, W = func_jvp(fwd, (f_t,), (v,))
            return W

        # vmap over the direction dim (n_dirs in parallel on GPU)
        W_all = func_vmap(jvp_one)(V_dirs_t)         # (n_dirs, Nx, Nt+1)
        W_all_np = W_all.detach().cpu().numpy()
        if not np.isfinite(W_all_np).all():
            raise RuntimeError(f"Non-finite JVP for {label} sample {sample_index}")

        sample_out = np.zeros((args.n_dirs, args.Nx_out, args.Nt_out),
                              dtype=np.float32)
        max_sample_w = 0.0
        # Serial interp on CPU (fast: ~1ms each, dominated by JVP cost)
        for d in range(args.n_dirs):
            w_fno = interpolate_to_fno_grid(W_all_np[d], Nx, Nt,
                                             args.Nx_out, args.Nt_out, T_FINAL)
            if not np.isfinite(w_fno).all():
                raise RuntimeError(
                    f"Non-finite interpolated JVP for {label} sample {sample_index}, dir {d}"
                )
            max_w = float(np.max(np.abs(w_fno)))
            if args.max_abs_jvp > 0 and max_w > args.max_abs_jvp:
                raise RuntimeError(
                    f"Unstable JVP for {label} sample {sample_index}, dir {d}: "
                    f"max |w|={max_w:.3e} > {args.max_abs_jvp:.3e}"
                )
            if max_w > np.finfo(np.float32).max:
                raise RuntimeError(
                    f"Float32-overflow JVP for {label} sample {sample_index}, dir {d}: "
                    f"max |w|={max_w:.3e}"
                )
            w32 = w_fno.astype(np.float32)
            if not np.isfinite(w32).all():
                raise RuntimeError(
                    f"Non-finite float32 JVP for {label} sample {sample_index}, dir {d}"
                )
            sample_out[d] = w32
            max_sample_w = max(max_sample_w, max_w)
        return sample_out, max_sample_w

    def gen_jvps_for_split(c_split, f_split, u_split, V_dirs_t, label):
        """For each sample, vmap-batched JVP over all n_dirs directions.
        Returns ndarray (N, n_dirs, Nx_out, Nt_out)."""
        N = c_split.shape[0]
        out = np.zeros((N, args.n_dirs, args.Nx_out, args.Nt_out), dtype=np.float32)
        resampled_indices = []
        t0 = time.time()
        for i in range(N):
            last_error = None
            for attempt in range(1, args.max_retries + 1):
                try:
                    if attempt == 1:
                        c_i = c_split[i].astype(np.float64)
                        f32 = None
                        u32 = None
                    else:
                        c_i = rng_resample.randn(args.M).astype(np.float64) * args.sigma_c
                        f_grid, u_grid = solve_one_sample(
                            c_i, args.M, T_FINAL, args.sigma_rbf,
                            args.Nx_solve, args.Nt_solve,
                            args.Nx_out, args.Nt_out,
                        )
                        f32, u32 = validate_sample(f_grid, u_grid, max_abs_u)

                    sample_jvps, max_sample_w = compute_sample_jvps(
                        c_i, V_dirs_t, label, i)
                    out[i] = sample_jvps
                    if attempt > 1:
                        c_split[i] = c_i.astype(np.float32)
                        f_split[i] = f32
                        u_split[i] = u32
                        resampled_indices.append(i)
                        print(f"  [{label}] resampled sample {i} after {attempt-1} rejects; "
                              f"max|u|={float(np.abs(u32).max()):.3f}, "
                              f"max|w|={max_sample_w:.3e}", flush=True)
                    break
                except Exception as e:
                    last_error = e
                    if not args.resample_unstable:
                        raise
            else:
                raise RuntimeError(
                    f"Could not generate stable JVPs for {label} sample {i} "
                    f"after {args.max_retries} attempts; last error: {last_error}"
                )

            if (i + 1) % 50 == 0 or (i + 1) <= 3:
                el = time.time() - t0
                eta = el / (i + 1) * (N - i - 1)
                mw = float(np.abs(out[i]).max())
                print(f"  [{label}] [{i+1:4d}/{N}]  max|w|={mw:.4f}  "
                      f"t={el:.0f}s  eta={eta:.0f}s", flush=True)
        return out, np.array(resampled_indices, dtype=np.int32)

    print(f"\n=== Generating TRAIN jvps ({N_train} samples × {args.n_dirs} dirs) ===")
    train_jvps, train_resampled = gen_jvps_for_split(
        train_c, data["train_f"], data["train_u"], V_train_t, "train")

    if not args.skip_test:
        print(f"\n=== Generating TEST jvps ({N_test} samples × {args.n_dirs} dirs) ===")
        test_jvps, test_resampled = gen_jvps_for_split(
            test_c, data["test_f"], data["test_u"], V_test_t, "test ")
    else:
        test_resampled = np.array([], dtype=np.int32)

    if args.resample_unstable and (len(train_resampled) or len(test_resampled)):
        data["train_c"] = train_c.astype(np.float32)
        data["test_c"] = test_c.astype(np.float32)
        data["jvp_resampled_train_indices"] = train_resampled
        data["jvp_resampled_test_indices"] = test_resampled
        data["jvp_resample_seed"] = np.array(args.resample_seed, dtype=np.int32)
        data["jvp_resample_max_abs_u"] = np.array(max_abs_u, dtype=np.float32)
        clean_path = args.clean_data_path or args.data_path
        tmp_path = clean_path + ".tmp.npz"
        np.savez(tmp_path, **data)
        os.replace(tmp_path, clean_path)
        print(f"\nUpdated cleaned data: {clean_path}")
        print(f"  resampled train indices: {train_resampled.tolist()}")
        print(f"  resampled test indices: {test_resampled.tolist()}")

    os.makedirs(args.save_dir, exist_ok=True)
    p_train = os.path.join(args.save_dir, f"train_jvps{args.save_suffix}.npz")
    np.savez_compressed(p_train, jvps=train_jvps, delta_dirs=delta_train_fno,
                        seed=args.seed_train,
                        source="autograd_through_implicit_picard_vmap_v3")
    print(f"\nSaved: {p_train}  ({os.path.getsize(p_train)/1e6:.1f} MB)")
    print(f"  jvps shape: train={train_jvps.shape}")

    if not args.skip_test:
        p_test  = os.path.join(args.save_dir, f"test_jvps{args.save_suffix}.npz")
        np.savez_compressed(p_test,  jvps=test_jvps,  delta_dirs=delta_test_fno,
                            seed=args.seed_test,
                            source="autograd_through_implicit_picard_vmap_v3")
        print(f"       {p_test}   ({os.path.getsize(p_test)/1e6:.1f} MB)")
        print(f"  jvps shape: test ={test_jvps.shape}")
    print("Done.")


if __name__ == "__main__":
    main()
