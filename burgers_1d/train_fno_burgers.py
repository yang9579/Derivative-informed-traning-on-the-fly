#!/usr/bin/env python
"""
FNO Training for Burgers 1D OCP
================================
Trains an FNO2D on data-driven (f, u) pairs for the 1D Burgers PDE,
with optional sTCL or DIFNO (offline derivative-informed) regularization.

Input to FNO: (x, t, f(t)) at each (Nx, Nt) grid point -> 3 channels
Output: u(x, t) -> 1 channel

Usage:
  # Data-only (MSE)
  python train_fno_burgers.py --data_path fno_data/burgers_fno_data.npz --epochs 2000

  # MSE + sTCL
  python train_fno_burgers.py --data_path fno_data/burgers_fno_data.npz --epochs 2000 --lam_tan 1.0

  # MSE + DIFNO (offline JVP matching)
  python train_fno_burgers.py --data_path fno_data/burgers_fno_data.npz --epochs 2000 \
      --method difno --lam_jvp 1.0 --jvp_path fno_data/train_jvps_r200_v3.npz

  # Specify training set size
  python train_fno_burgers.py --data_path fno_data/burgers_fno_data.npz --N_train 512 --lam_tan 1.0
"""

import os
import sys
import argparse
import json

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PARENT_DIR)

from fno2d import FNO2D
from train import train_fno, evaluate_jacobian_error
from stcl_burgers import stcl_burgers


def expected_coefficients(n_train, n_test, n_val, m, sigma_c, seed):
    rng = np.random.RandomState(seed)
    n_total = n_train + n_test + n_val
    all_c = (rng.randn(n_total, m) * sigma_c).astype(np.float32)
    idx = np.arange(n_total)
    rng.shuffle(idx)
    return {
        "train_c": all_c[idx[:n_train]],
        "test_c": all_c[idx[n_train:n_train + n_test]],
        "val_c": all_c[idx[n_train + n_test:]],
    }


def check_paper_distribution(data, sigma_c, seed):
    required = ["train_c", "test_c", "val_c"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"Burgers data is missing coefficient arrays: {missing}")

    n_train = data["train_c"].shape[0]
    n_test = data["test_c"].shape[0]
    n_val = data["val_c"].shape[0]
    m = data["train_c"].shape[1]
    expected = expected_coefficients(n_train, n_test, n_val, m, sigma_c, seed)
    bad = []
    for key, exp in expected.items():
        got = data[key]
        if got.shape != exp.shape or not np.array_equal(got, exp):
            max_abs = float(np.max(np.abs(got - exp))) if got.shape == exp.shape else float("nan")
            bad.append(f"{key}: shape={got.shape}, expected={exp.shape}, max_abs_diff={max_abs:.6g}")
    if bad:
        msg = "\n".join(bad)
        raise ValueError(
            "Burgers base data does not match the paper/Drive control "
            f"distribution sigma_c={sigma_c}, seed={seed}.\n{msg}\n"
            "Regenerate it as described in burgers_1d/README.md "
            "or pass --skip_distribution_check for a non-paper dataset."
        )


def parse_args():
    p = argparse.ArgumentParser(description="Train FNO on Burgers 1D data")
    p.add_argument("--data_path", type=str,
                   default=os.path.join(SCRIPT_DIR, "fno_data", "burgers_fno_data.npz"))
    p.add_argument("--N_train", type=int, default=512,
                   help="Number of training samples; 512 is the default paper cell")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lam_tan", type=float, default=None,
                   help="sTCL EMA target weight; defaults to the paper value 1 for method=stcl")
    p.add_argument("--q_tan", type=int, default=4,
                   help="Number of random directions for sTCL")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--modes", type=int, default=12,
                   help="Fourier modes per dimension")
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", type=str, default=None,
                   help="Output directory (auto-named if not specified)")
    p.add_argument("--print_every", type=int, default=25)
    p.add_argument("--T", type=float, default=1.0,
                   help="Final time for the PDE")
    p.add_argument("--nu", type=float, default=0.01,
                   help="Viscosity")
    # DIFNO arguments
    p.add_argument("--method", type=str, default="mse",
                   choices=["mse", "stcl", "difno"],
                   help=("Training method: solution-only MSE, residual sTCL, "
                         "or offline DIFNO"))
    p.add_argument("--jvp_path", type=str, default=None,
                   help="Path to train_jvps_r200_v3.npz (for DIFNO)")
    p.add_argument("--test_jvp_path", type=str, default=None,
                   help="Path to test_jvps_r200_v3.npz (Metric-B Jacobian evaluation)")
    p.add_argument("--lam_jvp", type=float, default=1.0,
                   help="DIFNO JVP loss weight")
    p.add_argument("--expected_sigma_c", type=float, default=1.5,
                   help="Paper/Drive Burgers coefficient std checked before training.")
    p.add_argument("--data_seed", type=int, default=42,
                   help="Paper/Drive Burgers coefficient seed checked before training.")
    p.add_argument("--skip_distribution_check", action="store_true",
                   help="Allow training on a non-paper Burgers dataset.")
    p.add_argument("--precond_gamma", type=float, default=0.0,
                   help="Spectral preconditioner strength for sTCL (0=off, paper setting)")
    p.add_argument("--n_jvp_per_step", type=int, default=4,
                   help="Number of JVP directions per training step")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Preserve the legacy shorthand --lam_tan > 0 while making explicit
    # --method stcl use the paper default lambda=1.
    if args.method == "mse" and args.lam_tan is not None and args.lam_tan > 0:
        args.method = "stcl"
    if args.lam_tan is None:
        args.lam_tan = 1.0 if args.method == "stcl" else 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    data = np.load(args.data_path)
    if not args.skip_distribution_check:
        check_paper_distribution(data, args.expected_sigma_c, args.data_seed)
    train_f = data["train_f"]  # (N, Nx, Nt)
    train_u = data["train_u"]  # (N, Nx, Nt)
    val_f = data["val_f"]
    val_u = data["val_u"]
    test_f = data["test_f"]
    test_u = data["test_u"]
    x_grid = data["x_grid"]
    t_grid = data["t_grid"]

    Nx, Nt = train_f.shape[1], train_f.shape[2]

    # Subsample training data if requested
    if args.N_train is not None and args.N_train < train_f.shape[0]:
        train_f = train_f[:args.N_train]
        train_u = train_u[:args.N_train]
    N_train = train_f.shape[0]

    print(f"\n{'='*60}")
    print(f"  Burgers 1D FNO Training  [method={args.method}]")
    print(f"{'='*60}")
    print(f"  N_train={N_train}  N_val={val_f.shape[0]}  N_test={test_f.shape[0]}")
    print(f"  Grid: Nx={Nx}, Nt={Nt}")
    print(f"  FNO: width={args.width}, modes={args.modes}, layers={args.n_layers}")
    print(f"  Epochs={args.epochs}, lr={args.lr}, bs={args.batch_size}")
    if args.method == "stcl":
        print(f"  lam_tan={args.lam_tan}, q_tan={args.q_tan}")
    elif args.method == "difno":
        print(f"  lam_jvp={args.lam_jvp}, n_jvp_per_step={args.n_jvp_per_step}")
    print()

    # Build coordinate grid: (Nx, Nt, 2) with (x, t) at each point
    gx, gt = np.meshgrid(x_grid, t_grid, indexing='ij')
    coord_grid = torch.tensor(
        np.stack([gx, gt], axis=-1), dtype=torch.float32
    ).to(device)  # (Nx, Nt, 2)

    # Input: f(t) at each (x, t) point — shape (N, Nx, Nt, 1)
    # The train_f already has shape (N, Nx, Nt) with f broadcast to all x
    train_a = torch.tensor(train_f, dtype=torch.float32).unsqueeze(-1)  # (N, Nx, Nt, 1)
    train_u_t = torch.tensor(train_u, dtype=torch.float32).unsqueeze(-1)  # (N, Nx, Nt, 1)
    val_a = torch.tensor(val_f, dtype=torch.float32).unsqueeze(-1)
    val_u_t = torch.tensor(val_u, dtype=torch.float32).unsqueeze(-1)
    test_a = torch.tensor(test_f, dtype=torch.float32).unsqueeze(-1)
    test_u_t = torch.tensor(test_u, dtype=torch.float32).unsqueeze(-1)

    # Build FNO: 3 input channels (x, t, f), 1 output channel (u)
    fno = FNO2D(
        in_channels=3, out_channels=1,
        width=args.width,
        modes1=args.modes, modes2=args.modes,
        n_layers=args.n_layers,
    ).to(device)
    print(f"  FNO params: {fno.count_params():,}")

    resolved_jvp_path = None
    resolved_test_jvp_path = None

    if args.method == "difno":
        # ── DIFNO training path ──
        from train_difno_1d import train_difno_1d

        # Load JVP data
        jvp_path = args.jvp_path or os.path.join(
            SCRIPT_DIR, "fno_data", "train_jvps_r200_v3.npz")
        resolved_jvp_path = jvp_path
        print(f"  Loading JVPs from {jvp_path} ...", flush=True)
        jvp_data = np.load(jvp_path)
        train_jvps = jvp_data["jvps"][:N_train]  # (N, K, Nx, Nt)
        delta_dirs = jvp_data["delta_dirs"]       # (K, Nx, Nt)
        print(f"  JVPs: {train_jvps.shape}  dirs: {delta_dirs.shape}")

        # Auto-name save directory
        if args.save_dir is None:
            tag = f"fno_N{N_train}_difno_lam{args.lam_jvp}_seed{args.seed}"
            args.save_dir = os.path.join(SCRIPT_DIR, "fno_results", tag)

        history = train_difno_1d(
            fno, train_a, train_u_t, train_jvps, delta_dirs,
            val_a, val_u_t,
            steps=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lam_jvp=args.lam_jvp,
            n_jvp_per_step=args.n_jvp_per_step,
            save_dir=args.save_dir,
            print_every=args.print_every,
            coord_grid=coord_grid,
            device=device,
        )

    else:
        # ── MSE or sTCL training path ──
        stcl_fn = None
        if args.method == "stcl" and args.lam_tan > 0:
            def stcl_fn(fno_model, f_batch, cg, q):
                return stcl_burgers(fno_model, f_batch, cg, q=q,
                                    T=args.T, nu=args.nu,
                                    precond_gamma=args.precond_gamma)

        # Auto-name save directory
        if args.save_dir is None:
            tag = (
                f"fno_N{N_train}_{args.method}_lam{args.lam_tan}_"
                f"seed{args.seed}"
            )
            args.save_dir = os.path.join(SCRIPT_DIR, "fno_results", tag)

        # Train using the shared training loop
        history = train_fno(
            fno, train_a, train_u_t, val_a, val_u_t,
            stcl_fn=stcl_fn,
            steps=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lam_tan=args.lam_tan,
            q_tan=args.q_tan,
            save_dir=args.save_dir,
            print_every=args.print_every,
            coord_grid=coord_grid,
            device=device,
        )

    # Held-out test evaluation uses the validation-selected best_model.pth.
    # Validation alone drives ReduceLROnPlateau and checkpoint selection.
    best_path = os.path.join(args.save_dir, "best_model.pth")
    fno.load_state_dict(torch.load(best_path, map_location=device,
                                    weights_only=True))
    with torch.no_grad():
        test_a_d = test_a.to(device)
        test_u_d = test_u_t.to(device)
        coords_test = coord_grid.unsqueeze(0).expand(len(test_a_d), -1, -1, -1)
        pred_test = fno(torch.cat([coords_test, test_a_d], dim=-1))
        test_mse = torch.mean((pred_test - test_u_d) ** 2).item()
        per_sample = ((pred_test - test_u_d).flatten(1).norm(dim=1)
                      / (test_u_d.flatten(1).norm(dim=1) + 1e-20))
        test_rel = per_sample.mean().item()

    # Metric-B Jacobian error on the fixed test JVP bank, for every method.
    jac_err = None
    test_jvp_path = args.test_jvp_path or os.path.join(
        SCRIPT_DIR, "fno_data", "test_jvps_r200_v3.npz")
    if os.path.exists(test_jvp_path):
        resolved_test_jvp_path = test_jvp_path
        print("\nEvaluating Jacobian error on test set ...")
        test_jvp_data = np.load(test_jvp_path)
        jac_err = evaluate_jacobian_error(
            fno, test_f, test_jvp_data["jvps"],
            test_jvp_data["delta_dirs"],
            coord_grid, device=device)
        print(f"  Mean relative Jacobian error: {jac_err:.4f}")

    metrics_path = os.path.join(args.save_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics.update({
            "test_mse": test_mse,
            "test_rel_err": test_rel,
            "test_jac_rel_err": jac_err,
            "jacobian_rel_error": jac_err,
            "test_checkpoint": "best_model.pth",
        })
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
    print(f"  Held-out test: rel={test_rel:.4f}  mse={test_mse:.4e}")

    # Save extra metadata
    meta_path = os.path.join(args.save_dir, "config.json")
    config = {
        "data_path": args.data_path,
        "N_train": N_train,
        "N_val": int(val_f.shape[0]),
        "N_test": int(test_f.shape[0]),
        "Nx": Nx, "Nt": Nt,
        "width": args.width, "modes": args.modes,
        "n_layers": args.n_layers,
        "epochs": args.epochs,
        "lr": args.lr,
        "method": args.method,
        "lam_tan": args.lam_tan,
        "q_tan": args.q_tan,
        "precond_gamma": args.precond_gamma if args.method == "stcl" else None,
        "lam_jvp": args.lam_jvp if args.method == "difno" else None,
        "n_jvp_per_step": args.n_jvp_per_step if args.method == "difno" else None,
        "jvp_path": resolved_jvp_path,
        "test_jvp_path": resolved_test_jvp_path,
        "expected_sigma_c": args.expected_sigma_c,
        "data_seed": args.data_seed,
        "seed": args.seed,
        "nu": args.nu, "T": args.T,
    }
    with open(meta_path, "w") as fh:
        json.dump(config, fh, indent=2)

    print(f"\nConfig saved: {meta_path}")


if __name__ == "__main__":
    main()
