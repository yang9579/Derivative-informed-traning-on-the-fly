#!/usr/bin/env python
"""
Train FNO2D on Nonlinear Diffusion-Reaction — DIFNO Setup
===========================================================
  -div(exp(a) * grad(u)) + u^3 = f
  u = 0 on dOmega (homogeneous Dirichlet)

Matches DIFNO paper Section 6.2 architecture and training protocol.

Usage:
  # FNO baseline (MSE only)
  python train_fno_difno.py --n_use 512 --lam_tan 0.0 --epochs 2000

  # FNO + sTCL
  python train_fno_difno.py --n_use 512 --lam_tan 1.0 --epochs 2000
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from fno2d import FNO2D
from train import train_fno, evaluate_jacobian_error
from stcl_difno import stcl_nonlinear_diffusion_difno


def main():
    ap = argparse.ArgumentParser(
        description="Train FNO2D on nonlinear diffusion-reaction (DIFNO setup)")
    ap.add_argument("--data_dir", type=str,
                    default=os.path.join(SCRIPT_DIR, "data"))
    ap.add_argument("--save_dir", type=str, default=None)
    ap.add_argument("--n_use",    type=int, default=512,
                    help="Use first N training samples (0=all)")
    ap.add_argument("--epochs",   type=int, default=2000)
    ap.add_argument("--batch",    type=int, default=32)
    ap.add_argument("--lr",       type=float, default=1e-3)
    ap.add_argument("--lam_tan",  type=float, default=0.0)
    ap.add_argument("--q_tan",    type=int, default=4)
    ap.add_argument("--width",    type=int, default=32)
    ap.add_argument("--modes",    type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--seed",     type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Default save directory
    if args.save_dir is None:
        n_str = f"N{args.n_use}" if args.n_use > 0 else "Nall"
        lam_str = f"lam{args.lam_tan}"
        args.save_dir = os.path.join(SCRIPT_DIR, "results", f"{n_str}_{lam_str}")

    # Load data
    print(f"\nLoading data from {args.data_dir} ...", flush=True)
    train_data = np.load(os.path.join(args.data_dir, "train.npz"))
    val_data = np.load(os.path.join(args.data_dir, "val.npz"))
    test_data = np.load(os.path.join(args.data_dir, "test.npz"))

    a_train = train_data["a"]
    u_train = train_data["u"]
    a_val = val_data["a"]
    u_val = val_data["u"]
    a_test = test_data["a"]
    u_test = test_data["u"]

    if args.n_use > 0 and args.n_use < len(a_train):
        a_train = a_train[:args.n_use]
        u_train = u_train[:args.n_use]
        print(f"  Using first {args.n_use} of {train_data['a'].shape[0]} samples")

    print(f"  train: a={a_train.shape}  u={u_train.shape}")
    print(f"  test:  a={a_test.shape}   u={u_test.shape}")

    # Build surrogate (DIFNO-paper FNO defaults)
    fno = FNO2D(in_channels=3, out_channels=1,
                width=args.width, modes1=args.modes, modes2=args.modes,
                n_layers=args.n_layers).to(device)
    print(f"\nFNO2D: {fno.count_params():,} parameters")
    print(f"  width={args.width}  modes={args.modes}  layers={args.n_layers}")

    # Coordinate grid (65x65)
    Nx, Ny = a_train.shape[1], a_train.shape[2]
    xs = torch.linspace(0, 1, Nx)
    ys = torch.linspace(0, 1, Ny)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    coord_grid = torch.stack([gx, gy], dim=-1)

    # sTCL function
    stcl_fn = stcl_nonlinear_diffusion_difno if args.lam_tan > 0 else None

    print(f"\n{'='*60}")
    print(f"  Training FNO2D — DIFNO Setup (Section 6.2)")
    print(f"  N_train={len(a_train)}  epochs={args.epochs}  "
          f"lam_tan={args.lam_tan}  q={args.q_tan}")
    print(f"{'='*60}")

    history = train_fno(
        fno, a_train, u_train, a_val, u_val,
        stcl_fn=stcl_fn,
        steps=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        lam_tan=args.lam_tan,
        q_tan=args.q_tan,
        save_dir=args.save_dir,
        print_every=max(1, args.epochs // 20),
        coord_grid=coord_grid,
        device=device,
    )

    # Held-out test evaluation uses the validation-selected checkpoint.  The
    # test split does not drive the scheduler or checkpoint selection.
    best_path = os.path.join(args.save_dir, "best_model.pth")
    fno.load_state_dict(torch.load(best_path, map_location=device,
                                    weights_only=True))
    test_a_t = torch.tensor(a_test, dtype=torch.float32, device=device).unsqueeze(-1)
    test_u_t = torch.tensor(u_test, dtype=torch.float32, device=device).unsqueeze(-1)
    coords_t = coord_grid.to(device).unsqueeze(0).expand(len(a_test), -1, -1, -1)
    with torch.no_grad():
        pred_test = fno(torch.cat([coords_t, test_a_t], dim=-1))
        test_mse = F.mse_loss(pred_test, test_u_t).item()
        per_sample = ((pred_test - test_u_t).flatten(1).norm(dim=1)
                      / (test_u_t.flatten(1).norm(dim=1) + 1e-20))
        test_rel = per_sample.mean().item()

    jac_err = None
    if "jvps" in test_data and "delta_dirs" in test_data:
        print("\nEvaluating Jacobian error ...", flush=True)
        jac_err = evaluate_jacobian_error(
            fno, a_test, test_data["jvps"], test_data["delta_dirs"],
            coord_grid, device=device)
        print(f"  Mean relative Jacobian error: {jac_err:.4f}")

    import json
    metrics_path = os.path.join(args.save_dir, "metrics.json")
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

    print("\nDone.")


if __name__ == "__main__":
    main()
