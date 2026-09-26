#!/usr/bin/env python
"""
Derivative-Informed FNO Training (DIFNO-style) — DIFNO Setup
==============================================================
Replicates the DIFNO training approach (Yao et al., arXiv:2512.14086):
  Loss = MSE(u_pred, u_true) + lam_jvp * adaptive * (1/K) sum_k MSE(JVP_pred_k, JVP_true_k)

True JVPs: pre-computed offline via forward sensitivity equations (tangent PDE solves).
FNO JVPs:  computed via forward-mode AD on the FNO (torch.autograd.functional.jvp),
           matching the DIFNO paper Section 5.5.

Usage:
  python train_difno.py --n_use 512 --lam_jvp 1.0 --epochs 2000
"""

import os
import sys
import csv
import json
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jvp as _func_jvp  # ~2x faster than torch.autograd.functional.jvp(create_graph=True)

def jvp(fn, primals, tangents, create_graph=True):
    return _func_jvp(fn, primals, tangents)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from fno2d import FNO2D
from train import evaluate_jacobian_error
from runtime_profile import end as profile_end
from runtime_profile import epoch_end as profile_epoch_end
from runtime_profile import epoch_start as profile_epoch_start
from runtime_profile import start as profile_start


def train_difno(fno, train_a, train_u, train_jvps, delta_dirs,
                val_a, val_u,
                steps=500, batch_size=32, lr=1e-3,
                lam_jvp=1.0, n_jvp_per_step=4,
                save_dir=None, print_every=25,
                coord_grid=None, device=None):
    """
    Train FNO with output + derivative supervision (DIFNO paper approach).

    True JVPs from pre-computed data (sensitivity solves).
    FNO JVPs via forward-mode AD (torch.autograd.functional.jvp).
    """
    if device is None:
        device = next(fno.parameters()).device

    if isinstance(train_a, np.ndarray):
        train_a = torch.tensor(train_a, dtype=torch.float32)
        train_u = torch.tensor(train_u, dtype=torch.float32)
        train_jvps = torch.tensor(train_jvps, dtype=torch.float32)
        delta_dirs = torch.tensor(delta_dirs, dtype=torch.float32)
        val_a = torch.tensor(val_a, dtype=torch.float32)
        val_u = torch.tensor(val_u, dtype=torch.float32)

    if train_a.dim() == 3:
        train_a = train_a.unsqueeze(-1)
    if train_u.dim() == 3:
        train_u = train_u.unsqueeze(-1)
    if val_a.dim() == 3:
        val_a = val_a.unsqueeze(-1)
    if val_u.dim() == 3:
        val_u = val_u.unsqueeze(-1)

    # delta_dirs: (K, Nx, Ny) -> (K, Nx, Ny, 1)
    if delta_dirs.dim() == 3:
        delta_dirs = delta_dirs.unsqueeze(-1)

    N_train = train_a.shape[0]
    Nx, Ny = train_a.shape[1], train_a.shape[2]
    K_total = delta_dirs.shape[0]

    if coord_grid is None:
        xs = torch.linspace(0, 1, Nx)
        ys = torch.linspace(0, 1, Ny)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        coord_grid = torch.stack([gx, gy], dim=-1)
    coord_grid = coord_grid.to(device)

    optimizer = torch.optim.Adam(fno.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=50, factor=0.5, min_lr=1e-6)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    history = []
    best_val_mse = float("inf")
    best_val_rel = float("inf")
    best_epoch = None
    t0 = time.time()

    # Adaptive scaling (EMA)
    mse_ema = None
    jvp_ema = None
    ema_decay = 0.99

    for epoch in range(1, steps + 1):
        profile_epoch_start(epoch)
        fno.train()
        perm = torch.randperm(N_train)
        epoch_mse = 0.0
        epoch_jvp_loss = 0.0
        epoch_scale = 0.0
        n_batches = 0

        for i in range(0, N_train, batch_size):
            idx = perm[i:i + batch_size]
            a_batch = train_a[idx].to(device)  # (B, Nx, Ny, 1)
            u_batch = train_u[idx].to(device)
            B = a_batch.shape[0]

            coords = coord_grid.unsqueeze(0).expand(B, -1, -1, -1)

            # Output loss
            _pt = profile_start("data_forward")
            fno_input = torch.cat([coords, a_batch], dim=-1)
            u_pred = fno(fno_input)
            mse = F.mse_loss(u_pred, u_batch)
            profile_end(_pt)

            # Derivative loss via forward-mode AD (DIFNO paper Section 5.5)
            _deriv_pt = profile_start("derivative_total")
            dir_idx = torch.randperm(K_total)[:n_jvp_per_step]
            jvp_loss = torch.tensor(0.0, device=device)

            def fno_fn(a_field):
                inp = torch.cat([coords, a_field], dim=-1)
                return fno(inp)

            for dk in dir_idx:
                # True JVP (pre-computed via sensitivity equations)
                _pt = profile_start("label_h2d")
                true_jvp_k = train_jvps[idx, dk].to(device)  # (B, Nx, Ny)
                profile_end(_pt)

                # FNO JVP via forward-mode AD
                da = delta_dirs[dk].unsqueeze(0).expand(B, -1, -1, -1).to(device)
                _pt = profile_start("online_jvp")
                _, pred_jvp_k = jvp(fno_fn, (a_batch,), (da,),
                                     create_graph=True)
                profile_end(_pt)
                pred_jvp_k = pred_jvp_k.squeeze(-1)  # (B, Nx, Ny)

                jvp_loss = jvp_loss + F.mse_loss(pred_jvp_k, true_jvp_k)

            jvp_loss = jvp_loss / n_jvp_per_step
            profile_end(_deriv_pt)

            cur_scale = mse_ema / max(jvp_ema, 1e-10) if mse_ema is not None else 1.0

            loss = mse + lam_jvp * cur_scale * jvp_loss
            _pt = profile_start("backward_optimizer")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fno.parameters(), 1.0)
            optimizer.step()
            profile_end(_pt)
            mse_val = mse.item(); jvp_val = jvp_loss.item()

            if mse_ema is None:
                mse_ema = mse_val
                jvp_ema = max(jvp_val, 1e-10)
            else:
                mse_ema = ema_decay * mse_ema + (1 - ema_decay) * mse_val
                jvp_ema = ema_decay * jvp_ema + (1 - ema_decay) * max(jvp_val, 1e-10)
            adaptive_scale = cur_scale

            epoch_mse += mse_val
            epoch_jvp_loss += jvp_val
            epoch_scale += adaptive_scale
            n_batches += 1

        epoch_mse /= n_batches
        epoch_jvp_loss /= n_batches
        epoch_scale /= max(n_batches, 1)

        # Validation evaluation.  Test data is evaluated only after training.
        fno.eval()
        with torch.no_grad():
            val_a_d = val_a.to(device)
            B_val = val_a_d.shape[0]
            coords_val = coord_grid.unsqueeze(0).expand(B_val, -1, -1, -1)
            fno_input_val = torch.cat([coords_val, val_a_d], dim=-1)
            u_pred_val = fno(fno_input_val)
            val_u_d = val_u.to(device)
            val_mse = F.mse_loss(u_pred_val, val_u_d).item()
            val_rel = (torch.norm(u_pred_val - val_u_d) /
                       torch.norm(val_u_d)).item()

        scheduler.step(val_mse)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_val_rel = val_rel
            best_epoch = epoch
            if save_dir:
                torch.save(fno.state_dict(),
                           os.path.join(save_dir, "best_model.pth"))

        record = {
            "epoch": epoch,
            "train_mse": epoch_mse,
            "val_mse": val_mse,
            "val_rel": val_rel,
            "rel_err": val_rel,
            "jvp_loss": epoch_jvp_loss,
            "adapt_scale": epoch_scale,
            "lr": optimizer.param_groups[0]["lr"],
            "wall_time": time.time() - t0,
        }
        history.append(record)

        if epoch % print_every == 0 or epoch <= 3:
            print(f"  [{epoch:4d}/{steps}]  "
                  f"mse={epoch_mse:.4e}  val={val_mse:.4e}  "
                  f"rel={val_rel:.4f}  jvp={epoch_jvp_loss:.4e}  "
                  f"scale={epoch_scale:.2e}  "
                  f"lr={record['lr']:.1e}  t={record['wall_time']:.0f}s",
                  flush=True)
        profile_epoch_end(epoch, n_batches)

    # Save
    if save_dir:
        torch.save(fno.state_dict(),
                   os.path.join(save_dir, "last_model.pth"))

        if history:
            with open(os.path.join(save_dir, "train_history.csv"),
                      "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=history[0].keys())
                w.writeheader()
                w.writerows(history)

        cfg = {
            "steps": steps,
            "batch_size": batch_size,
            "lr": lr,
            "lam_jvp": lam_jvp,
            "n_jvp_per_step": n_jvp_per_step,
            "K_total": K_total,
            "N_train": N_train,
            "Nx": Nx, "Ny": Ny,
            "best_val_mse": best_val_mse,
            "best_val_rel": best_val_rel,
            "best_epoch": best_epoch,
            "final_val_rel": history[-1]["val_rel"],
            "method": "DIFNO_AD",
        }
        with open(os.path.join(save_dir, "metrics.json"), "w") as fh:
            json.dump(cfg, fh, indent=2)

        print(f"\nSaved -> {save_dir}")
        print(f"  best_val_mse={best_val_mse:.4e}  "
              f"best_val_rel={best_val_rel:.4f}  "
              f"final_val_rel={history[-1]['val_rel']:.4f}")

    return history


def main():
    ap = argparse.ArgumentParser(
        description="DIFNO-style derivative-informed FNO training (AD-based)")
    ap.add_argument("--data_dir", type=str,
                    default=os.path.join(SCRIPT_DIR, "data"))
    ap.add_argument("--save_dir", type=str, default=None)
    ap.add_argument("--n_use",    type=int, default=512)
    ap.add_argument("--epochs",   type=int, default=2000)
    ap.add_argument("--batch",    type=int, default=32)
    ap.add_argument("--lr",       type=float, default=1e-3)
    ap.add_argument("--lam_jvp",  type=float, default=1.0)
    ap.add_argument("--n_jvp_per_step", type=int, default=4)
    ap.add_argument("--width",    type=int, default=32)
    ap.add_argument("--modes",    type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--seed",     type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.save_dir is None:
        n_str = f"N{args.n_use}" if args.n_use > 0 else "Nall"
        args.save_dir = os.path.join(
            SCRIPT_DIR, "results", f"{n_str}_difno_lam{args.lam_jvp}")

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

    # Apply n_use filter to (a, u) BEFORE loading JVPs
    n_needed = args.n_use if args.n_use > 0 else len(a_train)
    n_needed = min(n_needed, len(a_train))
    a_train = a_train[:n_needed]
    u_train = u_train[:n_needed]

    # Load JVP data — try chunked (289 dirs) first, fall back to single file
    # Only load chunks that contain samples we'll use
    chunk_dir = os.path.join(args.data_dir, "train_jvps_chunks")
    if os.path.isdir(chunk_dir) and os.path.exists(os.path.join(chunk_dir, "meta.json")):
        meta = json.load(open(os.path.join(chunk_dir, "meta.json")))
        n_dirs_total = meta["n_dirs"]
        chunk_size = meta["chunk_size"]
        n_chunks = meta["n_chunks"]
        # Only load chunks needed for the first n_needed samples
        chunks_needed = (n_needed + chunk_size - 1) // chunk_size
        print(f"  Loading chunked JVPs: {n_dirs_total} dirs, "
              f"{chunks_needed}/{n_chunks} chunks (for {n_needed} samples)")

        # Load directions
        dirs_data = np.load(os.path.join(args.data_dir, "train_jvp_dirs.npz"))
        delta_dirs = dirs_data["delta_dirs"]

        chunks = []
        for ci in range(chunks_needed):
            cp = os.path.join(chunk_dir, f"chunk_{ci:04d}.npz")
            chunks.append(np.load(cp)["jvps"])
        train_jvps = np.concatenate(chunks, axis=0)[:n_needed]
        print(f"  Loaded JVPs: {train_jvps.shape} ({train_jvps.nbytes/1e9:.1f} GB)")
    else:
        jvp_data = np.load(os.path.join(args.data_dir, "train_jvps.npz"))
        train_jvps = jvp_data["jvps"][:n_needed]
        delta_dirs = jvp_data["delta_dirs"]
        print(f"  Loaded JVPs: {train_jvps.shape}")

    if args.n_use > 0:
        print(f"  Using first {args.n_use} samples")

    print(f"  train: a={a_train.shape}  u={u_train.shape}  "
          f"jvps={train_jvps.shape}")
    print(f"  test:  a={a_test.shape}   u={u_test.shape}")
    print(f"  directions: {delta_dirs.shape}")

    # Build surrogate
    fno = FNO2D(in_channels=3, out_channels=1,
                width=args.width, modes1=args.modes, modes2=args.modes,
                n_layers=args.n_layers).to(device)
    print(f"\nFNO2D: {fno.count_params():,} parameters")

    # Coordinate grid
    Nx, Ny = a_train.shape[1], a_train.shape[2]
    xs = torch.linspace(0, 1, Nx)
    ys = torch.linspace(0, 1, Ny)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    coord_grid = torch.stack([gx, gy], dim=-1)

    print(f"\n{'='*60}")
    print(f"  DIFNO Training — Forward-Mode AD (paper approach)")
    print(f"  N_train={len(a_train)}  epochs={args.epochs}  "
          f"lam_jvp={args.lam_jvp}")
    print(f"  K_total={delta_dirs.shape[0]}  "
          f"n_jvp_per_step={args.n_jvp_per_step}")
    print(f"{'='*60}")

    history = train_difno(
        fno, a_train, u_train, train_jvps, delta_dirs,
        a_val, u_val,
        steps=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        lam_jvp=args.lam_jvp,
        n_jvp_per_step=args.n_jvp_per_step,
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
