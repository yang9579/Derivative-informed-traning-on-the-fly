#!/usr/bin/env python
"""
DIFNO Training for 1D Time-Dependent PDEs (Burgers, Allen-Cahn)
================================================================
Derivative-Informed FNO training with offline JVP matching.

Loss = MSE(u_pred, u_true) + lam_jvp * adaptive_scale * (1/K) sum_k MSE(JVP_pred_k, JVP_true_k)

True JVPs: pre-computed offline via tangent equation solves.
FNO JVPs:  computed via forward-mode AD (torch.func.jvp).

The key difference from the 2D DIFNO training (train_difno.py in
nonlinear_diffusion_difno/) is:
- Input perturbation directions have shape (K, Nx, Nt) — space-uniform
  f(t) control, broadcast to the (x, t) grid.
- JVPs have shape (N_train, K, Nx, Nt) — tangent solutions w(x,t).

Usage: imported by burgers_1d/train_fno_burgers.py.
"""

import os
import csv
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jvp as _func_jvp
from runtime_profile import end as profile_end
from runtime_profile import epoch_end as profile_epoch_end
from runtime_profile import epoch_start as profile_epoch_start
from runtime_profile import start as profile_start


def train_difno_1d(fno, train_a, train_u, train_jvps, delta_dirs,
                   val_a, val_u,
                   steps=500, batch_size=32, lr=1e-3,
                   lam_jvp=1.0, n_jvp_per_step=4,
                   save_dir=None, print_every=25,
                   coord_grid=None, device=None):
    """
    Train FNO with output + derivative supervision (DIFNO approach)
    for 1D time-dependent PDEs.

    Args:
        fno: FNO2D model
        train_a: (N, Nx, Nt, 1) input fields
        train_u: (N, Nx, Nt, 1) output fields
        train_jvps: (N, K, Nx, Nt) pre-computed true JVPs
        delta_dirs: (K, Nx, Nt) perturbation directions (on FNO grid)
        val_a, val_u: validation data used for scheduling/checkpointing
        steps, batch_size, lr: training hyperparameters
        lam_jvp: weight for JVP matching loss
        n_jvp_per_step: number of directions to sample per batch step
        save_dir: output directory
        coord_grid: (Nx, Nt, 2) coordinate grid
        device: torch device

    Returns:
        history: list of dicts with training metrics
    """
    if device is None:
        device = next(fno.parameters()).device

    # Convert all inputs to tensors
    def _to_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=torch.float32)
        return x

    train_a = _to_tensor(train_a)
    train_u = _to_tensor(train_u)
    train_jvps = _to_tensor(train_jvps)
    delta_dirs = _to_tensor(delta_dirs)
    val_a = _to_tensor(val_a)
    val_u = _to_tensor(val_u)

    if train_a.dim() == 3:
        train_a = train_a.unsqueeze(-1)
    if train_u.dim() == 3:
        train_u = train_u.unsqueeze(-1)
    if val_a.dim() == 3:
        val_a = val_a.unsqueeze(-1)
    if val_u.dim() == 3:
        val_u = val_u.unsqueeze(-1)

    # delta_dirs: (K, Nx, Nt) -> (K, Nx, Nt, 1)
    if delta_dirs.dim() == 3:
        delta_dirs = delta_dirs.unsqueeze(-1)

    N_train = train_a.shape[0]
    Nx, Nt = train_a.shape[1], train_a.shape[2]
    K_total = delta_dirs.shape[0]

    if coord_grid is None:
        xs = torch.linspace(0, 1, Nx)
        ts = torch.linspace(0, 1, Nt)
        gx, gt = torch.meshgrid(xs, ts, indexing="ij")
        coord_grid = torch.stack([gx, gt], dim=-1)
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

    print(f"  DIFNO training: lam_jvp={lam_jvp}, K_total={K_total}, "
          f"n_jvp_per_step={n_jvp_per_step}")

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
            a_batch = train_a[idx].to(device)
            u_batch = train_u[idx].to(device)
            B = a_batch.shape[0]

            coords = coord_grid.unsqueeze(0).expand(B, -1, -1, -1)

            # Output loss
            _pt = profile_start("data_forward")
            fno_input = torch.cat([coords, a_batch], dim=-1)
            u_pred = fno(fno_input)
            mse = F.mse_loss(u_pred, u_batch)
            profile_end(_pt)

            # Derivative loss via forward-mode AD
            _deriv_pt = profile_start("derivative_total")
            dir_idx = torch.randperm(K_total)[:n_jvp_per_step]
            jvp_loss = torch.tensor(0.0, device=device)

            def fno_fn(a_field):
                inp = torch.cat([coords, a_field], dim=-1)
                return fno(inp)

            for dk in dir_idx:
                # True JVP (pre-computed)
                _pt = profile_start("label_h2d")
                true_jvp_k = train_jvps[idx, dk].to(device)  # (B, Nx, Nt)
                profile_end(_pt)

                # FNO JVP via forward-mode AD
                da = delta_dirs[dk].unsqueeze(0).expand(B, -1, -1, -1).to(device)
                _pt = profile_start("online_jvp")
                _, pred_jvp_k = _func_jvp(fno_fn, (a_batch,), (da,))
                profile_end(_pt)
                pred_jvp_k = pred_jvp_k.squeeze(-1)  # (B, Nx, Nt)

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
            mse_val = mse.item()
            jvp_val = jvp_loss.item()

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

        # Validation evaluation.  Test data is handled only after training by
        # the benchmark-specific entry point.
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
            "Nx": Nx, "Nt": Nt,
            "best_val_mse": best_val_mse,
            "best_val_rel": best_val_rel,
            "best_epoch": best_epoch,
            "final_val_rel": history[-1]["val_rel"],
            "method": "DIFNO_1D",
        }
        with open(os.path.join(save_dir, "metrics.json"), "w") as fh:
            json.dump(cfg, fh, indent=2)

        print(f"\nSaved -> {save_dir}")
        print(f"  best_val_mse={best_val_mse:.4e}  "
              f"best_val_rel={best_val_rel:.4f}  "
              f"final_val_rel={history[-1]['val_rel']:.4f}")

    return history
