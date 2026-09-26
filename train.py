#!/usr/bin/env python
"""
Shared FNO2D training loop with optional sTCL
==============================================
Trains FNO2D on (input_field, output_field) pairs with:
  L = MSE(FNO(a), u) + lam_tan * (MSE_ema / sTCL_ema) * sTCL(FNO, a)

The adaptive scaling ensures sTCL is balanced with MSE regardless of
raw magnitude. lam_tan controls relative weight (1.0 = equal weight).

Usage (called from PDE-specific scripts):
  from train import train_fno
  train_fno(fno, train_a, train_u, val_a, val_u, stcl_fn=my_stcl, ...)
"""

import os
import csv
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jvp as func_jvp
from runtime_profile import end as profile_end
from runtime_profile import epoch_end as profile_epoch_end
from runtime_profile import epoch_start as profile_epoch_start
from runtime_profile import start as profile_start


def evaluate_field_error(fno, data_a, data_u, coord_grid, device=None,
                         batch_size=32):
    """Return field MSE, dataset-global relative L2, and mean sample-relative L2."""
    if device is None:
        device = next(fno.parameters()).device
    if isinstance(data_a, np.ndarray):
        data_a = torch.tensor(data_a, dtype=torch.float32)
    if isinstance(data_u, np.ndarray):
        data_u = torch.tensor(data_u, dtype=torch.float32)
    if data_a.dim() == 3:
        data_a = data_a.unsqueeze(-1)
    if data_u.dim() == 3:
        data_u = data_u.unsqueeze(-1)

    fno.eval()
    coord_grid = coord_grid.to(device)
    error_sq = 0.0
    reference_sq = 0.0
    sample_rel_sum = 0.0
    n_values = 0
    with torch.no_grad():
        for start in range(0, len(data_a), batch_size):
            a_batch = data_a[start:start + batch_size].to(device)
            u_batch = data_u[start:start + batch_size].to(device)
            coords = coord_grid.unsqueeze(0).expand(
                len(a_batch), -1, -1, -1)
            prediction = fno(torch.cat([coords, a_batch], dim=-1))
            difference = prediction - u_batch
            error_sq += difference.square().sum().item()
            reference_sq += u_batch.square().sum().item()
            sample_rel_sum += (
                difference.flatten(1).norm(dim=1)
                / (u_batch.flatten(1).norm(dim=1) + 1e-20)
            ).sum().item()
            n_values += u_batch.numel()

    mse = error_sq / n_values
    global_rel = (error_sq / max(reference_sq, 1e-40)) ** 0.5
    mean_sample_rel = sample_rel_sum / len(data_a)
    return mse, global_rel, mean_sample_rel


def train_fno(fno, train_a, train_u, val_a, val_u,
              stcl_fn=None,
              steps=500, batch_size=32, lr=1e-3,
              lam_tan=0.0, q_tan=4,
              save_dir=None, print_every=25,
              coord_grid=None, device=None):
    """
    Train FNO2D with MSE + optional sTCL (physics-informed JVP loss).

    Uses adaptive loss balancing: sTCL is scaled so that lam_tan=1.0 means
    equal weight between MSE and sTCL. This prevents the PDE residual loss
    from dominating when its raw magnitude is orders of magnitude larger.
    """
    if device is None:
        device = next(fno.parameters()).device

    # Ensure arrays are tensors
    if isinstance(train_a, np.ndarray):
        train_a = torch.tensor(train_a, dtype=torch.float32)
        train_u = torch.tensor(train_u, dtype=torch.float32)
        val_a  = torch.tensor(val_a, dtype=torch.float32)
        val_u  = torch.tensor(val_u, dtype=torch.float32)

    # Ensure 4D: (N, Nx, Ny) -> (N, Nx, Ny, 1)
    if train_a.dim() == 3:
        train_a = train_a.unsqueeze(-1)
    if train_u.dim() == 3:
        train_u = train_u.unsqueeze(-1)
    if val_a.dim() == 3:
        val_a = val_a.unsqueeze(-1)
    if val_u.dim() == 3:
        val_u = val_u.unsqueeze(-1)

    use_stcl = (stcl_fn is not None and lam_tan > 0)
    if use_stcl:
        print(f"  sTCL enabled: lam_tan={lam_tan}, q={q_tan} "
              f"(adaptive scaling)")

    N_train = train_a.shape[0]
    Nx, Ny = train_a.shape[1], train_a.shape[2]

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

    # Adaptive scaling: EMA of MSE and sTCL magnitudes
    mse_ema = None
    stcl_ema = None
    ema_decay = 0.99

    for epoch in range(1, steps + 1):
        profile_epoch_start(epoch)
        fno.train()

        perm = torch.randperm(N_train)
        epoch_mse = 0.0
        epoch_stcl = 0.0
        epoch_scale = 0.0
        n_batches = 0

        for i in range(0, N_train, batch_size):
            idx = perm[i:i + batch_size]
            a_batch = train_a[idx].to(device)
            u_batch = train_u[idx].to(device)
            B = a_batch.shape[0]

            coords = coord_grid.unsqueeze(0).expand(B, -1, -1, -1)

            # Adaptive scale from the EMAs of previous steps.
            cur_adaptive_scale = mse_ema / max(stcl_ema, 1e-10) if mse_ema is not None else 1.0

            _data_pt = profile_start('data_forward')
            fno_input = torch.cat([coords, a_batch], dim=-1)
            u_pred = fno(fno_input)
            mse = F.mse_loss(u_pred, u_batch)
            profile_end(_data_pt)
            stcl_loss = torch.tensor(0.0, device=device)
            if use_stcl:
                _deriv_pt = profile_start('derivative_total')
                stcl_loss = stcl_fn(fno, a_batch, coord_grid, q_tan)
                profile_end(_deriv_pt)
                loss = mse + lam_tan * cur_adaptive_scale * stcl_loss
            else:
                loss = mse

            _pt = profile_start('backward_optimizer')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fno.parameters(), 1.0)
            optimizer.step()
            profile_end(_pt)
            mse_val = mse.item()
            stcl_val = stcl_loss.item()

            # Update EMAs after the step
            if use_stcl:
                if mse_ema is None:
                    mse_ema = mse_val
                    stcl_ema = max(stcl_val, 1e-10)
                else:
                    mse_ema = ema_decay * mse_ema + (1 - ema_decay) * mse_val
                    stcl_ema = ema_decay * stcl_ema + (1 - ema_decay) * max(stcl_val, 1e-10)
            adaptive_scale = cur_adaptive_scale

            epoch_mse += mse_val
            epoch_stcl += stcl_val
            epoch_scale += adaptive_scale
            n_batches += 1

        epoch_mse /= n_batches
        epoch_stcl /= n_batches
        epoch_scale /= max(n_batches, 1)

        # Validation evaluation.  The test split is evaluated only after
        # training by the benchmark-specific entry point.
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
            "stcl_raw": epoch_stcl,
            "adapt_scale": epoch_scale,
            "lr": optimizer.param_groups[0]["lr"],
            "wall_time": time.time() - t0,
        }
        history.append(record)

        if epoch % print_every == 0 or epoch <= 3:
            print(f"  [{epoch:4d}/{steps}]  "
                  f"mse={epoch_mse:.4e}  val={val_mse:.4e}  "
                  f"rel={val_rel:.4f}  stcl={epoch_stcl:.4e}  "
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
            "lam_tan": lam_tan,
            "q_tan": q_tan,
            "N_train": N_train,
            "Nx": Nx,
            "Ny": Ny,
            "best_val_mse": best_val_mse,
            "best_val_rel": best_val_rel,
            "best_epoch": best_epoch,
            "final_val_rel": history[-1]["val_rel"],
        }
        with open(os.path.join(save_dir, "metrics.json"), "w") as fh:
            json.dump(cfg, fh, indent=2)

        print(f"\nSaved -> {save_dir}")
        print(f"  best_val_mse={best_val_mse:.4e}  "
              f"best_val_rel={best_val_rel:.4f}  "
              f"final_val_rel={history[-1]['val_rel']:.4f}")

    return history


def evaluate_jacobian_error(fno, test_a, true_jvps, delta_a_dirs,
                            coord_grid, device=None, n_samples=None,
                            n_dirs=None, batch_size=32):
    """Evaluate Metric B: mean per-field/per-direction JVP relative error."""
    if device is None:
        device = next(fno.parameters()).device

    if isinstance(test_a, np.ndarray):
        test_a = torch.tensor(test_a, dtype=torch.float32)
    if isinstance(true_jvps, np.ndarray):
        true_jvps = torch.tensor(true_jvps, dtype=torch.float32)
    if isinstance(delta_a_dirs, np.ndarray):
        delta_a_dirs = torch.tensor(delta_a_dirs, dtype=torch.float32)

    if test_a.dim() == 3:
        test_a = test_a.unsqueeze(-1)

    fno.eval()
    coord_grid = coord_grid.to(device)
    sample_count = test_a.shape[0]
    direction_count = delta_a_dirs.shape[0]
    if n_samples is not None:
        sample_count = min(sample_count, n_samples)
    if n_dirs is not None:
        direction_count = min(direction_count, n_dirs)
    if true_jvps.shape[0] < sample_count:
        raise ValueError("true JVP bank has fewer fields than requested")
    if true_jvps.shape[1] < direction_count:
        raise ValueError("true JVP bank has fewer directions than requested")

    error_sum = 0.0
    error_count = 0
    with torch.no_grad():
        for d in range(direction_count):
            direction = delta_a_dirs[d:d + 1].unsqueeze(-1).to(device)
            for batch_start in range(0, sample_count, batch_size):
                batch_stop = min(batch_start + batch_size, sample_count)
                a_batch = test_a[batch_start:batch_stop].to(device)
                coords = coord_grid.unsqueeze(0).expand(
                    len(a_batch), -1, -1, -1)
                da_batch = direction.expand(len(a_batch), -1, -1, -1)

                def fno_fn(a_field):
                    return fno(torch.cat([coords, a_field], dim=-1))

                _, fno_jvp = func_jvp(
                    fno_fn, (a_batch,), (da_batch,))
                target = true_jvps[batch_start:batch_stop, d].to(device)
                relative = (
                    (fno_jvp.squeeze(-1) - target).flatten(1).norm(dim=1)
                    / (target.flatten(1).norm(dim=1) + 1e-10)
                )
                error_sum += relative.sum().item()
                error_count += len(a_batch)

    return error_sum / error_count
