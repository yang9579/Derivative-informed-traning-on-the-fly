#!/usr/bin/env python
"""
Nonlinear diffusion sTCL (residual form with scaled-Fourier preconditioner).

This is the "true" sTCL formulation: no tangent solve, no ground-truth
target — just a PDE-residual loss on the FNO's predicted JVP, weighted
by a cheap FFT-based approximate inverse of the tangent operator.

Loss:
    r(a, u_theta, δa) = A_y(u_theta) · (FNO'(a)·δa)  +  F_a(u_theta) · δa
    L_sTCL        = r^T · M · r    (scalar, differentiable wrt FNO params)

where
    A_y · v            = −div(exp(a) · ∇v) + 3 u² · v        (linearised PDE op)
    F_a · δa           = −div(exp(a) · δa · ∇u)              (input-direction term)
    M = D^(-1/2) (−Δ)^(-1) D^(-1/2)                          (scaled-Fourier)

Per-step cost: one FNO JVP + one apply_Ay_batch + one shifted-DST — no
iterative solver, no ground-truth tangent target.  Target per-epoch time
is within 1.1× of DIFNO training.
"""
import os, sys, csv, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jvp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from fno2d import FNO2D
from train import evaluate_field_error, evaluate_jacobian_error
from generate_data_difno import sample_gp_matern
from runtime_profile import end as profile_end
from runtime_profile import epoch_end as profile_epoch_end
from runtime_profile import epoch_start as profile_epoch_start
from runtime_profile import start as profile_start
from fd_preconditioner import (
    apply_Ay_batch,
    apply_M,
    compute_derivatives_full,
    full_to_unk_torch,
    get_fd_matrices,
    get_inv_lam,
)


def train_stcl_preconditioned_loss(fno, train_a, train_u, val_a, val_u,
                                   steps=500, batch_size=32, lr=1e-3,
                                   lam_jvp=1.0, q=4,
                                   save_dir=None, print_every=25, device=None,
                                   direction_seed=2026):
    if device is None:
        device = next(fno.parameters()).device
    print("  sTCL form: direct residual loss r^T M r")
    train_a_t = torch.tensor(train_a, dtype=torch.float32) if isinstance(train_a, np.ndarray) else train_a
    train_u_t = torch.tensor(train_u, dtype=torch.float32) if isinstance(train_u, np.ndarray) else train_u
    val_a_t = torch.tensor(val_a, dtype=torch.float32) if isinstance(val_a, np.ndarray) else val_a
    val_u_t = torch.tensor(val_u, dtype=torch.float32) if isinstance(val_u, np.ndarray) else val_u
    if train_a_t.dim() == 3:
        train_a_t = train_a_t.unsqueeze(-1)
        train_u_t = train_u_t.unsqueeze(-1)
        val_a_t = val_a_t.unsqueeze(-1)
        val_u_t = val_u_t.unsqueeze(-1)

    N_train = train_a_t.shape[0]
    Nx, Ny = train_a_t.shape[1], train_a_t.shape[2]
    if q < 1:
        raise ValueError(f"q must be positive, got {q}")

    xs = torch.linspace(0, 1, Nx, device=device)
    ys = torch.linspace(0, 1, Ny, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    coord_grid = torch.stack([gx, gy], dim=-1)

    L2d, Dx, Dy = get_fd_matrices(Nx, Ny, device)
    inv_lam = get_inv_lam(Nx, Ny, device)

    optimizer = torch.optim.Adam(fno.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=50, factor=0.5, min_lr=1e-6)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    mse_ema, stcl_ema, ema_decay = None, None, 0.99
    history = []
    best_val_mse = float("inf")
    t0 = time.time()
    dir_counter = 0

    # Pre-sample a large direction pool so the GP cosine-basis allocation
    # (the slow part of sample_gp_matern) is done once
    pool_size = 2048
    print(f"  Pre-sampling {pool_size} GP directions ...", flush=True)
    t_pool = time.time()
    pool_np = sample_gp_matern(
        pool_size, Nx, Ny, n_modes=40, mean=0.0, seed=direction_seed)
    norms = np.linalg.norm(pool_np.reshape(pool_size, -1), axis=1)
    pool_np /= (norms[:, None, None] + 1e-10)
    pool_t = torch.tensor(pool_np, dtype=torch.float32, device=device)
    print(f"    done in {time.time() - t_pool:.1f}s")

    parameters = [
        parameter for parameter in fno.parameters()
        if parameter.requires_grad
    ]

    def compute_losses(a_batch, u_batch, deltas):
        B = a_batch.shape[0]
        coords = coord_grid.unsqueeze(0).expand(B, -1, -1, -1)

        _data_pt = profile_start('data_forward')
        fno_input = torch.cat([coords, a_batch], dim=-1)
        u_pred = fno(fno_input)
        mse = F.mse_loss(u_pred, u_batch)
        profile_end(_data_pt)

        _deriv_pt = profile_start('derivative_total')
        # r = A_y(u_pred) J_theta(a) da + F_a(u_pred) da
        # L_stcl = r^T M r. No true solution, tangent solve, or JVP target.
        a_full = a_batch.squeeze(-1)
        u_state = u_pred.detach().squeeze(-1)
        dax_full, day_full = compute_derivatives_full(a_full, Nx, Ny)
        D = torch.exp(full_to_unk_torch(a_full, Nx, Ny))
        dax = full_to_unk_torch(dax_full, Nx, Ny)
        day = full_to_unk_torch(day_full, Nx, Ny)
        u_state_unk = full_to_unk_torch(u_state, Nx, Ny)
        D_sqrt_inv = 1.0 / torch.sqrt(D + 1e-10)

        lap_u = (L2d @ u_state_unk.T).T
        ux = (Dx @ u_state_unk.T).T
        uy = (Dy @ u_state_unk.T).T

        def fno_fn(a_field):
            return fno(torch.cat([coords, a_field], dim=-1))

        stcl_loss = torch.tensor(0.0, device=device)
        for delta in deltas:
            delta_2d = delta.unsqueeze(0).expand(B, -1, -1)
            d_dax_full, d_day_full = compute_derivatives_full(
                delta_2d, Nx, Ny)
            da = full_to_unk_torch(delta_2d, Nx, Ny)
            d_da_x = full_to_unk_torch(d_dax_full, Nx, Ny)
            d_da_y = full_to_unk_torch(d_day_full, Nx, Ny)
            F_a_delta = -(
                D * da * lap_u
                + D * (dax * da + d_da_x) * ux
                + D * (day * da + d_da_y) * uy
            )

            da_full = delta.unsqueeze(0).unsqueeze(-1).expand(
                B, -1, -1, -1)
            _pt = profile_start('online_jvp')
            _, du_pred_full = jvp(
                fno_fn, (a_batch,), (da_full,))
            profile_end(_pt)
            du_pred_unk = full_to_unk_torch(
                du_pred_full.squeeze(-1), Nx, Ny)

            _pt = profile_start('pde_residual')
            A_du = apply_Ay_batch(
                du_pred_unk, D, dax, day, u_state_unk,
                L2d, Dx, Dy)
            r = A_du + F_a_delta
            profile_end(_pt)
            _pt = profile_start('preconditioner_krylov')
            Mr = apply_M(r, D_sqrt_inv, Nx, Ny, inv_lam)
            profile_end(_pt)
            stcl_loss = stcl_loss + (r * Mr).sum() / B
        profile_end(_deriv_pt)
        return mse, stcl_loss / len(deltas)

    for epoch in range(1, steps + 1):
        profile_epoch_start(epoch)
        fno.train()
        perm = torch.randperm(N_train)
        epoch_mse = 0.0
        epoch_stcl = 0.0
        n_batches = 0

        for i in range(0, N_train, batch_size):
            idx = perm[i:i + batch_size]
            pool_idx = (
                torch.arange(q, device=device) + dir_counter * q
            ) % pool_size
            dir_counter += 1
            deltas = pool_t[pool_idx]

            a_batch = train_a_t[idx].to(device)
            u_batch = train_u_t[idx].to(device)
            mse, stcl_loss = compute_losses(a_batch, u_batch, deltas)
            mse_val = mse.item()
            stcl_val = stcl_loss.item()

            # ------ adaptive scaling ------
            if mse_ema is None:
                mse_ema = mse_val
                stcl_ema = max(stcl_val, 1e-20)
            else:
                mse_ema = (
                    ema_decay * mse_ema + (1 - ema_decay) * mse_val)
                stcl_ema = (
                    ema_decay * stcl_ema
                    + (1 - ema_decay) * max(stcl_val, 1e-20))
            adaptive_scale = mse_ema / max(stcl_ema, 1e-20)
            effective_lambda = lam_jvp * adaptive_scale

            optimizer.zero_grad(set_to_none=True)
            loss = mse + effective_lambda * stcl_loss
            _pt = profile_start('backward_optimizer')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            profile_end(_pt)

            epoch_mse += mse_val
            epoch_stcl += stcl_val
            n_batches += 1

        epoch_mse /= n_batches
        epoch_stcl /= n_batches

        val_mse, val_rel, _ = evaluate_field_error(
            fno, val_a_t, val_u_t, coord_grid, device=device,
            batch_size=batch_size)
        scheduler.step(val_mse)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            if save_dir:
                torch.save(
                    fno.state_dict(), os.path.join(save_dir, "best_model.pth"))

        wall_time = time.time() - t0
        history.append({
            "epoch": epoch,
            "train_mse": epoch_mse,
            "val_mse": val_mse,
            "val_rel": val_rel,
            "rel_err": val_rel,
            "stcl_loss": epoch_stcl,
            "lr": optimizer.param_groups[0]["lr"],
            "wall_time_s": wall_time,
            "wall_time": wall_time,
        })

        if epoch % print_every == 0 or epoch <= 3:
            print(f"  [{epoch:4d}/{steps}]  mse={epoch_mse:.3e}  "
                  f"val={val_mse:.3e}  rel={val_rel:.4f}  "
                  f"stcl={epoch_stcl:.3e}  "
                  f"t={history[-1]['wall_time_s']:.0f}s", flush=True)
        profile_epoch_end(epoch, n_batches)

    if save_dir:
        torch.save(fno.state_dict(), os.path.join(save_dir, "last_model.pth"))
        with open(os.path.join(save_dir, "train_history.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=history[0].keys())
            w.writeheader()
            w.writerows(history)

    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_use",   type=int, default=512)
    ap.add_argument("--q",       type=int, default=4)
    ap.add_argument("--lam_jvp", type=float, default=1.0)
    ap.add_argument("--epochs",  type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr",      type=float, default=1e-3)
    ap.add_argument("--save_dir", type=str, default=None)
    ap.add_argument("--data_dir", type=str, default=os.path.join(SCRIPT_DIR, "data"))
    ap.add_argument("--seed",    type=int, default=0)
    ap.add_argument("--direction_seed", type=int, default=2026)
    args = ap.parse_args()
    if args.q < 1:
        ap.error("--q must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = np.load(os.path.join(args.data_dir, "train.npz"))
    va = np.load(os.path.join(args.data_dir, "val.npz"))
    te = np.load(os.path.join(args.data_dir, "test.npz"))
    a_train = tr["a"][:args.n_use]
    u_train = tr["u"][:args.n_use]
    a_val = va["a"]
    u_val = va["u"]
    a_test = te["a"]
    u_test = te["u"]
    val_jvps = va["jvps"] if "jvps" in va.files else None
    val_delta_dirs = va["delta_dirs"] if "delta_dirs" in va.files else None
    if "jvps" not in te.files or "delta_dirs" not in te.files:
        raise ValueError("test data must contain jvps and delta_dirs")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    fno = FNO2D(in_channels=3, out_channels=1, width=32,
                modes1=8, modes2=8, n_layers=4).to(device)
    print(f"FNO2D: {sum(p.numel() for p in fno.parameters()):,} parameters")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    save_dir = args.save_dir or os.path.join(
        SCRIPT_DIR, "results",
        f"N{args.n_use}_stcl_preconditioned_loss_lam{args.lam_jvp}_q{args.q}")
    print(f"\nTraining sTCL preconditioned loss | N={args.n_use} q={args.q} "
          f"epochs={args.epochs}")
    history = train_stcl_preconditioned_loss(
        fno, a_train, u_train, a_val, u_val,
        steps=args.epochs, batch_size=args.batch_size, lr=args.lr,
        lam_jvp=args.lam_jvp, q=args.q, save_dir=save_dir,
        direction_seed=args.direction_seed)

    # Held-out test evaluation uses the validation-selected checkpoint.
    # Validation alone drives the scheduler and checkpoint selection.
    checkpoint_name = "best_model.pth"
    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    fno.load_state_dict(torch.load(
        checkpoint_path, map_location=device, weights_only=True))
    nx, ny = a_test.shape[1:3]
    xs = torch.linspace(0, 1, nx, device=device)
    ys = torch.linspace(0, 1, ny, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    coords = torch.stack([gx, gy], -1)
    val_mse_ckpt, val_global_rel, val_mean_sample_rel = evaluate_field_error(
        fno, a_val, u_val, coords, device=device, batch_size=args.batch_size)
    test_mse, test_global_rel, test_mean_sample_rel = evaluate_field_error(
        fno, a_test, u_test, coords, device=device, batch_size=args.batch_size)

    print(f"\nEvaluating {checkpoint_name} test JVP Metric B ...", flush=True)
    val_jac_err = None
    if val_jvps is not None and val_delta_dirs is not None:
        val_jac_err = evaluate_jacobian_error(
            fno, a_val, val_jvps, val_delta_dirs, coords,
            device=device, batch_size=args.batch_size)
    test_jac_err = evaluate_jacobian_error(
        fno, a_test, te["jvps"], te["delta_dirs"], coords,
        device=device, n_samples=len(a_test),
        n_dirs=te["delta_dirs"].shape[0], batch_size=args.batch_size)

    best_epoch = min(history, key=lambda row: row["val_mse"])["epoch"]
    metrics = {
        "method": "scaled_fourier_preconditioned",
        "N": len(a_train),
        "seed": args.seed,
        "best_epoch": best_epoch,
        "val_mse": val_mse_ckpt,
        "val_field_global_rel": val_global_rel,
        "val_field_mean_sample_rel": val_mean_sample_rel,
        "val_jvp_rel_metricB": val_jac_err,
        "test_mse": test_mse,
        "test_field_global_rel": test_global_rel,
        "test_field_mean_sample_rel": test_mean_sample_rel,
        "test_jvp_rel_metricB": test_jac_err,
        "test_rel_err": test_mean_sample_rel,
        "test_jac_rel_err": test_jac_err,
        "wall_time_s": history[-1]["wall_time_s"],
        "total_wall_time_s": history[-1]["wall_time_s"],
        "test_checkpoint": checkpoint_name,
        "n_epochs": args.epochs,
        "n_train": len(a_train),
        "n_val": len(a_val),
        "n_test": len(a_test),
        "grid_nx": nx,
        "grid_ny": ny,
        "q": args.q,
        "direction_seed": args.direction_seed,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "lambda_jvp": args.lam_jvp,
        "width": 32,
        "modes": 8,
        "n_layers": 4,
        "jacobian_n_dirs": int(te["delta_dirs"].shape[0]),
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if device.type == "cuda" else None),
    }
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    val_jac_text = "n/a" if val_jac_err is None else f"{val_jac_err:.4f}"
    print(f"\nDone. checkpoint={checkpoint_name}  best epoch={best_epoch}  "
          f"val field={val_global_rel:.4f}  val JVP={val_jac_text}")
    print(f"  test field={test_global_rel:.4f}  test JVP={test_jac_err:.4f}")


if __name__ == "__main__":
    main()
