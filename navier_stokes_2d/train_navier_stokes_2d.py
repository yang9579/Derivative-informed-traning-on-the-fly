#!/usr/bin/env python
"""Train naive FNO, DIFNO, and sTCL for the state map U -> (Y, P)."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jvp as func_jvp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))

try:
    from .fno_navier_stokes_2d import FNOAONN
    from .stcl_navier_stokes_2d import stcl_loss
except ImportError:  # Support direct script execution.
    from fno_navier_stokes_2d import FNOAONN
    from stcl_navier_stokes_2d import stcl_loss

from runtime_profile import end as profile_end
from runtime_profile import epoch_end as profile_epoch_end
from runtime_profile import epoch_start as profile_epoch_start
from runtime_profile import start as profile_start


def load_split(data_dir: str, split: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    z = np.load(os.path.join(data_dir, f"{split}.npz"))
    u = torch.tensor(z["U"], dtype=torch.float32)
    y = torch.tensor(z["Y"], dtype=torch.float32)
    p = torch.tensor(z["P"], dtype=torch.float32) if "P" in z.files else None
    return u, y, p, int(z["gy"]), int(z["gx"])


def load_jvps(data_dir: str, split: str, required: bool = False):
    path = os.path.join(data_dir, f"{split}_jvps.npz")
    if not os.path.exists(path):
        shard_dir = Path(data_dir) / f"{split}_jvps_shards"
        shard_paths = sorted(shard_dir.glob("shard_*.npz")) if shard_dir.is_dir() else []
        if shard_paths:
            metadata = []
            delta_dirs = None
            for shard_path in shard_paths:
                z = np.load(shard_path)
                start = int(z["sample_start"])
                stop = int(z["sample_stop"])
                source_n = int(z["source_n_samples"])
                dirs = z["delta_dirs"]
                if delta_dirs is None:
                    delta_dirs = torch.tensor(dirs, dtype=torch.float32)
                elif dirs.shape != tuple(delta_dirs.shape) or not np.allclose(dirs, delta_dirs.numpy()):
                    raise ValueError(f"direction bank mismatch in {shard_path}")
                if z["jvps"].shape[0] != stop - start:
                    raise ValueError(f"sample-range mismatch in {shard_path}")
                metadata.append((start, stop, source_n, shard_path))
            metadata.sort()
            source_ns = {m[2] for m in metadata}
            if len(source_ns) != 1:
                raise ValueError(f"inconsistent source sizes in {shard_dir}")
            source_n = source_ns.pop()
            expected = 0
            for start, stop, _, shard_path in metadata:
                if start != expected:
                    raise ValueError(f"non-contiguous shards at {shard_path}: expected start {expected}")
                expected = stop
            if expected != source_n:
                raise ValueError(f"incomplete shards in {shard_dir}: cover {expected}/{source_n}")
            first = np.load(metadata[0][3])
            tail = first["jvps"].shape[1:]
            jvps = torch.empty((source_n,) + tail, dtype=torch.float32)
            for start, stop, _, shard_path in metadata:
                z = np.load(shard_path)
                jvps[start:stop].copy_(torch.from_numpy(z["jvps"]))
            return delta_dirs, jvps
        if required:
            raise FileNotFoundError(path)
        return None, None
    z = np.load(path)
    delta_dirs = torch.tensor(z["delta_dirs"], dtype=torch.float32)   # (r, 2, gy, gx) shared
    jvps = torch.tensor(z["jvps"], dtype=torch.float32)               # (N, r, 2, gy, gx) labels
    return delta_dirs, jvps


@torch.no_grad()
def eval_solution(model: FNOAONN, u: torch.Tensor, y: torch.Tensor, device: torch.device, bs: int = 64):
    was_training = model.training
    model.eval()
    rels = []
    mse_sum = 0.0
    nb = 0
    for i in range(0, len(u), bs):
        ub = u[i : i + bs].to(device)
        yb = y[i : i + bs].to(device)
        pred = model(ub)
        mse_sum += float(F.mse_loss(pred, yb).item())
        nb += 1
        rel = (pred - yb).flatten(1).norm(dim=1) / (yb.flatten(1).norm(dim=1) + 1e-20)
        rels.extend(rel.cpu().tolist())
    model.train(was_training)
    return {"state_rel": float(np.mean(rels)), "mse": mse_sum / max(nb, 1)}


@torch.no_grad()
def eval_pressure(model: FNOAONN, u: torch.Tensor, p, device: torch.device, bs: int = 64):
    """Mean-subtracted (gauge-invariant) relative L2 error on the pressure field."""
    if p is None:
        return None
    was_training = model.training
    model.eval()
    rels = []
    for i in range(0, len(u), bs):
        ub = u[i : i + bs].to(device)
        pb = p[i : i + bs].to(device)
        pr = model.state_uvp(ub)[:, 2]
        pr = pr - pr.mean(dim=(1, 2), keepdim=True)
        pt = pb - pb.mean(dim=(1, 2), keepdim=True)
        rel = (pr - pt).flatten(1).norm(dim=1) / (pt.flatten(1).norm(dim=1) + 1e-20)
        rels.extend(rel.cpu().tolist())
    model.train(was_training)
    return float(np.mean(rels))


def eval_jvp(
    model: FNOAONN,
    u: torch.Tensor,
    delta_dirs: torch.Tensor,   # (r, 2, gy, gx) shared control-space directions
    jvps: torch.Tensor,         # (N, r, 2, gy, gx) per-sample tangent labels
    device: torch.device,
    bs: int = 32,
):
    was_training = model.training
    model.eval()
    errs = []
    n_dirs = jvps.shape[1]
    for d in range(n_dirs):
        for i in range(0, len(u), bs):
            ub = u[i : i + bs].to(device)
            du = delta_dirs[d].unsqueeze(0).expand(ub.shape[0], -1, -1, -1).to(device)
            true = jvps[i : i + bs, d].to(device)
            with torch.enable_grad():
                _, pred = func_jvp(model, (ub,), (du,))
            rel = (pred - true).flatten(1).norm(dim=1) / (true.flatten(1).norm(dim=1) + 1e-20)
            errs.extend(rel.detach().cpu().tolist())
    model.train(was_training)
    return {"jvp_state_rel": float(np.mean(errs))}


def save_history(path: str, history) -> None:
    if not history:
        return
    keys = []                                   # union of keys across rows (robust to resume)
    for row in history:
        for k in row:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, restval="")
        writer.writeheader()
        writer.writerows(history)


def residual_weight_dict(args) -> Dict[str, float]:
    return {
        "state": args.rw_state,
        "div_y": args.rw_div_y,
        "boundary": args.rw_boundary,
    }


def load_state(model: FNOAONN, path: str, device: torch.device) -> None:
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)


def train(args, device: torch.device):
    train_u, train_y, train_p, gy, gx = load_split(args.data_dir, "train")
    val_u, val_y, val_p, _, _ = load_split(args.data_dir, "val")
    test_u, test_y, test_p, _, _ = load_split(args.data_dir, "test")
    train_u = train_u[: args.n_use]
    train_y = train_y[: args.n_use]
    train_p = train_p[: args.n_use] if train_p is not None else None
    n_train = len(train_u)

    train_dirs = train_jvps = None
    # DIFNO uses the offline label bank (shared directions + per-sample tangent labels).
    # sTCL is label-free: it samples control-space directions ONLINE and reads no bank.
    if args.method == "difno":
        train_dirs, train_jvps = load_jvps(args.data_dir, "train", required=True)
        train_jvps = train_jvps[: args.n_use]
        print(f"  DIFNO label bank: shared dirs{tuple(train_dirs.shape)} "
              f"labels{tuple(train_jvps.shape)}", flush=True)

    # Matched directions + tangent labels for Metric-B Jacobian evaluation (val/test).
    val_dirs, val_jvps = load_jvps(args.data_dir, "val", required=False)
    test_dirs, test_jvps = load_jvps(args.data_dir, "test", required=False)

    model = FNOAONN(gy=gy, gx=gx, width=args.width, modes=args.modes, layers=args.layers).to(device)
    # Viscosity from the dataset (Re = vel_amp/mu); the sTCL residual uses model.mu.
    _meta = np.load(os.path.join(args.data_dir, "train.npz"))
    mu_ds = float(_meta["mu_fixed"])
    model.mu = mu_ds
    # Direction prior (so online sTCL matches the DIFNO/eval bank + input distribution).
    dir_mode = str(_meta["direction_mode"]) if "direction_mode" in _meta.files else "sine"
    dir_skmax = int(_meta["stream_kmax"]) if "stream_kmax" in _meta.files else 6
    dir_sdecay = float(_meta["stream_decay"]) if "stream_decay" in _meta.files else 1.5
    # Standardize the control input channels using train-set statistics.
    u_mean = train_u.mean(dim=(0, 2, 3))
    u_std = train_u.std(dim=(0, 2, 3))
    model.set_input_norm(u_mean.to(device), u_std.to(device))
    if args.init_from:
        load_state(model, args.init_from, device)   # warm-start from a prior checkpoint
        print(f"  warm-started weights from {args.init_from}", flush=True)
    print(
        f"FNO params={model.count_params()} grid={gy}x{gx} mu={model.mu} "
        f"u_mean={u_mean.tolist()} u_std={u_std.tolist()}",
        flush=True,
    )

    train_u_d = train_u.to(device)
    train_state_device = train_y.to(device)
    train_p_device = train_p.to(device) if train_p is not None else None
    use_pressure = train_p_device is not None and args.w_p > 0.0
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", patience=args.lr_patience, factor=args.lr_factor, min_lr=args.min_lr
    )

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    method = args.method
    best_val = float("inf")
    best_epoch = None
    history = []
    mse_ema = None
    deriv_ema = None
    blk_ema = {"mom": None, "div": None, "bc": None}   # per-block EMA for sTCL loss balancing
    ema_decay = 0.99
    t0 = time.time()
    rweights = residual_weight_dict(args)

    # Full-state resume (model + optimizer + scheduler + epoch + EMA + history).
    start_epoch = 0
    ckpt_path = os.path.join(args.save_dir, "ckpt.pth")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch = int(ck["epoch"])
        best_val = ck["best_val"]
        best_epoch = ck.get("best_epoch")
        history = ck["history"]
        mse_ema = ck.get("mse_ema")
        deriv_ema = ck.get("deriv_ema")
        blk_ema = ck.get("blk_ema", blk_ema)
        if "torch_rng_state" in ck:
            # The checkpoint is loaded with map_location=device so model and
            # optimizer tensors are ready for CUDA.  CPU RNG state must stay a
            # CPU ByteTensor, however, and CUDA RNG setters accept CPU state
            # tensors as well.
            torch.set_rng_state(ck["torch_rng_state"].cpu())
        if torch.cuda.is_available() and ck.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in ck["cuda_rng_state"]]
            )
        if "numpy_rng_state" in ck:
            np.random.set_state(ck["numpy_rng_state"])
        print(f"  resumed from {ckpt_path} at epoch {start_epoch}", flush=True)

    def save_ckpt(ep):
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
             "epoch": ep, "best_val": best_val, "best_epoch": best_epoch, "history": history,
             "mse_ema": mse_ema, "deriv_ema": deriv_ema, "blk_ema": blk_ema,
             "torch_rng_state": torch.get_rng_state(),
             "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
             "numpy_rng_state": np.random.get_state()},
            ckpt_path,
        )

    for ep in range(start_epoch + 1, args.epochs + 1):
        profile_epoch_start(ep)
        model.train()
        perm = torch.randperm(n_train, device=device)
        acc = {k: 0.0 for k in ("mse", "mse_uv", "mse_p", "deriv", "deriv_scaled",
                                "mom", "div", "bc", "jvp", "scale")}
        nb = 0
        for i in range(0, n_train, args.bs):
            idx = perm[i : i + args.bs]
            ub = train_u_d[idx]
            yb = train_state_device[idx]
            _pt = profile_start('data_forward')
            if use_pressure:
                pred3 = model.state_uvp(ub)                       # (B,3,gy,gx)
                mse_uv = F.mse_loss(pred3[:, :2], yb)
                pp = pred3[:, 2] - pred3[:, 2].mean(dim=(1, 2), keepdim=True)  # gauge-invariant
                pt = train_p_device[idx] - train_p_device[idx].mean(dim=(1, 2), keepdim=True)
                mse_p = F.mse_loss(pp, pt)
                mse = mse_uv + args.w_p * mse_p
                mse_uv_v, mse_p_v = float(mse_uv.detach()), float(mse_p.detach())
            else:
                mse = F.mse_loss(model(ub), yb)
                mse_uv_v, mse_p_v = float(mse.detach()), 0.0
            profile_end(_pt)

            parts = {}
            scale = 0.0
            if method == "naive":
                loss = mse
                deriv_loss = ub.new_zeros(())
            elif method == "difno":
                _deriv_pt = profile_start('derivative_total')
                # Offline labels: match DG_theta[delta_u] to the tangent-solve label delta_y
                # for q shared directions per batch.
                idx_cpu = idx.detach().cpu()
                n_dirs = train_jvps.shape[1]
                q = min(args.q, n_dirs)
                dirs = torch.randperm(n_dirs)[:q].tolist()
                deriv_loss = ub.new_zeros(())
                for d in dirs:
                    du = train_dirs[d].unsqueeze(0).expand(ub.shape[0], -1, -1, -1).to(device)
                    _pt = profile_start('label_h2d')
                    true_jvp = train_jvps[idx_cpu, d].to(device)
                    profile_end(_pt)
                    _pt = profile_start('online_jvp')
                    _, pred_jvp = func_jvp(model, (ub,), (du,))
                    profile_end(_pt)
                    deriv_loss = deriv_loss + F.mse_loss(pred_jvp, true_jvp)
                deriv_loss = deriv_loss / q
                profile_end(_deriv_pt)
                parts = {"jvp": float(deriv_loss.detach())}
            elif method == "stcl":
                # Label-free ONLINE sTCL: sample q fresh control-space directions each step
                # (same prior as the DIFNO/eval bank) and drive the tangent residual to zero.
                _deriv_pt = profile_start('derivative_total')
                deriv_loss, sparts = stcl_loss(
                    model, ub,
                    n_sketch=args.q, n_modes=args.sketch_modes,
                    direction_mode=dir_mode, stream_kmax=dir_skmax, stream_decay=dir_sdecay,
                    residual_weights=(None if args.balance_stcl else rweights),
                    precond=args.precond, precond_div=args.precond_div,
                    precond_tau=args.precond_tau, oseen_speed=args.oseen_speed, oseen_power=args.oseen_power, return_parts=True,
                )
                profile_end(_deriv_pt)
                if args.balance_stcl:
                    # Normalize each block by its own running scale so momentum / continuity /
                    # no-slip contribute in a fixed ratio (bw_*) regardless of the preconditioner
                    # scale (raw momentum can be 100-1000x the div/bc blocks, or vice-versa).
                    tw = {"mom": args.bw_mom, "div": args.bw_div, "bc": args.bw_bc}
                    deriv_loss = ub.new_zeros(())
                    for k in ("mom", "div", "bc"):
                        v = float(sparts[k].detach())
                        blk_ema[k] = (max(v, 1e-30) if blk_ema[k] is None
                                      else ema_decay * blk_ema[k] + (1 - ema_decay) * max(v, 1e-30))
                        deriv_loss = deriv_loss + tw[k] * sparts[k] / blk_ema[k]
                parts = {k: float(v.detach()) for k, v in sparts.items()}
            else:
                raise ValueError(method)

            if method != "naive":
                dval = max(float(deriv_loss.detach().item()), 1e-20)
                if mse_ema is None:
                    mse_ema = float(mse.detach().item())
                    deriv_ema = dval
                else:
                    mse_ema = ema_decay * mse_ema + (1.0 - ema_decay) * float(mse.detach().item())
                    deriv_ema = ema_decay * deriv_ema + (1.0 - ema_decay) * dval
                scale = args.jvp_weight * (mse_ema / max(deriv_ema, 1e-20))
                loss = mse + scale * deriv_loss

            _pt = profile_start('backward_optimizer')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            profile_end(_pt)

            dloss_v = float(deriv_loss.detach().item())
            acc["mse"] += float(mse.detach().item())
            acc["mse_uv"] += mse_uv_v
            acc["mse_p"] += mse_p_v
            acc["deriv"] += dloss_v
            acc["deriv_scaled"] += float(scale) * dloss_v
            acc["scale"] += float(scale)
            acc["mom"] += parts.get("mom", 0.0)
            acc["div"] += parts.get("div", 0.0)
            acc["bc"] += parts.get("bc", 0.0)
            acc["jvp"] += parts.get("jvp", 0.0)
            nb += 1

        for k in acc:
            acc[k] /= max(nb, 1)
        ep_mse = acc["mse"]
        ep_deriv = acc["deriv"]
        val = eval_solution(model, val_u, val_y, device, bs=args.eval_bs)
        sched.step(val["state_rel"])
        if val["state_rel"] < best_val:
            best_val = val["state_rel"]
            best_epoch = ep
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))

        vjvp = None
        if args.jvp_every > 0 and val_dirs is not None and (ep % args.jvp_every == 0 or ep == args.epochs):
            ns = min(getattr(args, "jvp_eval_n", 64), len(val_u))
            vjvp = eval_jvp(model, val_u[:ns], val_dirs, val_jvps[:ns], device,
                            bs=max(1, args.eval_bs // 2))["jvp_state_rel"]

        row = {
            "epoch": ep,
            "val_jvp_rel": vjvp,                 # periodic eval JVP error (val subset), if --jvp_every>0
            "train_mse": ep_mse,                 # total data loss (uv + w_p*p)
            "train_mse_uv": acc["mse_uv"],       # velocity data MSE
            "train_mse_p": acc["mse_p"],         # pressure data MSE (mean-subtracted)
            "train_deriv_loss": ep_deriv,        # raw derivative loss (unweighted)
            "train_deriv_scaled": acc["deriv_scaled"],  # EMA-scaled contribution to total loss
            "ema_scale": acc["scale"],           # jvp_weight * EMA(data)/EMA(deriv)
            "train_stcl_mom": acc["mom"],        # sTCL momentum block (H^-1 or raw)
            "train_stcl_div": acc["div"],        # sTCL continuity block
            "train_stcl_bc": acc["bc"],          # sTCL no-slip block
            "train_difno_jvp": acc["jvp"],       # DIFNO JVP-matching loss
            "val_mse": val["mse"],
            "val_state_rel": val["state_rel"],
            "lr": opt.param_groups[0]["lr"],
            "wall_time": time.time() - t0,
        }
        history.append(row)
        if ep % args.ckpt_every == 0 or ep == args.epochs:
            torch.save(model.state_dict(), os.path.join(args.save_dir, "last_model.pth"))
            save_ckpt(ep)
            save_history(os.path.join(args.save_dir, "train_history.csv"), history)
        if ep <= 3 or ep % args.print_every == 0:
            print(
                f"  [{ep:4d}/{args.epochs}] mse={ep_mse:.3e} deriv={ep_deriv:.3e} "
                f"val_y_rel={val['state_rel']:.4f}",
                flush=True,
            )
        profile_epoch_end(ep, nb)

    torch.save(model.state_dict(), os.path.join(args.save_dir, "last_model.pth"))
    save_history(os.path.join(args.save_dir, "train_history.csv"), history)

    last = eval_solution(model, test_u, test_y, device, bs=args.eval_bs)
    last["p_rel"] = eval_pressure(model, test_u, test_p, device, bs=args.eval_bs)
    last_val = eval_solution(model, val_u, val_y, device, bs=args.eval_bs)
    if test_jvps is not None:
        last.update(eval_jvp(model, test_u, test_dirs, test_jvps, device, bs=max(1, args.eval_bs // 2)))
    if val_jvps is not None:
        last_val.update(eval_jvp(model, val_u, val_dirs, val_jvps, device, bs=max(1, args.eval_bs // 2)))

    best_path = os.path.join(args.save_dir, "best_model.pth")
    best = None
    best_val_eval = None
    if os.path.exists(best_path):
        load_state(model, best_path, device)
        best = eval_solution(model, test_u, test_y, device, bs=args.eval_bs)
        best["p_rel"] = eval_pressure(model, test_u, test_p, device, bs=args.eval_bs)
        best_val_eval = eval_solution(model, val_u, val_y, device, bs=args.eval_bs)
        if test_jvps is not None:
            best.update(eval_jvp(model, test_u, test_dirs, test_jvps, device, bs=max(1, args.eval_bs // 2)))
        if val_jvps is not None:
            best_val_eval.update(eval_jvp(model, val_u, val_dirs, val_jvps, device, bs=max(1, args.eval_bs // 2)))

    metrics = {
        "method": method,
        "n_train": n_train,
        "seed": args.seed,
        "best_val_state_rel": best_val,
        "best_epoch": best_epoch,
        "last": last,
        "last_val": last_val,
        "best": best,
        "best_val": best_val_eval,
        "total_wall_time_s": history[-1]["wall_time"] if history else 0.0,
    }
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    best_msg = "NA" if best is None else f"{best['state_rel']:.4f}"
    jvp_msg = "NA" if "jvp_state_rel" not in last else f"{last['jvp_state_rel']:.4f}"
    print(
        f"[{method} N={n_train} seed={args.seed}] "
        f"LAST test_y_rel={last['state_rel']:.4f} test_jvp_y={jvp_msg} "
        f"BEST test_y_rel={best_msg}",
        flush=True,
    )
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--method", required=True, choices=["naive", "difno", "stcl"])
    ap.add_argument("--n_use", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5000, help="training epochs")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--eval_bs", type=int, default=64)
    ap.add_argument("--jvp_every", type=int, default=50, help="log eval JVP error every N epochs (0=off)")
    ap.add_argument("--jvp_eval_n", type=int, default=128, help="val subset size for periodic JVP eval")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_factor", type=float, default=0.5)
    ap.add_argument("--lr_patience", type=int, default=50)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--jvp_weight", type=float, default=None,
                    help="Paper-selected EMA derivative weight; inferred from method and n_use when omitted")
    ap.add_argument("--w_p", type=float, default=0.05, help="pressure weight in the data loss (0 disables)")
    ap.add_argument("--q", type=int, default=4, help="DIFNO directions or sTCL sketches per batch")
    ap.add_argument("--sketch_modes", type=int, default=8)
    ap.add_argument("--precond", choices=["hinv", "hinv2", "none", "leray", "leray_raw", "leray_oseen", "oseen", "dst", "leray_dst"], default="leray_oseen",
                    help="sTCL momentum metric; see README for formulas")
    ap.add_argument("--precond_div", choices=["hinv", "hinv2", "none"], default="none",
                    help="sTCL continuity-block conditioning (default none/raw L2)")
    ap.add_argument("--precond_tau", type=float, default=0.005,
                    help="dimensionless spectral shift")
    ap.add_argument("--oseen_speed", type=float, default=3.0,
                    help="characteristic advection speed for Oseen metrics")
    ap.add_argument("--oseen_power", type=float, default=1.0,
                    help="power of the Oseen residual metric symbol")
    ap.add_argument("--rw_state", type=float, default=1.0,
                    help="sTCL momentum-block weight")
    ap.add_argument("--rw_div_y", type=float, default=0.05,
                    help="sTCL continuity-block weight")
    ap.add_argument("--rw_boundary", type=float, default=1.0,
                    help="sTCL no-slip boundary-block weight")
    # sTCL loss balancing: per-block EMA normalization so momentum/continuity/no-slip
    # contribute in the ratio bw_mom:bw_div:bw_bc regardless of preconditioner scale.
    ap.add_argument("--balance_stcl", type=int, default=0,
                    help="1=EMA-balance sTCL blocks. WARNING: normalizing tiny div/bc blocks to "
                         "equal weight amplifies their gradients ~1/scale and destabilizes training; "
                         "the natural imbalance is protective. Default off.")
    ap.add_argument("--bw_mom", type=float, default=1.0)
    ap.add_argument("--bw_div", type=float, default=1.0)
    ap.add_argument("--bw_bc", type=float, default=1.0)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--data_dir", default=os.path.join(SCRIPT_DIR, "data"))
    ap.add_argument("--save_dir", default=None)
    ap.add_argument("--init_from", default=None, help="warm-start model weights from this .pth checkpoint")
    ap.add_argument("--resume", action="store_true", help="resume full state (model+opt+sched+epoch) from save_dir/ckpt.pth if present")
    ap.add_argument("--ckpt_every", type=int, default=100,
                    help="checkpoint interval in epochs")
    ap.add_argument("--print_every", type=int, default=100,
                    help="training-log interval in epochs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.jvp_weight is None:
        if args.method == "naive":
            args.jvp_weight = 0.0
        else:
            paper_lambdas = (
                {512: 0.5, 1024: 1.0, 2048: 0.5}
                if args.method == "difno"
                else {512: 0.25, 1024: 0.25, 2048: 0.5}
            )
            if args.n_use not in paper_lambdas:
                raise ValueError(
                    f"No paper-selected Navier--Stokes lambda for N={args.n_use}; "
                    "pass --jvp_weight explicitly."
                )
            args.jvp_weight = paper_lambdas[args.n_use]
            print(f"Using paper-selected lambda={args.jvp_weight} for {args.method}, N={args.n_use}")

    if args.save_dir is None:
        args.save_dir = os.path.join(SCRIPT_DIR, f"results_{args.method}_N{args.n_use}_seed{args.seed}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    print(
        f"\nTraining state PDE {args.method} | N={args.n_use} "
        f"epochs={args.epochs} device={device}",
        flush=True,
    )
    train(args, device)


if __name__ == "__main__":
    main()
