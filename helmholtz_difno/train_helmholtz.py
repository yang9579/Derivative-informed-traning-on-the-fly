#!/usr/bin/env python
"""
Unified training harness for the Helmholtz benchmark (DIFNO Sec 6.3 setup).

All three methods share the same FNO, data, optimiser, and schedule — they
differ only in the tangent target:

  --method fno       : MSE-only baseline (no JVPs)
  --method difno     : offline DI — precomputed JVPs loaded from disk
  --method fminres   : sTCL — online GPU-batched MINRES with shifted-Laplacian
                        Fourier preconditioner (default 5 iters, shift 0.5)

Usage:
  python train_helmholtz.py --method fno --n_use 512 --epochs 2000
  python train_helmholtz.py --method difno --n_use 512 --epochs 2000
  python train_helmholtz.py --method fminres --n_use 512 --q 4 --epochs 2000
"""
import os, sys, time, csv, json, argparse
import numpy as np
import scipy.sparse.linalg as spla
import torch
import torch.nn.functional as F
from torch.func import jvp as _func_jvp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from fno2d import FNO2D
from generate_data_difno import (
    sample_gp_matern, KAPPA,
    build_fd_laplacian_dirichlet, build_helmholtz_operator, solve_tangent
)
from minres_solver import (
    apply_A_batch, get_inv_shifted_lam, shifted_fourier_solve, minres_fourier
)
from runtime_profile import end as profile_end
from runtime_profile import epoch_end as profile_epoch_end
from runtime_profile import epoch_start as profile_epoch_start
from runtime_profile import start as profile_start


def jvp(fn, primals, tangents):
    return _func_jvp(fn, primals, tangents)


# ---------- helper: full grid <-> interior, batched on torch ----------
def full_to_unk_torch(field, nx, ny):
    """field: (B, nx, ny) -> (B, (ny-2)*(nx-2))  in (y, x) row-major order."""
    return field[:, 1:-1, 1:-1].permute(0, 2, 1).reshape(field.shape[0], -1)


def unk_to_full_torch(unk, nx, ny):
    B = unk.shape[0]
    out = torch.zeros(B, nx, ny, device=unk.device, dtype=unk.dtype)
    out[:, 1:-1, 1:-1] = unk.view(B, ny - 2, nx - 2).permute(0, 2, 1)
    return out


def exp2a_unk(a_batch, nx, ny):
    """a_batch: (B, nx, ny, 1) -> interior D = exp(2a) shape (B, Nint)"""
    a = a_batch.squeeze(-1)
    return full_to_unk_torch(torch.exp(2.0 * a), nx, ny)


# ---------- evaluation helpers ----------
def evaluate_solution(fno, test_a, test_u, coords, device):
    """Return MSE and the mean per-sample relative L2 error

        bar{E}_u = mean_i ||u_theta(a_i) - u_i|| / ||u_i||.
    """
    fno.eval()
    with torch.no_grad():
        a = test_a.to(device)
        u = test_u.to(device)
        B = a.shape[0]
        c = coords.unsqueeze(0).expand(B, -1, -1, -1)
        inp = torch.cat([c, a], dim=-1)
        pred = fno(inp)
        mse = F.mse_loss(pred, u).item()
        rel = ((pred - u).flatten(1).norm(dim=1)
               / (u.flatten(1).norm(dim=1) + 1e-20)).mean().item()
    fno.train()
    return mse, rel


def evaluate_jacobian(fno, test_a, test_jvps, delta_dirs, coords, device):
    """Metric B Jacobian relative error: per-(sample, direction) average

        bar{E} = mean_{i,d} ||J(a_i)v_d - J_theta(a_i)v_d|| / ||J(a_i)v_d||.

    This matches the paper (Tab. 3 caption) and ``train.evaluate_jacobian_error``
    used for the other PDEs.  It is NOT the aggregate Frobenius-norm ratio
    (norm over the whole test batch per direction), which inflates the Helmholtz
    Jacobian error by ~2-3x relative to the reported numbers.
    """
    fno.eval()
    with torch.no_grad():
        a = test_a.to(device)
        B = a.shape[0]
        n_dirs = test_jvps.shape[1]
        c = coords.unsqueeze(0).expand(B, -1, -1, -1)
        errs = []

        def fno_fn(x):
            return fno(torch.cat([c, x], dim=-1))

        for d in range(n_dirs):
            delta = torch.tensor(delta_dirs[d], dtype=torch.float32, device=device)
            delta = delta.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1)
            _, pred_jvp = jvp(fno_fn, (a,), (delta,))
            pred_jvp = pred_jvp.squeeze(-1)
            true_jvp = torch.tensor(test_jvps[:, d], dtype=torch.float32, device=device)
            # per-sample relative error, then averaged over (sample, direction)
            per = (pred_jvp - true_jvp).flatten(1).norm(dim=1) / \
                  (true_jvp.flatten(1).norm(dim=1) + 1e-10)
            errs.extend(per.cpu().tolist())
    fno.train()
    return float(np.mean(errs))


# ---------- sTCL target computation for FMINRES ----------
def compute_fminres_targets(a_batch, u_batch, deltas, q, nx, ny, dx, dy,
                             inv_lam, n_iter=5):
    """Returns (B, q, nx, ny) tensor of tangent solutions via GPU MINRES."""
    device = a_batch.device
    B = a_batch.shape[0]

    D_batch = exp2a_unk(a_batch, nx, ny)
    u_unk = full_to_unk_torch(u_batch.squeeze(-1), nx, ny)

    def A_fn(w):
        return apply_A_batch(w, D_batch, nx, ny, dx, dy)

    def M_inv(f):
        return shifted_fourier_solve(f, nx, ny, inv_lam)

    targets = torch.zeros(B, q, nx, ny, device=device)
    with torch.no_grad():
        for k in range(q):
            delta = deltas[k]
            delta_b = delta.unsqueeze(0).expand(B, -1, -1)  # (B, nx, ny)
            _pt = profile_start("pde_residual")
            rhs_full = 2.0 * (KAPPA**2) * torch.exp(2.0 * a_batch.squeeze(-1)) \
                       * u_batch.squeeze(-1) * delta_b
            rhs = full_to_unk_torch(rhs_full, nx, ny)
            profile_end(_pt)
            _pt = profile_start("preconditioner_krylov")
            du_unk = minres_fourier(A_fn, rhs, M_inv, n_iter)
            profile_end(_pt)
            targets[:, k] = unk_to_full_torch(du_unk, nx, ny)
    return targets


# ---------- training loop ----------
def train(fno, data, args, save_dir, device):
    nx, ny = data['train_a'].shape[1:3]
    method = args.method

    # Tensors
    train_a = torch.tensor(data['train_a'], dtype=torch.float32)
    train_u = torch.tensor(data['train_u'], dtype=torch.float32)
    val_a   = torch.tensor(data['val_a'], dtype=torch.float32)
    val_u   = torch.tensor(data['val_u'], dtype=torch.float32)
    test_a  = torch.tensor(data['test_a'], dtype=torch.float32)
    test_u  = torch.tensor(data['test_u'], dtype=torch.float32)
    test_jvps = data.get('test_jvps')
    test_dirs = data.get('test_delta_dirs')

    if train_a.dim() == 3:
        train_a = train_a.unsqueeze(-1)
        train_u = train_u.unsqueeze(-1)
        val_a = val_a.unsqueeze(-1)
        val_u = val_u.unsqueeze(-1)
        test_a = test_a.unsqueeze(-1)
        test_u = test_u.unsqueeze(-1)

    N = train_a.shape[0]
    xs = torch.linspace(0, 1, nx, device=device)
    ys = torch.linspace(0, 1, ny, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing='ij')
    coords = torch.stack([gx, gy], dim=-1)

    # Method-specific setup
    train_jvps_t = None
    train_dirs = None
    inv_lam = None
    dx = dy = 1.0 / (nx - 1)

    if method == 'difno':
        train_jvps_t = torch.tensor(data['train_jvps'], dtype=torch.float32)
        train_dirs = torch.tensor(data['train_delta_dirs'], dtype=torch.float32).to(device)
        K_total = train_jvps_t.shape[1]
        print(f"  DIFNO: loaded precomputed JVPs  {tuple(train_jvps_t.shape)}")
    elif method == 'fminres':
        inv_lam = get_inv_shifted_lam(nx, ny, args.shift_scale, device)
        print(f"  FMINRES: shift={args.shift_scale}  n_iter={args.minres_iters}")

    # Pre-sample a pool of GP directions to avoid recomputing the cosine basis
    # on every batch.  sample_gp_matern is expensive (40x40x65x65 basis alloc).
    if method == 'fminres':
        POOL_SIZE = 2048
        print(f"  Pre-sampling {POOL_SIZE} GP directions (pool) ...", flush=True)
        t_pool0 = time.time()
        dir_pool_np = sample_gp_matern(POOL_SIZE, nx, ny, n_modes=40, seed=args.dir_seed)
        for k in range(POOL_SIZE):
            dir_pool_np[k] /= np.linalg.norm(dir_pool_np[k]) + 1e-10
        dir_pool_t = torch.tensor(dir_pool_np, dtype=torch.float32, device=device)
        print(f"    done in {time.time()-t_pool0:.1f}s")

    # Optimizer
    opt = torch.optim.Adam(fno.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=50, factor=0.5, min_lr=1e-6)

    # EMA adaptive scaling
    mse_ema = None
    jvp_ema = None
    ema_decay = 0.99

    history = []
    best_val_mse = float('inf')
    best_val_rel = float('inf')
    best_epoch = None
    t_start = time.time()
    dir_counter = 0
    os.makedirs(save_dir, exist_ok=True)

    def fno_fn_factory(c):
        def f(x): return fno(torch.cat([c, x], dim=-1))
        return f

    for epoch in range(1, args.epochs + 1):
        profile_epoch_start(epoch)
        fno.train()
        perm = torch.randperm(N)
        epoch_mse = 0.0
        epoch_jvp = 0.0
        n_batches = 0

        for i in range(0, N, args.batch_size):
            idx = perm[i:i + args.batch_size]
            a_b = train_a[idx].to(device)
            u_b = train_u[idx].to(device)
            B = a_b.shape[0]
            c = coords.unsqueeze(0).expand(B, -1, -1, -1)
            fno_fn = fno_fn_factory(c)

            _pt = profile_start("data_forward")
            u_pred = fno_fn(a_b)
            mse = F.mse_loss(u_pred, u_b)
            profile_end(_pt)

            if method == 'fno':
                loss = mse
            else:
                _deriv_pt = profile_start("derivative_total")
                # ------ sample / fetch q directions ------
                if method == 'difno':
                    di = torch.randperm(K_total)[:args.q]
                    deltas_t = train_dirs[di]                   # (q, nx, ny)
                    _pt = profile_start("label_h2d")
                    true_jvps = train_jvps_t[idx][:, di].to(device)  # (B, q, nx, ny)
                    profile_end(_pt)
                else:  # fminres: tangent targets from a few MINRES iterations
                    pool_idx = (torch.arange(args.q) + dir_counter * args.q) % POOL_SIZE
                    dir_counter += 1
                    deltas_t = dir_pool_t[pool_idx]
                    true_jvps = compute_fminres_targets(
                        a_b, u_b, deltas_t, args.q, nx, ny, dx, dy,
                        inv_lam, n_iter=args.minres_iters)

                # ------ FNO JVPs at the training anchors ------
                jvp_loss = torch.tensor(0.0, device=device)
                for k in range(args.q):
                    da = deltas_t[k].unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1)
                    _pt = profile_start("online_jvp")
                    _, pred_jvp = jvp(fno_fn, (a_b,), (da,))
                    profile_end(_pt)
                    pred_jvp = pred_jvp.squeeze(-1)
                    jvp_loss = jvp_loss + F.mse_loss(pred_jvp, true_jvps[:, k])
                jvp_loss = jvp_loss / args.q
                profile_end(_deriv_pt)

                jvp_val = jvp_loss.item()
                if mse_ema is None:
                    mse_ema = mse.item()
                    jvp_ema = max(jvp_val, 1e-20)
                else:
                    mse_ema = ema_decay * mse_ema + (1 - ema_decay) * mse.item()
                    jvp_ema = ema_decay * jvp_ema + (1 - ema_decay) * max(jvp_val, 1e-20)
                scale = mse_ema / max(jvp_ema, 1e-20)
                loss = mse + args.lam_jvp * scale * jvp_loss
                epoch_jvp += jvp_val

            _pt = profile_start("backward_optimizer")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fno.parameters(), 1.0)
            opt.step()
            profile_end(_pt)
            epoch_mse += mse.item()
            n_batches += 1

        epoch_mse /= n_batches
        epoch_jvp /= max(n_batches, 1)

        val_mse, val_rel = evaluate_solution(fno, val_a, val_u, coords, device)
        sched.step(val_mse)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_val_rel = val_rel
            best_epoch = epoch
            torch.save(fno.state_dict(), os.path.join(save_dir, 'best_model.pth'))

        history.append({
            'epoch': epoch,
            'train_mse': epoch_mse,
            'val_mse': val_mse,
            'rel_err': val_rel,
            'jvp_loss': epoch_jvp,
            'lr': opt.param_groups[0]['lr'],
            'wall_time': time.time() - t_start,
        })

        if epoch % args.print_every == 0 or epoch <= 3:
            print(f"  [{epoch:4d}/{args.epochs}]  mse={epoch_mse:.3e}  "
                  f"val={val_mse:.3e}  rel={val_rel:.4f}  jvp={epoch_jvp:.3e}  "
                  f"t={history[-1]['wall_time']:.0f}s", flush=True)
        profile_epoch_end(epoch, n_batches)

    # Save
    torch.save(fno.state_dict(), os.path.join(save_dir, 'last_model.pth'))
    with open(os.path.join(save_dir, 'train_history.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)

    # Held-out test evaluation uses the validation-selected checkpoint.  The
    # test split does not drive the scheduler or checkpoint selection.
    fno.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth'),
                                   map_location=device, weights_only=True))
    test_mse, test_rel = evaluate_solution(fno, test_a, test_u, coords, device)
    jac_err = None
    if test_jvps is not None and test_dirs is not None:
        jac_err = evaluate_jacobian(fno, test_a, test_jvps, test_dirs, coords, device)

    metrics = {
        'method': method,
        'best_epoch': best_epoch,
        'best_val_mse': best_val_mse,
        'best_val_rel': best_val_rel,
        'test_mse': test_mse,
        'test_rel_err': test_rel,
        'test_jac_rel_err': jac_err,
        'test_checkpoint': 'best_model.pth',
        'total_wall_time_s': history[-1]['wall_time'],
        'n_epochs': args.epochs,
        'n_train': N,
    }
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=['fno', 'difno', 'fminres'])
    ap.add_argument('--n_use', type=int, default=512)
    ap.add_argument('--q', type=int, default=4)
    ap.add_argument('--jvp_bank_r', type=int, default=289,
                    help='Stored offline JVP bank size for DIFNO; q is directions sampled per step.')
    ap.add_argument('--lam_jvp', type=float, default=None,
                    help='Paper-selected EMA derivative weight; inferred from method and n_use when omitted.')
    ap.add_argument('--epochs', type=int, default=2000)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--minres_iters', type=int, default=5)
    ap.add_argument('--shift_scale', type=float, default=0.5)
    ap.add_argument('--dir_seed', type=int, default=2027)
    ap.add_argument('--print_every', type=int, default=25)
    ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--modes', type=int, default=8)
    ap.add_argument('--data_dir', type=str, default=os.path.join(SCRIPT_DIR, 'data'))
    ap.add_argument('--save_dir', type=str, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='Torch device used for training and evaluation.')
    args = ap.parse_args()

    if args.lam_jvp is None:
        if args.method == "fno":
            args.lam_jvp = 0.0
        else:
            paper_lambdas = (
                {128: 0.1, 512: 0.5, 1024: 0.9}
                if args.method == "difno"
                else {128: 0.1, 512: 0.5, 1024: 0.7}
            )
            if args.n_use not in paper_lambdas:
                raise ValueError(
                    f"No paper-selected Helmholtz lambda for N={args.n_use}; "
                    "pass --lam_jvp explicitly."
                )
            args.lam_jvp = paper_lambdas[args.n_use]
            print(f"Using paper-selected lambda={args.lam_jvp} for {args.method}, N={args.n_use}")

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load data
    tr = np.load(os.path.join(args.data_dir, 'train.npz'))
    va = np.load(os.path.join(args.data_dir, 'val.npz'))
    te = np.load(os.path.join(args.data_dir, 'test.npz'))

    data = {
        'train_a': tr['a'][:args.n_use],
        'train_u': tr['u'][:args.n_use],
        'val_a': va['a'],
        'val_u': va['u'],
        'test_a': te['a'],
        'test_u': te['u'],
        'test_jvps': te.get('jvps'),
        'test_delta_dirs': te.get('delta_dirs'),
    }

    if args.method == 'difno':
        # Load or compute offline training JVPs
        tj_path = os.path.join(args.data_dir, f'train_jvps_r{args.jvp_bank_r}.npz')
        if not os.path.exists(tj_path):
            print(f"  Generating offline JVPs at {tj_path} ...", flush=True)
            nx_data, ny_data = tr['a'].shape[1:3]
            L2d = build_fd_laplacian_dirichlet(nx_data, ny_data)
            n_dirs = args.jvp_bank_r
            dirs = sample_gp_matern(n_dirs, nx_data, ny_data, n_modes=40, seed=123456)
            for d in range(n_dirs):
                dirs[d] /= np.linalg.norm(dirs[d]) + 1e-10
            N_full = len(tr['a'])
            t0 = time.time()
            jvps_all = np.zeros((N_full, n_dirs, nx_data, ny_data), dtype=np.float32)
            for i in range(N_full):
                A = build_helmholtz_operator(tr['a'][i], nx_data, ny_data, L2d).tocsc()
                A_lu = spla.splu(A)
                for d in range(n_dirs):
                    jvps_all[i, d] = solve_tangent(tr['a'][i], tr['u'][i], dirs[d],
                                                    nx_data, ny_data, L2d, A_lu)
                if (i + 1) % max(1, N_full // 5) == 0:
                    print(f"    [{i+1}/{N_full}]  t={time.time()-t0:.0f}s")
            t_offline = time.time() - t0
            print(f"  Offline JVP generation: {t_offline:.1f}s  ({t_offline/60:.1f} min)")
            np.savez(tj_path, jvps=jvps_all, delta_dirs=dirs,
                     offline_seconds=t_offline)
        tj = np.load(tj_path)
        data['train_jvps'] = tj['jvps'][:args.n_use]
        data['train_delta_dirs'] = tj['delta_dirs']
        if 'offline_seconds' in tj.files:
            print(f"  Using offline JVPs generated in {float(tj['offline_seconds']):.1f}s")

    # Surrogate
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    fno = FNO2D(in_channels=3, out_channels=1, width=args.width,
                modes1=args.modes, modes2=args.modes, n_layers=4).to(device)
    print(f"FNO2D: {sum(p.numel() for p in fno.parameters()):,} parameters  "
          f"(w={args.width}, m={args.modes})")

    save_dir = args.save_dir
    if save_dir is None:
        tag = f'N{args.n_use}_{args.method}'
        if args.method != 'fno':
            tag += f'_q{args.q}_lam{args.lam_jvp}'
        if args.method == 'fminres':
            tag += f'_m{args.minres_iters}_s{args.shift_scale}'
        save_dir = os.path.join(SCRIPT_DIR, 'results', tag)

    print(f"\nTraining {args.method} | N={args.n_use}  epochs={args.epochs}")
    metrics = train(fno, data, args, save_dir, device)
    print(f"\nDone. best epoch={metrics['best_epoch']}  "
          f"val rel={metrics['best_val_rel']:.4f}  "
          f"test rel={metrics['test_rel_err']:.4f}  "
          f"jac={metrics['test_jac_rel_err']}")
    print(f"Saved: {save_dir}")


if __name__ == '__main__':
    main()
