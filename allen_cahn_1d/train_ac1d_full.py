#!/usr/bin/env python
"""FNO baseline, DIFNO, and sTCL training for 1D Allen--Cahn.

  --method fno/data: data MSE on the trajectory only (``fno`` kept for compatibility)
  --method difno  : data MSE + offline tangent labels (stored true du/da . v from <split>_jvps)
  --method stcl   : data MSE + online sTCL residual (exact forward-mode JVP of phys; NO labels)

Protocol matches the paper's shared setup (App. experiment details): FNO 4 layers, width 32,
GELU, Adam lr=1e-3, 1500 epochs, batch 32, sketch q=4, ReduceLROnPlateau, grad-clip 1.0.
Allen-Cahn is the parabolic space-time / Burgers class -> modes=12, raw-residual sTCL, lam=2.

Metrics: per-sample solution rel-L2 over the trajectory, and Metric-B Jacobian error
mean_{i,d} ||J_theta v - J v|| / ||J v|| over the stored validation/test directions.
Every epoch also records label-based validation JVP MSE/relative error on a fixed bank
subset, plus the raw validation sTCL residual for sTCL runs. The reported test metrics use
the best-validation checkpoint. DIFNO labels and sTCL directions are smooth-GRF fields
matched to the input prior.
"""
import os, sys, time, json, csv, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jvp as func_jvp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from fno_ac1d import FNO
from stcl_ac1d import stcl_loss
from runtime_profile import end as profile_end
from runtime_profile import epoch_end as profile_epoch_end
from runtime_profile import epoch_start as profile_epoch_start
from runtime_profile import start as profile_start


def load_split(d, name):
    z = np.load(os.path.join(d, f'{name}.npz'))
    from ac1d_time_solver import ELL, TAU
    ell = float(z['ell']) if 'ell' in z.files else ELL
    tau = float(z['tau']) if 'tau' in z.files else TAU
    return (torch.tensor(z['a'], dtype=torch.float32), torch.tensor(z['U'], dtype=torch.float32),
            int(z['nt']), int(z['nx']), float(z['T']), float(z['eps']), ell, tau)


def load_bank(d, name, nt=None):
    """Returns (jvps (N,r,F,nx), dirs (r,nx), frames (F,)).  frames = stored time indices."""
    p = os.path.join(d, f'{name}_jvps.npz')
    if not os.path.exists(p):
        return None, None, None
    z = np.load(p)
    fr = torch.tensor(z['frames'], dtype=torch.long) if 'frames' in z.files else \
        (torch.arange(nt) if nt is not None else None)
    return torch.from_numpy(z['jvps']), torch.tensor(z['dirs'], dtype=torch.float32), fr


@torch.no_grad()
def eval_field(model, A, U, dev, bs=32, frame=None):
    """Per-sample relative L2, then averaged (MC estimate, paper Eq. 34a):
    mean_i ||G_theta(a_i) - G(a_i)|| / ||G(a_i)||.
    frame=None -> over the whole trajectory; frame=k -> on time slice k only
    (frame=-1 gives an auxiliary final-state diagnostic)."""
    model.eval(); errs = []
    for i in range(0, len(A), bs):
        p = model.field(A[i:i+bs].to(dev)); t = U[i:i+bs].to(dev)
        if frame is not None:
            p, t = p[:, frame], t[:, frame]
        num = (p - t).flatten(1).norm(dim=1)
        den = t.flatten(1).norm(dim=1)
        errs.extend((num / (den + 1e-20)).cpu().tolist())
    model.train()
    return float(np.mean(errs))


def eval_jvp_metrics(model, A, jvps, dirs, frames, dev, bs=16,
                     n_samples=None, n_dirs=None):
    """Return label-based JVP MSE and Metric-B relative error.

    ``n_samples`` and ``n_dirs`` select deterministic prefixes of the fixed
    validation bank. This makes per-epoch curves comparable while allowing
    their cost to be controlled independently of the full checkpoint metrics.
    """
    was_training = model.training
    model.eval()
    requested_samples = len(A) if n_samples is None else int(n_samples)
    requested_dirs = dirs.shape[0] if n_dirs is None else int(n_dirs)
    ns = min(requested_samples, len(A), len(jvps))
    nd = min(requested_dirs, dirs.shape[0], jvps.shape[1])
    if ns <= 0 or nd <= 0:
        raise ValueError(f'JVP evaluation requires positive samples/directions; got {ns}/{nd}')

    errs = []
    squared_error = 0.0
    n_values = 0
    fr = frames.to(dev)
    for d in range(nd):
        vd = dirs[d].to(dev)
        for i in range(0, ns, bs):
            j = min(i + bs, ns)
            vb = vd.unsqueeze(0).expand(j - i, -1)
            true = jvps[i:j, d].to(dev)
            with torch.enable_grad():
                _, pred = func_jvp(model.field, (A[i:j].to(dev),), (vb,))
            pred = pred[:, fr]                                    # match stored frames
            delta = pred - true
            squared_error += float(delta.detach().square().sum())
            n_values += delta.numel()
            rel = delta.flatten(1).norm(dim=1) / (true.flatten(1).norm(dim=1) + 1e-10)
            errs.extend(rel.detach().cpu().tolist())
    model.train(was_training)
    return {
        'mse': squared_error / max(n_values, 1),
        'rel': float(np.mean(errs)),
        'n_samples': ns,
        'n_dirs': nd,
        'n_frames': int(len(frames)),
    }


def eval_jac(model, A, jvps, dirs, frames, dev, bs=16):
    """Metric-B: per-(sample,direction) Jacobian relative error, averaged."""
    return eval_jvp_metrics(model, A, jvps, dirs, frames, dev, bs=bs)['rel']


def eval_stcl(model, A, dirs, dev, bs=32, n_samples=None, n_dirs=None):
    """Raw sTCL tangent-residual MSE on fixed held-out directions."""
    was_training = model.training
    model.eval()
    requested_samples = len(A) if n_samples is None else int(n_samples)
    requested_dirs = dirs.shape[0] if n_dirs is None else int(n_dirs)
    ns = min(requested_samples, len(A))
    nd = min(requested_dirs, dirs.shape[0])
    if ns <= 0 or nd <= 0:
        raise ValueError(f'sTCL evaluation requires positive samples/directions; got {ns}/{nd}')

    squared_residual = 0.0
    n_values = 0
    for d in range(nd):
        vd = dirs[d].to(dev)
        for i in range(0, ns, bs):
            j = min(i + bs, ns)
            a = A[i:j].to(dev)
            vb = vd.unsqueeze(0).expand(j - i, -1)
            with torch.enable_grad():
                _, residual_jvp = func_jvp(model.phys, (a,), (vb,))
            squared_residual += float(residual_jvp.detach().square().sum())
            n_values += residual_jvp.numel()
    model.train(was_training)
    return squared_residual / max(n_values, 1)


def write_history(save_dir, history):
    """Persist per-epoch training/validation errors after every epoch."""
    if not save_dir:
        return
    json_path = os.path.join(save_dir, 'history.json')
    csv_path = os.path.join(save_dir, 'history.csv')
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=2)
    if history:
        keys = []
        for row in history:
            for key in row:
                if key not in keys:
                    keys.append(key)
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, restval='')
            w.writeheader()
            w.writerows(history)


def clone_state_cpu(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_json(path, default):
    if not path or not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def resume_paths(resume_from):
    """Resolve resume checkpoint/history paths from a run dir or checkpoint path."""
    if not resume_from:
        return None
    if os.path.isdir(resume_from):
        base = resume_from
        return {
            'base': base,
            'last': os.path.join(base, 'last_model.pth'),
            'best': os.path.join(base, 'best_model.pth'),
            'meta': os.path.join(base, 'best_model_meta.json'),
            'history': os.path.join(base, 'history.json'),
        }
    base = os.path.dirname(os.path.abspath(resume_from))
    return {
        'base': base,
        'last': resume_from,
        'best': os.path.join(base, 'best_model.pth'),
        'meta': os.path.join(base, 'best_model_meta.json'),
        'history': os.path.join(base, 'history.json'),
    }


def eval_snapshot(model, state, tag, Atr, Utr, Ava, Uva, Ate, Ute,
                  val_bank, test_bank, dev, nt):
    """Evaluate one checkpoint snapshot with field errors and Metric-B JVP errors."""
    model.load_state_dict(state)
    val_j, val_d, val_fr = val_bank
    test_j, test_d, test_fr = test_bank
    out = {
        'checkpoint': tag,
        'train_rel': eval_field(model, Atr, Utr, dev),
        'val_rel': eval_field(model, Ava, Uva, dev),
        'test_rel': eval_field(model, Ate, Ute, dev),
        'test_rel_final': eval_field(model, Ate, Ute, dev, frame=-1),
        'val_jac': (eval_jac(model, Ava, val_j, val_d, val_fr, dev) if val_j is not None else None),
        'test_jac': (eval_jac(model, Ate, test_j, test_d, test_fr, dev) if test_j is not None else None),
        'val_jac_n_dirs': (int(val_d.shape[0]) if val_d is not None else None),
        'test_jac_n_dirs': (int(test_d.shape[0]) if test_d is not None else None),
        'val_jac_n_frames': (int(len(val_fr)) if val_fr is not None else None),
        'test_jac_n_frames': (int(len(test_fr)) if test_fr is not None else None),
    }
    out['val_jac_frame_mode'] = None if val_fr is None else ('full' if len(val_fr) == nt else 'subset')
    out['test_jac_frame_mode'] = None if test_fr is None else ('full' if len(test_fr) == nt else 'subset')
    return out


def train(args, dev):
    Atr, Utr, nt, nx, T, eps, ell, tau = load_split(args.data_dir, 'train')
    Ava, Uva, *_ = load_split(args.data_dir, 'val')
    Ate, Ute, *_ = load_split(args.data_dir, 'test')
    Atr, Utr = Atr[:args.n_use], Utr[:args.n_use]; N = len(Atr)
    Atr_d, Utr_d = Atr.to(dev), Utr.to(dev)

    val_bank = load_bank(args.data_dir, 'val', nt)
    test_bank = load_bank(args.data_dir, 'test', nt)
    val_jvps, val_dirs, val_frames = val_bank
    periodic_val_jvp = args.val_jvp_every > 0
    if periodic_val_jvp and val_jvps is None:
        raise FileNotFoundError(
            f'--val_jvp_every={args.val_jvp_every} requires {args.data_dir}/val_jvps.npz')
    if periodic_val_jvp:
        val_jvp_n_samples = min(args.val_jvp_n, len(Ava), len(val_jvps))
        val_jvp_n_dirs = min(args.val_jvp_dirs, val_dirs.shape[0], val_jvps.shape[1])
        if val_jvp_n_samples <= 0 or val_jvp_n_dirs <= 0:
            raise ValueError('validation JVP sample and direction counts must be positive')
        print(f'  per-epoch val JVP: N={val_jvp_n_samples} dirs={val_jvp_n_dirs} '
              f'frames={len(val_frames)}/{nt} every={args.val_jvp_every}', flush=True)
    else:
        val_jvp_n_samples = val_jvp_n_dirs = 0

    train_jvps = train_dirs = train_fr = None
    if args.method == 'difno':
        train_jvps, train_dirs, train_fr = load_bank(args.data_dir, 'train', nt)
        train_jvps = train_jvps[:args.n_use]; train_dirs = train_dirs.to(dev); train_fr = train_fr.to(dev)
        r_tot = train_dirs.shape[0]
        print(f"  difno bank {tuple(train_jvps.shape)} r={r_tot} frames={len(train_fr)}/{nt}", flush=True)

    model = FNO(nt, nx, args.width, tuple(args.modes), args.nlayers, eps=eps, T=T).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"fno params={n_params:,} N={N} method={args.method}", flush=True)

    data_only = args.method in ('fno', 'data')

    history = []
    start_epoch = 0
    resume_elapsed = 0.0
    resume_last_lr = None
    best_val = 1e9
    best_epoch = 0
    best_state = None
    mse_ema = jvp_ema = None
    rp = resume_paths(args.resume_from)
    if rp is not None:
        if not os.path.exists(rp['last']):
            raise FileNotFoundError(f"resume checkpoint not found: {rp['last']}")
        model.load_state_dict(torch.load(rp['last'], map_location=dev))
        history = load_json(rp['history'], [])
        if history:
            last_hist = history[-1]
            start_epoch = int(last_hist.get('ep', len(history)))
            resume_elapsed = float(last_hist.get('t', 0.0))
            resume_last_lr = last_hist.get('lr', None)
            best_val = float(last_hist.get('best_val_rel', best_val))
            best_epoch = int(last_hist.get('best_epoch', best_epoch))
            if not data_only:
                mse_ema = float(last_hist.get('train_mse', last_hist.get('mse', 0.0)))
                jvp_ema = max(float(last_hist.get('jvp', 1e-20)), 1e-20)
        meta = load_json(rp['meta'], {})
        if meta:
            best_epoch = int(meta.get('best_epoch', best_epoch))
            best_val = float(meta.get('best_val_rel', best_val))
        if os.path.exists(rp['best']):
            best_state = torch.load(rp['best'], map_location='cpu')
        else:
            best_state = clone_state_cpu(model)
        print(f"  resumed from {rp['last']} at epoch {start_epoch}; "
              f"best_ep={best_epoch} best_val={best_val:.6g}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.resume_lr_from_history and resume_last_lr is not None:
        for group in opt.param_groups:
            group['lr'] = float(resume_last_lr)
        print(f"  resumed optimizer lr from history: {float(resume_last_lr):.3e}", flush=True)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=20, factor=0.5, min_lr=1e-6)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        json.dump(vars(args), open(os.path.join(args.save_dir, 'config.json'), 'w'), indent=2)
        if history:
            write_history(args.save_dir, history)
        if best_state is not None:
            torch.save(best_state, os.path.join(args.save_dir, 'best_model.pth'))
            json.dump({'best_epoch': best_epoch, 'best_val_rel': best_val},
                      open(os.path.join(args.save_dir, 'best_model_meta.json'), 'w'), indent=2)

    ema = 0.99; t0 = time.time() - resume_elapsed
    if start_epoch >= args.epochs:
        print(f"  requested epochs={args.epochs}; checkpoint already at epoch {start_epoch}", flush=True)
    for ep in range(start_epoch + 1, args.epochs + 1):
        profile_epoch_start(ep)
        model.train(); perm = torch.randperm(N, device=dev); emse = ejvp = 0.0; nb = 0
        opt.zero_grad(set_to_none=True)
        for i in range(0, N, args.bs):
            idx = perm[i:i+args.bs]; a = Atr_d[idx]; u = Utr_d[idx]; B = a.shape[0]
            _pt = profile_start('data_forward')
            pred = model.field(a); mse = F.mse_loss(pred, u)
            profile_end(_pt)
            if data_only:
                loss = mse
            else:
                _deriv_pt = profile_start('derivative_total')
                if args.method == 'difno':
                    di = torch.randperm(r_tot, device=dev)[:args.q].tolist(); idx_c = idx.cpu()
                    def direction_loss(d):
                        vb = train_dirs[d].unsqueeze(0).expand(B, -1)
                        _jvp_pt = profile_start('online_jvp')
                        _, pj = func_jvp(model.field, (a,), (vb,))
                        profile_end(_jvp_pt)
                        _label_pt = profile_start('label_h2d')
                        true_jvp = train_jvps[idx_c, d].to(dev)
                        profile_end(_label_pt)
                        return F.mse_loss(pj[:, train_fr], true_jvp)
                else:
                    def direction_loss(_):
                        return stcl_loss(model, a, n_sketch=1, ell=ell, tau=tau)

                direction_ids = di if args.method == 'difno' else range(args.q)
                jl = a.new_zeros(())
                for d in direction_ids:
                    jl = jl + direction_loss(d)
                jl = jl / args.q
                jv = float(jl.item())
                profile_end(_deriv_pt)

                if mse_ema is None: mse_ema, jvp_ema = mse.item(), max(jv, 1e-20)
                else:
                    mse_ema = ema * mse_ema + (1 - ema) * mse.item()
                    jvp_ema = ema * jvp_ema + (1 - ema) * max(jv, 1e-20)
                gamma = args.lam * (mse_ema / max(jvp_ema, 1e-20))
                loss = mse + gamma * jl
                ejvp += jv
            _backward_pt = profile_start('backward_optimizer')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            profile_end(_backward_pt)
            emse += mse.item(); nb += 1
        emse /= nb; ejvp /= max(nb, 1)
        val_rel = eval_field(model, Ava, Uva, dev); sched.step(val_rel)
        train_rel = eval_field(model, Atr, Utr, dev)        # per-sample train rel-L2 (same metric as val)
        val_jvp_mse = val_jvp_rel = val_stcl = None
        if periodic_val_jvp and (ep % args.val_jvp_every == 0 or ep == args.epochs):
            val_jvp = eval_jvp_metrics(
                model, Ava, val_jvps, val_dirs, val_frames, dev,
                bs=args.val_jvp_bs, n_samples=val_jvp_n_samples, n_dirs=val_jvp_n_dirs)
            val_jvp_mse = val_jvp['mse']
            val_jvp_rel = val_jvp['rel']
            if args.method == 'stcl':
                val_stcl = eval_stcl(
                    model, Ava, val_dirs, dev, bs=args.val_jvp_bs,
                    n_samples=val_jvp_n_samples, n_dirs=val_jvp_n_dirs)
        if val_rel < best_val:
            best_val = val_rel
            best_epoch = ep
            best_state = clone_state_cpu(model)
            if args.save_dir:
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
                json.dump({'best_epoch': best_epoch, 'best_val_rel': best_val},
                          open(os.path.join(args.save_dir, 'best_model_meta.json'), 'w'), indent=2)
        history.append({'ep': ep,
                        'train_mse': emse, 'mse': emse,
                        'train_rel': train_rel, 'val_rel': val_rel,
                        'best_val_rel': best_val, 'best_epoch': best_epoch,
                        'jvp': ejvp, 'val_stcl': val_stcl,
                        'val_jvp_mse': val_jvp_mse, 'val_jvp_rel': val_jvp_rel,
                        'val_jvp_n_samples': val_jvp_n_samples,
                        'val_jvp_n_dirs': val_jvp_n_dirs,
                        'val_jvp_n_frames': (int(len(val_frames)) if val_frames is not None else 0),
                        'lr': opt.param_groups[0]['lr'], 't': time.time() - t0})
        write_history(args.save_dir, history)
        if ep % args.print_every == 0 or ep <= 2:
            val_jvp_msg = 'NA' if val_jvp_rel is None else f'{val_jvp_rel:.4f}'
            val_stcl_msg = 'NA' if val_stcl is None else f'{val_stcl:.3e}'
            print(f"  [{ep:4d}/{args.epochs}] mse={emse:.3e} train_rel={train_rel:.4f} "
                  f"val_rel={val_rel:.4f} best_ep={best_epoch} jvp={ejvp:.3e} "
                  f"val_stcl={val_stcl_msg} val_jvp_rel={val_jvp_msg} "
                  f"lr={opt.param_groups[0]['lr']:.1e} t={time.time()-t0:.0f}s", flush=True)
        if args.save_dir and (ep % max(1, args.save_last_every) == 0 or ep == args.epochs):
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'last_model.pth'))
        profile_epoch_end(ep, nb)

    last_state = clone_state_cpu(model)
    if best_state is None:
        best_state = last_state

    last_metrics = eval_snapshot(model, last_state, 'last', Atr, Utr, Ava, Uva, Ate, Ute,
                                 val_bank, test_bank, dev, nt)
    best_metrics = eval_snapshot(model, best_state, 'best', Atr, Utr, Ava, Uva, Ate, Ute,
                                 val_bank, test_bank, dev, nt)
    # Leave the model on the last snapshot after reporting, matching last_model.pth.
    model.load_state_dict(last_state)

    metrics = dict(method=args.method, architecture='fno', model_params=n_params,
                   n_train=N, lam=(None if data_only else args.lam),
                   best_epoch=best_epoch, best_val_rel=best_val,
                   last=last_metrics, best=best_metrics,
                   wall_s=time.time() - t0)
    for prefix, snap in (('last', last_metrics), ('best', best_metrics)):
        for key, value in snap.items():
            if key == 'checkpoint':
                continue
            metrics[f'{key}_{prefix}'] = value
    # The flat keys report the validation-selected (best) checkpoint, as in the paper.
    for key, value in best_metrics.items():
        if key != 'checkpoint':
            metrics[key] = value
    if args.save_dir:
        json.dump(metrics, open(os.path.join(args.save_dir, 'metrics.json'), 'w'), indent=2)
    jac = 'NA' if metrics['test_jac'] is None else f"{metrics['test_jac']:.4f}"
    print(f"[{args.method} N={N} lam={args.lam}] best_ep={best_epoch} "
          f"test_rel={metrics['test_rel']:.4f} test_jac={jac} "
          f"(final_frame_rel={metrics['test_rel_final']:.4f}, "
          f"jac_frames={metrics['test_jac_n_frames']})", flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--method', required=True,
        choices=['fno', 'data', 'difno', 'stcl'],
        help="data-only, offline-JVP DIFNO, or online residual sTCL; 'fno' aliases data")
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--n_use', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=1500)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--q', type=int, default=4)
    ap.add_argument('--val_jvp_every', type=int, default=1,
                    help='Record validation JVP metrics every N epochs (0 disables).')
    ap.add_argument('--val_jvp_n', type=int, default=128,
                    help='Number of fixed validation samples used for the per-epoch JVP curve.')
    ap.add_argument('--val_jvp_dirs', type=int, default=4,
                    help='Number of fixed validation-bank directions used per epoch.')
    ap.add_argument('--val_jvp_bs', type=int, default=32,
                    help='Batch size for per-epoch validation JVP and sTCL evaluation.')
    ap.add_argument('--lam', type=float, default=2.0)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--modes', type=int, nargs=2, default=[12, 12])
    ap.add_argument('--nlayers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save_dir', default=None)
    ap.add_argument('--resume_from', default=None,
                    help='Run directory or checkpoint path to continue from.')
    ap.add_argument('--resume_lr_from_history', action='store_true',
                    help='When resuming, initialize Adam lr from the last history row.')
    ap.add_argument('--save_last_every', type=int, default=100,
                    help='Save last_model.pth every this many epochs.')
    ap.add_argument('--print_every', type=int, default=50)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    if args.val_jvp_every < 0:
        ap.error('--val_jvp_every must be nonnegative')
    if args.val_jvp_every > 0 and (args.val_jvp_n <= 0 or args.val_jvp_dirs <= 0 or args.val_jvp_bs <= 0):
        ap.error('--val_jvp_n, --val_jvp_dirs, and --val_jvp_bs must be positive')
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"\nfno/{args.method} N={args.n_use} seed={args.seed} "
          f"epochs={args.epochs} bs={args.bs}", flush=True)
    train(args, torch.device(args.device))


if __name__ == '__main__':
    main()
