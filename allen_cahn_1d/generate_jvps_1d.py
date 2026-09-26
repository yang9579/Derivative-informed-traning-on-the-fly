#!/usr/bin/env python
"""Generate true-JVP banks for the 1D Allen-Cahn data (DIFNO labels + Metric-B test dirs).

For each split, draw r smooth-GRF direction fields v_d (matched to the input prior, unit-norm)
and store the exact tangent du/da . v_d for every sample, computed by forward-mode AD through
the differentiable spectral solver.  These are DIFNO's offline labels (train bank) and the
matched probe directions for the Metric-B Jacobian metric (val/test).

In 1D the full trajectory bank is small, so we store ALL nt frames by default (no frame
subset needed, unlike the 13 GB 2D bank).

  <split>_jvps.npz : jvps (N, r, nt, nx) float32, dirs (r, nx) float32, frames (nt,)
"""
import os, argparse, time
import numpy as np
import torch

from ac1d_time_solver import true_jvp, grf_ic, ELL, TAU


def gen_bank(a, r, eps, nt, T, nsub, frames, gen, dev, ell=ELL, tau=TAU, bs=2048):
    """Probe/label directions are drawn from the SAME Matern prior as the dataset input
    (ell, tau read from the dataset metadata) so the JVP bank matches the input distribution.

    NB: the solve is a long (nt*nsub ~ 5000-step) loop of tiny per-step FFT kernels, so it is
    GPU-launch-overhead bound, not compute bound -- batch as many samples as fit (large bs)
    to amortize the launches over the batch (≈ N/bs speedup)."""
    N, nx = a.shape[0], a.shape[-1]
    F = len(frames)
    dirs = grf_ic(r, nx, gen, ell=ell, tau=tau, tanh=False)          # smooth field directions
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    jvps = np.empty((N, r, F, nx), dtype=np.float32)
    fr = torch.tensor(frames, device=dev)
    t0 = time.time()
    for d in range(r):
        vd = dirs[d:d+1].to(dev).double()
        for i in range(0, N, bs):
            j = min(i + bs, N)
            _, w = true_jvp(a[i:j].to(dev).double(), vd.expand(j - i, -1),
                            eps=eps, nt=nt, T=T, n_substep=nsub)
            jvps[i:j, d] = w[:, fr].float().cpu().numpy()
        if (d + 1) % 32 == 0:
            print(f"    dir {d+1}/{r}  ({time.time()-t0:.0f}s)", flush=True)
    return jvps, dirs.cpu().float().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    ap.add_argument('--r_train', type=int, default=200)
    ap.add_argument('--r_eval', type=int, default=200)
    ap.add_argument('--n_frames', type=int, default=0,
                    help='0 = full trajectory (default/recommended); positive = frame subset, e.g. 1 keeps final only')
    ap.add_argument('--n_max', type=int, default=0, help='0 = all samples; else cap train to first n_max')
    ap.add_argument('--dt_internal', type=float, default=1e-3)
    ap.add_argument('--bs', type=int, default=2048, help='samples per solve; large = fewer kernel launches (faster)')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    dev = torch.device(args.device)
    gen = torch.Generator().manual_seed(args.seed)

    for name in args.splits:
        z = np.load(os.path.join(args.data_dir, f'{name}.npz'))
        a = torch.tensor(z['a'], dtype=torch.float32)
        if args.n_max > 0 and name == 'train':
            a = a[:args.n_max]
        nt, nx, T, eps = int(z['nt']), int(z['nx']), float(z['T']), float(z['eps'])
        # direction prior MUST match the dataset's input distribution (avoid silent drift)
        ell = float(z['ell']) if 'ell' in z.files else ELL
        tau = float(z['tau']) if 'tau' in z.files else TAU
        nsub = max(1, round((T / (nt - 1)) / args.dt_internal))
        r = args.r_train if name == 'train' else args.r_eval
        if args.n_frames <= 0:
            frames = list(range(nt))                              # full trajectory
        else:
            frames = sorted(set(int(round(x)) for x in np.linspace(0, nt - 1, args.n_frames + 1)[1:]))
        F = len(frames)
        print(f"[{name}] N={a.shape[0]} r={r} frames={F}/{nt} nx={nx} T={T} eps={eps} "
              f"ell={ell} tau={tau} nsub={nsub} -> {a.shape[0]*r*F*nx*4/1e6:.0f} MB", flush=True)
        jvps, dirs = gen_bank(a, r, eps, nt, T, nsub, frames, gen, dev, ell=ell, tau=tau, bs=args.bs)
        out = os.path.join(args.data_dir, f'{name}_jvps.npz')
        np.savez(out, jvps=jvps, dirs=dirs, frames=np.array(frames, dtype=np.int64),
                 ell=ell, tau=tau)                                # record prior for traceability
        print(f"  saved {out}", flush=True)


if __name__ == '__main__':
    main()
