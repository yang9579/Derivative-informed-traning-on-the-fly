#!/usr/bin/env python
"""Generate the 1D Allen-Cahn initial->trajectory dataset (fields only).

Input  a(x) ~ smooth Matern GRF (ell, tau, tanh).  Output u(x,t) trajectory from the
differentiable spectral solver.  Saves <split>.npz with a (N,nx), U (N,nt,nx).
JVP banks (for DIFNO / Metric-B) come from generate_jvps_1d.py.
"""
import os, argparse
import numpy as np
import torch

from ac1d_time_solver import solve, grf_ic, EPS, T_FINAL, ELL, TAU, NX, NT


def gen_fields(n, nx, nt, ell, tau, sigma, nsub, eps, T, gen, dev, bs=64):
    A, U = [], []
    for i in range(0, n, bs):
        b = min(bs, n - i)
        a = grf_ic(b, nx, gen, ell=ell, tau=tau, sigma=sigma, tanh=True)
        u = solve(a.to(dev), eps=eps, nt=nt, T=T, n_substep=nsub)
        A.append(a.float()); U.append(u.cpu().float())
    return torch.cat(A), torch.cat(U)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default=os.path.join(os.path.dirname(__file__), 'data_1d'))
    ap.add_argument('--n_train', type=int, default=2048)
    ap.add_argument('--n_val', type=int, default=128)
    ap.add_argument('--n_test', type=int, default=128)
    ap.add_argument('--nx', type=int, default=NX)
    ap.add_argument('--nt', type=int, default=NT)
    ap.add_argument('--ell', type=float, default=ELL)
    ap.add_argument('--tau', type=float, default=TAU)
    ap.add_argument('--sigma', type=float, default=1.0)
    ap.add_argument('--eps', type=float, default=EPS, help='interface sharpness; smaller = harder')
    ap.add_argument('--T', type=float, default=T_FINAL, help='final time; larger = coarsening (harder field)')
    ap.add_argument('--dt_internal', type=float, default=1e-3)
    ap.add_argument('--n_substep', type=int, default=0, help='0 = auto from dt_internal')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device(args.device)
    gen = torch.Generator().manual_seed(args.seed)
    nsub = args.n_substep if args.n_substep > 0 else max(1, round((args.T / (args.nt - 1)) / args.dt_internal))
    print(f"T={args.T} eps={args.eps} ell={args.ell} nx={args.nx} nt={args.nt} n_substep={nsub}", flush=True)

    for name, n in dict(train=args.n_train, val=args.n_val, test=args.n_test).items():
        a, u = gen_fields(n, args.nx, args.nt, args.ell, args.tau, args.sigma, nsub, args.eps, args.T, gen, dev)
        np.savez(os.path.join(args.out_dir, f'{name}.npz'),
                 a=a.numpy(), U=u.numpy(), nt=args.nt, nx=args.nx,
                 eps=args.eps, T=args.T, ell=args.ell, tau=args.tau, sigma=args.sigma)
        well = 100 * (u[:, -1].abs() > 0.9).float().mean().item()
        print(f"{name}: N={n}  a{tuple(a.shape)} U{tuple(u.shape)}  "
              f"u(T) in [{u[:,-1].min():.2f},{u[:,-1].max():.2f}]  well%={well:.1f}  "
              f"({u.numel()*4/1e6:.1f} MB)", flush=True)
    print(f"saved -> {args.out_dir}")


if __name__ == '__main__':
    main()
