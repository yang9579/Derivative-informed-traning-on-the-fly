# Allen--Cahn

Unforced phase-field benchmark on the periodic unit interval, initial condition to full
trajectory:

$$
u_t=\varepsilon^2 u_{xx}+u-u^3,\qquad x\in(0,1)\ \text{periodic},\qquad u(x,0)=a(x),\qquad
\varepsilon=0.03,\quad T=5 .
$$

The operator maps the initial field `a(x)` to the trajectory `u(x, t)` at 40 stored frames
(including `t=0`). Initial conditions are periodic Matern Gaussian random fields (correlation
length 0.07, smoothness exponent 4), normalized samplewise and mapped through `tanh`. The
reference solver uses 128 Fourier points and IMEX Euler stepping (diffusion implicit in Fourier
space, `u - u^3` explicit, internal step `1e-3`).

For an initial perturbation `v`, the tangent satisfies `w_t = eps^2 w_xx + (1 - 3u^2) w`,
`w(0) = v`. The surrogate is a 2-D FNO over `(t, x)` with a hard initial-condition ansatz
`u_theta(a)(t, x) = a(x) + (t / T) N_theta(a, t / T, x)`, so the initial condition and its JVP
are exact. sTCL uses the raw space--time L2 tangent residual over all noninitial frames,
obtained as the forward-mode JVP of the strong-form residual (centered differences in time, a
second-order backward difference at `T`, spectral Laplacian in space). Probe directions are
unit-normalized Matern fields matched to the input prior.

## Files

| File | Purpose |
|---|---|
| `ac1d_time_solver.py` | Differentiable spectral IMEX solver, initial-condition sampler, and reference JVPs. |
| `generate_data_1d.py` | Writes `{train,val,test}.npz` (`a`, trajectory `U`, and the PDE/prior metadata). |
| `generate_jvps_1d.py` | Writes `{train,val,test}_jvps.npz`: offline-DI labels (train) and fixed Jacobian probes (val/test), over the full trajectory. |
| `fno_ac1d.py` | FNO with the hard initial-condition ansatz and the strong-form residual. |
| `stcl_ac1d.py` | sTCL loss and the matched Matern direction sampler. |
| `train_ac1d_full.py` | Training and evaluation for all three methods. |

## 1. Generate data

```bash
cd allen_cahn_1d
python generate_data_1d.py --out_dir data_1d --n_train 2048 --n_val 128 --n_test 128
python generate_jvps_1d.py --data_dir data_1d --r_train 200 --r_eval 200 --n_frames 0
```

`--n_frames 0` stores JVPs on all trajectory frames. The training bank is used by offline DI;
the validation bank is used for per-epoch Jacobian monitoring and the test bank for the
reported Jacobian error.

## 2. Train

```bash
# Data-only FNO
python train_ac1d_full.py --method fno   --data_dir data_1d --n_use 512 --seed 0 \
  --save_dir results/N512_fno_seed0
# Offline DI: q=4 directions per update from the stored 200-direction bank
python train_ac1d_full.py --method difno --data_dir data_1d --n_use 512 --seed 0 --lam 2.0 \
  --save_dir results/N512_difno_seed0
# sTCL: raw space-time residual, q=4 fresh directions per update
python train_ac1d_full.py --method stcl  --data_dir data_1d --n_use 512 --seed 0 --lam 2.0 \
  --save_dir results/N512_stcl_seed0
```

Defaults follow the paper: FNO with 4 layers, width 32 and 12 x 12 modes; Adam (lr `1e-3`),
batch 32, 1500 epochs, ReduceLROnPlateau on the validation error (patience 20, factor 0.5),
gradient clipping at 1.0, `q=4`, and EMA-normalized weight `lambda=2`. Other options:

- `--val_jvp_every` (default 1): epoch interval of the validation Jacobian curve in
  `history.csv`; 0 disables it. It is evaluation only and does not affect training.
- `--resume_from RUN_DIR`: continue an interrupted run from its `last_model.pth`
  (saved every `--save_last_every` epochs).
- `--device`.

## 3. Outputs

Each run directory contains `last_model.pth`, `best_model.pth` (lowest validation trajectory
error), `history.csv` / `history.json`, `config.json`, and `metrics.json` with

- `test_rel`: mean per-sample relative L2 error over the full trajectory;
- `test_jac`: Jacobian error, the per-(sample, direction) mean of
  `||J v - J_theta v|| / ||J v||` over all stored frames and the 200 test directions.

Both are evaluated on the validation-selected checkpoint `best_model.pth`, as in the paper.

## Reproducing the paper

`lambda=2` was selected for both derivative methods on the PDE-level validation grid
`{0.1, 0.3, 0.5, 1.0, 2.0}` (N=512, seed 0) and is shared across all training sizes.

Main tables (`N in {512, 1024, 2048}`, seeds 0, 1, 2, `lambda=2`):

```bash
for seed in 0 1 2; do
  for n in 512 1024 2048; do
    python train_ac1d_full.py --method fno --data_dir data_1d --n_use $n --seed $seed \
      --save_dir results/N${n}_fno_seed${seed}
    for method in difno stcl; do
      python train_ac1d_full.py --method $method --data_dir data_1d --n_use $n --seed $seed \
        --lam 2.0 --save_dir results/N${n}_${method}_seed${seed}
    done
  done
done
```

The long-run convergence check uses the same commands with `--n_use 1024 --epochs 4000`.
