# Burgers

Parabolic, advection-dominated benchmark with a time-dependent control:

$$
u_t + u\,u_x = \nu\,u_{xx} + f(t),\qquad x\in(0,1),\ t\in(0,1],\qquad
u(0,t)=u(1,t)=0,\quad u(x,0)=0,\quad \nu=10^{-2}.
$$

The spatially uniform control is `f(t) = sum_{j=1}^{16} c_j exp(-(t - t_j)^2 / (2 * 0.2^2))` with
`c_j ~ N(0, 1.5^2)` and uniformly spaced centers `t_j`; it is broadcast in `x` as the input
channel. Reference trajectories come from an implicit finite-difference solver with three Picard
iterations per step (128 spatial unknowns, 200 time steps) and are interpolated to the
64 x 100 `(x, t)` grid used by the FNO. Nonfinite draws and trajectories with `||u||_inf > 20`
are rejected and resampled.

For a control perturbation `v(t)`, the tangent `w = DS(f)[v]` satisfies
`w_t + (u w)_x - nu w_xx = v`. sTCL evaluates this residual at the surrogate trajectory
(detached `u_theta`) with centered finite differences at interior points and noninitial frames,
and divides the per-sample residual energy by the energy of the right-hand side. The optional
spatial preconditioner (`--precond_gamma`) is off in the paper.

## Files

| File | Purpose |
|---|---|
| `adjoint_burgers.py` | Implicit-Picard finite-difference Burgers solver (also runnable as a standalone adjoint optimal-control example). |
| `generate_burgers_fno_data.py` | Samples controls, solves the PDE, and writes `fno_data/burgers_fno_data.npz`. |
| `generate_burgers_jvps_v2.py` | JVPs by forward-mode differentiation through the implicit-Picard solver (used by v3). |
| `generate_burgers_jvps_v3.py` | Offline-DI training bank and test probe bank (direction loop batched with `vmap`). |
| `stcl_burgers.py` | sTCL residual loss. |
| `train_fno_burgers.py` | Training and evaluation for all three methods. |

## 1. Generate data

```bash
cd burgers_1d
python generate_burgers_fno_data.py --sigma_c 1.5
python generate_burgers_jvps_v3.py --n_dirs 200 --save_suffix _r200_v3 --sigma_c 1.5 --resample_unstable
```

This writes `fno_data/burgers_fno_data.npz` (2048 / 128 / 128 train / validation / test samples)
and `fno_data/{train,test}_jvps_r200_v3.npz` (200 directions each). `--resample_unstable`
replaces any sample whose tangent solve overflows and records the replaced indices in the data
file; no replacement is expected with the paper settings. Before training, the trainer checks
that the control coefficients match the paper distribution (`sigma_c=1.5`, seed 42); pass
`--skip_distribution_check` to train on a different dataset.

## 2. Train

```bash
# Data-only FNO
python train_fno_burgers.py --method mse   --N_train 512 --seed 0 --save_dir results/N512_fno_seed0
# Offline DI: q=4 directions per update from the stored 200-direction bank
python train_fno_burgers.py --method difno --N_train 512 --seed 0 --lam_jvp 1.0 \
  --save_dir results/N512_difno_seed0
# sTCL: RHS-normalized space-time residual, q=4 fresh directions per update
python train_fno_burgers.py --method stcl  --N_train 512 --seed 0 --lam_tan 1.0 \
  --save_dir results/N512_stcl_seed0
```

Defaults follow the paper: FNO with 4 layers, width 32 and 12 modes; Adam (lr `1e-3`), batch 32,
2000 epochs, ReduceLROnPlateau (patience 50, factor 0.5), gradient clipping at 1.0, `q=4`
(`--n_jvp_per_step` for offline DI, `--q_tan` for sTCL), and EMA-normalized weight `lambda=1`.
The banks are read from `fno_data/` by default; use `--data_path`, `--jvp_path`, and
`--test_jvp_path` for other locations.

## 3. Outputs

Each run directory contains `last_model.pth`, `best_model.pth` (lowest validation MSE),
`train_history.csv`, `config.json`, and `metrics.json` with

- `test_rel_err`: mean per-sample relative L2 error on the test set;
- `test_jac_rel_err`: Jacobian error, the per-(sample, direction) mean of
  `||J v - J_theta v|| / ||J v||` over the 200 test directions.

Both are evaluated on the validation-selected checkpoint `best_model.pth`, as in the paper.

## Reproducing the paper

Main tables (`N in {32, 128, 512}`, seeds 0, 1, 2, `lambda=1` for both derivative methods):

```bash
for seed in 0 1 2; do
  for n in 32 128 512; do
    python train_fno_burgers.py --method mse --N_train $n --seed $seed \
      --save_dir results/N${n}_fno_seed${seed}
    python train_fno_burgers.py --method difno --lam_jvp 1.0 --N_train $n --seed $seed \
      --save_dir results/N${n}_difno_seed${seed}
    python train_fno_burgers.py --method stcl --lam_tan 1.0 --N_train $n --seed $seed \
      --save_dir results/N${n}_stcl_seed${seed}
  done
done
```

The long-run convergence check uses the same commands with `--N_train 512 --epochs 4000`.
