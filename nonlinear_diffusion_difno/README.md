# Nonlinear diffusion--reaction

SPD elliptic benchmark (DIFNO setup):

$$
-\nabla\cdot\big(e^{a(x)}\nabla u(x)\big)+u(x)^3=f(x),\qquad u|_{\partial\Omega}=0,\qquad \Omega=(0,1)^2 .
$$

The operator maps the log-diffusivity `a` to `u` on a 65 x 65 grid. The input prior is
`C = (10/3 I - 1/30 Delta)^(-2)` with Neumann covariance boundary conditions (cosine KLE, 40 modes
per coordinate, 800 largest-variance mode pairs). The source is the sum of four Gaussian bumps
(amplitude 10, width 0.1). Reference states are computed with centered finite differences and
Newton iteration on the 63 x 63 interior grid.

The tangent operator `A_y(u) du = -div(e^a grad du) + 3 u^2 du` is symmetric positive definite
but stiff. sTCL measures the tangent residual

$$
r_\theta = A_y(u_\theta)\,DS_\theta(a)[\delta a] - \nabla\cdot(e^a\,\delta a\,\nabla u_\theta)
$$

in the metric `M = D^(-1/2) (-Delta_h)^(-1) D^(-1/2)`, `D = e^a`, where the inverse Dirichlet
Laplacian is applied exactly with a DST-I. The state `u_theta` is detached inside the residual.
No tangent solve and no stored labels are used.

## Files

| File | Purpose |
|---|---|
| `generate_data_difno.py` | Samples inputs, solves the PDE, and writes `data/{train,val,test}.npz`. With `--jvps`, `test.npz` also stores the probe directions and reference JVPs. |
| `generate_train_jvps.py` | Offline-DI label bank: `data/train_jvps_chunks/`, `data/train_jvp_dirs.npz`. |
| `fd_preconditioner.py` | Finite-difference tangent operator and the DST preconditioner `M`. |
| `stcl_difno.py` | Residual sTCL without the DST metric (RHS-normalized residual, white-noise probes), used by `train_fno_difno.py --lam_tan > 0`. |
| `train_fno_difno.py` | Data-only FNO (`--lam_tan 0`). |
| `train_difno.py` | Offline DI. |
| `train_stcl_preconditioned_loss.py` | sTCL with the DST-preconditioned residual (paper method). |

## 1. Generate data

```bash
cd nonlinear_diffusion_difno
python generate_data_difno.py --n_train 4096 --n_val 128 --n_test 128 --jvps --n_jvp_dirs 10
python generate_train_jvps.py --n_dirs 289 --chunk_size 256
```

`--jvps` adds the 10 test probe directions used for the Jacobian error. The 289-direction
training bank is written in chunks and is only needed for offline DI. Add `--val_jvps` to the
first command if you also want a validation Jacobian error in the sTCL metrics.

## 2. Train

```bash
# Data-only FNO
python train_fno_difno.py --n_use 512 --lam_tan 0.0 --seed 0 --save_dir results/N512_fno_seed0
# Offline DI: q=4 directions per update from the stored 289-direction bank
python train_difno.py --n_use 512 --lam_jvp 1.0 --n_jvp_per_step 4 --seed 0 \
  --save_dir results/N512_difno_seed0
# sTCL: DST-preconditioned residual, q=4 fresh directions per update
python train_stcl_preconditioned_loss.py --n_use 512 --lam_jvp 1.0 --q 4 --seed 0 \
  --save_dir results/N512_stcl_seed0
```

Defaults follow the paper: FNO with 4 layers, width 32 and 8 modes; Adam (lr `1e-3`), batch 32,
2000 epochs, ReduceLROnPlateau (patience 50, factor 0.5), gradient clipping at 1.0, and an
EMA-normalized derivative weight `lambda=1`.

For the residual without the DST metric (not the form reported in the paper), use
`python train_fno_difno.py --n_use 512 --lam_tan 1.0 --q_tan 4`.

## 3. Outputs

Each run directory contains `last_model.pth`, `best_model.pth` (lowest validation MSE),
`train_history.csv`, and `metrics.json` with

- `test_rel_err`: mean per-sample relative L2 error on the test set;
- `test_jac_rel_err`: Jacobian error, the per-(sample, direction) mean of
  `||J v - J_theta v|| / ||J v||` over the 10 test directions.

Both are evaluated on the validation-selected checkpoint `best_model.pth`, as in the paper.

## Reproducing the paper

Main tables (`N in {128, 512, 1024}`, seeds 0, 1, 2, `lambda=1` for both derivative methods):

```bash
for seed in 0 1 2; do
  for n in 128 512 1024; do
    python train_fno_difno.py --n_use $n --lam_tan 0.0 --seed $seed \
      --save_dir results/N${n}_fno_seed${seed}
    python train_difno.py --n_use $n --lam_jvp 1.0 --n_jvp_per_step 4 --seed $seed \
      --save_dir results/N${n}_difno_seed${seed}
    python train_stcl_preconditioned_loss.py --n_use $n --lam_jvp 1.0 --q 4 --seed $seed \
      --save_dir results/N${n}_stcl_seed${seed}
  done
done
```

The long-run convergence check uses the same commands with `--n_use 1024 --epochs 4000`.
