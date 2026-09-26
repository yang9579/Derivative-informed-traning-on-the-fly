# Helmholtz

Indefinite elliptic benchmark:

$$
(-\Delta-\kappa^2 e^{2a(x)})\,u(x)=f(x),\qquad u|_{\partial\Omega}=0,\qquad \Omega=(0,1)^2,\quad \kappa=5 .
$$

The operator maps the log-coefficient field `a` to the solution `u` on a 65 x 65 grid.
Inputs are drawn in a 40 x 40 cosine basis from the centered prior
`C = (12.5 I - 0.5 Delta)^(-2)`; the source is a unit Gaussian centered at (0.258, 0.833)
with width 0.08. Reference solutions come from sparse direct solves, and draws with
`||u||_2 > 3 x median` of the candidate pool are rejected to avoid near-resonant outliers.

The tangent equation `(-Delta - kappa^2 e^{2a}) du = 2 kappa^2 e^{2a} da u` is symmetric but
indefinite. sTCL therefore uses **MINRES target matching**: for every sketch direction, five
preconditioned MINRES iterations (shifted-Laplacian preconditioner
`(-Delta_h + 0.5 kappa^2 I)^(-1)`, applied exactly with a DST-I) build an approximate tangent
target on the fly, and the FNO JVP is trained to match it. No tangent labels are stored.

## Files

| File | Purpose |
|---|---|
| `generate_data_difno.py` | Samples inputs, solves the PDE, and writes `data/{train,val,test}.npz`. `test.npz` also contains 8 probe directions and their reference JVPs. |
| `generate_train_jvps.py` | Offline-DI label bank `data/train_jvps_r{r}.npz` (parallel over samples, one LU per sample). |
| `minres_solver.py` | Batched GPU MINRES and the DST shifted-Laplacian preconditioner used by sTCL. |
| `train_helmholtz.py` | Training and evaluation for all three methods. |

## 1. Generate data

```bash
cd helmholtz_difno
python generate_data_difno.py --n_train 2048 --n_val 128 --n_test 128      # data/{train,val,test}.npz
python generate_train_jvps.py --n_train 2048 --n_dirs 289 --n_workers 16   # data/train_jvps_r289.npz
```

The label bank is only needed for offline DI (`--method difno`).

## 2. Train

```bash
# Data-only FNO
python train_helmholtz.py --method fno     --n_use 512 --seed 0 --save_dir results/N512_fno_seed0
# Offline DI: FNO JVPs matched to the stored 289-direction bank
python train_helmholtz.py --method difno   --n_use 512 --seed 0 --save_dir results/N512_difno_seed0
# sTCL: MINRES-5 target matching computed on the fly
python train_helmholtz.py --method fminres --n_use 512 --seed 0 --save_dir results/N512_stcl_seed0
```

Defaults follow the paper: FNO with 4 layers, width 32 and 8 modes; Adam (lr `1e-3`), batch 32,
2000 epochs, ReduceLROnPlateau (patience 50, factor 0.5), gradient clipping at 1.0, `q=4`
directions per update, EMA-normalized derivative weight. Useful options:

- `--lam_jvp`: derivative weight. If omitted, the validation-selected paper value for
  `N in {128, 512, 1024}` is used: 0.1 / 0.1, 0.5 / 0.5, and 0.9 (offline DI) / 0.7 (sTCL).
- `--minres_iters` (default 5) and `--shift_scale` (default 0.5): sTCL MINRES settings.
- `--jvp_bank_r` (default 289): size of the offline-DI bank to load.
- `--device`, `--data_dir`, `--epochs`, `--q`.

## 3. Outputs

Each run directory contains `last_model.pth`, `best_model.pth` (lowest validation MSE),
`train_history.csv`, and `metrics.json` with

- `test_rel_err`: mean per-sample relative L2 error on the test set;
- `test_jac_rel_err`: Jacobian error, the per-(sample, direction) mean of
  `||J v - J_theta v|| / ||J v||` over the 8 test directions.

Both are evaluated on the validation-selected checkpoint `best_model.pth`, as in the paper.

## Reproducing the paper

Main tables (`N in {128, 512, 1024}`, seeds 0, 1, 2):

```bash
for seed in 0 1 2; do
  for n in 128 512 1024; do
    for method in fno difno fminres; do
      python train_helmholtz.py --method $method --n_use $n --seed $seed \
        --save_dir results/N${n}_${method}_seed${seed}
    done
  done
done
```

MINRES iteration ablation (N=512, fixed `lambda=0.1`):

```bash
for m in 1 2 3 5 10 25; do
  for seed in 0 1 2; do
    python train_helmholtz.py --method fminres --minres_iters $m --lam_jvp 0.1 \
      --n_use 512 --seed $seed --save_dir results_minres/N512_m${m}_seed${seed}
  done
done
```

The long-run convergence check uses the same commands with `--n_use 1024 --epochs 4000`.
