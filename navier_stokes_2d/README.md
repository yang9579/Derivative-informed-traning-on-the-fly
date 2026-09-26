# Steady Navier--Stokes

Steady incompressible flow on the unit square, driven by a distributed forcing `u`:

$$
-\mu\Delta y+(y\cdot\nabla)y+\nabla p=u,\qquad \nabla\cdot y=0,\qquad y|_{\partial\Omega}=0,
\qquad \Omega=(0,1)^2,\quad \mu=0.1,
$$

with the pressure gauge `int p = 0`. The operator maps the forcing to velocity and pressure,
`S: u -> (y, p)`. The reported function and Jacobian errors use the velocity; the pressure is
an auxiliary output with weight 0.05 in the data loss.

Forcings come from a solenoidal stream-function prior (`k, l = 1..6`, coefficients decaying
like `(k^2 + l^2)^(-3/2)`), scaled to `||u||_inf ~ U(16, 24)`. The 64 x 64 centered
finite-difference solver uses a stabilized Stokes initial guess followed by Newton iteration;
the continuity row is stabilized as `D y - alpha h_x h_y L p = 0` with `alpha = 1e-3` to remove
the collocated pressure checkerboard mode.

The tangent `(w, pi) = DS(u)[s]` solves the Oseen saddle system. sTCL differentiates the
momentum, continuity, and boundary residuals of the surrogate. It applies an approximate Leray
projector and an inverse Oseen-symbol metric (characteristic speed 3, shift 0.005) to the
momentum block, adds `0.05 * mean(r_div^2) + mean(r_bc^2)`, and uses no tangent solve or
stored label.

## Files

| File | Purpose |
|---|---|
| `navier_stokes_2d.py` | Finite-difference operators, forcing prior, and residual helpers. |
| `ns_forward.py` | Newton solver for the steady state. |
| `generate_data_forward.py` | Samples forcings, solves the PDE, and writes `data/{train,val,test}.npz`. |
| `generate_jvps.py` | Reference velocity tangents from the stabilized Oseen system: `data/{train,val,test}_jvps.npz`. |
| `fno_navier_stokes_2d.py` | FNO wrapper predicting `(y1, y2, p)` from normalized forcing channels. |
| `stcl_navier_stokes_2d.py` | sTCL loss, direction sampler, and the Leray--Oseen metric. |
| `train_navier_stokes_2d.py` | Training and evaluation for all three methods. |

## 1. Generate data

```bash
cd navier_stokes_2d
python generate_data_forward.py --workers 16    # data/{train,val,test}.npz (2048 / 128 / 128, mu = 0.1)
python generate_jvps.py                         # data/{train,val,test}_jvps.npz
```

The defaults of `generate_data_forward.py` are the paper settings (`--mu 0.1 --alpha 1e-3
--stream_kmax 6 --stream_decay 1.5 --vel_amp_min 16 --vel_amp_max 24`). `generate_jvps.py`
stores 200 directions for the training bank (offline DI) and 16 for validation and test.

The training bank is the expensive part. It can be split over sample ranges, for example on
several machines; when `data/train_jvps.npz` is absent, the trainer merges
`data/train_jvps_shards/shard_*.npz` automatically:

```bash
python generate_jvps.py --splits train --start 0 --stop 64 \
  --output data/train_jvps_shards/shard_00000_00064.npz
# ... repeat for [64, 128), ..., [1984, 2048)
python generate_jvps.py --splits val test
```

## 2. Train

```bash
# Data-only FNO
python train_navier_stokes_2d.py --method naive --n_use 512 --seed 0 --save_dir results/N512_naive_seed0
# Offline DI: q=4 directions per update from the stored 200-direction bank
python train_navier_stokes_2d.py --method difno --n_use 512 --seed 0 --save_dir results/N512_difno_seed0
# sTCL: Leray--Oseen-preconditioned residual, q=4 fresh directions per update
python train_navier_stokes_2d.py --method stcl  --n_use 512 --seed 0 --save_dir results/N512_stcl_seed0
```

Defaults follow the paper: FNO with 4 layers, width 32 and 12 modes; Adam (lr `1e-3`), batch
32, 5000 epochs, ReduceLROnPlateau on the validation velocity error (patience 50, factor 0.5),
gradient clipping at 1.0, `q=4`, pressure weight `--w_p 0.05`, and the Leray--Oseen metric
(`--precond leray_oseen`). Other options:

- `--jvp_weight`: EMA-normalized derivative weight. If omitted, the validation-selected paper
  value for `N in {512, 1024, 2048}` is used: 0.5 / 0.25, 1.0 / 0.25, and 0.5 / 0.5
  (offline DI / sTCL).
- `--jvp_every` (default 50): epoch interval of the validation Jacobian curve; 0 disables it.
- `--resume`: continue from `save_dir/ckpt.pth` (written every `--ckpt_every` epochs).
- `--data_dir`, `--device`.

## 3. Outputs

Each run directory contains `last_model.pth`, `best_model.pth` (lowest validation velocity
error), `ckpt.pth`, `train_history.csv`, `config.json`, and `metrics.json`. For the
validation-selected checkpoint (`best`) and the final checkpoint (`last`), `metrics.json`
stores

- `state_rel`: mean per-sample relative L2 error of the velocity on the test set;
- `jvp_state_rel`: velocity Jacobian error, the per-(sample, direction) mean of
  `||J s - J_theta s|| / ||J s||` over the 16 test directions;
- `p_rel`: mean-subtracted pressure error.

The paper reports `best.state_rel` and `best.jvp_state_rel`.

## Reproducing the paper

Main tables (`N in {512, 1024, 2048}`, seeds 0, 1, 2, paper-selected `lambda`):

```bash
for seed in 0 1 2; do
  for n in 512 1024 2048; do
    for method in naive difno stcl; do
      python train_navier_stokes_2d.py --method $method --n_use $n --seed $seed \
        --jvp_every 0 --save_dir results/N${n}_${method}_seed${seed}
    done
  done
done
```

The long-run convergence check uses `--n_use 2048 --epochs 5000 --jvp_weight 0.5` for both
derivative methods.
