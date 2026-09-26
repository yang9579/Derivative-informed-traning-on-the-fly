# Derivative-Informed Training of Neural Operators On-the-Fly via Sketched Tangent Consistency

This repository contains the code for the paper

> **Derivative-Informed Training of Neural Operators On-the-Fly via Sketched Tangent Consistency**<br>
> Xinhan Yang, Lu Lu, and Shancong Mou

Derivative-informed training supervises the input-output sensitivities of a neural operator,
which matters when the operator is used as a differentiable surrogate in inverse problems,
PDE-constrained optimization, design, and control. Existing methods rely on derivative labels
generated offline. The sketched tangent-consistency loss (sTCL) builds this supervision during
training instead: at each step it samples a few input perturbation directions, computes the
surrogate's Jacobian--vector products by forward-mode automatic differentiation, and penalizes
the forward sensitivity equation of the PDE, so no tangent labels are stored. The loss is
conditioned according to the tangent operator of each PDE. With this, sTCL reaches solution
and Jacobian accuracy comparable to offline derivative-informed training on five PDE
benchmarks without the offline derivative-data pipeline.

Each benchmark trains the same FNO backbone with three objectives: a data-only FNO, offline
derivative-informed training (offline DI, DIFNO-style), and sTCL.

## Benchmarks

| Benchmark | Directory | Tangent operator | sTCL form | Training sizes N | Epochs |
|---|---|---|---|---|---:|
| Helmholtz | [`helmholtz_difno/`](helmholtz_difno/README.md) | indefinite elliptic | MINRES-5 target matching | 128, 512, 1024 | 2000 |
| Nonlinear diffusion--reaction | [`nonlinear_diffusion_difno/`](nonlinear_diffusion_difno/README.md) | SPD elliptic | DST inverse-Laplacian residual | 128, 512, 1024 | 2000 |
| Burgers | [`burgers_1d/`](burgers_1d/README.md) | parabolic, advective | RHS-normalized space--time residual | 32, 128, 512 | 2000 |
| Allen--Cahn | [`allen_cahn_1d/`](allen_cahn_1d/README.md) | parabolic, phase field | raw space--time L2 residual | 512, 1024, 2048 | 1500 |
| Steady Navier--Stokes | [`navier_stokes_2d/`](navier_stokes_2d/README.md) | saddle point, non-self-adjoint | Leray--Oseen-preconditioned residual | 512, 1024, 2048 | 5000 |

Each benchmark directory has its own README with the PDE setup, the commands to generate the
data and train the three methods, the default settings, the reported metrics, and the loops
that reproduce the paper's tables.

## Installation

```bash
git clone https://github.com/yang9579/Derivative-informed-traning-on-the-fly.git
cd Derivative-informed-traning-on-the-fly
pip install -r requirements.txt
```

The code needs Python 3.9 or newer with PyTorch 2.x, NumPy, and SciPy. The trainers use a GPU
when one is available.

## Quick start

Every benchmark has two steps: generate the data (and, for offline DI, the tangent-label bank),
then train. For example, Helmholtz with N=512 training samples:

```bash
cd helmholtz_difno
python generate_data_difno.py --n_train 2048 --n_val 128 --n_test 128
python generate_train_jvps.py --n_train 2048 --n_dirs 289 --n_workers 16   # offline DI only
python train_helmholtz.py --method fno     --n_use 512 --seed 0 --save_dir results/N512_fno_seed0
python train_helmholtz.py --method difno   --n_use 512 --seed 0 --save_dir results/N512_difno_seed0
python train_helmholtz.py --method fminres --n_use 512 --seed 0 --save_dir results/N512_stcl_seed0
```

Each run writes its checkpoints, training history, and a `metrics.json` with the test
function-value and Jacobian errors to its `--save_dir`.

## Methods

- **Data-only FNO**: mean-squared error on the response.
- **Offline DI** (DIFNO-style): data loss plus matching the FNO Jacobian--vector products to
  tangent labels that were solved offline and stored for `r` directions per training sample;
  `q=4` of them are sampled per update.
- **sTCL**: data loss plus the sketched tangent-consistency loss, built on the fly from `q=4`
  unlabeled probe directions per update. It needs no tangent labels.

Both derivative-informed methods use matched probe-direction distributions and the same
EMA-normalized weighting: the derivative loss is multiplied by
`lambda * EMA(data loss) / EMA(derivative loss)` (decay 0.99), so `lambda` is the target ratio
of derivative to data loss for either method.

## Common protocol

FNO backbones with 4 Fourier layers and width 32, Adam at learning rate `1e-3`, batch size 32,
ReduceLROnPlateau, gradient clipping at 1.0, and training seeds 0, 1, 2. Validation and test
splits have 128 samples each. The function-value error is the mean per-sample relative L2
error, and the Jacobian error is the per-(sample, direction) average of
`||J(a) v - J_theta(a) v|| / ||J(a) v||` over a fixed test probe bank. Every benchmark reports
the checkpoint with the lowest validation error (`best_model.pth`), and the same checkpoint
supplies both errors.

## Repository layout

| Path | Purpose |
|---|---|
| `helmholtz_difno/`, `nonlinear_diffusion_difno/`, `burgers_1d/`, `allen_cahn_1d/`, `navier_stokes_2d/` | One directory per benchmark: data generation, tangent-label banks, sTCL loss, and trainer. |
| `fno2d.py` | 2-D FNO backbone used by Helmholtz, nonlinear diffusion, Burgers, and Navier--Stokes. |
| `train.py` | Shared training loop (data-only and sTCL) and the field / Jacobian error evaluation. |
| `train_difno_1d.py` | Offline-DI training loop for Burgers. |
| `runtime_profile.py` | Optional CUDA timing of training components; inactive unless `STCL_PROFILE_JSON=/path/profile.json` is set. |
| `requirements.txt` | Python dependencies. |

Datasets, tangent-label banks, checkpoints, results, logs, and figures are written inside the
benchmark directories and are ignored by Git.

## Citation

If you use this code, please cite the paper:

```bibtex
@misc{yang2026derivative,
  title  = {Derivative-Informed Training of Neural Operators On-the-Fly via Sketched Tangent Consistency},
  author = {Yang, Xinhan and Lu, Lu and Mou, Shancong},
  year   = {2026}
}
```
