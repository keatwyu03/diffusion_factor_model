# Diffusion Factor Model — Project Overview

## What It Does

Trains a DDPM-style diffusion model (2D U-Net) on macro financial cross-sections to learn the joint distribution of `(unemp, sp500, baa)` returns. After training, generates synthetic samples and evaluates how well the learned distribution matches real data via marginal KDE plots and pairwise joint density contours.

---

## Directory Structure

```
diffusion_factor_model/
├── config/
│   └── config.py                          # All hyperparameters — edit here, nowhere else
├── diffusion_factor_model/
│   ├── diffusion_factor_model.py          # Unet, GaussianDiffusion, Trainer, WarmUpCosineAnnealingWarmRestarts
│   └── attend.py                          # Flash/standard attention module
├── eval/
│   ├── mean_cov.py                        # Sample mean/cov with Bayes-Stein, OLSE, Ledoit-Wolf shrinkage
│   ├── simulation_eval.py                 # Frobenius norm error on latent subspace recovery
│   ├── mv_portfolio_eval.py               # Mean-variance portfolio optimization via MOSEK
│   └── ft_portfolio_eval.py               # Factor timing portfolios (PCA, POET, RP-PCA)
├── diffusion_model_analysis/
│   ├── unconditional_generation.py        # Load checkpoint → generate → plot marginals + joints
│   └── conditional_generation.py          # (placeholder for conditional generation)
├── explore/
│   ├── macro_data_new.csv                 # Main dataset (FRED + yfinance)
│   └── import_data.py                     # Script that originally built macro_data_new.csv
├── simulation_experiment_data/
│   └── training_data_example.npy          # Example simulation data (num_samples, H, W)
├── empirical_analysis_data/
│   └── training_data_example.npy          # Example empirical data (num_samples, assets)
├── train.py                               # Main training script — run this first
├── requirements.txt
└── setup.py
```

**Generated at runtime:**
```
checkpoints/dfm_macro/                     # Model checkpoints (model-epoch-N.pt, norm_stats.npy)
results/                                   # Output plots
model_results/                             # Legacy output from original train.py
samples/                                   # Legacy sampled .npy files
```

---

## Data

**File:** `explore/macro_data_new.csv`
**Source:** FRED API + yfinance (built by `explore/import_data.py`)

**Columns:**
| Column | Description |
|---|---|
| `unemp` | US unemployment rate (monthly, forward-filled to daily) |
| `unemp_flag` | Binary: unemployment moved ≥ 0.3pp in a month |
| `sp500` | S&P 500 log return (daily) |
| `baa` | Moody's BAA corporate bond yield log return (daily) |
| `baa_flag` | Binary: BAA log return ≥ 5% in absolute value |

**Active tickers for training:** `["unemp", "sp500", "baa"]` — set in `config.TICKERS`

**Total rows:** ~19,195 daily observations (1950–present)

**Train/test split:** last `TEST_DAYS = 3000` days held out as test set

**Preprocessing (done inside `train.py`):**
1. Load CSV, select `TICKERS` columns, drop NaNs
2. Z-score standardize using **all data** statistics (mean/std per column)
3. Reshape `(N, 3)` → `(N, 1, 3, 1)` for 2D U-Net input

> **Note:** `train.py` standardizes on all data. `unconditional_generation.py` standardizes using train-only statistics for a cleaner train/test comparison in plots. This is a known inconsistency — generated samples are in "all-data" standardized space while the plot's real distributions use train-only stats.

---

## Model Architecture

### Input Shape
Each sample = one day's returns for 3 assets, treated as a `3×1` "image":

```
(batch, channels, height, width) = (B, 1, 3, 1)

  channel 0:
    row 0 → unemp
    row 1 → sp500
    row 2 → baa
```

The model learns the **joint daily cross-sectional distribution** of the 3 assets. No temporal structure — each day is an independent sample.

### U-Net (`diffusion_factor_model/diffusion_factor_model.py`)
- 2D convolutional U-Net (`nn.Conv2d` throughout)
- `dim = 256` base channels
- `dim_mults = (1,)` — minimal config, no downsampling (forced by `min_dim = min(3,1) = 1`)
- `filter_size = 7` with padding to preserve spatial dims
- Sinusoidal time embeddings → MLP → injected via ResNet scale/shift
- Full self-attention at the single resolution level
- EMA of model weights maintained during training (`ema_decay = 0.999`)

### GaussianDiffusion
- **Type:** DDPM (discrete timesteps)
- **Objective:** `pred_noise` — model predicts the noise ε added at each step
- **Beta schedule:** cosine
- **Timesteps:** 200
- **Sampling:** DDPM reverse process (or DDIM if `sampling_timesteps < timesteps`)
- **Auto-normalize:** False — data is pre-standardized before entering the model

---

## Training

### Run
```bash
python train.py --data_path explore/macro_data.npy --seed 42 --gpu 0
```

Or first convert CSV to `.npy`:
```bash
python3 -c "
import pandas as pd, numpy as np
df = pd.read_csv('explore/macro_data_new.csv', index_col='Date')
np.save('explore/macro_data.npy', df[['unemp','sp500','baa']].dropna().values)
"
python train.py --data_path explore/macro_data.npy --seed 42 --gpu 0
```

### On SLURM cluster (e.g. Solar)
```bash
#!/bin/bash
#SBATCH --job-name=dfm_train
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out

cd ~/diffusion_factor_model
python train.py --data_path explore/macro_data.npy --seed 42 --gpu 0
```
```bash
sbatch run_train.sh
```

### Checkpoint output
Saved to `model_results/<exp_id>/model-epoch-N.pt` every `SAVE_INTERVAL` epochs.

Each checkpoint contains:
- `model` — raw U-Net weights
- `optimizer` — AdamW state
- `ema` — EMA model weights (used for sampling/inference)

---

## Learning Rate Schedule

```
LR
1e-4 |        *
     |      *   *
     |    *       *
     |  *           *
1e-6 |*______________*_______
     0   50         450  600
         ↑           ↑    ↑
      warmup     cosine  flat
      ends       ends   (stays at 1e-5)
```

- **Epochs 0–50:** linear warmup `0 → 1e-4`
- **Epochs 50–450:** cosine annealing `1e-4 → 1e-5` (`COSINE_CYCLE_LENGTH = 400`)
- **Epochs 450–600:** flat at `1e-5` (cosine phase done, scheduler does nothing)

`T_MULT = 1` means each restart cycle would be the same length (400 epochs), but there are not enough epochs for a restart to occur.

**Weight decay:** constant `0.01` throughout via AdamW — not scheduled, separate from LR.

---

## Config Reference (`config/config.py`)

| Parameter | Value | Description |
|---|---|---|
| `TICKERS` | `["unemp","sp500","baa"]` | Assets to train on |
| `CSV_PATH` | `explore/macro_data_new.csv` | Input data |
| `TEST_DAYS` | `3000` | Days held out for test set |
| `N_SAMPLES` | `5000` | Samples to generate for analysis |
| `CKPT_DIR` | `checkpoints/dfm_macro` | Where analysis scripts look for checkpoints |
| `RESULTS_DIR` | `results/` | Where plots are saved |
| `MODEL_DIM` | `256` | Base U-Net channel width |
| `TIMESTEPS` | `200` | DDPM diffusion steps |
| `BETA_SCHEDULE` | `cosine` | Noise schedule |
| `OBJECTIVE` | `pred_noise` | Model predicts noise ε |
| `EPOCHS` | `600` | Total training epochs |
| `BATCH_SIZE` | `50` | Mini-batch size |
| `LEARNING_RATE` | `1e-4` | Peak LR after warmup |
| `WARMUP_STEPS` | `50` | Linear warmup epochs |
| `COSINE_CYCLE_LENGTH` | `400` | Epochs for one cosine decay sweep |
| `COSINE_LR_MIN` | `1e-5` | LR floor |
| `WEIGHT_DECAY` | `0.01` | AdamW weight decay (constant) |
| `EMA_DECAY` | `0.999` | EMA smoothing factor |
| `SAVE_INTERVAL` | `100` | Save checkpoint every N epochs |
| `AUTO_NORMALIZE` | `False` | Data is pre-standardized — skip internal normalization |

---

## Analysis

### Unconditional Generation
After training, run:
```bash
python diffusion_model_analysis/unconditional_generation.py
```

Loads the checkpoint from `CKPT_DIR`, generates `N_SAMPLES` synthetic daily cross-sections, and produces two figures in `results/`:

1. **`unconditional_marginals.png`** — KDE per asset (real train, real test, generated). X-axis: standardized return. Y-axis: density.
2. **`unconditional_joint.png`** — Pairwise joint density contours for all 3 asset pairs (real vs. generated).

---

## Known Issues

1. **Standardization inconsistency:** `train.py` standardizes on all data; `unconditional_generation.py` standardizes on train-only. Generated samples and plotted real distributions are in slightly different scales.

2. **`SAVE_INTERVAL` must be ≤ `EPOCHS`:** If `SAVE_INTERVAL > EPOCHS`, no checkpoint is ever saved and training is wasted. Default is `100`.

3. **Tiny spatial dims:** Input is `(B, 1, 3, 1)` — only 3 spatial positions. Forces `dim_mults=(1,)` with no downsampling. Model capacity is limited; all expressiveness comes from the 256 feature channels.

4. **MOSEK license required** for portfolio evaluation scripts (`eval/mv_portfolio_eval.py`, `eval/ft_portfolio_eval.py`). Free academic license available at mosek.com.
