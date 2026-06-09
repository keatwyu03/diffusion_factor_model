# Diffusion Factor Model — Project Overview

## What It Does

Trains a DDPM-style diffusion model (2D U-Net) on macro financial cross-sections to learn the joint distribution of equity returns and a macro conditioning signal. Supports both **unconditional generation** (sample from the full learned distribution) and **conditional generation** (sample only from regions where the conditioning event exceeds a threshold, via Doob's h-transform / classifier guidance).

The conditioning event is controlled entirely by `config.COND_EVENT` — changing it in the config automatically flows through data preparation, H-function training, and analysis.

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
│   ├── conditional_generation.py          # Plots and analysis for conditional samples
│   ├── cov_estimation.py                  # Covariance estimation analysis
│   └── loss_plotter.py                    # Plots score function loss + H-function loss/accuracy
├── explore/
│   ├── macro_data_new.csv                 # Main dataset (FRED + yfinance)
│   └── import_data.py                     # Script that builds macro_data_new.csv
├── train.py                               # Step 1: train diffusion model + H-function
├── h_function.py                          # HFunctionMLP + HFunctionTrainer
├── conditional_sampling.py                # Step 2: load checkpoints → generate conditional samples
├── requirements.txt
└── setup.py
```

**Generated at runtime:**
```
model_results/<exp_id>/
  ├── model-epoch-N.pt        # Score model checkpoints (every SAVE_INTERVAL epochs)
  ├── hfunction.pt            # H-function weights
  ├── train_losses.csv        # Score model loss per epoch
  └── h_function_losses.csv   # H-function loss, accuracy, pos_ratio per epoch

samples/<exp_id>/             # Generated samples (.pt files)
results/                      # Output plots
latest_run.txt                # Points to the most recent model_results/<exp_id>/ — auto-updated by train.py
```

Each training run creates a **unique timestamped directory** — new runs never overwrite old ones. All analysis scripts read `latest_run.txt` to find the most recent run automatically.

---

## Data

**File:** `explore/macro_data_new.csv`
**Source:** FRED API + yfinance (built by `explore/import_data.py`)

**Columns:**
| Column | Description |
|---|---|
| `CPI` | Monthly CPI month-over-month difference (forward-filled to daily) |
| `CPI_flag` | Binary: `\|CPI diff\|` ≥ `H_EVENT_THRESHOLD` |
| `baa` | Moody's BAA corporate bond yield log return (daily) |
| `baa_flag` | Binary: BAA log return ≥ 5% in absolute value |
| `AAPL` | Apple log return (daily) |
| `ORCL` | Oracle log return (daily) |
| `MSFT` | Microsoft log return (daily) |
| `IBM` | IBM log return (daily) |

**Active tickers for training:** `["CPI", "baa", "AAPL", "ORCL", "MSFT", "IBM"]` — set in `config.TICKERS`

**Conditioning event:** `config.COND_EVENT = "CPI"` — the flag column and H-function event are both driven by this single config value.

**Flag computation:** `import_data.py` uses `cond_series.diff(1)` (raw month-over-month difference) for the COND_EVENT flag, with threshold `cfg.H_EVENT_THRESHOLD`.

**Total rows:** ~19,195 daily observations (1950–present)

**Train/test split:** last `TEST_DAYS = 3000` days held out as test set

**Preprocessing (done inside `train.py`):**
1. Load CSV, select `TICKERS` columns, drop NaNs
2. Z-score standardize using **all data** statistics (mean/std per column)
3. Reshape `(N, D)` → `(N, 1, D, 1)` for 2D U-Net input

> **Note:** `train.py` standardizes on all data. `unconditional_generation.py` standardizes using train-only statistics for a cleaner train/test comparison in plots. This is a known inconsistency — generated samples are in "all-data" standardized space while the plot's real distributions use train-only stats.

---

## Model Architecture

### Input Shape
Each sample = one day's returns for D assets, treated as a `D×1` "image":

```
(batch, channels, height, width) = (B, 1, D, 1)

  channel 0:
    row 0 → CPI
    row 1 → baa
    row 2 → AAPL
    row 3 → ORCL
    row 4 → MSFT
    row 5 → IBM
```

The model learns the **joint daily cross-sectional distribution** of the assets. No temporal structure — each day is an independent sample.

### U-Net (`diffusion_factor_model/diffusion_factor_model.py`)
- 2D convolutional U-Net (`nn.Conv2d` throughout)
- `dim = 256` base channels
- `dim_mults` auto-selected based on spatial size
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
python train.py --data_path explore/macro_data_new.csv --seed 42
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
python train.py --data_path explore/macro_data_new.csv --seed 42
```
```bash
sbatch run_train.sh
```

### Checkpoint output
Each run saves to a unique `model_results/<exp_id>/` directory — no run ever overwrites another. `latest_run.txt` is updated at the end of every run and is what all analysis scripts use to find the most recent model.

Each score model checkpoint contains:
- `model` — raw U-Net weights
- `optimizer` — AdamW state
- `ema` — EMA model weights (used for sampling/inference)

The H-function checkpoint is saved alongside as `hfunction.pt`.

### Skipping H-function training
```bash
python train.py --data_path explore/macro_data_new.csv --skip_hfunction
# or load a pre-trained one
python train.py --data_path explore/macro_data_new.csv --h_ckpt path/to/hfunction.pt
```

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
- **Epochs 450–600:** flat at `1e-5`

**Weight decay:** constant `0.01` throughout via AdamW.

---

## Config Reference (`config/config.py`)

| Parameter | Value | Description |
|---|---|---|
| `TICKERS` | `["CPI","baa","AAPL","ORCL","MSFT","IBM"]` | Assets to train on |
| `COND_EVENT` | `"CPI"` | Conditioning event — drives flag column name, H-function target, and analysis labels |
| `CSV_PATH` | `explore/macro_data_new.csv` | Input data |
| `TEST_DAYS` | `3000` | Days held out for test set |
| `N_SAMPLES` | `5000` | Samples to generate for analysis |
| `RESULTS_DIR` | `results/` | Where plots are saved |
| `MODEL_DIM` | `256` | Base U-Net channel width |
| `TIMESTEPS` | `200` | DDPM diffusion steps |
| `BETA_SCHEDULE` | `cosine` | Noise schedule |
| `OBJECTIVE` | `pred_noise` | Model predicts noise ε |
| `EPOCHS` | `200` | Total training epochs |
| `BATCH_SIZE` | `50` | Mini-batch size |
| `LEARNING_RATE` | `1e-4` | Peak LR after warmup |
| `WARMUP_STEPS` | `50` | Linear warmup epochs |
| `COSINE_CYCLE_LENGTH` | `400` | Epochs for one cosine decay sweep |
| `COSINE_LR_MIN` | `1e-5` | LR floor |
| `WEIGHT_DECAY` | `0.01` | AdamW weight decay (constant) |
| `EMA_DECAY` | `0.999` | EMA smoothing factor |
| `SAVE_INTERVAL` | `50` | Save checkpoint every N epochs |
| `AUTO_NORMALIZE` | `False` | Data is pre-standardized — skip internal normalization |
| `H_EMBED_DIM` | `128` | Time embedding dimension for H-function MLP |
| `H_EPOCHS` | `1000` | H-function training epochs |
| `H_LEARNING_RATE` | `1e-4` | H-function optimizer LR |
| `H_WEIGHT_DECAY` | `1e-4` | H-function AdamW weight decay |
| `H_BATCH_SIZE` | `2048` | H-function mini-batch size |
| `H_EVENT_ASSET_IDX` | `0` | Column index of the conditioned asset (CPI = 0) |
| `H_EVENT_THRESHOLD` | `1.5` | Event fires if `\|COND_EVENT diff\|` > this value |

---

## H-Function (`h_function.py`)

Implements Doob's h-transform for conditional generation. The H-function learns `P(event | x_t, t)` — the probability that a noisy sample `x_t` at diffusion step `t` came from a clean sample where the conditioning event exceeded the threshold.

### Architecture (`HFunctionMLP`)
- Input: flatten `(B, 1, D, 1)` → `(B, D)` concatenated with a Gaussian Fourier time embedding
- MLP: `(D + 128) → 128 → 64 → 1 → Sigmoid`
- Output: scalar probability in `[0, 1]`

### Training (`HFunctionTrainer`)
For each training step:
1. Sample a random batch of clean `x_0` from training data
2. Sample a random diffusion timestep `t ∈ {0, ..., 199}`
3. Corrupt `x_0` → `x_t` via `q_sample` (forward diffusion)
4. Label = 1 if `x_0[COND_EVENT] > H_EVENT_THRESHOLD`, else 0
5. Train H to predict that label from `(x_t, t)` via MSE loss

### Loss Recording
`HFunctionTrainer` stores `self.train_losses` as a list of `(epoch, loss, accuracy, pos_ratio)` tuples. After training, `train.py` saves these to `model_results/<exp_id>/h_function_losses.csv`.

---

## Analysis

### Loss Plotter
```bash
python diffusion_model_analysis/loss_plotter.py
# or specify a path directly:
python diffusion_model_analysis/loss_plotter.py --loss_path model_results/<exp_id>/train_losses.csv
```

Reads from `latest_run.txt` by default. Saves `results/train_loss.png` with three panels:
1. **Score Function Loss** — diffusion model training loss per epoch
2. **H-Function Loss** — H-function MSE loss per epoch
3. **H-Function Accuracy & Pos Rate** — classification accuracy and event positive rate per epoch

Panels 2 and 3 are only shown if `h_function_losses.csv` exists (falls back to 1 panel for old runs).

### Unconditional Generation
```bash
python diffusion_model_analysis/unconditional_generation.py --ckpt path/to/model-epoch-N.pt
```

Generates `N_SAMPLES` synthetic daily cross-sections and produces figures in `results/`:
1. **`unconditional_marginals.png`** — KDE per asset (real train, real test, generated)
2. **`unconditional_diagnostics.png`** — Pairwise joint density contours (real vs. generated)

### Conditional Generation
```bash
python diffusion_model_analysis/conditional_generation.py
```

Reads `latest_run.txt` to find checkpoint and H-function. Runs DDPM reverse diffusion with classifier guidance — at each denoising step adds `guidance_scale * ∇ log h(x_t, t)` to steer samples toward the conditioning event region. Produces figures in `results/`.

---

## Known Issues

1. **Standardization inconsistency:** `train.py` standardizes on all data; `unconditional_generation.py` standardizes on train-only. Generated samples and plotted real distributions are in slightly different scales.

2. **`SAVE_INTERVAL` must be ≤ `EPOCHS`:** If `SAVE_INTERVAL > EPOCHS`, no checkpoint is ever saved and training is wasted. Default is `50`.

3. **Tiny spatial dims:** Input is `(B, 1, D, 1)` — only D spatial positions. For small D this forces minimal `dim_mults` with no downsampling. Model capacity is limited; all expressiveness comes from the feature channels.

4. **MOSEK license required** for portfolio evaluation scripts (`eval/mv_portfolio_eval.py`, `eval/ft_portfolio_eval.py`). Free academic license available at mosek.com.
