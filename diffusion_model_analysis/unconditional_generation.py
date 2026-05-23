import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
from ema_pytorch import EMA
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from diffusion_factor_model.diffusion_factor_model import Unet, GaussianDiffusion
import config.config as cfg

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True,
                    help="Path to checkpoint file, e.g. model_results/.../model-epoch-600.pt")
args = parser.parse_args()

# ── Settings ──────────────────────────────────────────────────────────────────
TICKERS       = ["unemp", "sp500", "baa"]
CSV_PATH      = os.path.join(ROOT, "explore", "macro_data_new.csv")
TEST_DAYS     = 3000
N_SAMPLES     = 5000
HEIGHT, WIDTH = 3, 1        # (N, 3 assets) → (N, 1, 3, 1) for 2D UNet
DIM_MULTS     = (1,)        # min_dim=1 → DIM_MULTS_MINIMAL
CKPT_DIR      = os.path.join(ROOT, "checkpoints", "dfm_macro")
CKPT_PATH     = os.path.join(CKPT_DIR, f"model-epoch-{cfg.EPOCHS}.pt")
RESULTS_DIR   = os.path.join(ROOT, "results")

# ── Data ──────────────────────────────────────────────────────────────────────
df       = pd.read_csv(CSV_PATH, index_col="Date")
data_np  = df[TICKERS].dropna().values          # (N, 3)

train_np = data_np[:-TEST_DAYS]                 # (N-3000, 3)
test_np  = data_np[-TEST_DAYS:]                 # (3000, 3)

# Standardize using train statistics only
mean      = train_np.mean(axis=0)
std       = train_np.std(axis=0)
train_std = (train_np - mean) / std
test_std  = (test_np  - mean) / std

print(f"Train samples: {len(train_std)}, Test samples: {len(test_std)}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = Unet(
    dim=cfg.MODEL_DIM,
    channels=cfg.MODEL_CHANNELS,
    filter_size=cfg.MODEL_FILTER_SIZE,
    dim_mults=DIM_MULTS,
)
diffusion = GaussianDiffusion(
    model,
    image_size=(HEIGHT, WIDTH),
    timesteps=cfg.TIMESTEPS,
    objective=cfg.OBJECTIVE,
    beta_schedule=cfg.BETA_SCHEDULE,
    auto_normalize=cfg.AUTO_NORMALIZE,
)

# ── Load checkpoint ───────────────────────────────────────────────────────────
if not os.path.exists(args.ckpt):
    raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}\nRun train.py first.")

print(f"Loading checkpoint: {args.ckpt}")
ckpt = torch.load(args.ckpt, map_location=cfg.DEVICE, weights_only=True)
diffusion.load_state_dict(ckpt["model"])
ema = EMA(diffusion, beta=cfg.EMA_DECAY, update_every=4)
ema.load_state_dict(ckpt["ema"])
diffusion_for_sampling = ema.ema_model.to(cfg.DEVICE)

# ── Generate ──────────────────────────────────────────────────────────────────
print(f"Generating {N_SAMPLES} unconditional samples...")
diffusion_for_sampling.eval()
with torch.no_grad():
    uncond = diffusion_for_sampling.sample(batch_size=N_SAMPLES).cpu()  # (N, 1, 3, 1)

gen_np = uncond.reshape(N_SAMPLES, len(TICKERS)).numpy()  # (N, 3)

# ── Diagnostics ───────────────────────────────────────────────────────────────
for i, ticker in enumerate(TICKERS):
    print(f"\n{ticker.upper()}")
    print(f"  real train — mean: {train_std[:, i].mean():.3f}  std: {train_std[:, i].std():.3f}")
    print(f"  generated  — mean: {gen_np[:, i].mean():.3f}  std: {gen_np[:, i].std():.3f}")
    print(f"  real train q[1,5,50,95,99]: {np.quantile(train_std[:, i], [.01,.05,.5,.95,.99]).round(3)}")
    print(f"  generated  q[1,5,50,95,99]: {np.quantile(gen_np[:, i],   [.01,.05,.5,.95,.99]).round(3)}")

# ── Plot settings ─────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
PLOT_TICKERS = ["sp500", "baa"]
plot_idx     = [TICKERS.index(t) for t in PLOT_TICKERS]

# ── Marginal distributions: rows=assets, cols=train|test ──────────────────────
fig, axes = plt.subplots(len(PLOT_TICKERS), 2, figsize=(12, 4 * len(PLOT_TICKERS)))

for row, (ticker, idx) in enumerate(zip(PLOT_TICKERS, plot_idx)):
    for col, (split, real_data) in enumerate([("train", train_std), ("test", test_std)]):
        ax    = axes[row, col]
        r     = real_data[:, idx]
        g     = gen_np[:, idx]
        x_min = min(r.min(), g.min()) - 0.3
        x_max = max(r.max(), g.max()) + 0.3
        x     = np.linspace(x_min, x_max, 500)
        for vals, color, label in [
            (r, "darkorange", f"Real (n={len(r)})"),
            (g, "steelblue",  f"Generated (n={len(g)})"),
        ]:
            kde = gaussian_kde(vals, bw_method="silverman")
            ax.plot(x, kde(x), linewidth=2, color=color, label=label)
            ax.fill_between(x, kde(x), alpha=0.12, color=color)
        split_label = "In-Sample (Train)" if split == "train" else "Out-of-Sample (Test)"
        ax.set_title(f"{ticker.upper()} — {split_label}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Standardized Return", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

fig.suptitle("Unconditional Generation — Marginal Distributions", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "unconditional_marginals.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Joint distributions: left=train, right=test ────────────────────────────────
i, j   = plot_idx[0], plot_idx[1]
t1, t2 = PLOT_TICKERS[0], PLOT_TICKERS[1]

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

for col, (split, real_data) in enumerate([("train", train_std), ("test", test_std)]):
    ax = axes2[col]
    real_t1, real_t2 = real_data[:, i], real_data[:, j]
    gen_t1,  gen_t2  = gen_np[:, i],    gen_np[:, j]
    x_min = min(real_t1.min(), gen_t1.min()) - 0.5
    x_max = max(real_t1.max(), gen_t1.max()) + 0.5
    y_min = min(real_t2.min(), gen_t2.min()) - 0.5
    y_max = max(real_t2.max(), gen_t2.max()) + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 80), np.linspace(y_min, y_max, 80))
    grid   = np.vstack([xx.ravel(), yy.ravel()])
    zz_real = gaussian_kde(np.vstack([real_t1, real_t2]), bw_method="silverman")(grid).reshape(xx.shape)
    zz_gen  = gaussian_kde(np.vstack([gen_t1,  gen_t2]),  bw_method="silverman")(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz_real, levels=10, cmap="Oranges", alpha=0.5)
    ax.contour( xx, yy, zz_real, levels=10, colors="darkorange", linewidths=0.8, alpha=0.8)
    ax.contourf(xx, yy, zz_gen,  levels=10, cmap="Blues",   alpha=0.5)
    ax.contour( xx, yy, zz_gen,  levels=10, colors="steelblue",  linewidths=0.8, alpha=0.8)
    split_label = "In-Sample (Train)" if split == "train" else "Out-of-Sample (Test)"
    ax.set_title(f"Joint {t1.upper()} × {t2.upper()} — {split_label}", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"{t1.upper()} Standardized Return", fontsize=10)
    ax.set_ylabel(f"{t2.upper()} Standardized Return", fontsize=10)
    ax.legend(handles=[
        Patch(color="darkorange", alpha=0.7, label=f"Real (n={len(real_t1)})"),
        Patch(color="steelblue",  alpha=0.7, label=f"Generated (n={len(gen_t1)})"),
    ], fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

fig2.suptitle("Unconditional Generation — Joint Distribution", fontsize=13, fontweight="bold")
fig2.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "unconditional_joint.png"), dpi=150, bbox_inches="tight")
plt.close()
