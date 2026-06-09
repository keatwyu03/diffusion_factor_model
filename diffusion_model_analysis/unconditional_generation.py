import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
from ema_pytorch import EMA
import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from diffusion_factor_model.diffusion_factor_model import Unet, GaussianDiffusion
import config.config as cfg

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default=None,
                    help="Path to checkpoint file. Auto-resolves to latest if omitted.")
args = parser.parse_args()

if args.ckpt is None:
    hits = glob.glob(os.path.join(ROOT, "model_results", "**", "model-epoch-*.pt"), recursive=True)
    if not hits:
        raise FileNotFoundError("No checkpoint found in model_results/. Run train.py first.")
    args.ckpt = max(hits, key=os.path.getmtime)

print(f"Using checkpoint: {args.ckpt}")

# ── Settings ──────────────────────────────────────────────────────────────────
TICKERS       = cfg.TICKERS
CSV_PATH      = cfg.CSV_PATH
TEST_DAYS     = cfg.TEST_DAYS
N_SAMPLES     = 5000
HEIGHT        = len(TICKERS)
WIDTH         = 1
min_dim       = min(HEIGHT, WIDTH)
DIM_MULTS     = (cfg.DIM_MULTS_LARGE  if min_dim >= 32 else cfg.DIM_MULTS_MEDIUM if min_dim >= 16
            else cfg.DIM_MULTS_SMALL  if min_dim >= 8  else cfg.DIM_MULTS_TINY   if min_dim >= 4
            else cfg.DIM_MULTS_MINIMAL)
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
os.makedirs(RESULTS_DIR, exist_ok=True)
rows = []
for i, ticker in enumerate(TICKERS):
    qs_r = np.quantile(train_std[:, i], [.01,.05,.5,.95,.99]).round(3)
    qs_g = np.quantile(gen_np[:, i],    [.01,.05,.5,.95,.99]).round(3)
    rows.append([ticker.upper(), "real train",
                 f"{train_std[:, i].mean():.3f}", f"{train_std[:, i].std():.3f}",
                 str(qs_r)])
    rows.append(["", "generated",
                 f"{gen_np[:, i].mean():.3f}", f"{gen_np[:, i].std():.3f}",
                 str(qs_g)])

col_labels = ["Asset", "Split", "Mean", "Std", "q[1,5,50,95,99]"]
fig_d, ax_d = plt.subplots(figsize=(14, 0.4 * len(rows) + 1.5))
ax_d.axis("off")
tbl = ax_d.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="left")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.auto_set_column_width(col=list(range(len(col_labels))))
fig_d.suptitle("Unconditional Generation — Diagnostics", fontsize=12, fontweight="bold")
fig_d.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "unconditional_diagnostics.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Plot settings ─────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
PLOT_TICKERS = [t for t in TICKERS if t != TICKERS[cfg.H_EVENT_ASSET_IDX]]
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