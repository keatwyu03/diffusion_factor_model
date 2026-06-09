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
from h_function import HFunctionMLP

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default=None,
                    help="Path to diffusion checkpoint. If omitted, uses latest in model_results/.")
parser.add_argument("--h_ckpt", type=str, default=None,
                    help="Path to H-function checkpoint. If omitted, uses latest in model_results/.")
parser.add_argument("--guidance_scale", type=float, default=1.0)
parser.add_argument("--n_samples", type=int, default=2000)
parser.add_argument("--event_threshold", type=float, default=cfg.H_EVENT_THRESHOLD)
args = parser.parse_args()

# ── Auto-resolve checkpoints ───────────────────────────────────────────────────
def _latest(pattern):
    hits = glob.glob(os.path.join(ROOT, "model_results", "**", pattern), recursive=True)
    return max(hits, key=os.path.getmtime) if hits else None

if args.ckpt is None:
    args.ckpt = _latest("model-epoch-*.pt")
    if args.ckpt is None:
        raise FileNotFoundError("No diffusion checkpoint found in model_results/. Run train.py first.")
if args.h_ckpt is None:
    args.h_ckpt = _latest("hfunction.pt")
    if args.h_ckpt is None:
        raise FileNotFoundError("No hfunction.pt found in model_results/. Run train.py first.")

print(f"Diffusion checkpoint : {args.ckpt}")
print(f"H-function checkpoint: {args.h_ckpt}")

# ── Settings ───────────────────────────────────────────────────────────────────
TICKERS      = cfg.TICKERS
PLOT_TICKERS = [t for t in TICKERS if t != TICKERS[cfg.H_EVENT_ASSET_IDX]]
CSV_PATH     = cfg.CSV_PATH
TEST_DAYS    = cfg.TEST_DAYS
HEIGHT, WIDTH = len(TICKERS), 1
min_dim = min(HEIGHT, WIDTH)
DIM_MULTS = (cfg.DIM_MULTS_LARGE  if min_dim >= 32 else cfg.DIM_MULTS_MEDIUM if min_dim >= 16
        else cfg.DIM_MULTS_SMALL  if min_dim >= 8  else cfg.DIM_MULTS_TINY   if min_dim >= 4
        else cfg.DIM_MULTS_MINIMAL)
RESULTS_DIR  = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Data ───────────────────────────────────────────────────────────────────────
df      = pd.read_csv(CSV_PATH, index_col="Date")
data_np = df[TICKERS].dropna().values

train_np  = data_np[:-TEST_DAYS]
test_np   = data_np[-TEST_DAYS:]
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

ckpt = torch.load(args.ckpt, map_location=cfg.DEVICE, weights_only=True)
diffusion.load_state_dict(ckpt["model"])
ema = EMA(diffusion, beta=cfg.EMA_DECAY, update_every=4)
ema.load_state_dict(ckpt["ema"])
diffusion_model = ema.ema_model.to(cfg.DEVICE)
diffusion_model.eval()

h_model = HFunctionMLP(asset_dim=len(TICKERS), embed_dim=cfg.H_EMBED_DIM).to(cfg.DEVICE)
h_model.load_state_dict(torch.load(args.h_ckpt, map_location=cfg.DEVICE))
h_model.eval()

# ── Conditional sampling ───────────────────────────────────────────────────────
def p_sample_conditional(x, t, guidance_scale):
    b = x.shape[0]
    batched_t = torch.full((b,), t, device=x.device, dtype=torch.long)
    with torch.no_grad():
        model_mean, _, model_log_variance, _ = diffusion_model.p_mean_variance(
            x=x, t=batched_t, clip_denoised=False
        )
    x_in  = x.detach().requires_grad_(True)
    h_val = h_model(x_in, batched_t.float())
    log_h = torch.log(h_val.clamp(min=1e-6)).sum()
    grad  = torch.autograd.grad(log_h, x_in)[0]
    posterior_variance = torch.exp(model_log_variance)
    guided_mean = model_mean + guidance_scale * posterior_variance * grad.detach()
    noise = torch.randn_like(x) if t > 0 else 0.0
    return guided_mean + (0.5 * model_log_variance).exp() * noise

def sample_conditional(n_samples, guidance_scale):
    x = torch.randn((n_samples, cfg.MODEL_CHANNELS, HEIGHT, WIDTH), device=cfg.DEVICE)
    for t in reversed(range(0, diffusion_model.num_timesteps)):
        x = p_sample_conditional(x, t, guidance_scale)
    return diffusion_model.unnormalize(x)

print(f"Generating {args.n_samples} conditional samples (guidance_scale={args.guidance_scale})...")
cond = sample_conditional(args.n_samples, args.guidance_scale)
gen_np = cond.reshape(args.n_samples, len(TICKERS)).detach().cpu().numpy()

# ── Diagnostics ────────────────────────────────────────────────────────────────
event_rate_gen   = (gen_np[:, 0] > args.event_threshold).mean()
event_rate_train = (train_std[:, 0] > args.event_threshold).mean()
print(f"\nEvent rate — generated: {event_rate_gen:.3f}  |  real train: {event_rate_train:.3f}")
for i, ticker in enumerate(TICKERS):
    print(f"\n{ticker.upper()}")
    print(f"  real train  — mean: {train_std[:, i].mean():.3f}  std: {train_std[:, i].std():.3f}")
    print(f"  conditional — mean: {gen_np[:, i].mean():.3f}  std: {gen_np[:, i].std():.3f}")

# ── Plot: marginals — one plot per asset, cols=train|test ─────────────────────
plot_idx = [TICKERS.index(t) for t in PLOT_TICKERS]

for ticker, idx in zip(PLOT_TICKERS, plot_idx):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for col, (split, real_data) in enumerate([("train", train_std), ("test", test_std)]):
        ax    = axes[col]
        r     = real_data[:, idx]
        g     = gen_np[:, idx]
        x_min = min(r.min(), g.min()) - 0.3
        x_max = max(r.max(), g.max()) + 0.3
        x     = np.linspace(x_min, x_max, 500)
        for vals, color, label in [
            (r, "darkorange", f"Real (n={len(r)})"),
            (g, "steelblue",  f"Conditional (n={len(g)})"),
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
    fig.suptitle(
        f"Conditional Generation (unemp > {args.event_threshold}, guidance={args.guidance_scale}) — {ticker.upper()}",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    out_marginals = os.path.join(RESULTS_DIR, f"conditional_marginal_{ticker}.png")
    plt.savefig(out_marginals, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_marginals}")
