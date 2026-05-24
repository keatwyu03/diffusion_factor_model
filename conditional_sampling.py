"""
Conditional sampling script for Diffusion Factor Model.
Loads a trained diffusion model and H-function, generates samples
conditioned on unemp > threshold via classifier guidance.
Run train.py first to produce both checkpoints.
"""
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

ROOT = os.path.dirname(os.path.abspath(__file__))

from diffusion_factor_model.diffusion_factor_model import Unet, GaussianDiffusion
import config.config as cfg
from h_function import HFunctionMLP

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default=None,
                    help="Path to diffusion model checkpoint. If omitted, reads latest_run.txt.")
parser.add_argument("--h_ckpt", type=str, default=None,
                    help="Path to H-function checkpoint. If omitted, reads latest_run.txt.")
parser.add_argument("--guidance_scale", type=float, default=1.0,
                    help="Guidance strength")
parser.add_argument("--n_samples", type=int, default=2000,
                    help="Number of conditional samples to generate")
parser.add_argument("--event_threshold", type=float, default=cfg.H_EVENT_THRESHOLD,
                    help="Standardized unemp threshold for the event")
args = parser.parse_args()

# ── Resolve checkpoint paths from latest_run.txt if not provided ───────────────
if args.ckpt is None or args.h_ckpt is None:
    latest_run_path = os.path.join(ROOT, "latest_run.txt")
    if not os.path.exists(latest_run_path):
        raise FileNotFoundError(
            "No --ckpt provided and latest_run.txt not found. Run train.py first."
        )
    with open(latest_run_path) as f:
        model_dir = f.read().strip()

    if args.ckpt is None:
        candidates = glob.glob(os.path.join(model_dir, "model-epoch-*.pt"))
        if not candidates:
            # Fall back to searching all of model_results/
            candidates = glob.glob(os.path.join(ROOT, "model_results", "**", "model-epoch-*.pt"), recursive=True)
        if not candidates:
            raise FileNotFoundError(f"No model checkpoints found in {model_dir} or model_results/")
        args.ckpt = max(candidates, key=os.path.getmtime)

    if args.h_ckpt is None:
        h_candidate = os.path.join(model_dir, "hfunction.pt")
        if not os.path.exists(h_candidate):
            # Fall back to most recently modified hfunction.pt across all runs
            all_h = glob.glob(os.path.join(ROOT, "model_results", "**", "hfunction.pt"), recursive=True)
            h_candidate = max(all_h, key=os.path.getmtime) if all_h else h_candidate
        args.h_ckpt = h_candidate

print(f"Diffusion checkpoint : {args.ckpt}")
print(f"H-function checkpoint: {args.h_ckpt}")

# ── Settings ───────────────────────────────────────────────────────────────────
TICKERS       = ["unemp", "sp500", "baa"]
CSV_PATH      = os.path.join(ROOT, "explore", "macro_data_new.csv")
TEST_DAYS     = 3000
HEIGHT, WIDTH = 3, 1
DIM_MULTS     = (1,)
RESULTS_DIR   = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Data (for diagnostics and plots only) ─────────────────────────────────────
df       = pd.read_csv(CSV_PATH, index_col="Date")
data_np  = df[TICKERS].dropna().values

train_np = data_np[:-TEST_DAYS]
test_np  = data_np[-TEST_DAYS:]

mean      = train_np.mean(axis=0)
std       = train_np.std(axis=0)
train_std = (train_np - mean) / std
test_std  = (test_np  - mean) / std

print(f"Train samples: {len(train_std)}, Test samples: {len(test_std)}")

# ── Diffusion model ────────────────────────────────────────────────────────────
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

if not os.path.exists(args.ckpt):
    raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}\nRun train.py first.")

print(f"Loading diffusion checkpoint: {args.ckpt}")
ckpt = torch.load(args.ckpt, map_location=cfg.DEVICE, weights_only=True)
diffusion.load_state_dict(ckpt["model"])
ema = EMA(diffusion, beta=cfg.EMA_DECAY, update_every=4)
ema.load_state_dict(ckpt["ema"])
diffusion_model = ema.ema_model.to(cfg.DEVICE)
diffusion_model.eval()

# ── H-function ─────────────────────────────────────────────────────────────────
if not os.path.exists(args.h_ckpt):
    raise FileNotFoundError(f"H-function checkpoint not found: {args.h_ckpt}\nRun train.py first.")

print(f"Loading H-function checkpoint: {args.h_ckpt}")
h_model = HFunctionMLP(asset_dim=3, embed_dim=cfg.H_EMBED_DIM).to(cfg.DEVICE)
h_model.load_state_dict(torch.load(args.h_ckpt, map_location=cfg.DEVICE))
h_model.eval()

# ── Conditional sampling (DDPM + classifier guidance) ─────────────────────────
def p_sample_conditional(x, t, guidance_scale):
    b = x.shape[0]
    batched_t = torch.full((b,), t, device=x.device, dtype=torch.long)

    with torch.no_grad():
        model_mean, _, model_log_variance, _ = diffusion_model.p_mean_variance(
            x=x, t=batched_t, clip_denoised=False
        )

    x_in = x.detach().requires_grad_(True)
    h_val = h_model(x_in, batched_t.float())
    log_h = torch.log(h_val.clamp(min=1e-6)).sum()
    grad = torch.autograd.grad(log_h, x_in)[0]

    posterior_variance = torch.exp(model_log_variance)
    guided_mean = model_mean + guidance_scale * posterior_variance * grad.detach()

    noise = torch.randn_like(x) if t > 0 else 0.0
    return guided_mean + (0.5 * model_log_variance).exp() * noise


def sample_conditional(n_samples, guidance_scale):
    shape = (n_samples, cfg.MODEL_CHANNELS, HEIGHT, WIDTH)
    x = torch.randn(shape, device=cfg.DEVICE)

    for t in reversed(range(0, diffusion_model.num_timesteps)):
        x = p_sample_conditional(x, t, guidance_scale)

    return diffusion_model.unnormalize(x)


print(f"Generating {args.n_samples} conditional samples (guidance_scale={args.guidance_scale})...")
cond_samples = sample_conditional(args.n_samples, args.guidance_scale)
gen_np = cond_samples.reshape(args.n_samples, len(TICKERS)).detach().cpu().numpy()

# ── Diagnostics ────────────────────────────────────────────────────────────────
event_rate_gen   = (gen_np[:, 0] > args.event_threshold).mean()
event_rate_train = (train_std[:, 0] > args.event_threshold).mean()
print(f"\nEvent rate — generated: {event_rate_gen:.3f}  |  real train: {event_rate_train:.3f}")

for i, ticker in enumerate(TICKERS):
    print(f"\n{ticker.upper()}")
    print(f"  real train  — mean: {train_std[:, i].mean():.3f}  std: {train_std[:, i].std():.3f}")
    print(f"  conditional — mean: {gen_np[:, i].mean():.3f}  std: {gen_np[:, i].std():.3f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(TICKERS), figsize=(14, 4))

for i, (ticker, ax) in enumerate(zip(TICKERS, axes)):
    r = train_std[:, i]
    g = gen_np[:, i]
    x_min = min(r.min(), g.min()) - 0.3
    x_max = max(r.max(), g.max()) + 0.3
    x_vals = np.linspace(x_min, x_max, 500)
    for vals, color, label in [
        (r, "darkorange", f"Real train (n={len(r)})"),
        (g, "steelblue",  f"Conditional (n={len(g)})"),
    ]:
        kde = gaussian_kde(vals, bw_method="silverman")
        ax.plot(x_vals, kde(x_vals), linewidth=2, color=color, label=label)
        ax.fill_between(x_vals, kde(x_vals), alpha=0.12, color=color)
    if ticker == "unemp":
        ax.axvline(args.event_threshold, color="red", linestyle="--",
                   linewidth=1.2, label=f"threshold={args.event_threshold}")
    ax.set_title(f"{ticker.upper()}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Standardized Value", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle(
    f"Conditional Generation — unemp > {args.event_threshold}  "
    f"(guidance_scale={args.guidance_scale})",
    fontsize=13, fontweight="bold"
)
fig.tight_layout()
out_path = os.path.join(RESULTS_DIR, "conditional_marginals.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved to {out_path}")
