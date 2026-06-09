import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# ── Args (all optional — checkpoints auto-resolved from model_results/) ────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",            type=str,   default=None)
parser.add_argument("--h_ckpt",          type=str,   default=None)
parser.add_argument("--guidance_scale",  type=float, default=1.0)
parser.add_argument("--n_samples",       type=int,   default=2000)
parser.add_argument("--event_threshold", type=float, default=cfg.H_EVENT_THRESHOLD)
args = parser.parse_args()

def _latest(pat):
    hits = glob.glob(os.path.join(ROOT, "model_results", "**", pat), recursive=True)
    return max(hits, key=os.path.getmtime) if hits else None

if args.ckpt is None:
    args.ckpt = _latest("model-epoch-*.pt")
    if args.ckpt is None:
        raise FileNotFoundError("No diffusion checkpoint found. Run train.py first.")
if args.h_ckpt is None:
    args.h_ckpt = _latest("hfunction.pt")
    if args.h_ckpt is None:
        raise FileNotFoundError("No hfunction.pt found. Run train.py first.")

print(f"Diffusion checkpoint : {args.ckpt}")
print(f"H-function checkpoint: {args.h_ckpt}")

# ── Data ──────────────────────────────────────────────────────────────────────
TICKERS   = cfg.TICKERS
N         = len(TICKERS)
HEIGHT    = N

df        = pd.read_csv(cfg.CSV_PATH, index_col="Date")
train_np  = df[TICKERS].dropna().values[:-cfg.TEST_DAYS]
mean, std = train_np.mean(0), train_np.std(0)
train_std = (train_np - mean) / std

event_lbl = TICKERS[cfg.H_EVENT_ASSET_IDX].upper()
real_cond = train_std[train_std[:, cfg.H_EVENT_ASSET_IDX] > args.event_threshold]
print(f"Train: {len(train_std)} samples | High-{event_lbl}: {len(real_cond)} ({len(real_cond)/len(train_std)*100:.1f}%)")

# ── Load models ───────────────────────────────────────────────────────────────
min_dim   = min(HEIGHT, 1)
dim_mults = (cfg.DIM_MULTS_LARGE   if min_dim >= 32 else
             cfg.DIM_MULTS_MEDIUM  if min_dim >= 16 else
             cfg.DIM_MULTS_SMALL   if min_dim >= 8  else
             cfg.DIM_MULTS_TINY    if min_dim >= 4  else
             cfg.DIM_MULTS_MINIMAL)

model = Unet(dim=cfg.MODEL_DIM, channels=cfg.MODEL_CHANNELS,
             filter_size=cfg.MODEL_FILTER_SIZE, dim_mults=dim_mults)
diffusion = GaussianDiffusion(model, image_size=(HEIGHT, 1), timesteps=cfg.TIMESTEPS,
                              objective=cfg.OBJECTIVE, beta_schedule=cfg.BETA_SCHEDULE,
                              auto_normalize=cfg.AUTO_NORMALIZE)

ckpt = torch.load(args.ckpt, map_location=cfg.DEVICE, weights_only=True)
diffusion.load_state_dict(ckpt["model"])
ema = EMA(diffusion, beta=cfg.EMA_DECAY, update_every=4)
ema.load_state_dict(ckpt["ema"])
dm = ema.ema_model.to(cfg.DEVICE)
dm.eval()

hm = HFunctionMLP(asset_dim=N, embed_dim=cfg.H_EMBED_DIM).to(cfg.DEVICE)
hm.load_state_dict(torch.load(args.h_ckpt, map_location=cfg.DEVICE))
hm.eval()

# ── Unconditional samples ─────────────────────────────────────────────────────
print(f"\nGenerating {args.n_samples} unconditional samples...")
with torch.no_grad():
    uncond_np = dm.sample(batch_size=args.n_samples).cpu().reshape(args.n_samples, N).numpy()

# ── Conditional samples (classifier guidance via h-function) ──────────────────
def p_sample_cond(x, t):
    bt = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
    with torch.no_grad():
        mu, _, log_var, _ = dm.p_mean_variance(x=x, t=bt, clip_denoised=False)
    x_in  = x.detach().requires_grad_(True)
    log_h = torch.log(hm(x_in, bt.float()).clamp(min=1e-6)).sum()
    grad  = torch.autograd.grad(log_h, x_in)[0]
    guided_mu = mu + args.guidance_scale * log_var.exp() * grad.detach()
    noise     = torch.randn_like(x) if t > 0 else 0.0
    return guided_mu + (0.5 * log_var).exp() * noise

print(f"Generating {args.n_samples} conditional samples (guidance_scale={args.guidance_scale})...")
x = torch.randn((args.n_samples, cfg.MODEL_CHANNELS, HEIGHT, 1), device=cfg.DEVICE)
for t in reversed(range(dm.num_timesteps)):
    x = p_sample_cond(x, t)
cond_np = dm.unnormalize(x).reshape(args.n_samples, N).detach().cpu().numpy()

# ── Matrices ──────────────────────────────────────────────────────────────────
panels = [
    ("Real Train (all)",                                    train_std),
    (f"Real Train ({event_lbl} > {args.event_threshold})", real_cond),
    ("Unconditional Generated",                             uncond_np),
    ("Conditional Generated",                               cond_np),
]
corr_mats = [(lbl, np.corrcoef(arr.T)) for lbl, arr in panels]
cov_mats  = [(lbl, np.cov(arr.T))     for lbl, arr in panels]

print("\n── Correlation Matrices ──")
for lbl, C in corr_mats:
    print(f"\n{lbl}:")
    print(pd.DataFrame(C, index=TICKERS, columns=TICKERS).round(3).to_string())

print("\n── Covariance Matrices ──")
for lbl, C in cov_mats:
    print(f"\n{lbl}:")
    print(pd.DataFrame(C, index=TICKERS, columns=TICKERS).round(4).to_string())

# ── Plot helper ───────────────────────────────────────────────────────────────
def plot_matrices(mat_list, title, fname, vmin, vmax, fmt):
    tick_lbl  = [t.upper() for t in TICKERS]
    font_size = max(8, min(13, 40 // N))
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (lbl, C) in zip(axes.ravel(), mat_list):
        im = ax.imshow(C, vmin=vmin, vmax=vmax, cmap="RdBu_r")
        ax.set_xticks(range(N)); ax.set_xticklabels(tick_lbl, fontsize=10)
        ax.set_yticks(range(N)); ax.set_yticklabels(tick_lbl, fontsize=10)
        ax.set_title(lbl, fontsize=11, fontweight="bold", pad=8)
        for r in range(N):
            for c in range(N):
                v = C[r, c]
                ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=font_size,
                        fontweight="bold", color="white" if abs(v) > 0.6 * vmax else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"{title}\n(conditional: {event_lbl} > {args.event_threshold},  guidance = {args.guidance_scale})",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    out = os.path.join(ROOT, "results", fname)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out}")

# ── Save both figures ─────────────────────────────────────────────────────────
all_cov_vals = np.concatenate([C.ravel() for _, C in cov_mats])
cov_lim = float(np.abs(all_cov_vals).max())

plot_matrices(corr_mats, "Asset Correlation Matrices — Real vs Generated",
              "correlation_matrices.png", vmin=-1, vmax=1, fmt="{:.2f}")
plot_matrices(cov_mats,  "Asset Covariance Matrices — Real vs Generated",
              "covariance_matrices.png", vmin=-cov_lim, vmax=cov_lim, fmt="{:.3f}")
