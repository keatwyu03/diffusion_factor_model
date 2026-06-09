import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config.config as cfg

parser = argparse.ArgumentParser()
parser.add_argument("--loss_path", type=str, default=None,
                    help="Path to train_losses.csv. Defaults to latest_run.txt model dir.")
args = parser.parse_args()

if args.loss_path:
    model_dir = os.path.dirname(args.loss_path)
    loss_path = args.loss_path
else:
    latest_run_txt = os.path.join(ROOT, "latest_run.txt")
    if not os.path.exists(latest_run_txt):
        raise FileNotFoundError("No --loss_path given and latest_run.txt not found. Run train.py first.")
    with open(latest_run_txt) as f:
        model_dir = f.read().strip()
    loss_path = os.path.join(model_dir, cfg.LOSS_FILENAME)

h_loss_path = os.path.join(model_dir, "h_function_losses.csv")

score_df = pd.read_csv(loss_path)
has_h = os.path.exists(h_loss_path)
if has_h:
    h_df = pd.read_csv(h_loss_path)

ncols = 3 if has_h else 1
fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
if ncols == 1:
    axes = [axes]

# Score function loss
axes[0].plot(score_df["epoch"], score_df["loss"], linewidth=1.5, color="steelblue")
axes[0].set_xlabel("Epoch", fontsize=12)
axes[0].set_ylabel("Loss", fontsize=12)
axes[0].set_title("Score Function Loss", fontsize=13, fontweight="bold")
axes[0].grid(True, alpha=0.3)

if has_h:
    # H-function loss
    axes[1].plot(h_df["epoch"], h_df["loss"], linewidth=1.5, color="darkorange")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Loss", fontsize=12)
    axes[1].set_title("H-Function Loss", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    # H-function accuracy and pos_ratio
    axes[2].plot(h_df["epoch"], h_df["accuracy"], linewidth=1.5, color="seagreen", label="Accuracy")
    axes[2].plot(h_df["epoch"], h_df["pos_ratio"], linewidth=1.5, color="mediumpurple", linestyle="--", label="Pos Rate")
    axes[2].set_xlabel("Epoch", fontsize=12)
    axes[2].set_ylabel("Value", fontsize=12)
    axes[2].set_title("H-Function Accuracy & Pos Rate", fontsize=13, fontweight="bold")
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

fig.tight_layout()

os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
out_path = os.path.join(cfg.RESULTS_DIR, "train_loss.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved to: {out_path}")
