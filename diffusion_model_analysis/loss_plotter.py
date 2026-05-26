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
    loss_path = args.loss_path
else:
    latest_run_txt = os.path.join(ROOT, "latest_run.txt")
    if not os.path.exists(latest_run_txt):
        raise FileNotFoundError("No --loss_path given and latest_run.txt not found. Run train.py first.")
    with open(latest_run_txt) as f:
        model_dir = f.read().strip()
    loss_path = os.path.join(model_dir, cfg.LOSS_FILENAME)

df = pd.read_csv(loss_path)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["epoch"], df["loss"], linewidth=1.5, color="steelblue")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("Training Loss", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)
fig.tight_layout()

os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
out_path = os.path.join(cfg.RESULTS_DIR, "train_loss.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved to: {out_path}")
