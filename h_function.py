"""
H-function for conditional generation (adapted for DFM cross-section data)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps"""

    def __init__(self, embed_dim: int, scale: float = 30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, t):
        if t.ndim == 1:
            t = t[:, None]
        t_proj = t * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)


class HFunctionMLP(nn.Module):
    """
    H-function: P(event | x_t, t) for DFM data.
    Input x has shape (B, 1, 3, 1) or (B, 3); t is discrete diffusion step.
    """

    def __init__(self, asset_dim: int = 3, embed_dim: int = 128):
        super().__init__()
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(asset_dim + embed_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, t):
        if x.dim() == 4:
            x = x.reshape(x.shape[0], -1)  # (B, 1, 3, 1) -> (B, 3)
        t_float = t.float()
        if t_float.dim() == 1:
            t_float = t_float[:, None]
        t_emb = self.time_embed(t_float)
        return self.fc(torch.cat([x, t_emb], dim=1))


class HFunctionTrainer:
    """
    Trains H to predict P(unemp > threshold | x_t, t).

    For each training step:
      - sample a clean x_0 from train_data
      - sample a random diffusion timestep t
      - corrupt x_0 to x_t via q_sample
      - label = 1 if x_0[event_asset_idx] > event_threshold, else 0
      - train H(x_t, t) to predict that label
    """

    def __init__(
        self,
        diffusion,
        asset_dim: int = 3,
        embed_dim: int = 128,
        event_asset_idx: int = 0,
        event_threshold: float = 1.2,
        device: str = "cpu",
    ):
        self.diffusion = diffusion
        self.event_asset_idx = event_asset_idx
        self.event_threshold = event_threshold
        self.device = device
        self.model = HFunctionMLP(asset_dim=asset_dim, embed_dim=embed_dim).to(device)

    def train(
        self,
        train_data: torch.Tensor,
        n_epochs: int = 1000,
        batch_size: int = 2048,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        scheduler_patience: int = 50,
        scheduler_factor: float = 0.5,
        use_wandb: bool = False,
    ) -> None:
        """
        Args:
            train_data: clean training samples, shape (N, 1, 3, 1), standardized
        """
        if use_wandb:
            import wandb

        N = len(train_data)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience)
        loss_fn = nn.MSELoss()

        print("Starting H-function training...")
        for epoch in tqdm(range(n_epochs), desc="H-Function Training"):
            idx = torch.randint(0, N, (batch_size,))  # keep on CPU to index train_data
            x_0 = train_data[idx].to(self.device)  # (B, 1, 3, 1)

            t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=self.device).long()
            with torch.no_grad():
                x_t = self.diffusion.q_sample(x_start=x_0, t=t)

            # label from clean x_0: event fires if unemp > threshold
            unemp = x_0[:, 0, self.event_asset_idx, 0]  # (B,)
            target = (unemp > self.event_threshold).float().unsqueeze(1)  # (B, 1)

            pred = self.model(x_t, t)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(loss)

            acc = ((pred > 0.5).float() == target).float().mean().item()
            pos_ratio = target.mean().item()

            if use_wandb:
                wandb.log({
                    "hfunction/loss": loss.item(),
                    "hfunction/accuracy": acc,
                    "hfunction/pos_ratio": pos_ratio,
                    "hfunction/learning_rate": optimizer.param_groups[0]["lr"],
                    "hfunction/epoch": epoch,
                })

            if epoch % 100 == 0:
                tqdm.write(
                    f"Epoch {epoch:04d} | Loss: {loss.item():.6f} | "
                    f"Acc: {acc:.4f} | PosRatio: {pos_ratio:.3f}"
                )

        print("H-function training complete!")

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
        print(f"H-function saved to {path}")

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"H-function loaded from {path}")
