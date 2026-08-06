"""Action-chunk VAE for latent-space residual RL.

Played action: a = a_base + λ * decode(w), with w = π_residual(s, a_base).
Fits on warmup base-policy chunks (same data as PCA), then freezes the decoder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _ActionVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden: int = 128, linear: bool = False):
        super().__init__()
        self.latent_dim = int(latent_dim)
        if linear:
            # Linear encoder/decoder ≈ PCA when trained with Gaussian latents.
            self.enc = nn.Identity()
            self.mu = nn.Linear(input_dim, latent_dim)
            self.logvar = nn.Linear(input_dim, latent_dim)
            self.dec = nn.Linear(latent_dim, input_dim)
        else:
            self.enc = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )
            self.mu = nn.Linear(hidden, latent_dim)
            self.logvar = nn.Linear(hidden, latent_dim)
            self.dec = nn.Sequential(
                nn.Linear(latent_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, input_dim),
            )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.dec(self.reparam(mu, logvar)), mu, logvar

    def decode(self, z):
        return self.dec(z)


def _vae_loss(recon, x, mu, logvar, beta: float):
    recon_l = nn.functional.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_l + beta * kl, float(recon_l.detach()), float(kl.detach())


@dataclass
class ResidualActionVAE:
    """Frozen decoder mapping K-dim SAC codes -> (query_freq, action_dim) residual."""

    mean: np.ndarray  # (D,) chunk mean used for training normalization
    std: np.ndarray  # (D,)
    query_freq: int
    action_dim: int
    latent_dim: int
    hidden: int = 128
    linear: bool = False
    state_dict: dict | None = None
    _model: nn.Module | None = None

    @property
    def num_basis(self) -> int:
        return int(self.latent_dim)

    @property
    def feature_dim(self) -> int:
        return int(self.mean.shape[0])

    @property
    def basis_type(self) -> str:
        return "vae_linear" if self.linear else "vae"

    @property
    def explained_variance_ratio(self):
        return None

    def _ensure_model(self) -> nn.Module:
        if self._model is None:
            model = _ActionVAE(
                self.feature_dim, self.latent_dim, hidden=self.hidden, linear=self.linear
            )
            if self.state_dict is not None:
                model.load_state_dict(self.state_dict)
            model.eval()
            self._model = model
        return self._model

    def coeffs_to_chunk(self, c, scale: float = 1.0) -> np.ndarray:
        """Map K SAC latents to residual chunk (no mean add; matches PCA residual)."""
        w = np.asarray(c, dtype=np.float32).reshape(1, -1)
        if w.shape[1] != self.latent_dim:
            raise ValueError(f"Expected latent dim {self.latent_dim}, got {w.shape[1]}")
        model = self._ensure_model()
        with torch.no_grad():
            flat = model.decode(torch.from_numpy(w)).numpy()[0]
        # Unnormalize without +mean so decode(0)≈0 residual after centering train.
        flat = flat * self.std
        return (float(scale) * flat).reshape(self.query_freq, self.action_dim).astype(np.float32)

    def sample_action_chunks(self, n: int, seed: int = 0) -> np.ndarray:
        """Draw n absolute action chunks from the prior z~N(0,I), decode + unnormalize."""
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n, self.latent_dim)).astype(np.float32)
        model = self._ensure_model()
        with torch.no_grad():
            flat = model.decode(torch.from_numpy(z)).numpy()
        chunks = flat * self.std[None, :] + self.mean[None, :]
        return chunks.reshape(n, self.query_freq, self.action_dim).astype(np.float32)

    @classmethod
    def fit(
        cls,
        chunks: np.ndarray,
        latent_dim: int,
        query_freq: int,
        action_dim: int,
        *,
        hidden: int = 128,
        linear: bool = False,
        epochs: int = 120,
        batch_size: int = 256,
        lr: float = 1e-3,
        beta: float = 1e-3,
        seed: int = 0,
    ) -> "ResidualActionVAE":
        data = np.asarray(chunks, dtype=np.float32)
        if data.ndim == 3:
            data = data.reshape(data.shape[0], -1)
        if data.shape[0] < 2:
            raise ValueError(f"Need >=2 action chunks for VAE, got {data.shape[0]}")
        expected = int(query_freq) * int(action_dim)
        if data.shape[1] != expected:
            raise ValueError(f"Chunk feature dim {data.shape[1]} != {expected}")

        torch.manual_seed(seed)
        mean = data.mean(axis=0)
        std = data.std(axis=0).clip(min=1e-6)
        x = torch.from_numpy((data - mean) / std)
        loader = DataLoader(
            TensorDataset(x),
            batch_size=min(int(batch_size), len(x)),
            shuffle=True,
        )
        model = _ActionVAE(expected, int(latent_dim), hidden=int(hidden), linear=bool(linear))
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        for epoch in range(int(epochs)):
            totals = np.zeros(3, dtype=np.float64)
            n = 0
            for (batch,) in loader:
                recon, mu, logvar = model(batch)
                loss, recon_l, kl = _vae_loss(recon, batch, mu, logvar, beta)
                opt.zero_grad()
                loss.backward()
                opt.step()
                totals += np.array([float(loss), recon_l, kl])
                n += 1
            ep1 = epoch + 1
            if ep1 % max(1, int(epochs) // 5) == 0 or ep1 == int(epochs):
                print(
                    f"  vae z={latent_dim} epoch {ep1:03d}  "
                    f"loss={totals[0]/n:.4f} recon={totals[1]/n:.4f} kl={totals[2]/n:.4f}"
                )
        model.eval()
        return cls(
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            query_freq=int(query_freq),
            action_dim=int(action_dim),
            latent_dim=int(latent_dim),
            hidden=int(hidden),
            linear=bool(linear),
            state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
            _model=model,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model = self._ensure_model()
        torch.save(
            {
                "state_dict": model.state_dict(),
                "mean": self.mean,
                "std": self.std,
                "query_freq": self.query_freq,
                "action_dim": self.action_dim,
                "latent_dim": self.latent_dim,
                "hidden": self.hidden,
                "linear": self.linear,
            },
            path,
        )
        meta = {
            "num_basis": self.num_basis,
            "latent_dim": self.latent_dim,
            "query_freq": self.query_freq,
            "action_dim": self.action_dim,
            "feature_dim": self.feature_dim,
            "basis_type": self.basis_type,
            "hidden": self.hidden,
            "linear": self.linear,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ResidualActionVAE":
        path = Path(path)
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        obj = cls(
            mean=np.asarray(ckpt["mean"], dtype=np.float32),
            std=np.asarray(ckpt["std"], dtype=np.float32),
            query_freq=int(ckpt["query_freq"]),
            action_dim=int(ckpt["action_dim"]),
            latent_dim=int(ckpt["latent_dim"]),
            hidden=int(ckpt.get("hidden", 128)),
            linear=bool(ckpt.get("linear", False)),
            state_dict=ckpt["state_dict"],
        )
        obj._ensure_model()
        return obj
