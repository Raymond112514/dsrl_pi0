"""Action-chunk conditional flow matching for residual RL.

Played action: a = a_base + λ * flow(w), with w = π_residual(s, a_base) ∈ R^D
where D = query_freq * action_dim (no separate latent / num_basis hyperparameter).

Fits a velocity field on warmup base-policy chunks (OT / rectified-flow CFM),
then freezes it. At env step time, treat SAC output as x_0 and integrate to x_1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _VelocityNet(nn.Module):
    """v_θ(x_t, t) with scalar time concatenated to the state."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return self.net(torch.cat([x, t], dim=-1))


@dataclass
class ResidualActionFlow:
    """Frozen CFM mapping D-dim SAC noise -> (query_freq, action_dim) residual."""

    mean: np.ndarray  # (D,) chunk mean used for training normalization
    std: np.ndarray  # (D,)
    query_freq: int
    action_dim: int
    hidden: int = 128
    n_steps: int = 10
    state_dict: dict | None = None
    _model: nn.Module | None = None

    @property
    def feature_dim(self) -> int:
        return int(self.mean.shape[0])

    @property
    def num_basis(self) -> int:
        # SAC action dim = full chunk dim (no bottleneck).
        return self.feature_dim

    @property
    def basis_type(self) -> str:
        return "flow"

    @property
    def explained_variance_ratio(self):
        return None

    def _ensure_model(self) -> nn.Module:
        if self._model is None:
            model = _VelocityNet(self.feature_dim, hidden=self.hidden)
            if self.state_dict is not None:
                model.load_state_dict(self.state_dict)
            model.eval()
            self._model = model
        return self._model

    def _integrate(self, x0: torch.Tensor) -> torch.Tensor:
        """Euler integrate x_0 -> x_1 with frozen velocity net."""
        model = self._ensure_model()
        x = x0
        n = max(1, int(self.n_steps))
        dt = 1.0 / n
        with torch.no_grad():
            for i in range(n):
                t = torch.full((x.shape[0], 1), i * dt, dtype=x.dtype)
                x = x + dt * model(x, t)
        return x

    def coeffs_to_chunk(self, c, scale: float = 1.0) -> np.ndarray:
        """Map D-dim SAC output (as flow noise x_0) to residual chunk (no +mean)."""
        w = np.asarray(c, dtype=np.float32).reshape(1, -1)
        if w.shape[1] != self.feature_dim:
            raise ValueError(f"Expected dim {self.feature_dim}, got {w.shape[1]}")
        x1 = self._integrate(torch.from_numpy(w)).numpy()[0]
        flat = x1 * self.std
        return (float(scale) * flat).reshape(self.query_freq, self.action_dim).astype(np.float32)

    def sample_action_chunks(self, n: int, seed: int = 0) -> np.ndarray:
        """Draw n absolute action chunks: x_0~N(0,I) → integrate → unnormalize."""
        rng = np.random.default_rng(seed)
        x0 = rng.standard_normal((n, self.feature_dim)).astype(np.float32)
        x1 = self._integrate(torch.from_numpy(x0)).numpy()
        chunks = x1 * self.std[None, :] + self.mean[None, :]
        return chunks.reshape(n, self.query_freq, self.action_dim).astype(np.float32)

    @classmethod
    def fit(
        cls,
        chunks: np.ndarray,
        query_freq: int,
        action_dim: int,
        *,
        hidden: int = 128,
        epochs: int = 120,
        batch_size: int = 256,
        lr: float = 1e-3,
        n_steps: int = 10,
        seed: int = 0,
    ) -> "ResidualActionFlow":
        data = np.asarray(chunks, dtype=np.float32)
        if data.ndim == 3:
            data = data.reshape(data.shape[0], -1)
        if data.shape[0] < 2:
            raise ValueError(f"Need >=2 action chunks for flow matching, got {data.shape[0]}")
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
        model = _VelocityNet(expected, hidden=int(hidden))
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        for epoch in range(int(epochs)):
            total = 0.0
            n = 0
            for (batch,) in loader:
                x1 = batch
                x0 = torch.randn_like(x1)
                t = torch.rand(x1.shape[0], 1)
                xt = (1.0 - t) * x0 + t * x1
                target = x1 - x0
                pred = model(xt, t)
                loss = nn.functional.mse_loss(pred, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach())
                n += 1
            ep1 = epoch + 1
            if ep1 % max(1, int(epochs) // 5) == 0 or ep1 == int(epochs):
                print(f"  flow epoch {ep1:03d}  loss={total / max(n, 1):.4f}")
        model.eval()
        return cls(
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            query_freq=int(query_freq),
            action_dim=int(action_dim),
            hidden=int(hidden),
            n_steps=int(n_steps),
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
                "hidden": self.hidden,
                "n_steps": self.n_steps,
            },
            path,
        )
        meta = {
            "num_basis": self.num_basis,
            "feature_dim": self.feature_dim,
            "query_freq": self.query_freq,
            "action_dim": self.action_dim,
            "basis_type": self.basis_type,
            "hidden": self.hidden,
            "n_steps": self.n_steps,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ResidualActionFlow":
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
            hidden=int(ckpt.get("hidden", 128)),
            n_steps=int(ckpt.get("n_steps", 10)),
            state_dict=ckpt["state_dict"],
        )
        obj._ensure_model()
        return obj
