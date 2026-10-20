"""Flow matching on pairwise action differences. Same fit as examples/residual_flow.py.

``fit`` samples δ = a - a' from the raw chunks. Decode is
std * ODE(z) + mean, without residual_scale. SAC width is the full chunk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from latent_rl import register
from latent_rl.base import LatentModel, flat_chunks

_FILENAME = "residual_flow.pkl"


@register("flow")
class FlowLatentModel(LatentModel):
    def __init__(self, hparams: Any, seed: int = 0):
        self.epochs = int(getattr(hparams, "epochs", 100))
        self.batch_size = int(getattr(hparams, "batch_size", 256))
        self.lr = float(getattr(hparams, "lr", 1e-3))
        self.n_steps = int(getattr(hparams, "steps", 10))
        self.hidden_dims = tuple(getattr(hparams, "hidden_dims", (256, 256, 256)))
        self.seed = int(seed)
        self.query_freq: int | None = None
        self.env_action_dim: int | None = None
        self._model = None
        self._fitted = False

    @classmethod
    def add_args(cls, parser) -> None:
        parser.add_argument("--flow_epochs", type=int, default=100)
        parser.add_argument("--flow_batch_size", type=int, default=256)
        parser.add_argument("--flow_lr", type=float, default=1e-3)
        parser.add_argument("--flow_steps", type=int, default=10, help="Euler steps at decode time.")
        parser.add_argument(
            "--flow_path",
            default="",
            type=str,
            help="Optional pre-fit residual_flow.pkl.",
        )

    @classmethod
    def action_dim(cls, variant: Any) -> int:
        query_freq = int(getattr(variant, "query_freq", 0) or 0)
        action_dim = int(getattr(variant, "env_action_dim", 0) or 0)
        if query_freq < 1 or action_dim < 1:
            raise ValueError("flow action_dim needs variant.query_freq and variant.env_action_dim")
        return query_freq * action_dim

    def fit(self, base_chunks: np.ndarray) -> None:
        from examples.residual_flow import fit_residual_flow

        if self.query_freq is None or self.env_action_dim is None:
            raise RuntimeError("FlowLatentModel.fit called before configure")
        self._model = fit_residual_flow(
            flat_chunks(base_chunks).astype(np.float32),
            query_freq=int(self.query_freq),
            action_dim=int(self.env_action_dim),
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            hidden_dims=self.hidden_dims,
            n_steps=self.n_steps,
            seed=self.seed,
        )
        self._fitted = True

    def decode(self, z: np.ndarray) -> np.ndarray:
        return self._require().latent_to_chunk(z, scale=1.0)

    def jax_decode(self, z):
        from utils.flow_model import decode_flow_jax

        model = self._require()
        latents = jnp.asarray(z, dtype=jnp.float32)
        if latents.ndim == 1:
            latents = latents[None, :]
        return decode_flow_jax(
            model.params,
            latents,
            jnp.asarray(model.mean),
            jnp.asarray(model.std),
            int(model.feature_dim),
            tuple(model.hidden_dims),
            int(model.n_steps),
        )

    def save(self, output_dir: str) -> str:
        path = Path(output_dir) / _FILENAME
        self._require().save(path)
        return str(path)

    @classmethod
    def load(cls, path: str) -> "FlowLatentModel":
        from examples.residual_flow import ResidualFlow

        model = ResidualFlow.load(path)
        obj = cls(
            type("H", (), {
                "steps": int(model.n_steps),
                "hidden_dims": tuple(model.hidden_dims),
            })(),
            seed=0,
        )
        obj.configure(model.query_freq, model.action_dim)
        obj._model = model
        obj._fitted = True
        return obj

    def _require(self):
        if not self._fitted or self._model is None:
            raise RuntimeError("FlowLatentModel used before fit")
        return self._model
