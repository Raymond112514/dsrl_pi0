"""VAE on pairwise action differences. Same fit as examples/residual_vae.py.

``fit`` samples δ = a - a' from the raw chunks. Decode is
std * decoder(z) + mean, without residual_scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from latent_rl import register
from latent_rl.base import LatentModel, flat_chunks

_FILENAME = "residual_vae.pkl"


@register("vae")
class VAELatentModel(LatentModel):
    def __init__(self, hparams: Any, seed: int = 0):
        self.latent_dim = int(getattr(hparams, "latent_dim", 8))
        self.beta = float(getattr(hparams, "beta", 1e-3))
        self.epochs = int(getattr(hparams, "epochs", 100))
        self.batch_size = int(getattr(hparams, "batch_size", 256))
        self.lr = float(getattr(hparams, "lr", 1e-3))
        self.hidden_dims = tuple(getattr(hparams, "hidden_dims", (256, 256)))
        self.seed = int(seed)
        self.query_freq: int | None = None
        self.env_action_dim: int | None = None
        self._model = None
        self._fitted = False

    @classmethod
    def add_args(cls, parser) -> None:
        parser.add_argument("--vae_latent_dim", type=int, default=8, help="VAE latent width K.")
        parser.add_argument("--vae_beta", type=float, default=1e-3)
        parser.add_argument("--vae_epochs", type=int, default=100)
        parser.add_argument("--vae_batch_size", type=int, default=256)
        parser.add_argument("--vae_lr", type=float, default=1e-3)
        parser.add_argument(
            "--vae_path",
            default="",
            type=str,
            help="Optional pre-fit residual_vae.pkl.",
        )

    @classmethod
    def action_dim(cls, variant: Any) -> int:
        hparams = getattr(variant, "vae", None)
        k = getattr(hparams, "latent_dim", None) if hparams is not None else None
        if k is None:
            k = getattr(variant, "vae_latent_dim", 8)
        return int(k)

    def fit(self, base_chunks: np.ndarray) -> None:
        from examples.residual_vae import fit_residual_vae

        if self.query_freq is None or self.env_action_dim is None:
            raise RuntimeError("VAELatentModel.fit called before configure")
        self._model = fit_residual_vae(
            flat_chunks(base_chunks).astype(np.float32),
            latent_dim=self.latent_dim,
            query_freq=int(self.query_freq),
            action_dim=int(self.env_action_dim),
            beta=self.beta,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            hidden_dims=self.hidden_dims,
            seed=self.seed,
        )
        self._fitted = True

    def decode(self, z: np.ndarray) -> np.ndarray:
        return self._require().latent_to_chunk(z, scale=1.0)

    def jax_decode(self, z):
        from utils.vae_model import decode_latents_jax

        model = self._require()
        latents = jnp.asarray(z, dtype=jnp.float32)
        if latents.ndim == 1:
            latents = latents[None, :]
        return decode_latents_jax(
            model.params,
            latents,
            jnp.asarray(model.mean),
            jnp.asarray(model.std),
            int(model.feature_dim),
            int(model.latent_dim),
            tuple(model.hidden_dims),
        )

    def save(self, output_dir: str) -> str:
        path = Path(output_dir) / _FILENAME
        self._require().save(path)
        return str(path)

    @classmethod
    def load(cls, path: str) -> "VAELatentModel":
        from examples.residual_vae import ResidualVAE

        model = ResidualVAE.load(path)
        obj = cls(
            type("H", (), {
                "latent_dim": int(model.latent_dim),
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
            raise RuntimeError("VAELatentModel used before fit")
        return self._model
