"""PCA latent residual. Same sklearn fit as examples/residual_basis.py.

Fit uses the raw warmup chunks. Decode is V @ z, without adding the mean
and without residual_scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np

from latent_rl import register
from latent_rl.base import LatentModel, flat_chunks

_FILENAME = "residual_basis.npz"


@register("pca")
class PCALatentModel(LatentModel):
    def __init__(self, hparams: Any, seed: int = 0):
        self.k = int(getattr(hparams, "k", 8))
        self.seed = int(seed)
        self.query_freq: int | None = None
        self.env_action_dim: int | None = None
        self._basis = None
        self._fitted = False

    @classmethod
    def add_args(cls, parser) -> None:
        parser.add_argument("--pca_k", type=int, default=8, help="PCA latent width K.")
        parser.add_argument(
            "--pca_path",
            default="",
            type=str,
            help="Optional pre-fit residual_basis.npz.",
        )

    @classmethod
    def action_dim(cls, variant: Any) -> int:
        hparams = getattr(variant, "pca", None)
        k = getattr(hparams, "k", None) if hparams is not None else None
        if k is None:
            k = getattr(variant, "pca_k", 8)
        return int(k)

    def fit(self, base_chunks: np.ndarray) -> None:
        from examples.residual_basis import ResidualActionBasis

        if self.query_freq is None or self.env_action_dim is None:
            raise RuntimeError("PCALatentModel.fit called before configure")
        data = flat_chunks(base_chunks)
        self._basis = ResidualActionBasis.fit_top_k(
            data,
            self.k,
            query_freq=int(self.query_freq),
            action_dim=int(self.env_action_dim),
        )
        self.k = int(self._basis.num_basis)
        self._fitted = True

    def decode(self, z: np.ndarray) -> np.ndarray:
        basis = self._require()
        return basis.coeffs_to_chunk(z, scale=1.0)

    def jax_decode(self, z):
        basis = self._require()
        coeffs = jnp.asarray(z, dtype=jnp.float32)
        if coeffs.ndim == 1:
            coeffs = coeffs[None, :]
        flat = coeffs @ jnp.asarray(basis.V, dtype=jnp.float32).T
        return flat

    def fit_metrics(self) -> Optional[dict]:
        basis = self._basis
        if basis is None or basis.explained_variance_ratio is None:
            return None
        explained = np.asarray(basis.explained_variance_ratio, dtype=np.float64)
        metrics = {
            "num_components": int(basis.num_basis),
            "total_explained_variance": float(explained.sum()),
        }
        for idx, ratio in enumerate(explained):
            metrics[f"pc_{idx}_explained_variance"] = float(ratio)
        return metrics

    def save(self, output_dir: str) -> str:
        path = Path(output_dir) / _FILENAME
        self._require().save(path)
        return str(path)

    @classmethod
    def load(cls, path: str) -> "PCALatentModel":
        from examples.residual_basis import ResidualActionBasis

        basis = ResidualActionBasis.load(path)
        obj = cls(type("H", (), {"k": int(basis.num_basis)})(), seed=0)
        obj.configure(basis.query_freq, basis.action_dim)
        obj._basis = basis
        obj.k = int(basis.num_basis)
        obj._fitted = True
        return obj

    def _require(self):
        if not self._fitted or self._basis is None:
            raise RuntimeError("PCALatentModel used before fit")
        return self._basis
