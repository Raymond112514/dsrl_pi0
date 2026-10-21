"""PCA latent residual.

Fit uses the raw warmup chunks. Decode is V @ z, without adding the mean
and without residual_scale.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np

from latent_rl import register
from latent_rl.base import LatentModel, flat_chunks
from utils.action_eigenspace import fit_action_basis

_FILENAME = "residual_basis.npz"


@register("pca")
class PCALatentModel(LatentModel):
    def __init__(self, hparams: Any, seed: int = 0):
        self.k = int(getattr(hparams, "k", 8))
        self.seed = int(seed)
        self.query_freq: int | None = None
        self.env_action_dim: int | None = None
        self.mean: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self.explained_variance_ratio: np.ndarray | None = None
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
        if self.query_freq is None or self.env_action_dim is None:
            raise RuntimeError("PCALatentModel.fit called before configure")
        data = flat_chunks(base_chunks).astype(np.float64)
        if data.shape[0] < 2:
            raise ValueError(f"Need >=2 action chunks for PCA, got {data.shape[0]}")
        self.mean = data.mean(axis=0).astype(np.float32)
        V, explained = fit_action_basis(data, self.k)
        self.V = V.astype(np.float32)
        self.explained_variance_ratio = explained.astype(np.float32)
        self.k = int(self.V.shape[1])
        self._fitted = True

    @property
    def num_basis(self) -> int:
        if self.V is not None:
            return int(self.V.shape[1])
        return int(self.k)

    @property
    def feature_dim(self) -> int:
        if self.mean is None:
            raise RuntimeError("PCALatentModel used before fit")
        return int(self.mean.shape[0])

    def decode(self, z: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(z, dtype=np.float32).reshape(-1)
        flat = self._require_V() @ coeffs
        return flat.reshape(self.query_freq, self.env_action_dim).astype(np.float32)

    def jax_decode(self, z):
        coeffs = jnp.asarray(z, dtype=jnp.float32)
        if coeffs.ndim == 1:
            coeffs = coeffs[None, :]
        return coeffs @ jnp.asarray(self._require_V(), dtype=jnp.float32).T

    def make_jax_decoder(self):
        V = jnp.asarray(self._require_V(), dtype=jnp.float32)

        def decode_flat(z):
            coeffs = jnp.asarray(z, dtype=jnp.float32)
            if coeffs.ndim == 1:
                coeffs = coeffs[None, :]
            return coeffs @ V.T

        return decode_flat

    def fit_metrics(self) -> Optional[dict]:
        if self.explained_variance_ratio is None:
            return None
        explained = np.asarray(self.explained_variance_ratio, dtype=np.float64)
        metrics = {
            "num_components": int(self.num_basis),
            "total_explained_variance": float(explained.sum()),
        }
        for idx, ratio in enumerate(explained):
            metrics[f"pc_{idx}_explained_variance"] = float(ratio)
        return metrics

    def save(self, output_dir: str) -> str:
        path = Path(output_dir) / _FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mean=self.mean,
            V=self._require_V(),
            query_freq=np.array([self.query_freq], dtype=np.int32),
            action_dim=np.array([self.env_action_dim], dtype=np.int32),
            explained_variance_ratio=self.explained_variance_ratio,
        )
        meta = {
            "num_basis": self.num_basis,
            "query_freq": self.query_freq,
            "action_dim": self.env_action_dim,
            "feature_dim": self.feature_dim,
            "total_explained_variance": float(self.explained_variance_ratio.sum())
            if self.explained_variance_ratio is not None
            else None,
            "explained_variance_ratio": self.explained_variance_ratio.tolist()
            if self.explained_variance_ratio is not None
            else None,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        return str(path)

    @classmethod
    def load(cls, path: str) -> "PCALatentModel":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            if "V" in data:
                V = data["V"].astype(np.float32)
            else:
                V = data["components"].astype(np.float32).T
            explained = (
                data["explained_variance_ratio"]
                if "explained_variance_ratio" in data
                else None
            )
            query_freq = int(data["query_freq"][0])
            action_dim = int(data["action_dim"][0])
            mean = data["mean"].astype(np.float32)
        obj = cls(type("H", (), {"k": int(V.shape[1])})(), seed=0)
        obj.configure(query_freq, action_dim)
        obj.mean = mean
        obj.V = V
        obj.explained_variance_ratio = (
            explained.astype(np.float32) if explained is not None else None
        )
        obj.k = int(V.shape[1])
        obj._fitted = True
        return obj

    def _require_V(self) -> np.ndarray:
        if not self._fitted or self.V is None:
            raise RuntimeError("PCALatentModel used before fit")
        return self.V
