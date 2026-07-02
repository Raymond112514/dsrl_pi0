"""PCA basis for K-dimensional projected residual RL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class ResidualActionBasis:
    """Top-K PCA basis on flattened pi0 action chunks (query_freq * action_dim)."""

    mean: np.ndarray  # (D,)
    components: np.ndarray  # (K, D) sklearn row layout
    query_freq: int
    action_dim: int = 7
    explained_variance_ratio: np.ndarray | None = None

    @property
    def num_basis(self) -> int:
        return int(self.components.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.mean.shape[0])

    def coeffs_to_step_residual(self, coeffs, step_idx: int, scale: float = 1.0) -> np.ndarray:
        """Map K SAC coefficients to a 7-D residual at env step step_idx."""
        coeffs = np.asarray(coeffs, dtype=np.float32).reshape(-1)
        flat = coeffs @ self.components
        step = int(step_idx) % self.query_freq
        return (scale * flat.reshape(self.query_freq, self.action_dim)[step]).astype(np.float32)

    @classmethod
    def fit_top_k(
        cls,
        chunks: np.ndarray,
        num_basis: int,
        query_freq: int,
        action_dim: int = 7,
    ) -> ResidualActionBasis:
        data = np.asarray(chunks, dtype=np.float64)
        if data.shape[0] < 2:
            raise ValueError(f"Need >=2 action chunks for PCA, got {data.shape[0]}")
        mean = data.mean(axis=0)
        k = min(num_basis, data.shape[0], data.shape[1])
        pca = PCA(n_components=k)
        pca.fit(data - mean)
        return cls(
            mean=mean.astype(np.float32),
            components=pca.components_.astype(np.float32),
            query_freq=query_freq,
            action_dim=action_dim,
            explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mean=self.mean,
            components=self.components,
            query_freq=np.array([self.query_freq], dtype=np.int32),
            action_dim=np.array([self.action_dim], dtype=np.int32),
            explained_variance_ratio=self.explained_variance_ratio,
        )
        meta = {
            "num_basis": self.num_basis,
            "query_freq": self.query_freq,
            "action_dim": self.action_dim,
            "feature_dim": self.feature_dim,
            "total_explained_variance": float(self.explained_variance_ratio.sum())
            if self.explained_variance_ratio is not None
            else None,
            "explained_variance_ratio": self.explained_variance_ratio.tolist()
            if self.explained_variance_ratio is not None
            else None,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> ResidualActionBasis:
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            explained = data["explained_variance_ratio"] if "explained_variance_ratio" in data else None
            return cls(
                mean=data["mean"].astype(np.float32),
                components=data["components"].astype(np.float32),
                query_freq=int(data["query_freq"][0]),
                action_dim=int(data["action_dim"][0]),
                explained_variance_ratio=explained.astype(np.float32) if explained is not None else None,
            )

