"""PCA eigenbasis for K-dimensional projected residual RL."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.action_eigenspace import fit_action_basis


@dataclass
class ResidualActionBasis:
    """Top-K PCA basis on flattened pi0 action chunks (query_freq * action_dim)."""

    mean: np.ndarray  # (D,)
    V: np.ndarray  # (D, K) columns are eigenvectors
    query_freq: int
    action_dim: int = 7
    explained_variance_ratio: np.ndarray | None = None

    @property
    def num_basis(self) -> int:
        return int(self.V.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.mean.shape[0])

    def coeffs_to_chunk(self, c, scale: float = 1.0) -> np.ndarray:
        """Map K SAC coefficients to a (query_freq, action_dim) residual chunk."""
        coeffs = np.asarray(c, dtype=np.float32).reshape(-1)
        flat = self.V @ coeffs
        return (scale * flat).reshape(self.query_freq, self.action_dim).astype(np.float32)

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
        V, explained = fit_action_basis(data, num_basis)
        return cls(
            mean=mean.astype(np.float32),
            V=V.astype(np.float32),
            query_freq=query_freq,
            action_dim=action_dim,
            explained_variance_ratio=explained.astype(np.float32),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mean=self.mean,
            V=self.V,
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
            if "V" in data:
                V = data["V"].astype(np.float32)
            else:
                # Back-compat with sklearn row layout (K, D).
                V = data["components"].astype(np.float32).T
            explained = data["explained_variance_ratio"] if "explained_variance_ratio" in data else None
            return cls(
                mean=data["mean"].astype(np.float32),
                V=V,
                query_freq=int(data["query_freq"][0]),
                action_dim=int(data["action_dim"][0]),
                explained_variance_ratio=explained.astype(np.float32) if explained is not None else None,
            )
