"""PCA / random bases for K-dimensional projected residual RL."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def uses_projected_basis(variant) -> bool:
    """True when SAC acts in K-dim coefficient / VAE-latent residual space."""
    return bool(
        getattr(variant, "use_eigenbasis", False)
        or getattr(variant, "use_random_basis", False)
        or getattr(variant, "use_vae_basis", False)
    )


@dataclass
class ResidualActionBasis:
    """K-column basis on flattened pi0 action chunks (query_freq * action_dim)."""

    mean: np.ndarray  # (D,)
    V: np.ndarray  # (D, K) columns are basis vectors
    query_freq: int
    action_dim: int = 7
    explained_variance_ratio: np.ndarray | None = None
    basis_type: str = "pca"  # "pca" | "random"

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
        from utils.action_eigenspace import fit_action_basis

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
            basis_type="pca",
        )

    @classmethod
    def random(
        cls,
        num_basis: int,
        query_freq: int,
        action_dim: int = 7,
        seed: int = 0,
    ) -> ResidualActionBasis:
        """Random orthonormal columns via QR of a Gaussian matrix (ablation vs PCA)."""
        feature_dim = int(query_freq) * int(action_dim)
        k = int(num_basis)
        if k < 1:
            raise ValueError(f"num_basis must be >= 1, got {k}")
        if k > feature_dim:
            raise ValueError(f"num_basis={k} exceeds feature_dim={feature_dim}")
        rng = np.random.default_rng(seed)
        # Tall QR → orthonormal columns spanning a random K-plane in R^D.
        q, _ = np.linalg.qr(rng.standard_normal((feature_dim, k)), mode="reduced")
        return cls(
            mean=np.zeros(feature_dim, dtype=np.float32),
            V=q.astype(np.float32),
            query_freq=int(query_freq),
            action_dim=int(action_dim),
            explained_variance_ratio=None,
            basis_type="random",
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mean": self.mean,
            "V": self.V,
            "query_freq": np.array([self.query_freq], dtype=np.int32),
            "action_dim": np.array([self.action_dim], dtype=np.int32),
        }
        if self.explained_variance_ratio is not None:
            payload["explained_variance_ratio"] = self.explained_variance_ratio
        np.savez_compressed(path, **payload)
        meta = {
            "num_basis": self.num_basis,
            "query_freq": self.query_freq,
            "action_dim": self.action_dim,
            "feature_dim": self.feature_dim,
            "basis_type": self.basis_type,
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
        meta_path = path.with_suffix(".json")
        basis_type = "pca"
        if meta_path.is_file():
            try:
                basis_type = str(json.loads(meta_path.read_text()).get("basis_type", "pca"))
            except (json.JSONDecodeError, OSError):
                pass
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
                basis_type=basis_type,
            )
