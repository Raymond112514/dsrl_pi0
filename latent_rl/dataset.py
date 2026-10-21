"""Shared warmup dataset helpers for latent residual models."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def sample_pairwise_diffs(
    chunks: np.ndarray,
    num_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample δ = a - a' from (possibly different) chunk indices."""
    data = np.asarray(chunks, dtype=np.float32)
    n = data.shape[0]
    if n < 2:
        raise ValueError(f"Need >=2 action chunks for VAE diffs, got {n}")
    i = rng.integers(0, n, size=num_pairs)
    j = rng.integers(0, n, size=num_pairs)
    same = i == j
    while np.any(same):
        j[same] = rng.integers(0, n, size=int(same.sum()))
        same = i == j
    return data[i] - data[j]


def prepare_fit_arrays(
    chunks: np.ndarray,
    *,
    pairs_per_chunk: int,
    batch_size: int,
    val_frac: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """Sample δ = a - a' and split into train and val."""
    data = np.asarray(chunks, dtype=np.float32)
    n = int(data.shape[0])
    if n < 2:
        raise ValueError(f"Need >=2 action chunks for pairwise diffs, got {n}")
    num_pairs = max(int(pairs_per_chunk) * n, int(batch_size))
    data = sample_pairwise_diffs(data, num_pairs, rng)

    n_val = max(1, int(round(val_frac * len(data)))) if val_frac > 0 else 0
    n_val = min(n_val, len(data) - 1) if len(data) > 1 else 0
    perm = rng.permutation(len(data))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:] if n_val > 0 else perm
    train = data[train_idx]
    val = data[val_idx] if n_val > 0 else None
    info = {
        "num_chunks": float(n),
        "num_samples": float(len(data)),
        "num_pairs": float(num_pairs),
    }
    return train, val, info


def loaded_fit_residual(data: dict) -> bool:
    if "fit_residual" in data:
        return bool(data["fit_residual"])
    if data.get("target") == "actions":
        return False
    return True
