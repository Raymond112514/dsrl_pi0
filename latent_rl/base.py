"""Shared LatentModel contract for residual latent RL.

A new method is a new module in this package. The trainer only calls
``add_args``, ``action_dim``, ``build``, ``fit``, ``decode``, ``jax_decode``,
and ``save``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


def flat_chunks(chunks: np.ndarray) -> np.ndarray:
    """Accept (N, D) or (N, T, A) and return (N, D)."""
    data = np.asarray(chunks)
    if data.ndim == 3:
        data = data.reshape(data.shape[0], -1)
    if data.ndim != 2:
        raise ValueError(f"Expected chunks (N, D) or (N, T, A), got {data.shape}")
    return data


class LatentModel(ABC):
    name: str = ""

    @classmethod
    @abstractmethod
    def add_args(cls, parser) -> None:
        """Register CLI flags. Names start with ``cls.name + "_"``."""

    @classmethod
    @abstractmethod
    def action_dim(cls, variant: Any) -> int:
        """SAC width, known once ``query_freq`` and ``env_action_dim`` exist."""

    def configure(self, query_freq: int, action_dim: int) -> None:
        """Chunk shape used to reshape a flat residual back to (T, A)."""
        self.query_freq = int(query_freq)
        self.env_action_dim = int(action_dim)

    @abstractmethod
    def fit(self, base_chunks: np.ndarray) -> None:
        """Build this model's dataset from raw chunks (N, D) and fit."""

    @abstractmethod
    def decode(self, z: np.ndarray) -> np.ndarray:
        """Map one latent to a residual chunk (T, A). Do not apply residual_scale."""

    @abstractmethod
    def jax_decode(self, z):
        """JAX map of a batch of latents (B, K) to flat residuals (B, D).

        This is the decode the SAC update uses. It matches ``decode`` before
        the (T, A) reshape, including the mean for VAE and flow.
        """

    def make_jax_decoder(self):
        """Return a pure JAX decode(z) for the SAC updater."""
        return self.jax_decode

    @abstractmethod
    def save(self, output_dir: str) -> str:
        """Write this model's checkpoint under ``output_dir``. Return the path."""

    @property
    def fitted(self) -> bool:
        return bool(getattr(self, "_fitted", False))
