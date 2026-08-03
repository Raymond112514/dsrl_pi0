"""Helpers for training Q(s, a_exec) with a_exec = a_base + scale * residual."""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze


def strip_action_diffusion(observations: FrozenDict) -> FrozenDict:
    """Critic obs for BoN: pixels (+ state), without the base chunk."""
    obs = unfreeze(observations)
    obs.pop("action_diffusion", None)
    return freeze(obs)


def residual_to_executed(
    residual_actions: jnp.ndarray,
    action_diffusion: jnp.ndarray,
    residual_scale: float,
    use_basis: bool,
    basis_V: Optional[jnp.ndarray],
) -> jnp.ndarray:
    """Map SAC residual / coeffs to executed env action chunks.

    Args:
        residual_actions: (B, 1, A) or (B, A) SAC outputs.
        action_diffusion: (B, D, 1) or (B, D) flattened base chunk a_base.
        residual_scale: λ in a = a_base + λ * residual.
        use_basis: if True, residual_actions are K coeffs and basis_V is (D, K).
        basis_V: (D, K) orthonormal columns, or unused when use_basis is False.

    Returns:
        Executed actions shaped (B, 1, D).
    """
    batch = residual_actions.shape[0]
    a_base = action_diffusion.reshape(batch, -1)
    residual = residual_actions.reshape(batch, -1)
    if use_basis:
        # residual: (B, K), V: (D, K) -> (B, D)
        a_res = (residual @ basis_V.T) * residual_scale
    else:
        a_res = residual * residual_scale
    a_exec = a_base + a_res
    return a_exec.reshape(batch, 1, -1)
