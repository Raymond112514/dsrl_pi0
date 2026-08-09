"""Helpers for Q(s, a_exec) and actor/critic action-space remapping."""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze


def strip_action_diffusion(observations: FrozenDict) -> FrozenDict:
    """Critic obs for BoN: pixels (+ state), without the base chunk."""
    # Build explicitly (do not rely on pop) so the pytree keys are unambiguous.
    keep = {'pixels': observations['pixels']}
    if 'state' in observations:
        keep['state'] = observations['state']
    return freeze(keep)


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


def actor_action_to_critic_action(
    actor_actions: jnp.ndarray,
    basis_role: str,
    basis_V: Optional[jnp.ndarray],
) -> jnp.ndarray:
    """Map actor/replay actions to the critic's action input (residual space).

    Orthogonal to q_base_action (played a_exec). Does not add a_base.

    Roles (V is D×K with orthonormal columns):
      - both:   identity (c -> c, or r -> r)
      - critic: lift c -> V @ c  (D-dim residual in the PCA span)
      - actor:  project r -> V^T @ r = c  (coords of the in-subspace part)

    Projection is necessary only for role=actor: unconstrained r may leave the
    column span of V, so the K-dim critic must see c = V^T r.
    """
    batch = actor_actions.shape[0]
    a = actor_actions.reshape(batch, -1)
    if basis_role == 'both':
        out = a
    elif basis_role == 'critic':
        # (B, K) @ (K, D) -> (B, D)
        out = a @ basis_V.T
    elif basis_role == 'actor':
        # (B, D) @ (D, K) -> (B, K); least-squares coords since V^T V = I
        out = a @ basis_V
    else:
        raise ValueError(f'Unknown basis_role={basis_role!r}')
    return out.reshape(batch, 1, -1)
