import jax.numpy as jnp


def residual_to_full_actions(
    residual_actions,
    observations,
    residual_scale,
    jax_decode=None,
):
    """Map SAC residual / latents to the executed chunk.

    Always residual RL: a_full = a_base + residual_scale * D(z).
    When ``jax_decode`` is set, D is that model's JAX decode. Otherwise D is
    the identity (baseline SAC residual chunk).
    """
    a_base = observations['action_diffusion']
    batch_size = residual_actions.shape[0]
    a_base_flat = jnp.reshape(a_base, (batch_size, -1))
    decoded = jnp.reshape(residual_actions, (batch_size, -1))
    if jax_decode is not None:
        decoded = jax_decode(decoded)
    a_full = a_base_flat + residual_scale * decoded
    return jnp.reshape(a_full, (batch_size, 1, -1))


def actions_for_q(
    residual_actions,
    observations,
    residual_scale,
    jax_decode=None,
):
    """Q always scores the executed chunk when a base chunk is in the observation."""
    if 'action_diffusion' not in observations:
        return residual_actions
    return residual_to_full_actions(
        residual_actions,
        observations,
        residual_scale,
        jax_decode=jax_decode,
    )
