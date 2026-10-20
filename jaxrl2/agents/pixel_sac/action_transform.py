import jax.numpy as jnp

from utils.flow_model import decode_flow_jax
from utils.vae_model import decode_latents_jax


def residual_to_full_actions(
    residual_actions,
    observations,
    residual_scale,
    basis_V,
    use_eigenbasis: bool,
    use_vae: bool = False,
    vae_params=None,
    vae_mean=None,
    vae_std=None,
    vae_input_dim: int = 0,
    vae_latent_dim: int = 0,
    vae_hidden_dims=(256, 256),
    use_flow: bool = False,
    flow_params=None,
    flow_mean=None,
    flow_std=None,
    flow_dim: int = 0,
    flow_hidden_dims=(256, 256, 256),
    flow_steps: int = 10,
):
    """Map SAC residual / PCA coeffs / VAE latents / flow noise to the executed chunk.

    Always residual RL: a_full = a_base + residual_scale * D(z).
    D(z) is V @ c (PCA), std * decoder(z) + mean (VAE), or std * ODE(ε) + mean (flow).
    Flow does not compress, so ε has dim D.
    """
    a_base = observations['action_diffusion']
    batch_size = residual_actions.shape[0]
    a_base_flat = jnp.reshape(a_base, (batch_size, -1))
    decoded = jnp.reshape(residual_actions, (batch_size, -1))
    if use_flow:
        decoded = decode_flow_jax(
            flow_params,
            decoded,
            flow_mean,
            flow_std,
            flow_dim,
            flow_hidden_dims,
            flow_steps,
        )
    elif use_vae:
        decoded = decode_latents_jax(
            vae_params,
            decoded,
            vae_mean,
            vae_std,
            vae_input_dim,
            vae_latent_dim,
            vae_hidden_dims,
        )
    elif use_eigenbasis:
        decoded = decoded @ basis_V.T
    a_full = a_base_flat + residual_scale * decoded
    return jnp.reshape(a_full, (batch_size, 1, -1))


def maybe_residual_to_full_actions(
    residual_actions,
    observations,
    residual_scale,
    basis_V,
    use_eigenbasis: bool,
    q_on_full_actions: bool,
    use_vae: bool = False,
    vae_params=None,
    vae_mean=None,
    vae_std=None,
    vae_input_dim: int = 0,
    vae_latent_dim: int = 0,
    vae_hidden_dims=(256, 256),
    use_flow: bool = False,
    flow_params=None,
    flow_mean=None,
    flow_std=None,
    flow_dim: int = 0,
    flow_hidden_dims=(256, 256, 256),
    flow_steps: int = 10,
):
    if not q_on_full_actions:
        return residual_actions
    return residual_to_full_actions(
        residual_actions,
        observations,
        residual_scale,
        basis_V,
        use_eigenbasis,
        use_vae=use_vae,
        vae_params=vae_params,
        vae_mean=vae_mean,
        vae_std=vae_std,
        vae_input_dim=vae_input_dim,
        vae_latent_dim=vae_latent_dim,
        vae_hidden_dims=vae_hidden_dims,
        use_flow=use_flow,
        flow_params=flow_params,
        flow_mean=flow_mean,
        flow_std=flow_std,
        flow_dim=flow_dim,
        flow_hidden_dims=flow_hidden_dims,
        flow_steps=flow_steps,
    )
