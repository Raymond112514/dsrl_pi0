from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.agents.pixel_sac.action_transform import maybe_residual_to_full_actions
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey


def update_critic(
        key: PRNGKey, actor: TrainState, critic: TrainState,
        target_critic: TrainState, temp: TrainState, batch: DatasetDict,
        discount: float, backup_entropy: bool = False,
        critic_reduction: str = 'min',
        residual_scale: float = 1.0,
        basis_V: jnp.ndarray = None,
        use_eigenbasis: bool = False,
        q_on_full_actions: bool = False,
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
        flow_steps: int = 10) -> Tuple[TrainState, Dict[str, float]]:
    dist = actor.apply_fn({'params': actor.params}, batch['next_observations'])
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
    next_actions = maybe_residual_to_full_actions(
        next_actions,
        batch['next_observations'],
        residual_scale,
        basis_V,
        use_eigenbasis,
        q_on_full_actions,
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
    next_qs = target_critic.apply_fn({'params': target_critic.params},
                                     batch['next_observations'], next_actions)
    if critic_reduction == 'min':
        next_q = next_qs.min(axis=0)
    elif critic_reduction == 'mean':
        next_q = next_qs.mean(axis=0)
    else:
        raise NotImplemented()

    target_q = batch['rewards'] + batch["discount"] * batch['masks'] * next_q

    if backup_entropy:
        target_q -= batch["discount"] * batch['masks'] * temp.apply_fn(
            {'params': temp.params}) * next_log_probs

    def critic_loss_fn(
            critic_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        qs = critic.apply_fn({'params': critic_params}, batch['observations'],
                             batch['actions'])
        critic_loss = ((qs - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q': qs.mean(),
            'target_actor_entropy': -next_log_probs.mean(),
            'next_actions_sampled': next_actions.mean(),
            'next_log_probs': next_log_probs.mean(),
            'next_q_pi': next_qs.mean(),
            'target_q': target_q.mean(),
            'next_actions_mean': next_actions.mean(),
            'next_actions_std': next_actions.std(),
            'next_actions_min': next_actions.min(),
            'next_actions_max': next_actions.max(),
            'next_log_probs': next_log_probs.mean(),
            
        }

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)

    return new_critic, info
