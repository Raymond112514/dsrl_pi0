from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.agents.pixel_sac.executed_action_q import (
    actor_action_to_critic_action,
    residual_to_executed,
    strip_action_diffusion,
)
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey


def update_critic(
        key: PRNGKey, actor: TrainState, critic: TrainState,
        target_critic: TrainState, temp: TrainState, batch: DatasetDict,
        discount: float, backup_entropy: bool = False,
        critic_reduction: str = 'min',
        q_base_action: bool = False,
        residual_scale: float = 1.0,
        use_basis: bool = False,
        basis_V: Optional[jnp.ndarray] = None,
        basis_role: str = 'both',
) -> Tuple[TrainState, Dict[str, float]]:
    dist = actor.apply_fn({'params': actor.params}, batch['next_observations'])
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)

    if q_base_action:
        critic_obs = strip_action_diffusion(batch['observations'])
        critic_next_obs = strip_action_diffusion(batch['next_observations'])
        actions = residual_to_executed(
            batch['actions'],
            batch['observations']['action_diffusion'],
            residual_scale,
            use_basis,
            basis_V,
        )
        next_actions = residual_to_executed(
            next_actions,
            batch['next_observations']['action_diffusion'],
            residual_scale,
            use_basis,
            basis_V,
        )
    else:
        critic_obs = batch['observations']
        critic_next_obs = batch['next_observations']
        actions = actor_action_to_critic_action(
            batch['actions'], basis_role, basis_V,
        )
        next_actions = actor_action_to_critic_action(
            next_actions, basis_role, basis_V,
        )

    next_qs = target_critic.apply_fn({'params': target_critic.params},
                                     critic_next_obs, next_actions)
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
        qs = critic.apply_fn({'params': critic_params}, critic_obs, actions)
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
