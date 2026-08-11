from typing import Dict, Optional, Tuple

import distrax
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


def _tensor_l2(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.sum(jnp.square(x)))


def update_actor(key: PRNGKey, actor: TrainState, critic: TrainState,
                 temp: TrainState, batch: DatasetDict, cross_norm:bool=False, critic_reduction:str='min',
                 q_base_action: bool = False,
                 residual_scale: float = 1.0,
                 use_basis: bool = False,
                 basis_V: Optional[jnp.ndarray] = None,
                 basis_role: str = 'both',
) -> Tuple[TrainState, Dict[str, float]]:
    
    key, key_act = jax.random.split(key, num=2)
    alpha = temp.apply_fn({'params': temp.params})

    def apply_actor(actor_params: Params):
        if hasattr(actor, 'batch_stats') and actor.batch_stats is not None:
            dist, new_model_state = actor.apply_fn(
                {'params': actor_params, 'batch_stats': actor.batch_stats},
                batch['observations'],
                mutable=['batch_stats'],
            )
            if cross_norm:
                next_dist = actor.apply_fn(
                    {'params': actor_params, 'batch_stats': actor.batch_stats},
                    batch['next_observations'],
                    mutable=['batch_stats'],
                )
            else:
                next_dist = actor.apply_fn(
                    {'params': actor_params, 'batch_stats': actor.batch_stats},
                    batch['next_observations'],
                )
            if type(next_dist) == tuple:
                next_dist, new_model_state = next_dist
        else:
            dist = actor.apply_fn({'params': actor_params}, batch['observations'])
            new_model_state = {}
        return dist, new_model_state

    def actions_to_critic_inputs(actions):
        if q_base_action:
            critic_obs = strip_action_diffusion(batch['observations'])
            critic_actions = residual_to_executed(
                actions,
                batch['observations']['action_diffusion'],
                residual_scale,
                use_basis,
                basis_V,
            )
        else:
            critic_obs = batch['observations']
            critic_actions = actor_action_to_critic_action(
                actions, basis_role, basis_V,
            )
        return critic_obs, critic_actions

    def evaluate_q(critic_obs, critic_actions):
        if hasattr(critic, 'batch_stats') and critic.batch_stats is not None:
            qs, _ = critic.apply_fn(
                {'params': critic.params, 'batch_stats': critic.batch_stats},
                critic_obs,
                critic_actions,
                mutable=['batch_stats'],
            )
        else:
            qs = critic.apply_fn({'params': critic.params}, critic_obs, critic_actions)
        if critic_reduction == 'min':
            return qs.min(axis=0)
        if critic_reduction == 'mean':
            return qs.mean(axis=0)
        raise ValueError(f"Invalid critic reduction: {critic_reduction}")

    def terms_from_dist(dist):
        actions, log_probs = dist.sample_and_log_prob(seed=key_act)
        critic_obs, critic_actions = actions_to_critic_inputs(actions)
        q = evaluate_q(critic_obs, critic_actions)
        entropy_term = (alpha * log_probs).mean()
        critic_term = (-q).mean()
        return entropy_term, critic_term, log_probs, q

    def actor_loss_fn(actor_params: Params):
        dist, new_model_state = apply_actor(actor_params)
        entropy_term, critic_term, log_probs, q = terms_from_dist(dist)
        actor_loss = entropy_term + critic_term

        mean_dist = dist.distribution._loc
        std_diag_dist = dist.distribution._scale_diag
        things_to_log = {
            'actor_loss': actor_loss,
            'actor_entropy_loss': entropy_term,
            'actor_critic_loss': critic_term,
            'entropy': -log_probs.mean(),
            'q_pi_in_actor': q.mean(),
            'mean_pi_norm': jnp.linalg.norm(mean_dist, axis=-1).mean(),
            'std_pi_norm': jnp.linalg.norm(std_diag_dist, axis=-1).mean(),
            'mean_pi_avg': mean_dist.mean(),
            'mean_pi_max': mean_dist.max(),
            'mean_pi_min': mean_dist.min(),
            'std_pi_avg': std_diag_dist.mean(),
            'std_pi_max': std_diag_dist.max(),
            'std_pi_min': std_diag_dist.min(),
        }
        return actor_loss, (things_to_log, new_model_state)

    grads, (info, new_model_state) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)

    # Diagnostics only: ∇_μ / ∇_logσ of entropy vs critic terms (not full θ).
    dist, _ = apply_actor(actor.params)
    means0 = jax.lax.stop_gradient(dist.distribution._loc)
    log_stds0 = jax.lax.stop_gradient(jnp.log(dist.distribution._scale_diag))
    bijector = dist.bijector

    def terms_from_gauss(means, log_stds):
        base = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds)
        )
        gauss_dist = distrax.Transformed(distribution=base, bijector=bijector)
        e_term, c_term, _, _ = terms_from_dist(gauss_dist)
        return e_term, c_term

    _, gauss_vjp = jax.vjp(terms_from_gauss, means0, log_stds0)
    g_mu_entropy, g_log_std_entropy = gauss_vjp((1.0, 0.0))
    g_mu_critic, g_log_std_critic = gauss_vjp((0.0, 1.0))
    info['sac_actor_stats/entropy/grad_mu_l2'] = _tensor_l2(g_mu_entropy)
    info['sac_actor_stats/entropy/grad_log_std_l2'] = _tensor_l2(g_log_std_entropy)
    info['sac_actor_stats/critic/grad_mu_l2'] = _tensor_l2(g_mu_critic)
    info['sac_actor_stats/critic/grad_log_std_l2'] = _tensor_l2(g_log_std_critic)

    if 'batch_stats' in new_model_state:
        new_actor = actor.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info
