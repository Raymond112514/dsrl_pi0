from audioop import cross
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.agents.pixel_sac.action_transform import maybe_residual_to_full_actions
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey


def update_actor(key: PRNGKey, actor: TrainState, critic: TrainState,
                 temp: TrainState, batch: DatasetDict, cross_norm:bool=False, critic_reduction:str='min',
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
    
    key, key_act = jax.random.split(key, num=2)

    def actor_loss_fn(
            actor_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        if hasattr(actor, 'batch_stats') and actor.batch_stats is not None:
            dist, new_model_state = actor.apply_fn({'params': actor_params, 'batch_stats': actor.batch_stats}, batch['observations'], mutable=['batch_stats'])
            if cross_norm:
                next_dist = actor.apply_fn({'params': actor_params, 'batch_stats': actor.batch_stats}, batch['next_observations'], mutable=['batch_stats'])
            else:
                next_dist = actor.apply_fn({'params': actor_params, 'batch_stats': actor.batch_stats}, batch['next_observations'])
            if type(next_dist) == tuple:
                next_dist, new_model_state = next_dist
        else:
            dist = actor.apply_fn({'params': actor_params}, batch['observations'])
            next_dist = actor.apply_fn({'params': actor_params}, batch['next_observations'])
            new_model_state = {}
        
        # For logging only
        mean_dist = dist.distribution._loc
        std_diag_dist = dist.distribution._scale_diag
        mean_dist_norm = jnp.linalg.norm(mean_dist, axis=-1)
        std_dist_norm = jnp.linalg.norm(std_diag_dist, axis=-1)

        
        actions, log_probs = dist.sample_and_log_prob(seed=key_act)
        actions = maybe_residual_to_full_actions(
            actions,
            batch['observations'],
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

        if hasattr(critic, 'batch_stats') and critic.batch_stats is not None:
            qs, _ = critic.apply_fn({'params': critic.params, 'batch_stats': critic.batch_stats}, batch['observations'],
                            actions, mutable=['batch_stats'])
        else:    
            qs = critic.apply_fn({'params': critic.params}, batch['observations'], actions)
        
        if critic_reduction == 'min':
            q = qs.min(axis=0)
        elif critic_reduction == 'mean':
            q = qs.mean(axis=0)
        else:
            raise ValueError(f"Invalid critic reduction: {critic_reduction}")
        actor_loss = (log_probs * temp.apply_fn({'params': temp.params}) - q).mean()

        things_to_log = {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'q_pi_in_actor': q.mean(),
            'mean_pi_norm': mean_dist_norm.mean(),
            'std_pi_norm': std_dist_norm.mean(),
            'mean_pi_avg': mean_dist.mean(),
            'mean_pi_max': mean_dist.max(),
            'mean_pi_min': mean_dist.min(),
            'std_pi_avg': std_diag_dist.mean(),
            'std_pi_max': std_diag_dist.max(),
            'std_pi_min': std_diag_dist.min(),
        }
        return actor_loss, (things_to_log, new_model_state)

    grads, (info, new_model_state) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    
    if 'batch_stats' in new_model_state:
        new_actor = actor.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info