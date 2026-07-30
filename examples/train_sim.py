#! /usr/bin/env python
import os
# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs from https://github.com/huggingface/gym-aloha/tree/main?tab=readme-ov-file#-gpu-rendering-egl
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

# import sys
# sys.path.append('/global/home/users/r112358/dsrl_pi0')
# sys.path.append('/global/home/users/r112358/LIBERO')
# sys.path.append('/global/home/users/r112358/dsrl_pi0/openpi')
# sys.path.append('/global/home/users/r112358/dsrl_pi0/openpi/src')

import pathlib, copy
import time
import jax
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.utils.general_utils import add_batch_dim
import numpy as np

import gymnasium as gym
import gym_aloha
from gym.spaces import Dict, Box

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from jaxrl2.data import ReplayBuffer
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
import tempfile
from functools import partial
from examples.train_utils_sim import trajwise_alternating_training_loop, log_basis_explained_variance
import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache

from openpi.training import config as openpi_config
from openpi.policies import policy_config
from openpi.shared import download
from examples.residual_basis import ResidualActionBasis, uses_projected_basis
home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(home_dir, 'jax_compilation_cache'))

def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def shard_batch(batch, sharding):
    """Shards a batch across devices along its first dimension.

    Args:
        batch: A pytree of arrays.
        sharding: A jax Sharding object with shape (num_devices,).
    """
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )


class DummyEnv(gym.ObservationWrapper):

    def __init__(self, variant):
        self.variant = variant
        self.image_shape = (variant.resize_image, variant.resize_image, 3 * variant.num_cameras, 1)
        obs_dict = {}
        obs_dict['pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        if variant.add_states:
            if variant.env == 'libero':
                state_dim = 8
            elif variant.env == 'aloha_cube':
                state_dim = 14
            obs_dict['state'] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)
        # Residual RL (Libero + Aloha): condition on base pi0 chunk; SAC acts in
        # residual space (full chunk) or K-dim PCA coefficient space.
        if variant.env in ('libero', 'aloha_cube'):
            residual_dim = variant.pi0_action_horizon * variant.env_action_dim
            obs_dict['action_diffusion'] = Box(
                low=-np.inf, high=np.inf, shape=(residual_dim, 1), dtype=np.float32
            )
            if uses_projected_basis(variant):
                sac_dim = variant.num_basis
            else:
                sac_dim = residual_dim
            self.action_space = Box(low=-1, high=1, shape=(1, sac_dim), dtype=np.float32)
        else:
            # Classic noise-steering DSRL (32-D pi0 noise).
            self.action_space = Box(low=-1, high=1, shape=(1, 32,), dtype=np.float32)
        self.observation_space = Dict(obs_dict)

def build_exp_name(variant):
    if variant.env == 'libero':
        parts = [f"task{variant.task_id}"]
    else:
        parts = [variant.env]
    if uses_projected_basis(variant):
        tag = "rand" if getattr(variant, 'use_random_basis', False) else "pca"
        parts.append(f"K{variant.num_basis}_{tag}")
    return "_".join(parts)


def maybe_init_basis(variant):
    """Load PCA basis, or create a random orthonormal basis for the ablation."""
    if getattr(variant, 'use_eigenbasis', False) and getattr(variant, 'use_random_basis', False):
        raise ValueError("Pass only one of --use_eigenbasis / --use_random_basis")

    if getattr(variant, 'use_random_basis', False):
        if variant.get('basis_path', ''):
            raise ValueError("--basis_path is only supported with --use_eigenbasis")
        action_dim = int(variant.env_action_dim)
        # Prefer query_freq so residual reshape matches collect; falls back to horizon.
        query_freq = int(variant.query_freq if variant.query_freq > 0 else variant.pi0_action_horizon)
        basis = ResidualActionBasis.random(
            num_basis=int(variant.num_basis),
            query_freq=query_freq,
            action_dim=action_dim,
            seed=int(variant.seed),
        )
        save_path = os.path.join(variant.outputdir, "residual_basis.npz")
        basis.save(save_path)
        print(f"Initialized random orthonormal basis K={basis.num_basis} D={basis.feature_dim} -> {save_path}")
        return basis

    if not getattr(variant, 'use_eigenbasis', False):
        return None
    basis_path = variant.get('basis_path', '')
    if basis_path and os.path.isfile(basis_path):
        print(f"Loading PCA basis from {basis_path}")
        basis = ResidualActionBasis.load(basis_path)
        num_basis = int(variant.num_basis)
        if basis.num_basis > num_basis:
            print(f"Slicing loaded basis from K={basis.num_basis} to K={num_basis}")
            explained = basis.explained_variance_ratio
            if explained is not None:
                explained = explained[:num_basis]
            basis = ResidualActionBasis(
                mean=basis.mean,
                V=basis.V[:, :num_basis],
                query_freq=basis.query_freq,
                action_dim=basis.action_dim,
                explained_variance_ratio=explained,
                basis_type=basis.basis_type,
            )
        elif basis.num_basis < num_basis:
            raise ValueError(
                f"Loaded basis has K={basis.num_basis} but --num_basis={num_basis}"
            )
        return basis
    return None

def main(variant):
    devices = jax.local_devices()
    num_devices = len(devices)
    assert variant.batch_size % num_devices == 0
    print('num devices', num_devices)
    print('batch size', variant.batch_size)
    # we shard the leading dimension (batch dimension) accross all devices evenly
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")
    
    kwargs = variant['train_kwargs']
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps
        
    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]

    # if variant.suffix:
    #     expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    # else:
    #     expname = create_exp_name(variant.prefix, seed=variant.seed)
    
    expname = build_exp_name(variant)

    output_root = variant.output_dir
    outputdir = os.path.join(output_root, f"{expname}_{time.strftime('%Y%m%d-%H%M%S')}_{variant.seed}")
    variant.outputdir = outputdir
    os.makedirs(outputdir, exist_ok=True)
    print('writing to output dir ', outputdir)
    
    if variant.env == 'libero':
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict["libero_90"]()
        task_id = variant.task_id
        task = task_suite.get_task(task_id)
        env, task_description = _get_libero_env(task, 256, variant.seed)
        eval_env = env
        variant.task_description = task_description
        variant.env_max_reward = 1
        variant.max_timesteps = 400
    elif variant.env == 'aloha_cube':
        from gymnasium.envs.registration import register
        register(
            id="gym_aloha/AlohaTransferCube-v0",
            entry_point="gym_aloha.env:AlohaEnv",
            max_episode_steps=400,
            nondeterministic=True,
            kwargs={"obs_type": "pixels", "task": "transfer_cube"},
        )
        env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
        eval_env = copy.deepcopy(env)
        variant.env_max_reward = 4
        variant.max_timesteps = 400
        

    if variant.env == 'libero':
        config = openpi_config.get_config("pi05_libero")
        checkpoint_dir = variant.pi0_checkpoint or download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
    elif variant.env == 'aloha_cube':
        # Same pi0 Aloha-sim checkpoint used by openpi (main/residual_rl branches + serve_policy).
        config = openpi_config.get_config("pi0_aloha_sim")
        checkpoint_dir = variant.pi0_checkpoint or download.maybe_download(
            "gs://openpi-assets/checkpoints/pi0_aloha_sim"
        )
    else:
        raise NotImplementedError()
    variant.pi0_action_horizon = config.model.action_horizon
    variant.pi0_action_dim = config.model.action_dim
    if variant.query_freq <= 0:
        variant.query_freq = variant.pi0_action_horizon
    if variant.env == 'libero':
        variant.env_action_dim = 7
    elif variant.env == 'aloha_cube':
        variant.env_action_dim = 14
    if variant.env in ('libero', 'aloha_cube'):
        if not hasattr(variant, 'residual_scale') or variant.residual_scale is None:
            variant.residual_scale = 0.01
    agent_dp = policy_config.create_trained_policy(config, checkpoint_dir)
    print("Loaded pi policy from %s", checkpoint_dir)

    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(variant.prefix != '', variant, variant.wandb_project, experiment_id=expname, output_dir=wandb_output_dir, group_name=group_name)

    dummy_env = DummyEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    print('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])
    print('sample action shape', sample_action.shape)

    agent = PixelSACLearner(
        variant.seed,
        sample_obs,
        sample_action,
        zero_init_actor_mean=(
            variant.env in ('libero', 'aloha_cube')
            and not getattr(variant, 'init_residual', False)
        ),
        **kwargs,
    )

    online_buffer_size = variant.max_steps  // variant.multi_grad_step
    online_replay_buffer = ReplayBuffer(dummy_env.observation_space, dummy_env.action_space, int(online_buffer_size))
    replay_buffer = online_replay_buffer
    replay_buffer.seed(variant.seed)
    variant.basis = maybe_init_basis(variant)
    if variant.basis is not None:
        log_basis_explained_variance(wandb_logger, variant.basis, step=0)
    trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger, shard_fn=shard_fn, agent_dp=agent_dp)
 
