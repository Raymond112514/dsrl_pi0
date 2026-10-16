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
from examples.train_utils_sim import (
    trajwise_alternating_training_loop,
    log_basis_explained_variance,
    log_vae_fit_metrics,
    log_flow_fit_metrics,
)
import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache

from openpi.training import config as openpi_config
from openpi.policies import policy_config
from openpi.shared import download

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
            elif variant.env in ('aloha_cube', 'aloha_insertion'):
                state_dim = 14
            obs_dict['state'] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)
        if variant.env in ('libero', 'aloha_cube', 'aloha_insertion'):
            residual_dim = variant.query_freq * variant.env_action_dim
            obs_dict['action_diffusion'] = Box(
                low=-np.inf, high=np.inf, shape=(residual_dim, 1), dtype=np.float32
            )
            if getattr(variant, 'use_flow', False):
                # Flow matching does not compress: SAC noise has the chunk dim.
                sac_dim = residual_dim
            elif getattr(variant, 'use_eigenbasis', False) or getattr(variant, 'use_vae', False):
                sac_dim = variant.num_basis
            else:
                sac_dim = residual_dim
            self.action_space = Box(low=-1, high=1, shape=(1, sac_dim), dtype=np.float32)
            self.full_action_space = Box(low=-1, high=1, shape=(1, residual_dim), dtype=np.float32)
        else:
            self.action_space = Box(low=-1, high=1, shape=(1, 32,), dtype=np.float32)
            self.full_action_space = self.action_space
        self.observation_space = Dict(obs_dict)

def build_exp_name(variant):
    if variant.env == 'libero':
        parts = [f"task{variant.task_id}"]
    else:
        parts = [variant.env]
    parts.append(f"qf{variant.query_freq}")
    if getattr(variant, 'use_eigenbasis', False):
        parts.append(f"K{variant.num_basis}")
    if getattr(variant, 'use_vae', False):
        parts.append(f"vaeK{variant.num_basis}")
        parts.append(f"beta{getattr(variant, 'vae_beta', 1e-3)}")
    if getattr(variant, 'use_flow', False):
        parts.append(f"flow{getattr(variant, 'flow_steps', 10)}")
    if not getattr(variant, 'fit_residual', True):
        parts.append("actions")
    if not getattr(variant, 'decode_add_mean', True):
        parts.append("nomean")
    if getattr(variant, 'vae_condition_on_state', False):
        parts.append("scond")
        if getattr(variant, 'fit_residual', True):
            parts.append(f"n{int(getattr(variant, 'vae_samples_per_state', 16))}")
    return "_".join(parts)


def resolve_pi0_checkpoint(variant, default_url):
    ckpt = getattr(variant, 'pi0_checkpoint', '') or ''
    if ckpt and os.path.isdir(ckpt):
        return ckpt
    scratch = os.environ.get(
        'OPENPI_DATA_HOME',
        '/global/scratch/users/r112358/pi0_checkpoint',
    )
    os.environ['OPENPI_DATA_HOME'] = scratch
    print(f"OpenPI checkpoint not found locally; downloading to {scratch}")
    return str(download.maybe_download(default_url))


def load_basis_if_provided(variant):
    if not getattr(variant, 'use_eigenbasis', False):
        return None
    basis_path = variant.get('basis_path', '')
    if basis_path and os.path.isfile(basis_path):
        from examples.residual_basis import ResidualActionBasis
        print(f"Loading PCA basis from {basis_path}")
        return ResidualActionBasis.load(basis_path)
    return None


def load_vae_if_provided(variant):
    if not getattr(variant, 'use_vae', False):
        return None
    vae_path = variant.get('vae_path', '') or variant.get('basis_path', '')
    if vae_path and os.path.isfile(vae_path):
        from examples.residual_vae import ResidualVAE
        print(f"Loading residual VAE from {vae_path}")
        model = ResidualVAE.load(vae_path)
        model.decode_add_mean = bool(getattr(variant, "decode_add_mean", True))
        return model
    return None


def load_flow_if_provided(variant):
    if not getattr(variant, 'use_flow', False):
        return None
    flow_path = variant.get('flow_path', '')
    if flow_path and os.path.isfile(flow_path):
        from examples.residual_flow import ResidualFlow
        print(f"Loading residual flow from {flow_path}")
        model = ResidualFlow.load(flow_path)
        model.decode_add_mean = bool(getattr(variant, "decode_add_mean", True))
        return model
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
    elif variant.env == 'aloha_insertion':
        from gymnasium.envs.registration import register
        register(
            id="gym_aloha/AlohaInsertion-v0",
            entry_point="gym_aloha.env:AlohaEnv",
            max_episode_steps=400,
            nondeterministic=True,
            kwargs={"obs_type": "pixels", "task": "insertion"},
        )
        env = gym.make("gym_aloha/AlohaInsertion-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
        eval_env = copy.deepcopy(env)
        variant.env_max_reward = 4
        variant.max_timesteps = 400

    agent_dp = None
    if variant.env == 'libero':
        config = openpi_config.get_config("pi05_libero")
        checkpoint_dir = resolve_pi0_checkpoint(
            variant, "gs://openpi-assets/checkpoints/pi05_libero"
        )
        variant.pi0_action_horizon = config.model.action_horizon
        variant.pi0_action_dim = config.model.action_dim
    elif variant.env == 'aloha_cube':
        config = openpi_config.get_config("pi0_aloha_sim")
        checkpoint_dir = resolve_pi0_checkpoint(
            variant, "s3://openpi-assets/checkpoints/pi0_aloha_sim"
        )
        variant.pi0_action_horizon = config.model.action_horizon
        variant.pi0_action_dim = config.model.action_dim
    elif variant.env == 'aloha_insertion':
        from examples.act_insertion_policy import ActInsertionPolicy
        agent_dp = ActInsertionPolicy(getattr(variant, 'act_checkpoint', None) or "lerobot/act_aloha_sim_insertion_human")
        variant.pi0_action_horizon = agent_dp.action_horizon
        variant.pi0_action_dim = agent_dp.action_dim
        print(f"Loaded ACT insertion policy from {agent_dp.repo_id}")
    else:
        raise NotImplementedError()
    if variant.query_freq <= 0:
        variant.query_freq = variant.pi0_action_horizon
    if variant.query_freq > variant.pi0_action_horizon:
        raise ValueError(
            f"query_freq={variant.query_freq} exceeds base policy action_horizon="
            f"{variant.pi0_action_horizon}"
        )
    if variant.env == 'libero':
        variant.env_action_dim = 7
    elif variant.env in ('aloha_cube', 'aloha_insertion'):
        variant.env_action_dim = 14
    if not hasattr(variant, 'residual_scale') or variant.residual_scale is None:
        variant.residual_scale = 0.01
    latent_modes = [
        name for name, flag in (
            ('--use_eigenbasis', getattr(variant, 'use_eigenbasis', False)),
            ('--use_vae', getattr(variant, 'use_vae', False)),
            ('--use_flow', getattr(variant, 'use_flow', False)),
        ) if flag
    ]
    if len(latent_modes) > 1:
        raise ValueError(f"{' and '.join(latent_modes)} are mutually exclusive")
    if not getattr(variant, 'fit_residual', True) and not (
        getattr(variant, 'use_vae', False) or getattr(variant, 'use_flow', False)
    ):
        raise ValueError("--fit_residual false requires --use_vae or --use_flow")
    if getattr(variant, 'vae_condition_on_state', False):
        if not getattr(variant, 'use_vae', False):
            raise ValueError("--vae_condition_on_state requires --use_vae")
        if not getattr(variant, 'add_states', 1):
            raise ValueError("--vae_condition_on_state requires --add_states 1")
        n_state_samples = int(getattr(variant, 'vae_samples_per_state', 16))
        if getattr(variant, 'fit_residual', True) and n_state_samples < 2:
            raise ValueError("--vae_samples_per_state must be >= 2 when fitting residuals")
    if getattr(variant, 'pca_k_curriculum', False):
        if not getattr(variant, 'use_eigenbasis', False):
            raise ValueError("--pca_k_curriculum requires --use_eigenbasis")
        if int(variant.pca_k_start) < 1 or int(variant.pca_k_start) > int(variant.num_basis):
            raise ValueError(
                f"--pca_k_start must be in [1, num_basis], got {int(variant.pca_k_start)} "
                f"and num_basis={int(variant.num_basis)}"
            )
        if int(variant.pca_k_inc) < 1 or int(variant.pca_k_every) < 1:
            raise ValueError("--pca_k_inc and --pca_k_every must be >= 1")
        print(
            f"PCA curriculum: k={int(variant.pca_k_start)} "
            f"+{int(variant.pca_k_inc)} every {int(variant.pca_k_every)} steps, "
            f"final {int(variant.num_basis)}"
        )
    print(
        f"SAC residual horizon = query_freq = {variant.query_freq} "
        f"(base policy action_horizon = {variant.pi0_action_horizon})"
    )

    expname = build_exp_name(variant)
    output_root = variant.output_dir
    outputdir = os.path.join(output_root, f"{expname}_{time.strftime('%Y%m%d-%H%M%S')}_{variant.seed}")
    variant.outputdir = outputdir
    os.makedirs(outputdir, exist_ok=True)
    print('writing to output dir ', outputdir)

    if agent_dp is None:
        agent_dp = policy_config.create_trained_policy(config, checkpoint_dir)
        print("Loaded pi policy from %s", checkpoint_dir)

    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_name = variant.prefix if variant.prefix else expname
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(variant.prefix != '', variant, variant.wandb_project, experiment_id=wandb_name, output_dir=wandb_output_dir, group_name=group_name)

    dummy_env = DummyEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    sample_full_action = add_batch_dim(dummy_env.full_action_space.sample())
    print('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])
    print('sample action shape', sample_action.shape)
    print('sample full action shape', sample_full_action.shape)

    q_on_full_actions = variant.env in ('libero', 'aloha_cube', 'aloha_insertion')
    agent = PixelSACLearner(
        variant.seed,
        sample_obs,
        sample_action,
        zero_init_actor_mean=variant.env in ('libero', 'aloha_cube', 'aloha_insertion'),
        critic_actions=sample_full_action,
        residual_scale=getattr(variant, 'residual_scale', 1.0),
        use_eigenbasis=getattr(variant, 'use_eigenbasis', False),
        use_vae=getattr(variant, 'use_vae', False),
        use_flow=getattr(variant, 'use_flow', False),
        q_on_full_actions=q_on_full_actions,
        **kwargs,
    )

    online_buffer_size = variant.max_steps  // variant.multi_grad_step
    online_replay_buffer = ReplayBuffer(dummy_env.observation_space, dummy_env.full_action_space, int(online_buffer_size))
    replay_buffer = online_replay_buffer
    replay_buffer.seed(variant.seed)
    if getattr(variant, 'use_vae', False):
        variant.basis = load_vae_if_provided(variant)
        if variant.basis is not None:
            agent.set_residual_vae(variant.basis)
            log_vae_fit_metrics(wandb_logger, variant.basis, step=0)
    elif getattr(variant, 'use_flow', False):
        variant.basis = load_flow_if_provided(variant)
        if variant.basis is not None:
            agent.set_residual_flow(variant.basis)
            log_flow_fit_metrics(wandb_logger, variant.basis, step=0)
    else:
        variant.basis = load_basis_if_provided(variant)
        if variant.basis is not None:
            agent.set_residual_basis(variant.basis.V)
            log_basis_explained_variance(wandb_logger, variant.basis, step=0)
    trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger, shard_fn=shard_fn, agent_dp=agent_dp)
 
