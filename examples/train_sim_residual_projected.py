#! /usr/bin/env python
import os

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

import sys

sys.path.append("/global/home/users/r112358/dsrl_pi0")
sys.path.append("/global/home/users/r112358/LIBERO")
sys.path.append("/global/home/users/r112358/dsrl_pi0/openpi")
sys.path.append("/global/home/users/r112358/dsrl_pi0/openpi/src")
sys.path.append("/global/home/users/r112358/dsrl_pi0/examples/classifier")
sys.path.append("/global/home/users/r112358/dsrl_pi0/examples/classifier")
sys.path.append("/global/home/users/r112358/dsrl_pi0/examples/utils")

import tempfile

import pathlib
import copy
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
from jaxrl2.utils.wandb_logger import WandBLogger
from functools import partial
from examples.train_utils_sim_residual_projected import trajwise_alternating_training_loop

from residual_basis import ResidualActionBasis
import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache

from openpi.training import config as openpi_config
from openpi.policies import policy_config
from openpi.shared import download

home_dir = os.environ["HOME"]
compilation_cache.initialize_cache(os.path.join(home_dir, "jax_compilation_cache"))


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def shard_batch(batch, sharding):
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
        obs_dict = {
            "pixels": Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8),
        }
        if variant.add_states:
            if variant.env == "libero":
                state_dim = 8
            elif variant.env == "aloha_cube":
                state_dim = 14
            else:
                raise NotImplementedError()
            obs_dict["state"] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)

        action_dim = 7 if variant.env == "libero" else 14
        obs_dict["action_diffusion"] = Box(
            low=-np.inf, high=np.inf, shape=(action_dim, 1), dtype=np.float32
        )
        self.observation_space = Dict(obs_dict)
        self.action_space = Box(
            low=-1,
            high=1,
            shape=(1, variant.num_basis),
            dtype=np.float32,
        )


def build_exp_name(variant):
    return f"task{variant.task_id}_K{variant.num_basis}_rs{variant.residual_scale}"


def load_basis_if_provided(variant):
    basis_path = variant.get("basis_path")
    if basis_path and os.path.isfile(basis_path):
        print(f"Loading PCA basis from {basis_path}")
        return ResidualActionBasis.load(basis_path)
    return None


def main(variant):
    devices = jax.local_devices()
    num_devices = len(devices)
    assert variant.batch_size % num_devices == 0
    print("num devices", num_devices)
    print("batch size", variant.batch_size)
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    tf.config.set_visible_devices([], "GPU")

    kwargs = variant["train_kwargs"]
    if kwargs.pop("cosine_decay", False):
        kwargs["decay_steps"] = variant.max_steps

    if not variant.prefix:
        import uuid

        variant.prefix = str(uuid.uuid4().fields[-1])[:5]

    expname = build_exp_name(variant)
    outputdir = os.path.join(
        "/global/scratch/users/r112358/pi0_exp",
        f"{expname}_{time.strftime('%Y%m%d-%H%M%S')}_{variant.seed}",
    )
    variant.outputdir = outputdir
    os.makedirs(outputdir, exist_ok=True)
    print("writing to output dir ", outputdir)

    if variant.env == "libero":
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict["libero_90"]()
        task = task_suite.get_task(variant.task_id)
        env, task_description = _get_libero_env(task, 256, variant.seed)
        eval_env = env
        variant.task_description = task_description
        variant.env_max_reward = 1
        variant.max_timesteps = 400
    elif variant.env == "aloha_cube":
        from gymnasium.envs.registration import register

        register(
            id="gym_aloha/AlohaTransferCube-v0",
            entry_point="gym_aloha.env:AlohaEnv",
            max_episode_steps=400,
            nondeterministic=True,
            kwargs={"obs_type": "pixels", "task": "transfer_cube"},
        )
        env = gym.make(
            "gym_aloha/AlohaTransferCube-v0",
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
        )
        eval_env = copy.deepcopy(env)
        variant.env_max_reward = 4
        variant.max_timesteps = 400
    else:
        raise NotImplementedError()

    if variant.env == "libero":
        config = openpi_config.get_config("pi0_libero")
        checkpoint_dir = "/global/scratch/users/r112358/checkpoints/pi0_libero"
    elif variant.env == "aloha_cube":
        config = openpi_config.get_config("pi0_aloha_sim")
        checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_aloha_sim")
    else:
        raise NotImplementedError()
    agent_dp = policy_config.create_trained_policy(config, checkpoint_dir)
    print("Loaded pi0 policy from", checkpoint_dir)

    basis = load_basis_if_provided(variant)
    variant.basis = basis

    group_name = variant.prefix + "_" + variant.launch_group_id
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(
        variant.prefix != "",
        variant,
        variant.wandb_project,
        experiment_id=expname,
        output_dir=wandb_output_dir,
        group_name=group_name,
    )

    dummy_env = DummyEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    print("sample obs shapes", [(k, v.shape) for k, v in sample_obs.items()])
    print("sample action shape", sample_action.shape)

    agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **kwargs)

    online_buffer_size = variant.max_steps // variant.multi_grad_step
    online_replay_buffer = ReplayBuffer(
        dummy_env.observation_space, dummy_env.action_space, int(online_buffer_size)
    )
    replay_buffer = online_replay_buffer
    replay_buffer.seed(variant.seed)
    try:
        trajwise_alternating_training_loop(
            variant,
            agent,
            env,
            eval_env,
            online_replay_buffer,
            replay_buffer,
            wandb_logger,
            shard_fn=shard_fn,
            agent_dp=agent_dp,
            basis=basis,
        )
    finally:
        wandb_logger.finish()

