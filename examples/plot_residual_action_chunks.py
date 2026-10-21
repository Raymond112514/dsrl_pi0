#!/usr/bin/env python3
"""Sample residual actions at a few real Aloha states and plot the chunks.

Follows serl/examples/async_drq_robomimic/plot_ckpt_action_stats.py: a short
rollout with the latest checkpoint visits real states, then every saved SAC
checkpoint is sampled at those states without stepping. Each sample is drawn
as three heatmaps:

  residual   decoded residual chunk, before residual_scale
  base       first query_freq steps of the pi0 chunk
  executed   base + residual_scale * residual
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flax.training import checkpoints
from gymnasium.spaces import Box, Dict

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "openpi" / "src"))

import gymnasium as gym
import gym_aloha
from examples.train_utils_sim import _executed_base_chunk, obs_to_img, obs_to_pi_zero_input, obs_to_qpos
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.utils.general_utils import add_batch_dim
from openpi.policies import policy_config
from openpi.training import config as openpi_config

ROOT = Path("/global/scratch/users/r112358/aloha_test")
OUT = Path.home() / "aloha_action_plots"
PI0_CKPT = Path("/global/scratch/users/r112358/pi0_checkpoint/pi0_aloha_sim")
QUERY_FREQ = 20
ACTION_DIM = 14
RESIDUAL_SCALE = 0.2
N_STATES = 3
N_SAMPLES = 4
WARMUP_CHUNKS = 2
IMAGE_SIZE = 64

METHODS = {
    "baseline": "aloha_test_baseline_qf20_rs0.2_utd10_seed100",
    "pca": "aloha_test_pca_qf20_K8_rs0.2_utd10_seed100",
    "vae": "aloha_test_vae_qf20_K8_beta0.0001_rs0.2_utd10_seed100",
    "flow": "aloha_test_flow_qf20_ep50_steps10_rs0.2_utd10_seed100",
}
JOINTS = ["L1", "L2", "L3", "L4", "L5", "L6", "LG", "R1", "R2", "R3", "R4", "R5", "R6", "RG"]


def _variant(pi0_horizon: int, pi0_action_dim: int) -> SimpleNamespace:
    return SimpleNamespace(
        env="aloha_cube",
        resize_image=IMAGE_SIZE,
        query_freq=QUERY_FREQ,
        env_action_dim=ACTION_DIM,
        pi0_action_horizon=pi0_horizon,
        pi0_action_dim=pi0_action_dim,
        residual_scale=RESIDUAL_SCALE,
    )


def _run_dir(name: str) -> Path:
    root = ROOT / name
    cands = [p for p in root.iterdir() if p.is_dir() and list(p.glob("checkpoint*"))]
    if not cands:
        raise FileNotFoundError(f"no checkpoints under {root}")
    return sorted(cands)[-1]


def _steps(run_dir: Path) -> list[tuple[str, Path, int | None]]:
    """(label, ckpt_dir, step). step None loads the only checkpoint in ckpt_dir."""
    found = []
    initial = run_dir / "initial"
    if list(initial.glob("checkpoint*")):
        found.append(("initial", initial, None))
    numbered = []
    for path in run_dir.glob("checkpoint*"):
        if path.is_dir() and path.name[len("checkpoint") :].isdigit():
            numbered.append((int(path.name[len("checkpoint") :]), path))
    for step, _path in sorted(numbered):
        found.append((str(step), run_dir, step))
    return found


def _make_agent(kind: str):
    full_dim = QUERY_FREQ * ACTION_DIM
    sac_dim = 8 if kind in ("pca", "vae") else full_dim
    obs_space = Dict(
        {
            "pixels": Box(low=0, high=255, shape=(IMAGE_SIZE, IMAGE_SIZE, 3, 1), dtype=np.uint8),
            "state": Box(low=-1.0, high=1.0, shape=(ACTION_DIM, 1), dtype=np.float32),
            "action_diffusion": Box(low=-np.inf, high=np.inf, shape=(full_dim, 1), dtype=np.float32),
        }
    )
    action_space = Box(low=-1, high=1, shape=(1, sac_dim), dtype=np.float32)
    full_action_space = Box(low=-1, high=1, shape=(1, full_dim), dtype=np.float32)
    agent = PixelSACLearner(
        100,
        add_batch_dim(obs_space.sample()),
        add_batch_dim(action_space.sample()),
        actor_lr=1e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        hidden_dims=(128, 128, 128),
        cnn_features=(32, 32, 32, 32),
        cnn_strides=(2, 1, 1, 1),
        cnn_padding="VALID",
        latent_dim=50,
        discount=0.999,
        tau=0.005,
        critic_reduction="mean",
        dropout_rate=0.0,
        encoder_type="small",
        encoder_norm="group",
        use_spatial_softmax=True,
        softmax_temperature=-1,
        target_entropy="auto",
        num_qs=10,
        action_magnitude=1.0,
        num_cameras=1,
        zero_init_actor_mean=True,
        critic_actions=add_batch_dim(full_action_space.sample()),
        residual_scale=RESIDUAL_SCALE,
        q_on_full_actions=True,
    )
    return agent


def _restore(agent, ckpt_dir: Path, step: int | None):
    restored = checkpoints.restore_checkpoint(
        ckpt_dir, agent._save_dict, step=step, prefix="checkpoint"
    )
    agent._actor = restored["actor"]
    agent._critic = restored["critic"]
    agent._target_critic_params = restored["target_critic_params"]
    agent._temp = restored["temp"]
    return agent


def _load_basis(kind: str, run_dir: Path):
    if kind == "pca":
        from latent_rl.pca import PCALatentModel

        return PCALatentModel.load(str(run_dir / "residual_basis.npz"))
    if kind == "vae":
        from latent_rl.vae import VAELatentModel

        return VAELatentModel.load(str(run_dir / "residual_vae.pkl"))
    if kind == "flow":
        from latent_rl.flow import FlowLatentModel

        return FlowLatentModel.load(str(run_dir / "residual_flow.pkl"))
    return None


def _base_chunk(agent_dp, obs, variant, key) -> np.ndarray:
    obs_pi = obs_to_pi_zero_input(obs, variant)
    noise = np.asarray(
        jax.device_get(jax.random.normal(key, (1, variant.pi0_action_horizon, variant.pi0_action_dim))),
        dtype=np.float32,
    )
    actions = agent_dp.infer(obs_pi, noise=noise)["actions"]
    return np.asarray(_executed_base_chunk(actions, variant), dtype=np.float32)


def _sac_obs(obs, variant, base: np.ndarray) -> dict:
    image = obs_to_img(obs, variant)
    qpos = np.asarray(obs_to_qpos(obs, variant), dtype=np.float32)
    return {
        "pixels": image[np.newaxis, ..., np.newaxis],
        "state": qpos[np.newaxis, ..., np.newaxis],
        "action_diffusion": base.reshape(1, -1, 1),
    }


def _decode(kind: str, latent: np.ndarray, basis) -> np.ndarray:
    flat = np.asarray(latent, dtype=np.float32).reshape(-1)
    if kind == "baseline":
        return flat.reshape(QUERY_FREQ, ACTION_DIM)
    return np.asarray(basis.decode(flat), dtype=np.float32)


def _sample_residual(agent, obs_dict) -> np.ndarray:
    action = agent.sample_actions(obs_dict)
    return np.asarray(jax.device_get(action), dtype=np.float32)


def _visit_states(env, agent, agent_dp, variant, kind: str, basis) -> list:
    states = []
    for r in range(N_STATES):
        obs, _ = env.reset(seed=1000 + r)
        rng = jax.random.PRNGKey(10_000 + r)
        for _chunk in range(WARMUP_CHUNKS):
            rng, key = jax.random.split(rng)
            base = _base_chunk(agent_dp, obs, variant, key)
            obs_dict = _sac_obs(obs, variant, base)
            residual = _decode(kind, _sample_residual(agent, obs_dict), basis)
            executed = base + RESIDUAL_SCALE * residual
            done = False
            for t in range(QUERY_FREQ):
                obs, _reward, terminated, truncated, _info = env.step(executed[t])
                if terminated or truncated:
                    done = True
                    break
            if done:
                break
        states.append(obs)
    return states


def _chunks_at_state(agent, agent_dp, obs, variant, kind: str, basis, key) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    key, base_key = jax.random.split(key)
    base = _base_chunk(agent_dp, obs, variant, base_key)
    obs_dict = _sac_obs(obs, variant, base)
    rows = []
    for _ in range(N_SAMPLES):
        residual = _decode(kind, _sample_residual(agent, obs_dict), basis)
        executed = base + RESIDUAL_SCALE * residual
        rows.append((residual, base, executed))
    return rows


def _heat(ax, matrix: np.ndarray, title: str, vmin: float, vmax: float):
    # Chunks are (time, joint). Draw joints on Y and chunk time on X.
    shown = np.asarray(matrix).T
    im = ax.imshow(shown, vmin=vmin, vmax=vmax, cmap="RdBu_r", aspect="auto")
    ax.set_yticks(range(ACTION_DIM), JOINTS, fontsize=7)
    ax.set_xticks(range(0, QUERY_FREQ, 2), [f"t{t}" for t in range(0, QUERY_FREQ, 2)], fontsize=7)
    ax.set_title(title, fontsize=8)
    return im


def _plot(samples: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]], path: Path, title: str):
    n_states = len(samples)
    flat = [matrix for rows in samples for triple in rows for matrix in triple]
    limit = max(float(np.max(np.abs(np.stack(flat)))), 1e-3)
    fig, axes = plt.subplots(
        n_states * N_SAMPLES,
        3,
        figsize=(15.5, 2.15 * n_states * N_SAMPLES),
        squeeze=False,
    )
    labels = ("residual", "base", "executed")
    last = None
    for s, rows in enumerate(samples):
        for j, triple in enumerate(rows):
            row = s * N_SAMPLES + j
            for col, (matrix, label) in enumerate(zip(triple, labels)):
                last = _heat(
                    axes[row, col],
                    matrix,
                    f"state {s + 1} sample {j + 1}  {label}",
                    -limit,
                    limit,
                )
    fig.subplots_adjust(right=0.90, top=0.96, hspace=0.65, wspace=0.28)
    cax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    fig.colorbar(last, cax=cax)
    fig.suptitle(title, fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    config = openpi_config.get_config("pi0_aloha_sim")
    variant = _variant(config.model.action_horizon, config.model.action_dim)
    print(f"loading pi0 from {PI0_CKPT}", flush=True)
    agent_dp = policy_config.create_trained_policy(config, PI0_CKPT)
    env = gym.make(
        "gym_aloha/AlohaTransferCube-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
    )
    n_written = 0
    for kind, dirname in METHODS.items():
        run_dir = _run_dir(dirname)
        steps = _steps(run_dir)
        if not steps:
            print(f"skip {kind}: no checkpoints in {run_dir}", flush=True)
            continue
        basis = _load_basis(kind, run_dir)
        print(f"{kind} {run_dir.name}  n_ckpt={len(steps)}", flush=True)
        agent = _make_agent(kind)
        latest_label, latest_dir, latest_step = steps[-1]
        _restore(agent, latest_dir, latest_step)
        states = _visit_states(env, agent, agent_dp, variant, kind, basis)
        for label, ckpt_dir, step in steps:
            _restore(agent, ckpt_dir, step)
            panels = [
                _chunks_at_state(
                    agent,
                    agent_dp,
                    obs,
                    variant,
                    kind,
                    basis,
                    jax.random.PRNGKey(2000 + s),
                )
                for s, obs in enumerate(states)
            ]
            out = OUT / kind / f"{label}.png"
            _plot(
                panels,
                out,
                (
                    f"{kind} step {label}   "
                    f"executed = base + {RESIDUAL_SCALE:g} * residual   "
                    f"{N_STATES} states x {N_SAMPLES} samples   qf={QUERY_FREQ}"
                ),
            )
            n_written += 1
            print(f"  wrote {out}  (states visited with {latest_label})", flush=True)
    env.close()
    print(f"done {n_written} figures under {OUT}", flush=True)


if __name__ == "__main__":
    main()
