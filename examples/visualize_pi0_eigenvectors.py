#!/usr/bin/env python3
"""Collect pi0 base-policy rollouts, fit PCA, and visualize eigenvectors on LIBERO.

Mirrors dimension/visualize_rollout_eigenvectors.py:
  1. Roll out pi0 with random noise (base policy) for num_rollouts episodes.
  2. Collect flattened action chunks (query_freq, 7) at each replan step.
  3. Fit PCA; keep PCs until variance_threshold (default 99%).
  4. For each PC, reset once and replay clip(mean + scale * PC) num_chunk_repeats times.

Example (from dsrl_pi0 repo root):
  python examples/visualize_pi0_eigenvectors.py \\
      --num-rollouts 20 --num-chunk-repeats 10 --query-freq 20 --task-id 57
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
from pathlib import Path

import imageio.v2 as imageio
import jax
import numpy as np
import tensorflow as tf
from tqdm import tqdm

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "openpi" / "src"))
sys.path.insert(0, str(REPO_ROOT / "openpi" / "packages" / "openpi-client" / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

# LIBERO is vendored under LIBERO/ but is not importable until `pip install -e LIBERO`.
# Add it to sys.path so this script works without a separate editable install.
_LIBERO_ROOT = REPO_ROOT / "LIBERO"
if _LIBERO_ROOT.is_dir():
    sys.path.insert(0, str(_LIBERO_ROOT))

try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Could not import `libero`. From the dsrl_pi0 repo root, run:\n"
        "  pip install -e LIBERO\n"
        "If LIBERO/ is missing, clone with submodules:\n"
        "  git clone --recurse-submodules ...\n"
        "  git submodule update --init --recursive"
    ) from exc

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as openpi_config
from openpi_client import image_tools

sys.path.insert(0, str(REPO_ROOT / "utils"))
from action_eigenspace import fit_action_basis_for_variance, save_action_basis

DIM_NAMES = ["dx", "dy", "dz", "dax", "day", "daz", "grip"]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "examples" / "output" / "pi0_eigenvector_viz"


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def obs_to_pi_zero_input(obs, task_description: str):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, 224, 224))
    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
        "prompt": str(task_description),
    }


def get_libero_env(task, resolution, seed):
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language


def frame_for_save(obs) -> np.ndarray:
    return np.asarray(obs["agentview_image"][::-1, ::-1], dtype=np.uint8)


def clip_action_chunk(action_chunk: np.ndarray, action_magnitude: float) -> np.ndarray:
    return np.clip(action_chunk, -action_magnitude, action_magnitude).astype(np.float32)


def collect_base_chunks(
    env,
    agent_dp,
    rng,
    *,
    task_description: str,
    num_rollouts: int,
    max_timesteps: int,
    query_frequency: int,
) -> np.ndarray:
    """Collect pi0 base action chunks, flattened to (query_freq * 7,)."""
    chunks = []
    for rollout_idx in tqdm(range(num_rollouts), desc="Collecting rollouts"):
        obs = env.reset()
        for t in range(max_timesteps):
            if t % query_frequency == 0:
                obs_pi = obs_to_pi_zero_input(obs, task_description)
                rng, key = jax.random.split(rng)
                noise = jax.random.normal(key, (1, 50, 32))
                actions = agent_dp.infer(obs_pi, noise=noise)["actions"]
                chunk = np.asarray(actions[:query_frequency], dtype=np.float64).reshape(-1)
                chunks.append(chunk)
            action_t = actions[t % query_frequency]
            obs, reward, done, _ = env.step(action_t)
            if done:
                break
        print(f"Rollout {rollout_idx + 1}/{num_rollouts}: {len(chunks)} chunks total")
    if len(chunks) < 2:
        raise ValueError(f"Need >=2 action chunks for PCA, got {len(chunks)}")
    return np.stack(chunks, axis=0)


def pc_summary(pc: np.ndarray, query_frequency: int) -> str:
    mat = pc.reshape(query_frequency, 7)
    parts = []
    for t in range(min(query_frequency, 4)):
        dominant_idx = int(np.argmax(np.abs(mat[t])))
        parts.append(f"t{t}:{DIM_NAMES[dominant_idx]}={mat[t, dominant_idx]:+.2f}")
    if query_frequency > 4:
        parts.append("...")
    return ", ".join(parts)


def save_gif(frames: list[np.ndarray], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, duration=1.0 / fps, loop=0)


def record_repeated_chunk_playback(
    env,
    action_chunk: np.ndarray,
    *,
    num_chunk_repeats: int,
) -> tuple[list[np.ndarray], int]:
    """One reset, then replay the same (query_freq, 7) chunk num_chunk_repeats times."""
    obs = env.reset()
    frames = [frame_for_save(obs)]
    env_steps = 0

    for _ in range(num_chunk_repeats):
        for substep in range(action_chunk.shape[0]):
            obs, reward, done, _ = env.step(action_chunk[substep])
            env_steps += 1
            frames.append(frame_for_save(obs))
            if done:
                return frames, env_steps

    return frames, env_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-rollouts", type=int, default=20)
    parser.add_argument("--num-chunk-repeats", type=int, default=10)
    parser.add_argument("--query-freq", type=int, default=20)
    parser.add_argument("--task-id", type=int, default=57)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-timesteps", type=int, default=400)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.99,
        help="Keep PCs until cumulative explained variance reaches this value.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Multiplier on eigenvector added to mean action chunk.",
    )
    parser.add_argument("--sign", type=float, default=1.0, choices=[-1.0, 1.0])
    parser.add_argument("--action-magnitude", type=float, default=1.5)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tf.config.set_visible_devices([], "GPU")

    if "SLURM_STEP_GPUS" in os.environ:
        os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_STEP_GPUS"]
        os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    task_suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = task_suite.get_task(args.task_id)
    env, task_description = get_libero_env(task, args.resolution, args.seed)

    config = openpi_config.get_config("pi0_libero")
    checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_libero")
    agent_dp = policy_config.create_trained_policy(config, checkpoint_dir)
    print(f"Loaded pi0 from {checkpoint_dir}")
    print(f"Task {args.task_id}: {task_description}")

    rng = jax.random.PRNGKey(args.seed)
    try:
        chunks = collect_base_chunks(
            env,
            agent_dp,
            rng,
            task_description=task_description,
            num_rollouts=args.num_rollouts,
            max_timesteps=args.max_timesteps,
            query_frequency=args.query_freq,
        )
        print(f"Collected {chunks.shape[0]} chunks, feature dim={chunks.shape[1]}")

        mean, V, explained, num_pcs = fit_action_basis_for_variance(
            chunks, variance_threshold=args.variance_threshold
        )
        cumvar = float(np.cumsum(explained)[-1])
        print(
            f"PCA: {num_pcs} PCs explain {cumvar * 100:.1f}% variance "
            f"(threshold={args.variance_threshold})"
        )
        for i, ratio in enumerate(explained):
            print(f"  PC{i + 1}: {ratio * 100:.2f}%")

        basis_path = output_dir / "fitted_basis.npz"
        save_action_basis(
            basis_path,
            V,
            explained,
            source="pi0_rollout",
            num_rollouts=args.num_rollouts,
            query_freq=args.query_freq,
            variance_threshold=args.variance_threshold,
            mean_action_chunk=mean.reshape(args.query_freq, 7).tolist(),
        )
        print(f"Saved basis -> {basis_path}")

        metadata = {
            "task_id": args.task_id,
            "task_language": task_description,
            "num_rollouts": args.num_rollouts,
            "num_pca_samples": int(chunks.shape[0]),
            "query_freq": args.query_freq,
            "feature_dim": int(chunks.shape[1]),
            "variance_threshold": args.variance_threshold,
            "num_pcs": num_pcs,
            "total_explained_variance": cumvar,
            "scale": args.scale,
            "sign": args.sign,
            "action_magnitude": args.action_magnitude,
            "num_chunk_repeats": args.num_chunk_repeats,
            "expected_env_steps": args.num_chunk_repeats * args.query_freq,
            "fps": args.fps,
            "mean_action_chunk": mean.reshape(args.query_freq, 7).tolist(),
            "basis_path": str(basis_path),
            "pcs": [],
        }

        for pc_idx in range(num_pcs):
            pc = V[:, pc_idx]
            var_pct = float(explained[pc_idx] * 100.0)
            summary = pc_summary(pc, args.query_freq)
            action_chunk = clip_action_chunk(
                (mean + args.sign * args.scale * pc).reshape(args.query_freq, 7),
                args.action_magnitude,
            )

            print(f"\nPC{pc_idx + 1} ({var_pct:.1f}%): {summary}")
            frames, env_steps = record_repeated_chunk_playback(
                env,
                action_chunk,
                num_chunk_repeats=args.num_chunk_repeats,
            )

            gif_path = output_dir / f"pc{pc_idx + 1:02d}.gif"
            save_gif(frames, gif_path, fps=args.fps)

            metadata["pcs"].append(
                {
                    "pc_index": pc_idx + 1,
                    "explained_variance_ratio": float(explained[pc_idx]),
                    "summary": summary,
                    "eigenvector": pc.reshape(args.query_freq, 7).tolist(),
                    "played_action_chunk": action_chunk.tolist(),
                    "gif_path": str(gif_path),
                    "num_frames": len(frames),
                    "env_steps": env_steps,
                }
            )
            print(
                f"[ok] PC{pc_idx + 1}: {summary}\n"
                f"     gif -> {gif_path.name} ({len(frames)} frames, {env_steps} env steps)"
            )

        meta_path = output_dir / "eigenvector_viz_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        print("-" * 72)
        print(f"Saved metadata: {meta_path}")
        print(f"Output dir:    {output_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

