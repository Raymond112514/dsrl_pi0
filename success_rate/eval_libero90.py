#!/usr/bin/env python3
"""Evaluate base pi policy success rate on all LIBERO-90 tasks."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from types import SimpleNamespace

import jax
import numpy as np
import tensorflow as tf
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from examples.train_sim import _get_libero_env
from examples.train_utils_sim import obs_to_pi_zero_input
from libero.libero import benchmark
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as openpi_config

DEFAULT_CHECKPOINTS = {
    "pi05_libero": "gs://openpi-assets/checkpoints/pi05_libero",
    "pi0_libero": "gs://openpi-assets/checkpoints/pi0_libero",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate base pi policy on all LIBERO-90 tasks."
    )
    parser.add_argument("--config", default="pi05_libero", choices=sorted(DEFAULT_CHECKPOINTS))
    parser.add_argument("--checkpoint", default="", help="Override checkpoint directory or URI.")
    parser.add_argument("--num_rollouts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--query_freq", type=int, default=-1, help="Defaults to model action horizon.")
    parser.add_argument("--max_timesteps", type=int, default=400)
    parser.add_argument("--task_start", type=int, default=0)
    parser.add_argument("--task_end", type=int, default=89, help="Inclusive task id.")
    parser.add_argument(
        "--output_csv",
        default="",
        help="Output CSV path. Defaults to success_rate/results/<config>_<timestamp>.csv",
    )
    return parser.parse_args()


def make_variant(task_description: str) -> SimpleNamespace:
    return SimpleNamespace(env="libero", task_description=task_description)


def eval_task(
    env,
    variant: SimpleNamespace,
    agent_dp,
    rng: jax.Array,
    *,
    num_rollouts: int,
    max_timesteps: int,
    query_freq: int,
    env_max_reward: int,
    action_horizon: int,
    action_dim: int,
) -> tuple[float, int, jax.Array]:
    successes: list[bool] = []

    for _ in range(num_rollouts):
        obs = env.reset()
        actions = None
        reward = 0.0

        for t in range(max_timesteps):
            if t % query_freq == 0:
                rng, key = jax.random.split(rng)
                obs_pi_zero = obs_to_pi_zero_input(obs, variant)
                noise = jax.random.normal(key, (1, action_horizon, action_dim))
                actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]

            action_t = actions[t % query_freq]
            obs, reward, done, _ = env.step(action_t)
            if done:
                break

        successes.append(reward == env_max_reward)

    return float(np.mean(successes)), int(np.sum(successes)), rng


def write_csv_row(csv_path: str, fieldnames: list[str], row: dict[str, object]) -> None:
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    tf.config.set_visible_devices([], "GPU")

    config = openpi_config.get_config(args.config)
    checkpoint_dir = args.checkpoint or download.maybe_download(DEFAULT_CHECKPOINTS[args.config])
    action_horizon = config.model.action_horizon
    action_dim = config.model.action_dim
    query_freq = action_horizon if args.query_freq <= 0 else args.query_freq

    output_csv = args.output_csv
    if not output_csv:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(REPO_ROOT, "success_rate", "results")
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(output_dir, f"{args.config}_{timestamp}.csv")

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    print(f"Loading policy: config={args.config}, checkpoint={checkpoint_dir}")
    agent_dp = policy_config.create_trained_policy(config, checkpoint_dir)

    task_suite = benchmark.get_benchmark_dict()["libero_90"]()
    num_tasks = task_suite.get_num_tasks()
    task_end = min(args.task_end, num_tasks - 1)
    if not (0 <= args.task_start <= task_end < num_tasks):
        raise ValueError(
            f"Invalid task range [{args.task_start}, {task_end}] for benchmark with {num_tasks} tasks."
        )

    fieldnames = [
        "task_id",
        "task_name",
        "task_description",
        "num_rollouts",
        "num_successes",
        "success_rate",
    ]

    rng = jax.random.PRNGKey(args.seed)
    total_successes = 0
    total_rollouts = 0

    for task_id in tqdm(range(args.task_start, task_end + 1), desc="LIBERO-90 tasks"):
        task = task_suite.get_task(task_id)
        env, task_description = _get_libero_env(task, 256, args.seed + task_id)
        variant = make_variant(task_description)

        success_rate, num_successes, rng = eval_task(
            env,
            variant,
            agent_dp,
            rng,
            num_rollouts=args.num_rollouts,
            max_timesteps=args.max_timesteps,
            query_freq=query_freq,
            env_max_reward=1,
            action_horizon=action_horizon,
            action_dim=action_dim,
        )
        env.close()

        total_successes += num_successes
        total_rollouts += args.num_rollouts

        row = {
            "task_id": task_id,
            "task_name": task.name,
            "task_description": task_description,
            "num_rollouts": args.num_rollouts,
            "num_successes": num_successes,
            "success_rate": f"{success_rate:.4f}",
        }
        write_csv_row(output_csv, fieldnames, row)
        print(
            f"task {task_id:02d}: success_rate={success_rate:.2%} "
            f"({num_successes}/{args.num_rollouts}) | {task_description}"
        )

    overall_success_rate = total_successes / total_rollouts if total_rollouts else 0.0
    summary_row = {
        "task_id": "overall",
        "task_name": "",
        "task_description": f"{args.config} aggregate over tasks {args.task_start}-{task_end}",
        "num_rollouts": total_rollouts,
        "num_successes": total_successes,
        "success_rate": f"{overall_success_rate:.4f}",
    }
    write_csv_row(output_csv, fieldnames, summary_row)

    print(f"\nOverall success rate: {overall_success_rate:.2%} ({total_successes}/{total_rollouts})")
    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    main()
