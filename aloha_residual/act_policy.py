"""Minimal LeRobot ACT adapter for the existing residual-RL training loop."""

from __future__ import annotations

import numpy as np
import torch

from lerobot.common import policies as _policies  # noqa: F401
from lerobot.common.envs.configs import AlohaEnv
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig


class ACTChunkPolicy:
    """Expose ACT's full action chunk through the pi0-style ``infer`` API."""

    def __init__(
        self,
        checkpoint: str = "lerobot/act_aloha_sim_insertion_human",
        device: str = "cuda",
    ):
        env_cfg = AlohaEnv(task="AlohaInsertion-v0", episode_length=400)
        config = PreTrainedConfig.from_pretrained(checkpoint)
        config.pretrained_path = checkpoint
        config.device = device
        config.use_amp = False

        self.policy = make_policy(config, env_cfg=env_cfg)
        self.policy.eval()
        self.device = torch.device(config.device)
        self.action_horizon = int(config.n_action_steps)
        self.action_dim = int(config.output_features["action"].shape[0])

    @torch.inference_mode()
    def infer(self, observation: dict, noise=None) -> dict[str, np.ndarray]:
        del noise
        raw_batch = {
            "pixels": {"top": np.asarray(observation["pixels"]["top"])[None]},
            "agent_pos": np.asarray(observation["agent_pos"])[None],
        }
        batch = preprocess_observation(raw_batch)
        batch = {key: value.to(self.device) for key, value in batch.items()}
        batch = self.policy.normalize_inputs(batch)
        batch["observation.images"] = [
            batch[key] for key in self.policy.config.image_features
        ]
        actions = self.policy.model(batch)[0][:, : self.action_horizon]
        actions = self.policy.unnormalize_outputs({"action": actions})["action"]
        return {"actions": actions[0].cpu().numpy().astype(np.float32)}
