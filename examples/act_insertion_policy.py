"""ACT policy trained on Aloha sim insertion.

The checkpoint is lerobot/act_aloha_sim_insertion_human. infer() returns a full
action chunk so the residual loop can take the first query_freq steps.
"""

from __future__ import annotations

import numpy as np


class ActInsertionPolicy:
    def __init__(self, repo_id: str = "lerobot/act_aloha_sim_insertion_human"):
        import torch
        from lerobot.common.policies.act.modeling_act import ACTPolicy

        self._torch = torch
        self.repo_id = repo_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = ACTPolicy.from_pretrained(repo_id)
        self.policy.to(self.device)
        self.policy.eval()
        self.action_horizon = int(self.policy.config.n_action_steps)
        self.action_dim = int(self.policy.config.output_features["action"].shape[0])
        image_shape = self.policy.config.input_features["observation.images.top"].shape
        self.image_hw = (int(image_shape[1]), int(image_shape[2]))

    def infer(self, obs, noise=None):
        torch = self._torch
        image = np.asarray(obs["pixels"]["top"])
        state = np.asarray(obs["agent_pos"], dtype=np.float32).reshape(-1)
        if image.shape[:2] != self.image_hw:
            import PIL.Image
            image = np.array(PIL.Image.fromarray(image).resize((self.image_hw[1], self.image_hw[0])))
        img = torch.from_numpy(np.ascontiguousarray(image)).to(self.device)
        if img.dtype != torch.float32:
            img = img.float()
        if float(img.max()) > 1.5:
            img = img / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0).contiguous()
        st = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
        batch = {
            "observation.images.top": img,
            "observation.state": st,
        }
        with torch.no_grad():
            batch = self.policy.normalize_inputs(batch)
            if self.policy.config.image_features:
                batch = dict(batch)
                batch["observation.images"] = [batch[key] for key in self.policy.config.image_features]
            actions = self.policy.model(batch)[0][:, : self.action_horizon]
            actions = self.policy.unnormalize_outputs({"action": actions})["action"]
        return {"actions": actions[0].detach().float().cpu().numpy()}
