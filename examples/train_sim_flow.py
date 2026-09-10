#!/usr/bin/env python3
"""Flow-matching residual RL entry (Libero + Aloha).

Fits a frozen action-chunk CFM on warmup base-policy rollouts, then trains SAC
in the full residual chunk space D = query_freq * action_dim:

    a = a_base + λ * flow(w),   w = π_residual(s, a_base) ∈ R^D

No --num_basis / latent dim: SAC outputs noise that is integrated by the flow.
Reuses the shared sim training loop in examples/train_sim.py.
"""

from __future__ import annotations

from examples.train_sim import main as train_sim_main


def main(variant):
    variant.use_flow_basis = True
    variant.use_vae_basis = False
    variant.use_eigenbasis = False
    variant.use_random_basis = False
    variant.use_dct_basis = False
    if getattr(variant, "q_base_action", False):
        raise ValueError("--q_base_action is not supported for flow residual RL")
    # num_basis is set in train_sim once query_freq / env_action_dim are known.
    print(
        f"Flow residual RL: warmup={getattr(variant, 'warmup_rollouts', 20)}, "
        f"flow_epochs={getattr(variant, 'flow_epochs', 120)}, "
        f"flow_n_steps={getattr(variant, 'flow_n_steps', 10)}"
    )
    train_sim_main(variant)
