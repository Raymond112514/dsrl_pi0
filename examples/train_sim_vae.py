#!/usr/bin/env python3
"""VAE-latent residual RL entry (Libero + Aloha).

Fits a frozen action-chunk VAE on warmup base-policy rollouts, then trains SAC
in the VAE latent space:

    a = a_base + λ * decode(w),   w = π_residual(s, a_base)

Reuses the shared sim training loop in examples/train_sim.py.
"""

from __future__ import annotations

from examples.train_sim import main as train_sim_main


def main(variant):
    variant.use_vae_basis = True
    variant.use_eigenbasis = False
    variant.use_random_basis = False
    variant.use_dct_basis = False
    # Align SAC / DummyEnv dim with VAE latent size.
    latent = int(getattr(variant, "vae_latent_dim", getattr(variant, "num_basis", 8)))
    variant.vae_latent_dim = latent
    variant.num_basis = latent
    if getattr(variant, "q_base_action", False):
        raise ValueError("--q_base_action is not supported for VAE residual RL")
    print(
        f"VAE residual RL: latent_dim={latent}, linear={bool(getattr(variant, 'vae_linear', False))}, "
        f"warmup={getattr(variant, 'warmup_rollouts', 20)}, "
        f"vae_epochs={getattr(variant, 'vae_epochs', 120)}"
    )
    train_sim_main(variant)
