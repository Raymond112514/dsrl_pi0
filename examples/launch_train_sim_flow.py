#!/usr/bin/env python3
"""Launch flow-matching residual RL (a = a_base + λ * flow(w))."""

import argparse
import sys

from examples.train_sim_flow import main
from jaxrl2.utils.launch_util import parse_training_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Residual RL with action-chunk flow matching (no latent / num_basis)."
    )

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--launch_group_id", default="")
    parser.add_argument("--eval_episodes", default=10, type=int)
    parser.add_argument("--env", default="libero", help="libero | aloha_cube | aloha_insertion")
    parser.add_argument("--log_interval", default=1000, type=int)
    parser.add_argument("--eval_interval", default=5000, type=int)
    parser.add_argument("--checkpoint_interval", default=-1, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_steps", default=int(1e6), type=int)
    parser.add_argument("--add_states", default=1, type=int)
    parser.add_argument("--wandb_project", default="dsrl_pi0", type=str)
    parser.add_argument("--start_online_updates", default=1000, type=int)
    parser.add_argument("--algorithm", default="pixel_sac", type=str)
    parser.add_argument("--prefix", default="", type=str)
    parser.add_argument("--suffix", default="", type=str)
    parser.add_argument("--multi_grad_step", default=1, type=int)
    parser.add_argument("--resize_image", default=-1, type=int)
    parser.add_argument(
        "--query_freq",
        default=-1,
        type=int,
        help="Env steps between base queries / residual chunk length. Default: full horizon.",
    )
    parser.add_argument("--task_id", default=44, type=int)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--pi0_checkpoint", default="", type=str)
    parser.add_argument(
        "--act_checkpoint",
        default="lerobot/act_aloha_sim_insertion_human",
        type=str,
    )
    parser.add_argument("--residual_scale", type=float, default=0.01)
    parser.add_argument("--init_residual", action="store_true")
    parser.add_argument("--init_scale", type=float, default=1.0)
    parser.add_argument("--collect_with_residual", action="store_true")
    parser.add_argument(
        "--warmup_rollouts",
        default=20,
        type=int,
        help="Base-policy rollouts used to fit the action flow before SAC updates",
    )
    parser.add_argument("--flow_hidden", default=128, type=int)
    parser.add_argument("--flow_epochs", default=120, type=int)
    parser.add_argument("--flow_batch_size", default=256, type=int)
    parser.add_argument("--flow_lr", default=1e-3, type=float)
    parser.add_argument(
        "--flow_n_steps",
        default=10,
        type=int,
        help="Euler ODE steps when mapping SAC noise -> residual chunk",
    )
    parser.add_argument(
        "--flow_path",
        default="",
        type=str,
        help="Optional pre-fit residual_flow.pt (skips online flow fit)",
    )
    parser.add_argument(
        "--basis_path",
        default="",
        type=str,
        help="Alias for --flow_path",
    )

    train_args_dict = dict(
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
        aug_next=1,
        use_bottleneck=True,
        encoder_type="small",
        encoder_norm="group",
        use_spatial_softmax=True,
        softmax_temperature=-1,
        target_entropy="auto",
        num_qs=10,
        action_magnitude=1.0,
        num_cameras=1,
    )

    variant, args = parse_training_args(train_args_dict, parser)
    # Force flow mode (not PCA / VAE).
    variant.use_flow_basis = True
    variant.use_vae_basis = False
    variant.use_eigenbasis = False
    variant.use_random_basis = False
    if getattr(variant, "flow_path", "") == "" and getattr(variant, "basis_path", ""):
        variant.flow_path = variant.basis_path
    print(variant)
    main(variant)
    sys.exit()
