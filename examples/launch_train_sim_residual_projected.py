import argparse
import sys
sys.path.append("/global/home/users/r112358/dsrl_pi0")
sys.path.append("/global/home/users/r112358/dsrl_pi0/LIBERO")
sys.path.append("/global/home/users/r112358/dsrl_pi0/examples/classifier")

from examples.train_sim_residual_projected import main
from jaxrl2.utils.launch_util import parse_training_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--launch_group_id", default="")
    parser.add_argument("--eval_episodes", default=10, type=int)
    parser.add_argument("--env", default="libero")
    parser.add_argument("--log_interval", default=1000, type=int)
    parser.add_argument("--eval_interval", default=5000, type=int)
    parser.add_argument("--checkpoint_interval", default=-1, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_steps", default=int(1e6), type=int)
    parser.add_argument("--add_states", default=1, type=int)
    parser.add_argument("--wandb_project", default="dsrl_pi0_residual_projected")
    parser.add_argument("--start_online_updates", default=500, type=int)
    parser.add_argument("--algorithm", default="pixel_sac")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--multi_grad_step", default=1, type=int)
    parser.add_argument("--resize_image", default=64, type=int)
    parser.add_argument("--query_freq", default=20, type=int)
    parser.add_argument("--task_id", default=44, type=int)

    parser.add_argument("--residual_scale", default=0.01, type=float)
    parser.add_argument("--num_basis", default=6, type=int, help="Top-K PCA components for residual policy")
    parser.add_argument("--warmup_rollouts", default=20, type=int, help="Pi0-only trajectories before SAC updates; also used to fit PCA")
    parser.add_argument("--basis_path", default="", type=str, help="Optional pre-fit basis .npz (skip warmup)")

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

    variant.residual_scale = args.residual_scale
    variant.num_basis = args.num_basis
    variant.warmup_rollouts = args.warmup_rollouts
    variant.basis_path = args.basis_path or None

    print(variant)
    main(variant)
    sys.exit()

