import argparse
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))              # for `examples.*`
sys.path.insert(0, str(REPO_ROOT / "LIBERO")) 

import sys
from examples.train_sim import main
from jaxrl2.utils.launch_util import parse_training_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, help='Random seed.', type=int)
    parser.add_argument('--launch_group_id', default='', help='group id used to group runs on wandb.')
    parser.add_argument('--eval_episodes', default=10,help='Number of episodes used for evaluation.', type=int)
    parser.add_argument('--env', default='libero', help='name of environment')
    parser.add_argument('--log_interval', default=1000, help='Logging interval.', type=int)
    parser.add_argument('--eval_interval', default=5000, help='Eval interval.', type=int)
    parser.add_argument('--checkpoint_interval', default=-1, help='checkpoint interval.', type=int)
    parser.add_argument('--batch_size', default=16, help='Mini batch size.', type=int)
    parser.add_argument('--max_steps', default=int(1e6), help='Number of training steps.', type=int)
    parser.add_argument('--add_states', default=1, help='whether to add low-dim states to the obervations', type=int)
    parser.add_argument('--wandb_project', default='dsrl_pi0', help='wandb project')
    parser.add_argument('--start_online_updates', default=1000, help='number of steps to collect before starting online updates', type=int)
    parser.add_argument('--algorithm', default='pixel_sac', help='type of algorithm')
    parser.add_argument('--prefix', default='', help='prefix to use for wandb')
    parser.add_argument('--suffix', default='', help='suffix to use for wandb')
    parser.add_argument('--multi_grad_step', default=1, help='Number of graident steps to take per environment step, aka UTD', type=int)
    parser.add_argument('--resize_image', default=-1, help='the size of image if need resizing', type=int)
    parser.add_argument(
        '--query_freq',
        default=-1,
        help='Replan interval and SAC residual horizon. Uses the first query_freq steps of the base policy chunk. Defaults to the base policy action_horizon.',
        type=int,
    )
    parser.add_argument('--task_id', default=44, help='task id', type=int)
    parser.add_argument('--output_dir', default='', help='Base directory for experiment outputs')
    parser.add_argument('--pi0_checkpoint', default='', help='Local pi0 checkpoint directory; download default if unset or missing')
    parser.add_argument(
        '--act_checkpoint',
        default='lerobot/act_aloha_sim_insertion_human',
        help='LeRobot ACT repo id or local path for aloha_insertion',
    )
    parser.add_argument('--residual_scale', type=float, default=0.01, help='Scale factor for residual actions')
    parser.add_argument(
        '--collect_with_residual',
        action='store_true',
        help='Apply SAC residual during rollout collection from the first trajectory',
    )
    parser.add_argument(
        '--use_eigenbasis',
        action='store_true',
        help='Train SAC in K-dim PCA coefficient space (a = a_base + lambda * V @ c)',
    )
    parser.add_argument('--num_basis', default=8, type=int, help='Number of PCA components / VAE latent dim.')
    parser.add_argument('--warmup_rollouts', default=20, type=int, help='Rollouts before PCA/VAE fit and SAC updates')
    parser.add_argument('--basis_path', default='', help='Optional pre-fit residual_basis.npz (skips online PCA)')
    parser.add_argument(
        '--use_vae',
        action='store_true',
        help='Train SAC in K-dim VAE latent space. The VAE is fit on pairwise chunk diffs. Execution is a_base + lambda * (std * D(z) + mean).',
    )
    parser.add_argument('--vae_beta', type=float, default=1e-3, help='KL weight β for residual VAE')
    parser.add_argument('--vae_epochs', type=int, default=100, help='Epochs to train residual VAE on warmup diffs')
    parser.add_argument('--vae_path', default='', help='Optional pre-fit residual_vae.pkl (skips online VAE fit)')
    parser.add_argument(
        '--use_flow',
        action='store_true',
        help=(
            'Train SAC on full-dim flow-matching noise. '
            'The flow is fit on pairwise chunk diffs. '
            'Executed action is a_base + lambda * (std * ODE(ε) + mean). '
            'Latent dim = query_freq * action_dim.'
        ),
    )
    parser.add_argument('--flow_epochs', type=int, default=100, help='Epochs to train residual flow on warmup diffs')
    parser.add_argument('--flow_steps', type=int, default=10, help='Euler steps when decoding flow noise to a residual')
    parser.add_argument('--flow_path', default='', help='Optional pre-fit residual_flow.pkl (skips online flow fit)')

    train_args_dict = dict(
        actor_lr=1e-4,
        critic_lr= 3e-4,
        temp_lr=3e-4,
        hidden_dims= (128, 128, 128),
        cnn_features= (32, 32, 32, 32),
        cnn_strides= (2, 1, 1, 1),
        cnn_padding= 'VALID',
        latent_dim= 50,
        discount= 0.999,
        tau= 0.005,
        critic_reduction = 'mean',
        dropout_rate=0.0,
        aug_next=1,
        use_bottleneck=True,
        encoder_type='small',
        encoder_norm='group',
        use_spatial_softmax=True,
        softmax_temperature=-1,
        target_entropy='auto',
        num_qs=10,
        action_magnitude=1.0,
        num_cameras=1,
        )

    variant, args = parse_training_args(train_args_dict, parser)
    print(variant)
    main(variant)
    sys.exit()
    
