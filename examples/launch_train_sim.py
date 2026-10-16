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
    parser.add_argument('--num_basis', default=8, type=int, help='Number of PCA components / VAE latent dim. With --pca_k_curriculum this is the final width; the actor is built at this K.')
    parser.add_argument(
        '--pca_k_curriculum',
        action='store_true',
        help='Decode only the first k(t) PCA coefficients. k starts at --pca_k_start and grows by --pca_k_inc every --pca_k_every training steps, up to --num_basis. Requires --use_eigenbasis.',
    )
    parser.add_argument('--pca_k_start', default=2, type=int, help='Live basis count at step 0 when --pca_k_curriculum is set.')
    parser.add_argument('--pca_k_inc', default=2, type=int, help='How many basis vectors to turn on at each curriculum boundary.')
    parser.add_argument('--pca_k_every', default=10000, type=int, help='Training steps between curriculum increases of the live basis count.')
    parser.add_argument('--warmup_rollouts', default=20, type=int, help='Rollouts before PCA/VAE fit and SAC updates')
    parser.add_argument('--basis_path', default='', help='Optional pre-fit residual_basis.npz (skips online PCA)')
    parser.add_argument(
        '--use_vae',
        action='store_true',
        help='Train SAC in K-dim VAE latent space. Execution is always a_base + lambda * D(z). --fit_residual false fits the VAE on action chunks instead of pairwise diffs.',
    )
    parser.add_argument('--vae_beta', type=float, default=1e-3, help='KL weight β for residual VAE')
    parser.add_argument('--vae_epochs', type=int, default=100, help='Epochs to train residual VAE on warmup diffs')
    parser.add_argument('--vae_path', default='', help='Optional pre-fit residual_vae.pkl (skips online VAE fit)')
    parser.add_argument(
        '--vae_condition_on_state',
        action='store_true',
        help=(
            'Condition the residual VAE on the query-time proprio vector '
            '(agent_pos, 14-d on ALOHA). Default is unconditional. '
            'With --fit_residual, targets are ordered pairwise differences of '
            '--vae_samples_per_state base chunks drawn at that same state.'
        ),
    )
    parser.add_argument(
        '--vae_samples_per_state',
        type=int,
        default=16,
        help=(
            'How many base-policy chunks to draw at each query state when '
            '--vae_condition_on_state and --fit_residual are set. The VAE is '
            'fit on all ordered pairs a_i - a_j, i != j. Default 16.'
        ),
    )
    parser.add_argument(
        '--decode_add_mean',
        default=True,
        type=lambda x: str(x).lower() in ('1', 'true', 'yes', 'y'),
        help=(
            'If true (default), the VAE/flow residual is std * D(z) + mean. '
            'If false, drop the dataset mean and use std * D(z). '
            'The executed action is still a_base + residual_scale * that residual.'
        ),
    )
    parser.add_argument(
        '--fit_residual',
        default=True,
        type=lambda x: str(x).lower() in ('1', 'true', 'yes', 'y'),
        help=(
            'If true (default), fit the VAE or flow on pairwise chunk diffs. '
            'If false, fit the action chunks themselves. Either way the executed '
            'action is a_base + residual_scale * D(z), with D the unnormalized decode.'
        ),
    )
    parser.add_argument(
        '--use_flow',
        action='store_true',
        help=(
            'Train SAC on full-dim flow-matching noise. '
            'Executed action is always a_base + lambda * ODE(ε). '
            '--fit_residual false fits the flow on action chunks instead of pairwise diffs. '
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
    
