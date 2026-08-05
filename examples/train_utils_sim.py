from tqdm import tqdm
import os
import numpy as np
import wandb
import jax
from openpi_client import image_tools
import math
import PIL

from examples.residual_basis import uses_projected_basis

def print_green(text):
    print(f'\033[92m{text}\033[0m')

def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def obs_to_img(obs, variant):
    '''
    Convert raw observation to resized image for DSRL actor/critic
    '''
    if variant.env == 'libero':
        curr_image = obs["agentview_image"][::-1, ::-1]
    elif variant.env in ('aloha_cube', 'aloha_insertion'):
        curr_image = obs["pixels"]["top"]
    else:
        raise NotImplementedError()
    if variant.resize_image > 0: 
        curr_image = np.array(PIL.Image.fromarray(curr_image).resize((variant.resize_image, variant.resize_image)))
    return curr_image

def obs_to_pi_zero_input(obs, variant):
    if variant.env == 'libero':
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        )
        
        obs_pi_zero = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": str(variant.task_description),
                    }
    elif variant.env == 'aloha_cube':
        img = np.ascontiguousarray(obs["pixels"]["top"])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        obs_pi_zero = {
            "state": obs["agent_pos"],
            "images": {"cam_high": np.transpose(img, (2,0,1))}
        }
    elif variant.env == 'aloha_insertion':
        obs_pi_zero = {
            "agent_pos": np.asarray(obs["agent_pos"], dtype=np.float32),
            "pixels": {"top": np.ascontiguousarray(obs["pixels"]["top"])},
        }
    else:
        raise NotImplementedError()
    return obs_pi_zero

def obs_to_qpos(obs, variant):
    if variant.env == 'libero':
        qpos = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        )
    elif variant.env in ('aloha_cube', 'aloha_insertion'):
        qpos = obs["agent_pos"]
    else:
        raise NotImplementedError()
    return qpos

def compute_action_chunk(
    variant, agent, agent_dp, rng, obs_pi_zero, obs_dict, *,
    use_residual, basis=None, residual_action_scale=1.0,
):
    rng, key = jax.random.split(rng)
    noise = jax.random.normal(key, (1, variant.pi0_action_horizon, variant.pi0_action_dim))
    a_base_full = np.asarray(agent_dp.infer(obs_pi_zero, noise=noise)["actions"], dtype=np.float32)
    expected_shape = (variant.pi0_action_horizon, variant.env_action_dim)
    if a_base_full.shape != expected_shape:
        raise ValueError(f"Base policy returned {a_base_full.shape}, expected {expected_shape}")
    # Execute / residual only over the first query_freq steps of the base chunk.
    query_freq = int(variant.query_freq)
    a_base = a_base_full[:query_freq]
    obs_dict = {**obs_dict, "action_diffusion": a_base.reshape(1, -1, 1)}

    if uses_projected_basis(variant):
        num_basis = variant.num_basis
        if use_residual and basis is not None:
            coeffs = agent.sample_actions(obs_dict)
            if coeffs.ndim == 1:
                coeffs = coeffs.reshape(1, -1)
            coeffs = residual_action_scale * coeffs
            res_chunk = basis.coeffs_to_chunk(coeffs[0], variant.residual_scale)
            actions = a_base + res_chunk
            stored_action = coeffs.astype(np.float32)
        else:
            actions = a_base
            stored_action = np.zeros((1, num_basis), dtype=np.float32)
        return rng, actions, stored_action, obs_dict

    if use_residual:
        a_res = agent.sample_actions(obs_dict).reshape(a_base.shape)
        scaled_residual = residual_action_scale * a_res
        actions = a_base + variant.residual_scale * scaled_residual
        residual_flat = scaled_residual.reshape(1, -1)
    else:
        actions = a_base
        residual_flat = np.zeros((1, np.prod(a_base.shape)), dtype=np.float32)
    return rng, actions, residual_flat, obs_dict


def log_basis_explained_variance(wandb_logger, basis, step=0):
    btype = getattr(basis, 'basis_type', 'pca')
    type_code = {'pca': 0, 'random': 1, 'vae': 2, 'vae_linear': 3}.get(btype, -1)
    metrics = {
        'basis/num_components': basis.num_basis,
        'basis/type': type_code,
    }
    if getattr(basis, 'explained_variance_ratio', None) is None:
        wandb_logger.log(metrics, step=step)
        print_green(f'Basis type={btype} K={basis.num_basis} D={basis.feature_dim}')
        return
    explained = np.asarray(basis.explained_variance_ratio, dtype=np.float64)
    metrics['basis/total_explained_variance'] = float(explained.sum())
    for idx, ratio in enumerate(explained):
        metrics[f'basis/pc_{idx}_explained_variance'] = float(ratio)
    wandb_logger.log(metrics, step=step)
    per_pc = ', '.join(f'PC{i}={explained[i]:.4f}' for i in range(len(explained)))
    print_green(f'Basis explained variance: total={explained.sum():.4f} ({per_pc})')


def fit_basis_from_warmup_chunks(variant, warmup_chunks):
    from examples.residual_basis import ResidualActionBasis

    action_dim = 7 if variant.env == 'libero' else 14
    basis = ResidualActionBasis.fit_top_k(
        np.stack(warmup_chunks, axis=0),
        variant.num_basis,
        query_freq=variant.query_freq,
        action_dim=action_dim,
    )
    save_path = os.path.join(variant.outputdir, "residual_basis.npz")
    basis.save(save_path)
    print_green(
        f"Fitted {basis.num_basis} PCs on {len(warmup_chunks)} chunks from "
        f"{variant.warmup_rollouts} warmup trajectories"
    )
    return basis


def fit_vae_from_warmup_chunks(variant, warmup_chunks):
    from examples.residual_vae import ResidualActionVAE

    action_dim = 7 if variant.env == 'libero' else 14
    basis = ResidualActionVAE.fit(
        np.stack(warmup_chunks, axis=0),
        latent_dim=int(variant.num_basis),
        query_freq=int(variant.query_freq),
        action_dim=action_dim,
        hidden=int(getattr(variant, 'vae_hidden', 128)),
        linear=bool(getattr(variant, 'vae_linear', False)),
        epochs=int(getattr(variant, 'vae_epochs', 120)),
        batch_size=int(getattr(variant, 'vae_batch_size', 256)),
        lr=float(getattr(variant, 'vae_lr', 1e-3)),
        beta=float(getattr(variant, 'vae_beta', 1e-3)),
        seed=int(variant.seed),
    )
    save_path = os.path.join(variant.outputdir, "residual_vae.pt")
    basis.save(save_path)
    print_green(
        f"Fitted VAE z={basis.latent_dim} ({basis.basis_type}) on {len(warmup_chunks)} chunks "
        f"from {variant.warmup_rollouts} warmup trajectories -> {save_path}"
    )
    return basis


def _needs_warmup_basis_fit(variant) -> bool:
    """Online PCA / VAE fit from warmup base chunks (no preloaded basis)."""
    if getattr(variant, 'basis', None) is not None:
        return False
    if getattr(variant, 'use_vae_basis', False):
        return not bool(getattr(variant, 'vae_path', '') or getattr(variant, 'basis_path', ''))
    if getattr(variant, 'use_eigenbasis', False):
        return not bool(getattr(variant, 'basis_path', ''))
    return False

def should_use_residual(variant, i):
    if getattr(variant, 'collect_with_residual', False) or (
        getattr(variant, 'init_residual', False) and i == 0
    ):
        return True
    return i > 0

def trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
                                       perform_control_evals=True, shard_fn=None, agent_dp=None):
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    total_env_steps = 0
    i = 0
    num_traj = 0
    warmup_rollouts = getattr(variant, 'warmup_rollouts', 20)
    basis = getattr(variant, 'basis', None)
    warmup_chunks = []
    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)
    wandb_logger.log({'env_steps': 0}, step=i)
    
    with tqdm(total=variant.max_steps, initial=0) as pbar:
        while i <= variant.max_steps:
            traj = collect_traj(variant, agent, env, i, agent_dp, basis=basis)
            num_traj += 1
            traj_id = online_replay_buffer._traj_counter
            add_online_data_to_buffer(variant, traj, online_replay_buffer)
            if basis is None and _needs_warmup_basis_fit(variant):
                warmup_chunks.extend(traj.get('pi0_chunks', []))
            total_env_steps += traj['env_steps']
            print('online buffer timesteps length:', len(online_replay_buffer))
            print('online buffer num traj:', traj_id + 1)
            print('total env steps:', total_env_steps)

            if basis is None and _needs_warmup_basis_fit(variant) and num_traj >= warmup_rollouts:
                if getattr(variant, 'use_vae_basis', False):
                    basis = fit_vae_from_warmup_chunks(variant, warmup_chunks)
                else:
                    basis = fit_basis_from_warmup_chunks(variant, warmup_chunks)
                variant.basis = basis
                log_basis_explained_variance(wandb_logger, basis, step=i)
                if (
                    getattr(variant, 'q_base_action', False)
                    and hasattr(agent, 'set_basis_matrix')
                    and getattr(basis, 'V', None) is not None
                ):
                    agent.set_basis_matrix(basis.V)
            
            if variant.get("num_online_gradsteps_batch", -1) > 0:
                num_gradsteps = variant.num_online_gradsteps_batch
            else:
                num_gradsteps = len(traj["rewards"])*variant.multi_grad_step

            if num_traj >= warmup_rollouts:
                print_green(f"Updating time at {i}")
                for _ in range(num_gradsteps):
                    if i == 0:
                        print('performing evaluation for initial checkpoint')
                        if perform_control_evals:
                            perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp, basis=basis)
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    # online perform update once we have some amount of online trajs
                    batch = next(replay_buffer_iterator)
                    update_info = agent.update(batch)

                    pbar.update()
                    i += 1
                        

                    if i % variant.log_interval == 0:
                        update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                        for k, v in update_info.items():
                            if v.ndim == 0:
                                wandb_logger.log({f'training/{k}': v}, step=i)
                            elif v.ndim <= 2:
                                wandb_logger.log_histogram(f'training/{k}', v, i)
                        # wandb_logger.log({'replay_buffer_size': len(online_replay_buffer)}, i)
                        wandb_logger.log({
                            'replay_buffer_size': len(online_replay_buffer),
                            'episode_return (exploration)': traj['episode_return'],
                            'is_success (exploration)': int(traj['is_success']),
                        }, i)

                    if i % variant.eval_interval == 0:
                        wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                        wandb_logger.log({'num_online_trajs': traj_id + 1}, step=i)
                        wandb_logger.log({'env_steps': total_env_steps}, step=i)
                        if perform_control_evals:
                            perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp, basis=basis)
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    if variant.checkpoint_interval != -1 and i % variant.checkpoint_interval == 0:
                        agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)

def add_online_data_to_buffer(variant, traj, online_replay_buffer):

    discount_horizon = variant.query_freq
    actions = np.array(traj['actions']) # (T, chunk_size, action_dim )
    episode_len = len(actions)
    rewards = np.array(traj['rewards'])
    masks = np.array(traj['masks'])

    for t in range(episode_len):
        obs = traj['observations'][t]
        next_obs = traj['observations'][t + 1]
        # remove batch dimension
        obs = {k: v[0] for k, v in obs.items()}
        next_obs = {k: v[0] for k, v in next_obs.items()}
        if not variant.add_states:
            obs.pop('state', None)
            next_obs.pop('state', None)
        
        insert_dict = dict(
            observations=obs,
            next_observations=next_obs,
            actions=actions[t],
            next_actions=actions[t + 1] if t < episode_len - 1 else actions[t],
            rewards=rewards[t],
            masks=masks[t],
            discount=variant.discount ** discount_horizon
        )
        online_replay_buffer.insert(insert_dict)
    online_replay_buffer.increment_traj_counter()

def collect_traj(variant, agent, env, i, agent_dp=None, basis=None):
    query_frequency = variant.query_freq
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward

    agent._rng, rng = jax.random.split(agent._rng)
    
    if 'libero' in variant.env:
        obs = env.reset()
    elif 'aloha' in variant.env:
        obs, _ = env.reset()
    
    image_list = [] # for visualization
    rewards = []
    action_list = []
    obs_list = []
    pi0_chunks = []

    for t in tqdm(range(max_timesteps)):
        curr_image = obs_to_img(obs, variant)
        
        qpos = obs_to_qpos(obs, variant)

        if variant.add_states:
            obs_dict = {
                'pixels': curr_image[np.newaxis, ..., np.newaxis],
                'state': qpos[np.newaxis, ..., np.newaxis],
            }
        else:
            obs_dict = {
                'pixels': curr_image[np.newaxis, ..., np.newaxis],
            }

        if t % query_frequency == 0:

            assert agent_dp is not None
            rng, key = jax.random.split(rng)
            obs_pi_zero = obs_to_pi_zero_input(obs, variant)
            use_residual = should_use_residual(variant, i) and (
                basis is not None or not uses_projected_basis(variant)
            )
            rng, actions, stored_action, obs_dict = compute_action_chunk(
                variant, agent, agent_dp, rng, obs_pi_zero, obs_dict,
                use_residual=use_residual,
                basis=basis,
                residual_action_scale=(
                    getattr(variant, 'init_scale', 1.0)
                    if getattr(variant, 'init_residual', False) and i == 0
                    else 1.0
                ),
            )
            action_list.append(stored_action)
            obs_list.append(obs_dict)
            if basis is None and _needs_warmup_basis_fit(variant):
                pi0_chunks.append(obs_dict['action_diffusion'].reshape(-1).astype(np.float64))
     
        action_t = actions[t % query_frequency]
        if 'libero' in variant.env:
            obs, reward, done, _ = env.step(action_t)
        elif 'aloha' in variant.env:
            obs, reward, terminated, truncated, _ = env.step(action_t)
            done = terminated or truncated
            
        rewards.append(reward)
        image_list.append(curr_image)
        if done:
            break

    # add last observation
    curr_image = obs_to_img(obs, variant)
    qpos = obs_to_qpos(obs, variant)
    obs_dict = {
        'pixels': curr_image[np.newaxis, ..., np.newaxis],
        'state': qpos[np.newaxis, ..., np.newaxis],
    }
    if variant.env in ('libero', 'aloha_cube', 'aloha_insertion') and agent_dp is not None:
        obs_pi_zero = obs_to_pi_zero_input(obs, variant)
        _, _, _, obs_dict = compute_action_chunk(
            variant, agent, agent_dp, rng, obs_pi_zero, obs_dict,
            use_residual=False, basis=basis,
        )
    obs_list.append(obs_dict)
    image_list.append(curr_image)
    
    # per episode
    rewards = np.array(rewards)
    episode_return = np.sum(rewards[rewards!=None])
    is_success = (reward == env_max_reward)
    print(f'Rollout Done: {episode_return=}, Success: {is_success}')
    
    
    '''
    We use sparse -1/0 reward to train the SAC agent.
    '''
    if is_success:
        query_steps = len(action_list)
        rewards = np.concatenate([-np.ones(query_steps - 1), [0]])
        masks = np.concatenate([np.ones(query_steps - 1), [0]])
    else:
        query_steps = len(action_list)
        rewards = -np.ones(query_steps)
        masks = np.ones(query_steps)

    return {
        'observations': obs_list,
        'actions': action_list,
        'rewards': rewards,
        'masks': masks,
        'is_success': is_success,
        'episode_return': episode_return,
        'images': image_list,
        'env_steps': t + 1,
        'pi0_chunks': pi0_chunks,
    }

def perform_control_eval(agent, env, i, variant, wandb_logger, agent_dp=None, basis=None):
    query_frequency = variant.query_freq
    print('query frequency', query_frequency)
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward
    episode_returns = []
    highest_rewards = []
    success_rates = []
    episode_lens = []

    rng = jax.random.PRNGKey(variant.seed+456)

    for rollout_id in range(variant.eval_episodes):
        if 'libero' in variant.env:
            obs = env.reset()
        elif 'aloha' in variant.env:
            obs, _ = env.reset()
            
        image_list = [] # for visualization
        rewards = []
        

        for t in tqdm(range(max_timesteps)):
            curr_image = obs_to_img(obs, variant)

            if t % query_frequency == 0:
                qpos = obs_to_qpos(obs, variant)
                if variant.add_states:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                        'state': qpos[np.newaxis, ..., np.newaxis],
                    }
                else:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                    }

                assert agent_dp is not None
                
                obs_pi_zero = obs_to_pi_zero_input(obs, variant)
                use_residual = i > 0 and (
                    basis is not None or not uses_projected_basis(variant)
                )
                rng, actions, _, obs_dict = compute_action_chunk(
                    variant, agent, agent_dp, rng, obs_pi_zero, obs_dict,
                    use_residual=use_residual, basis=basis,
                )
              
            action_t = actions[t % query_frequency]
            
            if 'libero' in variant.env:
                obs, reward, done, _ = env.step(action_t)
            elif 'aloha' in variant.env:
                obs, reward, terminated, truncated, _ = env.step(action_t)
                done = terminated or truncated
                
            rewards.append(reward)
            image_list.append(curr_image)
            if done:
                break

        # per episode
        episode_lens.append(t + 1)
        rewards = np.array(rewards)
        episode_return = np.sum(rewards)
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        is_success = (reward == env_max_reward)
        success_rates.append(is_success)
                
        print(f'Rollout {rollout_id} : {episode_return=}, Success: {is_success}')
        video = np.stack(image_list).transpose(0, 3, 1, 2)
        #wandb_logger.log({f'eval_video/{rollout_id}': wandb.Video(video, fps=50)}, step=i)


    success_rate = np.mean(np.array(success_rates))
    avg_return = np.mean(episode_returns)
    avg_episode_len = np.mean(episode_lens)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    wandb_logger.log({'evaluation/avg_return': avg_return}, step=i)
    wandb_logger.log({'evaluation/success_rate': success_rate}, step=i)
    wandb_logger.log({'evaluation/avg_episode_len': avg_episode_len}, step=i)
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / variant.eval_episodes
        wandb_logger.log({f'evaluation/Reward >= {r}': more_or_equal_r_rate}, step=i)
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{variant.eval_episodes} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

def make_multiple_value_reward_visulizations(agent, variant, i, replay_buffer, wandb_logger):
    trajs = replay_buffer.get_random_trajs(3)
    images = agent.make_value_reward_visulization(variant, trajs, shaped_rewards=None)
    wandb_logger.log({'reward_value_images': wandb.Image(images)}, step=i)
