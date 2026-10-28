from tqdm import tqdm
import os
import numpy as np
import wandb
import jax
from openpi_client import image_tools
import math
import PIL

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
    elif variant.env == 'aloha_cube':
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
    elif variant.env == 'aloha_cube':
        qpos = obs["agent_pos"]
    else:
        raise NotImplementedError()
    return qpos

def _executed_base_chunk(a_base, variant):
    """Use the first query_freq steps as the effective base-policy chunk."""
    chunk = np.asarray(a_base)
    if chunk.ndim == 3 and chunk.shape[0] == 1:
        chunk = chunk[0]
    env_action_dim = getattr(variant, 'env_action_dim', None)
    if env_action_dim is not None:
        chunk = chunk[..., :env_action_dim]
    query_freq = variant.query_freq
    if query_freq > chunk.shape[0]:
        raise ValueError(
            f"query_freq={query_freq} exceeds inferred action chunk length {chunk.shape[0]}"
        )
    return chunk[:query_freq]


def _add_warmup_noise(rng, actions, residual_scale):
    """Add uniform warmup noise in the range [-residual_scale, residual_scale] to the actions."""
    rng, key = jax.random.split(rng)
    scale = float(residual_scale)
    eps = np.asarray(
        jax.random.uniform(key, shape=np.shape(actions), minval=-scale, maxval=scale),
        dtype=np.float32,
    )
    return rng, np.asarray(actions, dtype=np.float32) + eps


def compute_action_chunk(variant, agent, agent_dp, rng, obs_pi_zero, obs_dict, *, mode, latent_model=None):
    """Build full action chunk. 
    * mode "base": return the base policy actions. 
    * mode "warmup": return the the base action plus uniform noise, used for warmup episodes. 
    * mode "policy": return the base action plus the SAC residual.
    """
    rng, key = jax.random.split(rng)
    noise = jax.random.normal(key, (1, variant.pi0_action_horizon, variant.pi0_action_dim))
    a_base = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]
    if getattr(variant, 'env_action_dim', None) is not None:
        a_exec = _executed_base_chunk(a_base, variant)
    else:
        a_exec = np.asarray(a_base)
        if a_exec.ndim == 3 and a_exec.shape[0] == 1:
            a_exec = a_exec[0]
    obs_dict = {**obs_dict, "action_diffusion": a_exec.reshape(1, -1, 1)}
    actions = np.asarray(a_exec, dtype=np.float32)

    if mode == "base":
        pass
    elif mode == "warmup":
        rng, actions = _add_warmup_noise(rng, actions, variant.residual_scale)
    elif mode == "policy":
        sample = agent.sample_actions(obs_dict)
        if latent_model is None:
            actions = actions + variant.residual_scale * np.asarray(sample).reshape(actions.shape)
        else:
            if sample.ndim == 1:
                sample = sample.reshape(1, -1)
            actions = actions + (variant.residual_scale * latent_model.decode(sample[0])).astype(np.float32)
    else:
        raise ValueError(f"Unknown action mode {mode}")

    stored = np.asarray(actions, dtype=np.float32).reshape(1, -1)
    return rng, actions, stored, obs_dict


def log_latent_fit_metrics(latent_model):
    """Return fit metrics. The caller logs them on the environment-step clock."""
    if latent_model is None or not hasattr(latent_model, "fit_metrics"):
        return {}
    metrics = latent_model.fit_metrics()
    if not metrics:
        return {}
    prefix = getattr(latent_model, "name", "latent") or "latent"
    print_green(f"{prefix} fit: " + "  ".join(f"{k}={v}" for k, v in list(metrics.items())[:6]))
    return {f"{prefix}/{k}": float(v) for k, v in metrics.items()}


class _EnvStepLog:
    """Holds a dictionary for all statistics for the current environment step.
    Calls at the same step are merged into that one commit.
    """

    def __init__(self, wandb_logger):
        self.wandb_logger = wandb_logger
        self.committed = -1
        self.step = None
        self.pending = {}

    def add(self, step, data):
        if not data or step <= self.committed:
            return
        if self.step is not None and step < self.step:
            return
        if self.step is not None and step > self.step:
            self.flush()
        self.step = step
        self.pending.update(data)

    def flush(self):
        if not self.pending or self.step is None or self.step <= self.committed:
            self.pending = {}
            return
        self.wandb_logger.log(self.pending, step=self.step)
        self.committed = self.step
        self.pending = {}


def trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
                                       perform_control_evals=True, shard_fn=None, agent_dp=None):
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    i = 0
    num_traj = 0
    warmup_rollouts = getattr(variant, 'warmup_rollouts', 20)
    latent_model = getattr(variant, 'latent_model', None)
    warmup_chunks = []
    logger = _EnvStepLog(wandb_logger)
    clock = {'env_steps': 0, 'update': None}

    def training_payload(update_info):
        if update_info is None:
            return {}
        payload = {}
        info = {k: jax.device_get(v) for k, v in update_info.items()}
        for k, v in info.items():
            if v.ndim == 0:
                payload[f'training/{k}'] = float(v)
            elif v.ndim <= 2:
                payload[f'training/{k}'] = wandb.Histogram(np.asarray(v).flatten())
        return payload

    def log_at(step):
        # Everything due at this environment step goes into one wandb commit.
        payload = {}
        if variant.log_interval > 0 and step % variant.log_interval == 0:
            payload.update(training_payload(clock['update']))
            payload['gradient_steps'] = i
            payload['replay_buffer_size'] = len(online_replay_buffer)
            payload['num_online_samples'] = len(online_replay_buffer)
            payload['num_online_trajs'] = num_traj
        if perform_control_evals and variant.eval_interval > 0 and step % variant.eval_interval == 0:
            payload.update(perform_control_eval(
                agent, eval_env, i, variant, agent_dp, latent_model=latent_model,
            ))
            if replay_buffer._traj_counter > 0 and hasattr(agent, 'perform_eval'):
                images = make_multiple_value_reward_visulizations(agent, variant, replay_buffer)
                payload['reward_value_images'] = wandb.Image(images)
        if payload:
            logger.add(step, payload)

    if latent_model is not None and latent_model.fitted:
        logger.add(0, log_latent_fit_metrics(latent_model))
    log_at(0)
    logger.flush()

    with tqdm(total=variant.max_steps, initial=0) as pbar:
        def on_env_step():
            # We check whether we need to log or perform eval at each environment step.
            clock['env_steps'] += 1
            pbar.update(1)
            log_at(clock['env_steps'])
            return clock['env_steps'] < variant.max_steps

        while clock['env_steps'] < variant.max_steps:
            traj = collect_traj(
                variant, agent, env, i, agent_dp,
                latent_model=latent_model, on_env_step=on_env_step,
            )
            num_traj += 1
            add_online_data_to_buffer(variant, traj, online_replay_buffer)
            if latent_model is not None and not latent_model.fitted:
                warmup_chunks.extend(traj.get('pi0_chunks', []))
            logger.add(clock['env_steps'], {
                'episode_return (exploration)': traj['episode_return'],
                'is_success (exploration)': int(traj['is_success']),
            })
            print('online buffer timesteps length:', len(online_replay_buffer))
            print('online buffer num traj:', num_traj)
            print('total env steps:', clock['env_steps'])

            if (
                latent_model is not None
                and not latent_model.fitted
                and num_traj >= warmup_rollouts
            ):
                latent_model.fit(np.stack(warmup_chunks, axis=0))
                latent_model.save(variant.outputdir)
                variant.latent_model = latent_model
                agent.set_latent_model(latent_model)
                logger.add(clock['env_steps'], log_latent_fit_metrics(latent_model))
                print_green(
                    f"Fitted latent model on {len(warmup_chunks)} chunks from "
                    f"{warmup_rollouts} warmup trajectories"
                )

            if variant.get("num_online_gradsteps_batch", -1) > 0:
                num_gradsteps = variant.num_online_gradsteps_batch
            else:
                num_gradsteps = len(traj["rewards"]) * variant.multi_grad_step

            if num_traj >= warmup_rollouts:
                print_green(f"Updating time at {i}")
                for _ in range(num_gradsteps):
                    batch = next(replay_buffer_iterator)
                    clock['update'] = agent.update(batch)
                    i += 1
                    if variant.checkpoint_interval != -1 and i % variant.checkpoint_interval == 0:
                        agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)
                # Same env step as this episode, still uncommitted, so this merges
                # rather than logging a second time at a step wandb already has.
                fresh = training_payload(clock['update'])
                fresh['gradient_steps'] = i
                logger.add(clock['env_steps'], fresh)
        logger.flush()

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

def collect_traj(variant, agent, env, i, agent_dp=None, latent_model=None, on_env_step=None):
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
            if i == 0:
                mode = "warmup" if variant.collect_warmup_with_noise else "base"
            else:
                mode = "policy"
            rng, actions, stored_action, obs_dict = compute_action_chunk(
                variant, agent, agent_dp, rng, obs_pi_zero, obs_dict,
                mode=mode, latent_model=latent_model,
            )
            action_list.append(stored_action)
            obs_list.append(obs_dict)
            if latent_model is not None and i == 0:
                chunk = obs_dict['action_diffusion'].reshape(-1).astype(np.float64)
                pi0_chunks.append(chunk)
     
        action_t = actions[t % query_frequency]
        if 'libero' in variant.env:
            obs, reward, done, _ = env.step(action_t)
        elif 'aloha' in variant.env:
            obs, reward, terminated, truncated, _ = env.step(action_t)
            done = terminated or truncated
            
        rewards.append(reward)
        image_list.append(curr_image)
        # `actions` still holds a[T+1:]. Eval runs here, then the loop continues that chunk.
        # A False return means the env-step budget is spent, so the rest of the chunk is dropped.
        if on_env_step is not None and on_env_step() is False:
            break
        if done:
            break

    # add last observation
    curr_image = obs_to_img(obs, variant)
    qpos = obs_to_qpos(obs, variant)
    obs_dict = {
        'pixels': curr_image[np.newaxis, ..., np.newaxis],
        'state': qpos[np.newaxis, ..., np.newaxis],
    }
    if agent_dp is not None:
        obs_pi_zero = obs_to_pi_zero_input(obs, variant)
        _, _, _, obs_dict = compute_action_chunk(
            variant, agent, agent_dp, rng, obs_pi_zero, obs_dict,
            mode="base", latent_model=latent_model,
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

def perform_control_eval(agent, env, i, variant, agent_dp=None, latent_model=None):
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
                mode = "base" if i == 0 else "policy"
                rng, actions, _, obs_dict = compute_action_chunk(
                    variant, agent, agent_dp, rng, obs_pi_zero, obs_dict,
                    mode=mode, latent_model=latent_model,
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
    log = {
        'evaluation/avg_return': avg_return,
        'evaluation/success_rate': success_rate,
        'evaluation/avg_episode_len': avg_episode_len,
    }
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / variant.eval_episodes
        log[f'evaluation/Reward >= {r}'] = more_or_equal_r_rate
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{variant.eval_episodes} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)
    return log

def make_multiple_value_reward_visulizations(agent, variant, replay_buffer):
    trajs = replay_buffer.get_random_trajs(3)
    return agent.make_value_reward_visulization(variant, trajs)
