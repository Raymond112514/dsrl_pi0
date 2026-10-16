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
    elif variant.env == 'aloha_insertion':
        # ACT was trained on the raw gym observation, not the pi0 camera packing.
        return obs
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
    elif variant.env in ('aloha_cube', 'aloha_insertion'):
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


def _sample_pi0_chunks_at_state(variant, agent_dp, rng, obs_pi_zero, first_chunk, n_samples):
    """Return n_samples flattened π0 chunks at one observation.

    `first_chunk` is the chunk already inferred for execution. The rest are
    new diffusion samples at the same observation.
    """
    n_samples = int(n_samples)
    if n_samples < 2:
        raise ValueError(f"Need >=2 chunks per state, got {n_samples}")
    samples = [np.asarray(first_chunk, dtype=np.float64).reshape(-1)]
    for _ in range(n_samples - 1):
        rng, key = jax.random.split(rng)
        noise = jax.random.normal(key, (1, variant.pi0_action_horizon, variant.pi0_action_dim))
        raw = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]
        samples.append(_executed_base_chunk(raw, variant).reshape(-1).astype(np.float64))
    return rng, samples


def _ordered_pairwise_diffs(samples):
    """All δ = a_i - a_j for i != j. Shape (K*(K-1), D)."""
    arr = np.stack(samples, axis=0)
    k = arr.shape[0]
    ii, jj = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    mask = ii != jj
    return arr[ii[mask]] - arr[jj[mask]]


def compute_action_chunk(variant, agent, agent_dp, rng, obs_pi_zero, obs_dict, *, use_residual, basis=None):
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

    def _full_action(action_chunk):
        return np.asarray(action_chunk, dtype=np.float32).reshape(1, -1)

    if (
        getattr(variant, 'use_eigenbasis', False)
        or getattr(variant, 'use_vae', False)
        or getattr(variant, 'use_flow', False)
    ):
        if use_residual and basis is not None:
            latents = agent.sample_actions(obs_dict)
            if latents.ndim == 1:
                latents = latents.reshape(1, -1)
            if getattr(variant, 'use_vae', False):
                state = None
                if getattr(basis, 'condition_on_state', False):
                    if 'state' not in obs_dict:
                        raise ValueError(
                            "State-conditioned VAE decode requires obs['state'] "
                            "(--add_states 1)"
                        )
                    state = np.asarray(obs_dict['state'], dtype=np.float32).reshape(-1)
                res_chunk = basis.latent_to_chunk(latents[0], variant.residual_scale, state=state)
                actions = a_exec + res_chunk
            elif getattr(variant, 'use_flow', False):
                res_chunk = basis.latent_to_chunk(latents[0], variant.residual_scale)
                actions = a_exec + res_chunk
            else:
                coeffs = np.asarray(latents, dtype=np.float32).reshape(-1)
                mask = np.asarray(getattr(agent, "latent_mask", 1.0), dtype=np.float32).reshape(-1)
                if mask.shape == coeffs.shape:
                    coeffs = coeffs * mask
                res_chunk = basis.coeffs_to_chunk(coeffs, variant.residual_scale)
                actions = a_exec + res_chunk
        else:
            actions = a_exec
        return rng, actions, _full_action(actions), obs_dict

    if use_residual:
        a_res = agent.sample_actions(obs_dict).reshape(a_exec.shape)
        actions = a_exec + variant.residual_scale * a_res
    else:
        actions = a_exec
    return rng, actions, _full_action(actions), obs_dict


def pca_k_at_step(variant, step):
    """Live PCA width at training step `step`. None when the curriculum is off.

    The actor is built at `num_basis`. k starts at pca_k_start and grows by
    pca_k_inc every pca_k_every training steps, the same counter as max_steps.
    """
    if not getattr(variant, "pca_k_curriculum", False):
        return None
    k = int(variant.pca_k_start) + int(variant.pca_k_inc) * (
        int(step) // int(variant.pca_k_every)
    )
    return min(int(variant.num_basis), k)


def sync_pca_curriculum(agent, variant, step, wandb_logger=None):
    k = pca_k_at_step(variant, step)
    if k is None or not hasattr(agent, "set_pca_latent_k"):
        return
    if agent.set_pca_latent_k(k):
        print_green(
            f"pca latent mask k={k}/{int(variant.num_basis)} at step={int(step)}"
        )
        if wandb_logger is not None:
            wandb_logger.log({"basis/active_k": int(k)}, step=int(step))


def uses_latent_residual(variant):
    return (
        getattr(variant, 'use_eigenbasis', False)
        or getattr(variant, 'use_vae', False)
        or getattr(variant, 'use_flow', False)
    )


def log_basis_explained_variance(wandb_logger, basis, step=0):
    if basis.explained_variance_ratio is None:
        return
    explained = np.asarray(basis.explained_variance_ratio, dtype=np.float64)
    metrics = {
        'basis/num_components': basis.num_basis,
        'basis/total_explained_variance': float(explained.sum()),
    }
    for idx, ratio in enumerate(explained):
        metrics[f'basis/pc_{idx}_explained_variance'] = float(ratio)
    wandb_logger.log(metrics, step=step)
    per_pc = ', '.join(f'PC{i}={explained[i]:.4f}' for i in range(len(explained)))
    print_green(f'Basis explained variance: total={explained.sum():.4f} ({per_pc})')


def log_vae_fit_metrics(wandb_logger, residual_vae, step=0):
    metrics = residual_vae.fit_metrics or {}
    log = {f'vae/{k}': float(v) for k, v in metrics.items()}
    log['vae/latent_dim'] = float(residual_vae.latent_dim)
    log['vae/feature_dim'] = float(residual_vae.feature_dim)
    log['vae/fit_residual'] = float(getattr(residual_vae, 'fit_residual', True))
    log['vae/condition_on_state'] = float(getattr(residual_vae, 'condition_on_state', False))
    log['vae/state_dim'] = float(getattr(residual_vae, 'cond_dim', 0))
    if log:
        wandb_logger.log(log, step=step)
    print_green(
        f"VAE fit: latent_dim={residual_vae.latent_dim}  "
        f"val_recon={metrics.get('val_recon_loss', float('nan')):.5f}  "
        f"val_kl={metrics.get('val_kl', float('nan')):.4f}  "
        f"val_ev={metrics.get('val_explained_var', float('nan')):.4f}"
    )


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


def log_flow_fit_metrics(wandb_logger, residual_flow, step=0):
    metrics = residual_flow.fit_metrics or {}
    log = {f'flow/{k}': float(v) for k, v in metrics.items()}
    log['flow/latent_dim'] = float(residual_flow.latent_dim)
    log['flow/n_steps'] = float(residual_flow.n_steps)
    log['flow/fit_residual'] = float(getattr(residual_flow, 'fit_residual', True))
    if log:
        wandb_logger.log(log, step=step)
    print_green(
        f"Flow fit: latent_dim={residual_flow.latent_dim}  "
        f"ode_steps={residual_flow.n_steps}  "
        f"val_loss={metrics.get('val_loss', float('nan')):.5f}  "
        f"val_recon={metrics.get('val_recon_loss', float('nan')):.5f}  "
        f"val_ev={metrics.get('val_explained_var', float('nan')):.4f}"
    )


def fit_flow_from_warmup_chunks(variant, warmup_chunks):
    from examples.residual_flow import fit_residual_flow

    action_dim = 7 if variant.env == 'libero' else 14
    residual_flow = fit_residual_flow(
        np.stack(warmup_chunks, axis=0),
        query_freq=variant.query_freq,
        action_dim=action_dim,
        epochs=getattr(variant, 'flow_epochs', 100),
        n_steps=getattr(variant, 'flow_steps', 10),
        seed=getattr(variant, 'seed', 0),
        fit_residual=getattr(variant, 'fit_residual', True),
    )
    residual_flow.decode_add_mean = bool(getattr(variant, "decode_add_mean", True))
    save_path = os.path.join(variant.outputdir, "residual_flow.pkl")
    residual_flow.save(save_path)
    kind = "residual" if residual_flow.fit_residual else "action"
    print_green(
        f"Fitted {kind} flow (D={residual_flow.latent_dim}, "
        f"ode_steps={residual_flow.n_steps}, epochs={getattr(variant, 'flow_epochs', 100)}, "
        f"decode_add_mean={residual_flow.decode_add_mean}) "
        f"on {len(warmup_chunks)} chunks from {variant.warmup_rollouts} warmup trajectories"
    )
    return residual_flow


def fit_vae_from_warmup_chunks(variant, warmup_chunks, warmup_states=None):
    from examples.residual_vae import fit_residual_vae

    action_dim = 7 if variant.env == 'libero' else 14
    condition = bool(getattr(variant, 'vae_condition_on_state', False))
    states = None
    if condition:
        if not warmup_states or len(warmup_states) != len(warmup_chunks):
            raise ValueError(
                f"State-conditioned VAE expected {len(warmup_chunks)} states, "
                f"got {0 if not warmup_states else len(warmup_states)}"
            )
        states = np.stack(warmup_states, axis=0)
    residual_vae = fit_residual_vae(
        np.stack(warmup_chunks, axis=0),
        latent_dim=variant.num_basis,
        query_freq=variant.query_freq,
        action_dim=action_dim,
        beta=getattr(variant, 'vae_beta', 1e-3),
        epochs=getattr(variant, 'vae_epochs', 100),
        seed=getattr(variant, 'seed', 0),
        fit_residual=getattr(variant, 'fit_residual', True),
        states=states,
        condition_on_state=condition,
    )
    residual_vae.decode_add_mean = bool(getattr(variant, "decode_add_mean", True))
    save_path = os.path.join(variant.outputdir, "residual_vae.pkl")
    residual_vae.save(save_path)
    kind = "residual" if residual_vae.fit_residual else "action"
    cond = ""
    if residual_vae.condition_on_state:
        n_per = int(getattr(variant, 'vae_samples_per_state', 16)) if residual_vae.fit_residual else 1
        cond = f", state_dim={residual_vae.cond_dim}, samples_per_state={n_per}"
    print_green(
        f"Fitted {kind} VAE (K={residual_vae.latent_dim}, beta={getattr(variant, 'vae_beta', 1e-3)}, "
        f"epochs={getattr(variant, 'vae_epochs', 100)}, decode_add_mean={residual_vae.decode_add_mean}"
        f"{cond}) "
        f"on {len(warmup_chunks)} targets from "
        f"{variant.warmup_rollouts} warmup trajectories"
    )
    return residual_vae

def should_use_residual(variant, i):
    if getattr(variant, 'collect_with_residual', False):
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
    warmup_states = []
    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)
    wandb_logger.log({'env_steps': 0}, step=i)
    
    with tqdm(total=variant.max_steps, initial=0) as pbar:
        while i <= variant.max_steps:
            sync_pca_curriculum(agent, variant, i, wandb_logger)
            traj = collect_traj(variant, agent, env, i, agent_dp, basis=basis)
            num_traj += 1
            traj_id = online_replay_buffer._traj_counter
            add_online_data_to_buffer(variant, traj, online_replay_buffer)
            if basis is None and uses_latent_residual(variant):
                warmup_chunks.extend(traj.get('pi0_chunks', []))
                warmup_states.extend(traj.get('pi0_states', []))
            total_env_steps += traj['env_steps']
            print('online buffer timesteps length:', len(online_replay_buffer))
            print('online buffer num traj:', traj_id + 1)
            print('total env steps:', total_env_steps)

            if (
                basis is None
                and uses_latent_residual(variant)
                and num_traj >= warmup_rollouts
            ):
                if getattr(variant, 'use_vae', False):
                    basis = fit_vae_from_warmup_chunks(variant, warmup_chunks, warmup_states)
                    variant.basis = basis
                    agent.set_residual_vae(basis)
                    log_vae_fit_metrics(wandb_logger, basis, step=i)
                elif getattr(variant, 'use_flow', False):
                    basis = fit_flow_from_warmup_chunks(variant, warmup_chunks)
                    variant.basis = basis
                    agent.set_residual_flow(basis)
                    log_flow_fit_metrics(wandb_logger, basis, step=i)
                else:
                    basis = fit_basis_from_warmup_chunks(variant, warmup_chunks)
                    variant.basis = basis
                    agent.set_residual_basis(basis.V)
                    log_basis_explained_variance(wandb_logger, basis, step=i)
            
            if variant.get("num_online_gradsteps_batch", -1) > 0:
                num_gradsteps = variant.num_online_gradsteps_batch
            else:
                num_gradsteps = len(traj["rewards"])*variant.multi_grad_step

            if num_traj >= warmup_rollouts:
                print_green(f"Updating time at {i}")
                for _ in range(num_gradsteps):
                    sync_pca_curriculum(agent, variant, i, wandb_logger)
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
    pi0_states = []

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
                basis is not None or not uses_latent_residual(variant)
            )
            rng, actions, stored_action, obs_dict = compute_action_chunk(
                variant, agent, agent_dp, rng, obs_pi_zero, obs_dict,
                use_residual=use_residual, basis=basis,
            )
            action_list.append(stored_action)
            obs_list.append(obs_dict)
            if basis is None and uses_latent_residual(variant):
                chunk = obs_dict['action_diffusion'].reshape(-1).astype(np.float64)
                if (
                    getattr(variant, 'use_vae', False)
                    and getattr(variant, 'vae_condition_on_state', False)
                ):
                    # Pair each target with this query's proprio. Residual targets
                    # are ordered pairwise diffs of K π0 samples at this state.
                    qpos_vec = np.asarray(qpos, dtype=np.float32).reshape(-1)
                    if getattr(variant, 'fit_residual', True):
                        n_samples = int(getattr(variant, 'vae_samples_per_state', 16))
                        rng, samples = _sample_pi0_chunks_at_state(
                            variant, agent_dp, rng, obs_pi_zero, chunk, n_samples,
                        )
                        for diff in _ordered_pairwise_diffs(samples):
                            pi0_chunks.append(diff)
                            pi0_states.append(qpos_vec)
                    else:
                        pi0_chunks.append(chunk)
                        pi0_states.append(qpos_vec)
                else:
                    pi0_chunks.append(chunk)
     
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
    if variant.env == 'libero' and agent_dp is not None:
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
        'pi0_states': pi0_states,
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
                    basis is not None or not uses_latent_residual(variant)
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
