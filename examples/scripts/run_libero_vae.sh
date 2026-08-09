#!/bin/bash
# VAE-latent residual RL on Libero (pi0.5).
# a = a_base + λ * decode(w), w = π_residual(s, a_base)
proj_name=DSRL_pi0_Libero_vae
device_id="${DEVICE_ID:-3}"

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=./openpi
export EXP=./logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/LIBERO:${REPO_ROOT}/openpi/src:${PYTHONPATH:-}"

pip install mujoco==3.2.3

python3 examples/launch_train_sim_vae.py \
  --algorithm pixel_sac \
  --env libero \
  --prefix dsrl_pi0_libero_vae \
  --wandb_project DSRL_pi0_Libero \
  --batch_size 256 \
  --discount 0.999 \
  --seed 10 \
  --max_steps 500000 \
  --eval_interval 10000 \
  --log_interval 500 \
  --eval_episodes 10 \
  --multi_grad_step 20 \
  --start_online_updates 500 \
  --resize_image 64 \
  --action_magnitude 1.0 \
  --residual_scale 0.01 \
  --query_freq 10 \
  --hidden_dims 128 \
  --output_dir "${EXP}" \
  --task_id 44 \
  --vae_latent_dim 8 \
  --warmup_rollouts 20 \
  --vae_epochs 120
