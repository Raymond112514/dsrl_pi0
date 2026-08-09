#!/bin/bash
# Flow-matching residual RL on Aloha cube (pi0).
# a = a_base + λ * flow(w), w = π_residual(s, a_base) ∈ R^{query_freq * 14}
# No --num_basis / latent dim.
proj_name=DSRL_pi0_Aloha_flow
device_id="${DEVICE_ID:-0}"

export DISPLAY=:0
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=./openpi
export EXP=./logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/LIBERO:${REPO_ROOT}/openpi/src:${PYTHONPATH:-}"

pip install mujoco==2.3.7

python3 examples/launch_train_sim_flow.py \
  --algorithm pixel_sac \
  --env aloha_cube \
  --prefix dsrl_pi0_aloha_flow \
  --wandb_project DSRL_pi0_Aloha \
  --batch_size 256 \
  --discount 0.999 \
  --seed 0 \
  --max_steps 3000000 \
  --eval_interval 10000 \
  --log_interval 500 \
  --eval_episodes 10 \
  --multi_grad_step 20 \
  --start_online_updates 1000 \
  --resize_image 64 \
  --action_magnitude 1.0 \
  --residual_scale 0.01 \
  --query_freq 50 \
  --hidden_dims 128 \
  --output_dir "${EXP}" \
  --warmup_rollouts 20 \
  --flow_epochs 40 \
  --flow_n_steps 10
