#!/bin/bash
proj_name=DSRL_pi0_Aloha
device_id=1

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

# Full residual RL: SAC outputs a (50 x 14) = 700-D residual on top of pi0_aloha_sim.
python3 examples/launch_train_sim.py \
  --algorithm pixel_sac \
  --env aloha_cube \
  --prefix dsrl_pi0_aloha \
  --wandb_project ${proj_name} \
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
  --output_dir ${EXP} \
  --init_residual 

# Eigenbasis residual (preferred for Aloha; see run_aloha_basis.sh):
# bash examples/scripts/run_aloha_basis.sh
