#!/bin/bash
set -euo pipefail

proj_name=DSRL_ACT_AlohaInsertion_basis
device_id="${DEVICE_ID:-0}"

export DISPLAY="${DISPLAY:-:0}"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="$device_id"
export EXP="${EXP:-./logs/$proj_name}"
export CUDA_VISIBLE_DEVICES="$device_id"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/LIBERO:${REPO_ROOT}/openpi/src:${PYTHONPATH:-}"

# ACT predicts 100x14 chunks. SAC learns K PCA coefficients for an additive
# residual: a = a_ACT + residual_scale * V @ c.
python3 examples/launch_train_sim.py \
  --algorithm pixel_sac \
  --env aloha_insertion \
  --act_checkpoint lerobot/act_aloha_sim_insertion_human \
  --prefix dsrl_act_aloha_insertion_basis \
  --wandb_project DSRL_ACT_AlohaInsertion \
  --batch_size 256 \
  --discount 0.999 \
  --seed 0 \
  --max_steps 3000000 \
  --eval_interval 10000 \
  --log_interval 500 \
  --eval_episodes 10 \
  --multi_grad_step 20 \
  --resize_image 64 \
  --action_magnitude 1.0 \
  --residual_scale 0.01 \
  --query_freq 100 \
  --hidden_dims 128 \
  --output_dir "${EXP}" \
  --use_eigenbasis \
  --num_basis 8 \
  --warmup_rollouts 20 \
  "$@"
