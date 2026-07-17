#!/bin/bash
proj_name=DSRL_pi0_Aloha_basis
device_id=0

export DISPLAY=:0
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=./openpi
export EXP=./logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/openpi/src:${PYTHONPATH:-}"

pip install mujoco==2.3.7

# Eigenbasis residual: SAC learns K PCA coeffs; a = a_base + lambda * V @ c.
# Default: fit basis online from warmup rollouts (K=8, ~96% variance on Aloha PCA).
python3 examples/launch_train_sim.py \
  --algorithm pixel_sac \
  --env aloha_cube \
  --prefix dsrl_pi0_aloha_basis \
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
  --use_eigenbasis \
  --num_basis 8 \
  --warmup_rollouts 20

# Optional: reuse precomputed PCA (K=15 @ 99% variance; sliced to --num_basis):
#   --basis_path success_rate/eigenvectors/pi0_aloha_sim_pca99/action_basis.npz \
#   --num_basis 8
