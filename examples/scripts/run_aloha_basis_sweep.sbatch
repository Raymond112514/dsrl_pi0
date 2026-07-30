#!/bin/bash

#SBATCH --job-name=dsrl_aloha_basis_sweep
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Residual RL (eigenbasis) sweep on Aloha TransferCube (pi0_aloha_sim):
#   residual_scale ∈ {0.3, 0.2, 0.1, 0.05}
#   UTD (multi_grad_step) ∈ {20, 10}
# 4 × 2 = 8 runs, one GPU job per array task.
#
# Submit from repo root:
#   mkdir -p logs
#   sbatch examples/scripts/run_aloha_basis_sweep.sbatch

# --------------------------
# ENV SETUP
# --------------------------

source ~/.bashrc
conda activate dsrl_pi0

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/global/home/users/r112358/.mujoco/mujoco210/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib/nvidia"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd ~/dsrl_pi0
REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/LIBERO:${REPO_ROOT}/openpi/src:${PYTHONPATH:-}"

mkdir -p logs

# Local OpenPI checkpoint (avoids gs:// download on compute nodes)
PI0_CKPT=/global/scratch/users/r112358/pi0_checkpoint/pi0_aloha_sim

# --------------------------
# SWEEP GRID
# --------------------------

RESIDUAL_SCALES=(0.3 0.2 0.1 0.05)
UTD_RATIOS=(20 10)

NUM_SCALES=${#RESIDUAL_SCALES[@]}   # 4
NUM_UTD=${#UTD_RATIOS[@]}           # 2
TOTAL_RUNS=$((NUM_SCALES * NUM_UTD)) # 8

index_to_pair () {
  local g=$1
  local scale_idx=$((g / NUM_UTD))
  local utd_idx=$((g % NUM_UTD))
  echo "${scale_idx} ${utd_idx}"
}

g=${SLURM_ARRAY_TASK_ID}

if (( g >= TOTAL_RUNS )); then
  echo "g=${g} >= TOTAL_RUNS=${TOTAL_RUNS} → no configuration to run"
  exit 0
fi

read scale_idx utd_idx < <(index_to_pair "${g}")

RS=${RESIDUAL_SCALES[$scale_idx]}
UTD=${UTD_RATIOS[$utd_idx]}

seed=0
run_name="aloha_cube_K8_pca_rs${RS}_utd${UTD}_seed${seed}"

echo "Running g=${g} / ${TOTAL_RUNS}"
echo "  residual_scale=${RS}"
echo "  utd (multi_grad_step)=${UTD}"
echo "  run_name=${run_name}"
echo "  pi0_checkpoint=${PI0_CKPT}"

# --------------------------
# COMMON RUNTIME CONFIG
# --------------------------

proj_name=DSRL_pi0_Aloha
device_id=0

export DISPLAY=:0
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=$device_id
export OPENPI_DATA_HOME=./openpi
export EXP=./logs/${proj_name}
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# --------------------------
# SINGLE RUN FOR THIS ARRAY INDEX
# --------------------------

python3 examples/launch_train_sim.py \
  --algorithm pixel_sac \
  --env aloha_cube \
  --prefix "${run_name}" \
  --launch_group_id aloha_basis_sweep \
  --wandb_project "${proj_name}" \
  --batch_size 256 \
  --discount 0.999 \
  --seed "${seed}" \
  --max_steps 500000 \
  --eval_interval 10000 \
  --log_interval 500 \
  --eval_episodes 10 \
  --multi_grad_step "${UTD}" \
  --start_online_updates 1000 \
  --resize_image 64 \
  --action_magnitude 1.0 \
  --residual_scale "${RS}" \
  --query_freq 50 \
  --hidden_dims 128 \
  --use_eigenbasis \
  --num_basis 8 \
  --warmup_rollouts 20 \
  --pi0_checkpoint "${PI0_CKPT}" \
  --output_dir "${EXP}/${run_name}"

echo "Completed g=${g} (residual_scale=${RS}, utd=${UTD})"
