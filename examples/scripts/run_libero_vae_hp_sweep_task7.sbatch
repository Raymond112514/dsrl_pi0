#!/bin/bash

#SBATCH --job-name=dsrl_vae_hp_sweep_t7
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Hyperparam sweep on Libero task 7 (VAE residual RL), max 200k steps.
# Grid: residual_scale {0.2,0.3} × vae_latent_dim {8,16} × vae_beta {1e-2,1e-3,1e-4}
#   → 2 × 2 × 3 = 12 runs (array 0-11)
#
# Submit from repo root:
#   mkdir -p logs
#   sbatch examples/scripts/run_libero_vae_hp_sweep_task7.sbatch

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

# Match openpi/examples/libero (3.3.x can make Franka colors look wrong).
pip install -q mujoco==3.2.3

mkdir -p logs

# --------------------------
# SWEEP GRID
# --------------------------

TASK_ID=7
SEED=10
UTD=10
QUERY_FREQ=5
WARMUP_ROLLOUTS=20
VAE_EPOCHS=120
MAX_STEPS=200000

RS_LIST=(0.2 0.3)
Z_LIST=(8 16)
BETA_LIST=(1e-2 1e-3 1e-4)

NUM_RS=${#RS_LIST[@]}
NUM_Z=${#Z_LIST[@]}
NUM_BETA=${#BETA_LIST[@]}
TOTAL_RUNS=$((NUM_RS * NUM_Z * NUM_BETA))

index_to_triple () {
  local g=$1
  local beta_idx=$((g % NUM_BETA))
  local tmp=$((g / NUM_BETA))
  local z_idx=$((tmp % NUM_Z))
  local rs_idx=$((tmp / NUM_Z))
  echo "${rs_idx} ${z_idx} ${beta_idx}"
}

g=${SLURM_ARRAY_TASK_ID}

if (( g >= TOTAL_RUNS )); then
  echo "g=${g} >= TOTAL_RUNS=${TOTAL_RUNS} → no configuration to run"
  exit 0
fi

read rs_idx z_idx beta_idx < <(index_to_triple "${g}")

RS=${RS_LIST[$rs_idx]}
Z=${Z_LIST[$z_idx]}
BETA=${BETA_LIST[$beta_idx]}

# Sanitize beta for filenames (1e-2 → 1e-2)
run_name="vae_task${TASK_ID}_z${Z}_rs${RS}_beta${BETA}_utd${UTD}_seed${SEED}"

echo "Running g=${g} / ${TOTAL_RUNS}"
echo "  task_id=${TASK_ID}"
echo "  seed=${SEED}"
echo "  residual_scale=${RS}"
echo "  vae_latent_dim=${Z}"
echo "  vae_beta=${BETA}"
echo "  utd=${UTD}"
echo "  max_steps=${MAX_STEPS}"
echo "  run_name=${run_name}"

# --------------------------
# COMMON RUNTIME CONFIG
# --------------------------

proj_name=pi05_residual_vae_hp
device_id=0

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id
export OPENPI_DATA_HOME=./openpi
export EXP=./logs/${proj_name}
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# --------------------------
# SINGLE RUN
# --------------------------

python3 examples/launch_train_sim_vae.py \
  --algorithm pixel_sac \
  --env libero \
  --prefix "${run_name}" \
  --launch_group_id vae_hp_sweep_task7 \
  --wandb_project "${proj_name}" \
  --batch_size 256 \
  --discount 0.999 \
  --seed "${SEED}" \
  --max_steps "${MAX_STEPS}" \
  --eval_interval 5000 \
  --log_interval 500 \
  --eval_episodes 20 \
  --task_id "${TASK_ID}" \
  --multi_grad_step "${UTD}" \
  --start_online_updates 500 \
  --resize_image 64 \
  --action_magnitude 1.0 \
  --residual_scale "${RS}" \
  --query_freq "${QUERY_FREQ}" \
  --hidden_dims 128 \
  --output_dir "${EXP}/${run_name}" \
  --vae_latent_dim "${Z}" \
  --vae_beta "${BETA}" \
  --warmup_rollouts "${WARMUP_ROLLOUTS}" \
  --vae_epochs "${VAE_EPOCHS}"

echo "Completed g=${g} (${run_name})"
