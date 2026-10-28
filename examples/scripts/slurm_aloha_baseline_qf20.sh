#!/bin/bash

#SBATCH --job-name=aloha_base_qf20
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --array=0-11
#SBATCH --output=/global/home/users/r112358/logs/%x_%A_%a.out
#SBATCH --error=/global/home/users/r112358/logs/%x_%A_%a.err
#SBATCH --exclude=n0139.savio4,n0141.savio4,n0142.savio4

# Aloha cube baseline residual SAC. query_freq fixed at 20, UTD fixed at 10.
#   residual_scale in {0.05, 0.1, 0.2}
#   seeds 10, 101, 10001, 1000001
# One array task is one node and one seed.
#   0-3   rs=0.05  seeds 10, 101, 10001, 1000001
#   4-7   rs=0.1   seeds 10, 101, 10001, 1000001
#   8-11  rs=0.2   seeds 10, 101, 10001, 1000001
# 12 nodes, 12 runs. W&B project aloha_cube_debug.
# The W&B name omits the seed so the four seeds of one scale group together.
#
#   bash examples/scripts/submit_aloha_qf20_sweeps.sh

source ~/.bashrc
set -eo pipefail

DSRL_PY=/global/scratch/users/r112358/miniconda3/envs/dsrl_pi0/bin/python
export PATH="/global/scratch/users/r112358/miniconda3/envs/dsrl_pi0/bin:${PATH}"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/global/home/users/r112358/.mujoco/mujoco210/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib/nvidia"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /global/home/users/r112358/dsrl_pi0
REPO_ROOT="$(pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/LIBERO:${REPO_ROOT}/openpi/src:${PYTHONPATH:-}"

mkdir -p logs /global/home/users/r112358/logs

"${DSRL_PY}" -m pip install -q mujoco==2.3.7

SCRATCH_CKPT_ROOT=/global/scratch/users/r112358/pi0_checkpoint
PI0_CKPT="${SCRATCH_CKPT_ROOT}/pi0_aloha_sim"
if [ ! -d "${PI0_CKPT}" ]; then
  echo "Local OpenPI checkpoint missing; downloading to ${SCRATCH_CKPT_ROOT}"
  export OPENPI_DATA_HOME="${SCRATCH_CKPT_ROOT}"
  "${DSRL_PY}" - <<'PY'
from openpi.shared import download
print(download.maybe_download("s3://openpi-assets/checkpoints/pi0_aloha_sim"))
PY
  if [ -d "${OPENPI_DATA_HOME}/openpi-assets/checkpoints/pi0_aloha_sim" ]; then
    PI0_CKPT="${OPENPI_DATA_HOME}/openpi-assets/checkpoints/pi0_aloha_sim"
  elif [ -d "${OPENPI_DATA_HOME}/checkpoints/pi0_aloha_sim" ]; then
    PI0_CKPT="${OPENPI_DATA_HOME}/checkpoints/pi0_aloha_sim"
  fi
fi

SCALES=(0.05 0.1 0.2)
SEEDS=(10 101 10001 1000001)
QF=20
UTD=10
NSEEDS=${#SEEDS[@]}

g=${SLURM_ARRAY_TASK_ID}
if (( g >= ${#SCALES[@]} * NSEEDS )); then
  echo "Array index ${g} has no config."
  exit 1
fi

scale_i=$((g / NSEEDS))
seed_i=$((g % NSEEDS))
RS=${SCALES[$scale_i]}
SEED=${SEEDS[$seed_i]}
proj_name=aloha_cube_debug

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
ALLOC_GPU="${CUDA_VISIBLE_DEVICES:-0}"
ALLOC_GPU="${ALLOC_GPU%%,*}"
export MUJOCO_EGL_DEVICE_ID="${ALLOC_GPU}"
export OPENPI_DATA_HOME="${SCRATCH_CKPT_ROOT}"
export EXP=./logs/${proj_name}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export WANDB_X_SERVICE_WAIT=300

echo "branch=$(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "Array ${g}: baseline query_freq=${QF} residual_scale=${RS} utd=${UTD}"
echo "  seed=${SEED}"
echo "  cuda=${CUDA_VISIBLE_DEVICES:-unset}  egl=${MUJOCO_EGL_DEVICE_ID}"
echo "  pi0_checkpoint=${PI0_CKPT}"

if ! "${DSRL_PY}" -c 'import jax; devs=jax.devices(); print(devs); assert any(d.platform=="gpu" for d in devs), devs'; then
  echo "No GPU visible to JAX (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
  exit 1
fi

run_seed () {
  local seed=$1
  local run_name="aloha_baseline_qf${QF}_rs${RS}_utd${UTD}"
  echo "Launching ${run_name} seed=${seed}"
  "${DSRL_PY}" examples/launch_train_sim.py \
    --algorithm pixel_sac \
    --env aloha_cube \
    --prefix "${run_name}" \
    --launch_group_id "${run_name}" \
    --wandb_project "${proj_name}" \
    --batch_size 256 \
    --discount 0.999 \
    --seed "${seed}" \
    --max_steps 300000 \
    --eval_interval 10000 \
    --log_interval 500 \
    --eval_episodes 10 \
    --multi_grad_step "${UTD}" \
    --start_online_updates 1000 \
    --resize_image 64 \
    --action_magnitude 1.0 \
    --residual_scale "${RS}" \
    --query_freq "${QF}" \
    --hidden_dims 128 \
    --latent none \
    --warmup_rollouts 20 \
    --pi0_checkpoint "${PI0_CKPT}" \
    --output_dir "${EXP}/${run_name}_seed${seed}"
}

run_seed "${SEED}"
echo "Completed array ${g}: baseline rs=${RS} seed=${SEED}"
