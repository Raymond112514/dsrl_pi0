#!/bin/bash

#SBATCH --job-name=aloha_vae_qf20
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --array=0-1
#SBATCH --output=/global/home/users/r112358/logs/%x_%A_%a.out
#SBATCH --error=/global/home/users/r112358/logs/%x_%A_%a.err
#SBATCH --exclude=n0139.savio4,n0141.savio4,n0142.savio4

# Aloha cube unconditional VAE residual SAC.
#   query_freq = 20, K = 8, beta = 1e-3, 100 epochs
#   residual_scale = 0.2, UTD = 10
#   seeds 10, 101, 10001, 1000001
# One array task is one node: two seeds in parallel on the allocated GPU.
#   0  seeds 10, 101
#   1  seeds 10001, 1000001
# 2 nodes, 4 runs. W&B project aloha_vae_qf20.
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

SEEDS=(10 101 10001 1000001)
QF=20
RS=0.2
UTD=10
LATENT_DIM=8
BETA=0.001
EPOCHS=100

g=${SLURM_ARRAY_TASK_ID}
if (( g >= 2 )); then
  echo "Array index ${g} has no config."
  exit 1
fi

SEED0=${SEEDS[$((g * 2))]}
SEED1=${SEEDS[$((g * 2 + 1))]}
proj_name=aloha_vae_qf20

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
ALLOC_GPU="${CUDA_VISIBLE_DEVICES:-0}"
ALLOC_GPU="${ALLOC_GPU%%,*}"
export MUJOCO_EGL_DEVICE_ID="${ALLOC_GPU}"
export OPENPI_DATA_HOME="${SCRATCH_CKPT_ROOT}"
export EXP=./logs/${proj_name}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40

echo "branch=$(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)"
echo "Array ${g}: vae query_freq=${QF} K=${LATENT_DIM} beta=${BETA} epochs=${EPOCHS}"
echo "  residual_scale=${RS} utd=${UTD} seeds=${SEED0} ${SEED1}"
echo "  cuda=${CUDA_VISIBLE_DEVICES:-unset}  egl=${MUJOCO_EGL_DEVICE_ID}"
echo "  pi0_checkpoint=${PI0_CKPT}"

if ! "${DSRL_PY}" -c 'import jax; devs=jax.devices(); print(devs); assert any(d.platform=="gpu" for d in devs), devs'; then
  echo "No GPU visible to JAX (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
  exit 1
fi

run_seed () {
  local seed=$1
  local run_name="aloha_vae_qf${QF}_K${LATENT_DIM}_beta${BETA}_ep${EPOCHS}_rs${RS}_utd${UTD}_seed${seed}"
  echo "Launching ${run_name}"
  "${DSRL_PY}" examples/launch_train_sim.py \
    --algorithm pixel_sac \
    --env aloha_cube \
    --prefix "${run_name}" \
    --launch_group_id aloha_vae_qf20 \
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
    --latent vae \
    --vae_latent_dim "${LATENT_DIM}" \
    --vae_beta "${BETA}" \
    --vae_epochs "${EPOCHS}" \
    --warmup_rollouts 20 \
    --pi0_checkpoint "${PI0_CKPT}" \
    --output_dir "${EXP}/${run_name}"
}

run_seed "${SEED0}" &
pid0=$!
run_seed "${SEED1}" &
pid1=$!

status=0
wait "${pid0}" || status=1
wait "${pid1}" || status=1
echo "Completed array ${g}: vae beta=${BETA} seeds=${SEED0},${SEED1}"
exit "${status}"
