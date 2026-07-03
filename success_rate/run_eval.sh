#!/bin/bash
set -euo pipefail

device_id="${DEVICE_ID:-3}"
config="${CONFIG:-pi05_libero}"
num_rollouts="${NUM_ROLLOUTS:-20}"
seed="${SEED:-10}"

export DISPLAY="${DISPLAY:-:0}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="$device_id"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-./openpi}"
export CUDA_VISIBLE_DEVICES="$device_id"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$(dirname "$0")/.."

python3 success_rate/eval_libero90.py \
  --config "$config" \
  --num_rollouts "$num_rollouts" \
  --seed "$seed" \
  "$@"
