#!/bin/bash
proj_name=DSRL_pi0_Libero
device_id=3

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl  
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=./openpi
export EXP=./logs/$proj_name; 
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

pip install mujoco==3.2.3

python3 examples/launch_train_sim.py \
--algorithm pixel_sac \
--env libero \
--prefix dsrl_pi0_libero \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.999 \
--seed 10 \
--max_steps 500000  \
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
--output_dir ${EXP} \
--task_id 44

# Eigenbasis residual (PCA warmup + K-dim SAC):
# python3 examples/launch_train_sim.py ... --use_eigenbasis --num_basis 8 --warmup_rollouts 20
